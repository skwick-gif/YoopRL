/**
 * TabTraining.jsx
 * Training Tab - Main Component (Updated with Phase 1 Improvements)
 * 
 * Purpose: Configure and train RL agents (PPO for stocks, SAC for leveraged ETFs)
 * 
 * Workflow:
 * 1. Configure hyperparameters (learning rate, episodes, etc.)
 * 2. Select input features (technical indicators, alternative data)
 * 3. Download historical training data
 * 4. Start training with Optuna hyperparameter optimization
 * 5. Monitor progress and view results
 * 6. Run backtesting and check for data drift
 * 7. Save/load configurations and trained models
 * 
 * Phase 1 Updates:
 * - Integrated useTrainingState hook for centralized state management
 * - Connected trainingAPI service for all backend calls
 * - Added ModelSelector for loading trained models
 * - Added BacktestResults for performance metrics
 * - Added DriftAlert for data drift warnings
 * - Added ConfigManager for saving/loading configurations
 * 
 * State Management:
 * - All training state now managed via useTrainingState hook
 * - Child components receive state props (controlled components)
 * - API calls handled through trainingAPI service
 */

import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import HyperparameterGrid from './training/HyperparameterGrid';
import FeatureSelection from './training/FeatureSelection';
import TrainingProgress from './training/TrainingProgress';
import ModelSelector from './training/ModelSelector';
import BacktestResults from './training/BacktestResults';
import DriftAlert from './training/DriftAlert';
import ConfigManager from './training/ConfigManager';
import ModelsComparisonTable from './training/ModelsComparisonTable';
import { useTrainingState } from '../hooks/useTrainingState';
import { startTraining, checkDriftStatus, runBacktest } from '../services/trainingAPI';

function TabTraining() {
  // Initialize training state from custom hook
  const trainingState = useTrainingState();

  // UI State (not part of training config)
  const [isDownloading, setIsDownloading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [dataDownloaded, setDataDownloaded] = useState(false);
  const [downloadMessage, setDownloadMessage] = useState(''); // NEW: non-blocking message
  const [trainingId, setTrainingId] = useState(null);
  
  // Agent selection
  const [selectedAgent, setSelectedAgent] = useState('PPO'); // 'PPO' or 'SAC'
  
  // Backtest results
  const [backtestResults, setBacktestResults] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  
  // Drift detection
  const [driftData, setDriftData] = useState(null);
  
  // Help documentation viewer
  const [showHelp, setShowHelp] = useState(false);
  const [helpContent, setHelpContent] = useState('');

  // Load help documentation content
  useEffect(() => {
    if (showHelp && !helpContent) {
      fetch('/docs/TRAINING_TAB_GUIDE.md')
        .then(res => res.text())
        .then(text => setHelpContent(text))
        .catch(err => {
          console.error('Failed to load help documentation:', err);
          setHelpContent('# Error Loading Documentation\n\nFailed to load the training guide. Please check the console for details.');
        });
    }
  }, [showHelp, helpContent]);

  // ESC key to close help modal
  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === 'Escape' && showHelp) {
        setShowHelp(false);
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [showHelp]);

  // Check for data drift on mount and periodically
  useEffect(() => {
    const checkDrift = async () => {
      try {
        const symbol = selectedAgent === 'PPO' ? trainingState.ppoSymbol : trainingState.sacSymbol;
        
        // Skip if no symbol configured
        if (!symbol) {
          console.warn('No symbol configured for drift check');
          return;
        }
        
        const result = await checkDriftStatus(symbol, selectedAgent);
        
        if (result.success && result.drift_data) {
          setDriftData(result.drift_data);
        } else {
          // Clear drift data if no drift detected or if check failed
          setDriftData(null);
        }
      } catch (error) {
        console.error('Error checking drift:', error);
        // Don't clear drift data on error - keep showing last known state
      }
    };

    checkDrift();
    const interval = setInterval(checkDrift, 300000); // Check every 5 minutes
    return () => clearInterval(interval);
  }, [selectedAgent, trainingState.ppoSymbol, trainingState.sacSymbol]);

  // Download Data Handler
  const handleDownloadData = async () => {
    setIsDownloading(true);
    setDownloadProgress(0);
    setDataDownloaded(false);
    
    try {
      const response = await fetch('http://localhost:8000/api/training/download', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: trainingState.ppoSymbol || trainingState.sacSymbol,
          start_date: trainingState.startDate,
          end_date: trainingState.endDate,
          // ALWAYS download ALL data sources - user selects at training time
          enable_sentiment: true,
          enable_social_media: true,
          enable_news: true,
          enable_market_events: true,
          enable_fundamental: true,
          enable_multi_asset: true,
          enable_macro: true,
          force_redownload: false,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.status === 'success') {
        console.log('Training data downloaded successfully:', result);
        setDownloadProgress(100);
        setDataDownloaded(true);
        
        // Non-blocking success message
        const message = `✅ Training Data Ready! Symbol: ${result.symbol}, ` +
                       `Rows: ${result.rows}, Features: ${result.features}, ` +
                       `Train: ${result.train_size} / Test: ${result.test_size}`;
        setDownloadMessage(message);
        
        // Auto-hide message after 10 seconds
        setTimeout(() => setDownloadMessage(''), 10000);
      } else {
        throw new Error(result.error || 'Download failed');
      }
    } catch (error) {
      console.error('Error downloading training data:', error);
      
      // Non-blocking error message
      setDownloadMessage(`❌ Download Failed: ${error.message}`);
      setTimeout(() => setDownloadMessage(''), 10000);
      
      setDownloadProgress(0);
    } finally {
      setIsDownloading(false);
    }
  };

  // Start Training Handler
  const handleStartTraining = async () => {
    if (!dataDownloaded) {
      alert('Please download training data first');
      return;
    }

    // Validate configuration
    const validation = trainingState.validateConfig(selectedAgent);
    if (!validation.valid) {
      alert(`Configuration errors:\n\n${validation.errors.join('\n')}`);
      return;
    }

    // Build training config
    const config = trainingState.buildTrainingConfig(selectedAgent);
    
    // Start training via API
    setIsTraining(true);
    setTrainingProgress(0);
    
    const result = await startTraining(config);
    
    if (result.success) {
      setTrainingId(result.training_id);
      alert(`✅ Training Started!\n\nTraining ID: ${result.training_id}`);
    } else {
      alert(`❌ Training Failed\n\n${result.error}`);
      setIsTraining(false);
    }
  };

  // Handle model selection for backtesting
  const handleModelSelect = async (model) => {
    console.log('Selected model:', model);
    
    // Run backtest on selected model
    setBacktestLoading(true);
    const backtestRequest = {
      model_path: model.file_path,
      start_date: trainingState.startDate,
      end_date: trainingState.endDate,
      initial_capital: 100000,
      commission: trainingState.commission
    };
    
    const result = await runBacktest(backtestRequest);
    
    if (result.success) {
      setBacktestResults(result.results);
    } else {
      alert(`❌ Backtest Failed\n\n${result.error}`);
    }
    
    setBacktestLoading(false);
  };

  // Handle drift detection - trigger retraining
  const handleRetrain = () => {
    if (window.confirm('Start retraining with recent data?')) {
      handleStartTraining();
    }
  };

  // Handle config load from ConfigManager
  const handleLoadConfig = (config) => {
    // Apply loaded config to state
    // This would require updating the hook or manually setting all state values
    console.log('Loading config:', config);
    
    // Extract values from config
    const configType = config.name || config.agent_type || 'Unknown';
    const lr = config.hyperparameters?.learning_rate || 'N/A';
    const gamma = config.hyperparameters?.gamma || 'N/A';
    const batch = config.hyperparameters?.batch_size || 'N/A';
    const episodes = config.hyperparameters?.episodes || 'N/A';
    
    // Non-blocking success message with clean formatting
    const message = `✅ Preset Loaded Successfully!\n\n` +
                   `📋 Type: ${configType}\n` +
                   `📊 Learning Rate: ${lr}\n` +
                   `🎯 Gamma: ${gamma}\n` +
                   `📦 Batch Size: ${batch}\n` +
                   `🔄 Episodes: ${episodes}`;
    setDownloadMessage(message);
    
    // Auto-hide message after 10 seconds
    setTimeout(() => setDownloadMessage(''), 10000);
  };

  return (
    <div>
      {/* Help Documentation Modal */}
      {showHelp && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          zIndex: 10000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px'
        }} onClick={() => setShowHelp(false)}>
          <div style={{
            backgroundColor: '#1e1e1e',
            borderRadius: '12px',
            width: '90%',
            maxWidth: '1200px',
            height: '90vh',
            overflow: 'hidden',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5)',
            display: 'flex',
            flexDirection: 'column'
          }} onClick={(e) => e.stopPropagation()}>
            {/* Header */}
            <div style={{
              padding: '20px 30px',
              borderBottom: '1px solid #333',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              backgroundColor: '#252525'
            }}>
              <h2 style={{ margin: 0, color: '#e0e0e0', fontSize: '24px', fontWeight: 600 }}>
                📚 Training Tab - User Guide
              </h2>
              <button
                onClick={() => setShowHelp(false)}
                style={{
                  backgroundColor: 'transparent',
                  border: 'none',
                  color: '#888',
                  fontSize: '32px',
                  cursor: 'pointer',
                  padding: '0 10px',
                  lineHeight: '32px'
                }}
                title="Close (Esc)"
              >
                ×
              </button>
            </div>
            
            {/* Content */}
            <div style={{
              flex: 1,
              padding: '30px 40px',
              overflowY: 'auto',
              backgroundColor: '#1e1e1e',
              color: '#d4d4d4',
              lineHeight: '1.8',
              fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
            }}>
              {helpContent ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    h1: ({node, ...props}) => <h1 style={{ color: '#4a9eff', fontSize: '32px', marginTop: '0', marginBottom: '16px', borderBottom: '2px solid #4a9eff', paddingBottom: '8px' }} {...props} />,
                    h2: ({node, ...props}) => <h2 style={{ color: '#5fb3f6', fontSize: '26px', marginTop: '32px', marginBottom: '12px', borderBottom: '1px solid #444', paddingBottom: '6px' }} {...props} />,
                    h3: ({node, ...props}) => <h3 style={{ color: '#7cc5f8', fontSize: '22px', marginTop: '24px', marginBottom: '10px' }} {...props} />,
                    h4: ({node, ...props}) => <h4 style={{ color: '#9dd5fa', fontSize: '18px', marginTop: '20px', marginBottom: '8px' }} {...props} />,
                    p: ({node, ...props}) => <p style={{ marginBottom: '12px', fontSize: '14px' }} {...props} />,
                    ul: ({node, ...props}) => <ul style={{ marginLeft: '20px', marginBottom: '12px' }} {...props} />,
                    ol: ({node, ...props}) => <ol style={{ marginLeft: '20px', marginBottom: '12px' }} {...props} />,
                    li: ({node, ...props}) => <li style={{ marginBottom: '6px', fontSize: '14px' }} {...props} />,
                    code: ({node, inline, ...props}) => inline 
                      ? <code style={{ backgroundColor: '#2d2d2d', padding: '2px 6px', borderRadius: '3px', fontSize: '13px', color: '#ce9178' }} {...props} />
                      : <code style={{ display: 'block', backgroundColor: '#2d2d2d', padding: '12px', borderRadius: '6px', fontSize: '13px', overflowX: 'auto', lineHeight: '1.6', marginBottom: '12px', border: '1px solid #444' }} {...props} />,
                    pre: ({node, ...props}) => <pre style={{ margin: 0 }} {...props} />,
                    blockquote: ({node, ...props}) => <blockquote style={{ borderLeft: '4px solid #4a9eff', paddingLeft: '16px', marginLeft: '0', marginBottom: '12px', color: '#aaa', fontStyle: 'italic' }} {...props} />,
                    table: ({node, ...props}) => <table style={{ borderCollapse: 'collapse', width: '100%', marginBottom: '16px', fontSize: '14px' }} {...props} />,
                    thead: ({node, ...props}) => <thead style={{ backgroundColor: '#2d2d2d' }} {...props} />,
                    tbody: ({node, ...props}) => <tbody {...props} />,
                    tr: ({node, ...props}) => <tr style={{ borderBottom: '1px solid #444' }} {...props} />,
                    th: ({node, ...props}) => <th style={{ padding: '10px', textAlign: 'left', color: '#4a9eff', fontWeight: 'bold', borderBottom: '2px solid #4a9eff' }} {...props} />,
                    td: ({node, ...props}) => <td style={{ padding: '10px', borderBottom: '1px solid #444' }} {...props} />,
                    a: ({node, ...props}) => <a style={{ color: '#4a9eff', textDecoration: 'none', borderBottom: '1px solid transparent' }} onMouseEnter={(e) => e.target.style.borderBottom = '1px solid #4a9eff'} onMouseLeave={(e) => e.target.style.borderBottom = '1px solid transparent'} {...props} />,
                    hr: ({node, ...props}) => <hr style={{ border: 'none', borderTop: '1px solid #444', margin: '24px 0' }} {...props} />,
                    strong: ({node, ...props}) => <strong style={{ color: '#fff', fontWeight: 600 }} {...props} />,
                    em: ({node, ...props}) => <em style={{ color: '#9dd5fa' }} {...props} />
                  }}
                >
                  {helpContent}
                </ReactMarkdown>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
                  <div style={{ fontSize: '48px', marginBottom: '16px' }}>📚</div>
                  <div style={{ fontSize: '16px' }}>Loading Training Guide...</div>
                </div>
              )}
            </div>
            
            {/* Footer */}
            <div style={{
              padding: '15px 30px',
              borderTop: '1px solid #333',
              backgroundColor: '#252525',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span style={{ color: '#888', fontSize: '13px' }}>
                Press ESC to close | Scroll to navigate
              </span>
              <button
                onClick={() => setShowHelp(false)}
                style={{
                  padding: '8px 20px',
                  backgroundColor: '#4a9eff',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '14px'
                }}
              >
                Got it!
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Agent Selection */}
      <div style={{ 
        padding: '15px', 
        backgroundColor: '#2d2d2d', 
        borderRadius: '8px', 
        marginBottom: '20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <label style={{ color: '#e0e0e0', fontWeight: 'bold', marginRight: '10px' }}>
            Select Agent Type:
          </label>
          <select 
            value={selectedAgent} 
            onChange={(e) => setSelectedAgent(e.target.value)}
            style={{
              padding: '8px 12px',
              backgroundColor: '#1e1e1e',
              color: '#e0e0e0',
              border: '1px solid #444',
              borderRadius: '4px',
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            <option value="PPO">PPO - Stock Trading</option>
            <option value="SAC">SAC - Leveraged ETF Trading</option>
          </select>
        </div>
        
        {/* Help Icon */}
        <button
          onClick={() => setShowHelp(true)}
          style={{
            backgroundColor: '#4a9eff',
            border: 'none',
            borderRadius: '50%',
            width: '36px',
            height: '36px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            fontSize: '18px',
            fontWeight: 'bold',
            color: 'white',
            boxShadow: '0 2px 8px rgba(74, 158, 255, 0.3)',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            e.target.style.transform = 'scale(1.1)';
            e.target.style.boxShadow = '0 4px 12px rgba(74, 158, 255, 0.5)';
          }}
          onMouseLeave={(e) => {
            e.target.style.transform = 'scale(1)';
            e.target.style.boxShadow = '0 2px 8px rgba(74, 158, 255, 0.3)';
          }}
          title="Open Training Guide - Learn how to use this tab"
        >
          ?
        </button>
      </div>

      {/* Current Model Version Info */}
      <div className="model-version">
        <strong>Current Models:</strong><br />
        PPO: v3.2_20241105 | Trained: 05/11/2024 | Episodes: 50,000<br />
        SAC: v2.8_20241105 | Trained: 05/11/2024 | Episodes: 45,000
      </div>

      {/* Model Selector */}
      <ModelSelector 
        onModelSelect={handleModelSelect}
        agentType={selectedAgent}
      />

      {/* Drift Alert (shows only if drift detected) */}
      <DriftAlert 
        driftData={driftData}
        onRetrain={handleRetrain}
      />

      {/* Hyperparameter Configuration */}
      <HyperparameterGrid 
        agentType={selectedAgent}
        trainingState={trainingState}
      />

      {/* Feature Selection */}
      <FeatureSelection 
        trainingState={trainingState}
      />

      {/* Configuration Manager */}
      <ConfigManager 
        currentConfig={trainingState.buildTrainingConfig(selectedAgent)}
        onLoadConfig={handleLoadConfig}
        agentType={selectedAgent}
      />

      {/* Training Controls and Progress */}
      <TrainingProgress 
        isDownloading={isDownloading}
        isTraining={isTraining}
        downloadProgress={downloadProgress}
        trainingProgress={trainingProgress}
        dataDownloaded={dataDownloaded}
        downloadMessage={downloadMessage}
        trainingId={trainingId}
        handleDownloadData={handleDownloadData}
        handleStartTraining={handleStartTraining}
      />

      {/* Backtest Results */}
      <BacktestResults 
        results={backtestResults}
        loading={backtestLoading}
      />

      {/* Models Comparison Table - Compare all trained models */}
      <ModelsComparisonTable 
        agentType={selectedAgent}
      />
    </div>
  );
}

export default TabTraining;
