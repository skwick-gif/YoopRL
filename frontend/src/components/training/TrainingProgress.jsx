/**
 * TrainingProgress.jsx (Updated - Phase 1)
 * Part of: Training Tab
 * 
 * Purpose: Displays training progress, logs, and results
 * Components:
 * - Action Bar: Download data, start/stop training, load model, save config
 * - Training Progress: Real-time progress display with download/training status
 * - Training Results: Best trial, average reward, training time
 * 
 * Phase 1 Updates:
 * - Added polling for live training progress
 * - Integrated with trainingAPI.getProgress()
 * - Shows training ID for active sessions
 * - Added stopTraining functionality
 * - Displays real-time episode/reward/loss updates
 * 
 * Props:
 * - isDownloading: Boolean for download state
 * - isTraining: Boolean for training state
 * - downloadProgress: Number (0-100)
 * - trainingProgress: Number (0-100)
 * - dataDownloaded: Boolean - enables training button
 * - trainingId: String - active training session ID
 * - handleDownloadData: Function to download data
 * - handleStartTraining: Function to start training
 * 
 * Wiring:
 * - Polls getProgress API every 5 seconds during training
 * - Updates progress display with episode/reward/loss
 * - Calls stopTraining API when stop button clicked
 */

import React, { useEffect, useRef, useState } from 'react';
import { Button, Card } from '../common/UIComponents';
import { getProgress, stopTraining } from '../../services/trainingAPI';

function TrainingProgress({ 
  isDownloading, 
  isTraining, 
  downloadProgress, 
  trainingProgress, 
  dataDownloaded,
  downloadMessage, // NEW: Non-blocking message
  trainingId,
  handleDownloadData,
  handleStartTraining,
  onTrainingStatusChange,
  onTrainingStopped
}) {
  const [progressData, setProgressData] = useState(null);
  const lastStatusRef = useRef();
  const lastProgressRef = useRef();

  useEffect(() => {
    if (isTraining) {
      lastStatusRef.current = undefined;
      lastProgressRef.current = undefined;
    }
  }, [isTraining, trainingId]);

  const resolveMessageStyle = (message) => {
    if (!message) {
      return {};
    }

    if (message.startsWith('✅')) {
      return {
        backgroundColor: '#1e4620',
        border: '1px solid #4CAF50',
        color: '#4CAF50'
      };
    }

    if (message.startsWith('❌')) {
      return {
        backgroundColor: '#4d1f1f',
        border: '1px solid #f44336',
        color: '#f44336'
      };
    }

    if (message.startsWith('⚠️')) {
      return {
        backgroundColor: 'rgba(187, 128, 9, 0.12)',
        border: '1px solid rgba(187, 128, 9, 0.6)',
        color: '#d29922'
      };
    }

    if (message.startsWith('🚀') || message.startsWith('⏳')) {
      return {
        backgroundColor: 'rgba(31, 111, 235, 0.12)',
        border: '1px solid rgba(88, 166, 255, 0.8)',
        color: '#58a6ff'
      };
    }

    return {
      backgroundColor: '#2d2d2d',
      border: '1px solid #444',
      color: '#e0e0e0'
    };
  };

  // Poll for training progress when training is active
  useEffect(() => {
    if (!isTraining || !trainingId) {
      return;
    }

    const pollProgress = async () => {
      const result = await getProgress(trainingId);
      if (result.success) {
        setProgressData(result.data);
      }
    };

    // Initial fetch
    pollProgress();

    // Poll every 5 seconds
    const interval = setInterval(pollProgress, 5000);

    return () => clearInterval(interval);
  }, [isTraining, trainingId]);

  useEffect(() => {
    if (!progressData) {
      return;
    }

    const { status, progress } = progressData;
    const numericProgress = typeof progress === 'number' ? progress : undefined;

    const hasStatusChanged = status && lastStatusRef.current !== status;
    const hasProgressChanged = typeof numericProgress === 'number' && lastProgressRef.current !== numericProgress;

    if ((hasStatusChanged || hasProgressChanged) && typeof onTrainingStatusChange === 'function') {
      onTrainingStatusChange({ status, progress: numericProgress });
    }

    if (hasStatusChanged) {
      lastStatusRef.current = status;
    }
    if (hasProgressChanged) {
      lastProgressRef.current = numericProgress;
    }

  }, [progressData, onTrainingStatusChange]);

  // Handle stop training
  const handleStopTraining = async () => {
    if (!trainingId) return;

    const result = await stopTraining(trainingId);
    if (result.success) {
      alert('Training stopped successfully');
      if (typeof onTrainingStatusChange === 'function') {
        onTrainingStatusChange({ status: 'stopped', progress: progressData?.progress });
      }
      if (typeof onTrainingStopped === 'function') {
        onTrainingStopped();
      }
    } else {
      alert(`Failed to stop training: ${result.error}`);
    }
  };
  return (
    <>
      {/* Action Bar - Main control buttons */}
      <div className="action-bar">
        {/* Download Data Button: 
            - Downloads historical market data for the selected symbol
            - Downloads all selected features (Price, Volume, RSI, MACD, etc.)
            - Downloads alternative data if selected (Fundamental, Sentiment, News, etc.)
            - Shows download progress in the Training Progress section below
            - Must complete successfully before training can start
        */}
        <Button 
          variant="secondary"
          onClick={handleDownloadData}
          disabled={isDownloading || isTraining}
        >
          {isDownloading ? 'Downloading...' : 'Download Training Data'}
        </Button>
        
        {/* Start Training Button:
            - Begins the training process using downloaded data
            - Runs Optuna trials to find best hyperparameters
            - Shows training progress, episodes, reward, and loss below
            - Can take hours depending on episodes and trials
            - ONLY ENABLED after data download completes successfully
        */}
        <Button 
          variant={dataDownloaded && !isTraining ? "primary" : "secondary"}
          onClick={handleStartTraining}
          disabled={!dataDownloaded || isDownloading || isTraining}
          style={{ 
            opacity: dataDownloaded && !isTraining ? 1 : 0.5,
            cursor: dataDownloaded && !isTraining ? 'pointer' : 'not-allowed'
          }}
        >
          {isTraining ? 'Training...' : 'Start Training'}
        </Button>
        
        {/* Stop Training Button: Halts training immediately */}
        <Button 
          variant="secondary" 
          onClick={handleStopTraining}
          disabled={!isTraining}
        >
          Stop Training
        </Button>
        
        {/* Load Model Button:
            - Loads a previously saved trained model from disk
            - Allows you to continue training or use for live trading
            - Skip training if you already have a good model
        */}
        <Button 
          variant="secondary"
          onClick={() => {
            // This will be handled by parent component (TabTraining)
            // by showing ModelSelector dropdown
            if (window.onLoadModelClick) {
              window.onLoadModelClick();
            } else {
              alert('Model selection will appear below. Scroll down to see "Select Trained Model" dropdown.');
            }
          }}
        >
          Load Model
        </Button>
        
        {/* Save Config Button: Saves current hyperparameter configuration */}
        <Button variant="secondary">Save Config</Button>
      </div>

      {/* Training Progress Display */}
      <Card style={{ marginTop: '12px' }}>
        <div className="control-title">Training Progress</div>
        <div className="log-display" style={{ maxHeight: '200px' }}>
          
          {/* Non-blocking Download Message */}
          {downloadMessage && (
            <div style={{
              padding: '12px',
              marginBottom: '10px',
              borderRadius: '4px',
              fontWeight: 600,
              fontSize: '14px',
              animation: 'fadeIn 0.3s ease-in',
              whiteSpace: 'pre-line',
              ...resolveMessageStyle(downloadMessage)
            }}>
              {downloadMessage}
            </div>
          )}
          
          {/* Training ID Display */}
          {trainingId && (
            <div className="log-info" style={{ color: '#4CAF50', fontWeight: 600 }}>
              Training ID: {trainingId}
            </div>
          )}

          {/* Download Progress Display */}
          {isDownloading && (
            <>
              <div className="log-info" style={{ fontWeight: 600 }}>
                ⬇️ Downloading Data... {downloadProgress}%
              </div>
              <div style={{ 
                width: '100%', 
                height: '8px', 
                background: '#21262d', 
                borderRadius: '4px', 
                overflow: 'hidden',
                margin: '8px 0' 
              }}>
                <div style={{ 
                  width: `${downloadProgress}%`, 
                  height: '100%', 
                  background: '#58a6ff',
                  transition: 'width 0.3s ease'
                }}></div>
              </div>
              <div className="log-debug">
                {downloadProgress < 100 
                  ? `Fetching historical data and selected features...`
                  : `✓ Download complete - Ready to train`
                }
              </div>
            </>
          )}

          {/* Training Progress Display */}
          {isTraining && progressData && (
            <>
              <div className="log-info">[{new Date().toLocaleTimeString()}] Training started: {progressData.status}</div>
              <div className="log-info" style={{ fontWeight: 600, marginTop: '8px' }}>
                🔄 Training... {progressData.progress || 0}%
              </div>
              <div style={{ 
                width: '100%', 
                height: '8px', 
                background: '#21262d', 
                borderRadius: '4px', 
                overflow: 'hidden',
                margin: '8px 0' 
              }}>
                <div style={{ 
                  width: `${progressData.progress || 0}%`, 
                  height: '100%', 
                  background: '#3fb950',
                  transition: 'width 0.3s ease'
                }}></div>
              </div>
              <div className="log-info">Episode: {progressData.current_episode?.toLocaleString() || 0} / {progressData.total_episodes?.toLocaleString() || 0}</div>
              <div className="log-debug">Avg Reward: {progressData.avg_reward?.toFixed(2) || 'N/A'} | Loss: {progressData.recent_loss?.toFixed(4) || 'N/A'}</div>
              <div className="log-debug">Elapsed: {progressData.elapsed_time || 'N/A'} | ETA: {progressData.eta || 'Calculating...'}</div>
              {progressData.progress === 100 && (
                <div className="log-success">✓ Training complete</div>
              )}
            </>
          )}

          {isTraining && !progressData && (
            <>
              <div className="log-info">[{new Date().toLocaleTimeString()}] Training started</div>
              <div className="log-debug">Waiting for progress updates...</div>
            </>
          )}

          {/* Initial State - Instructions */}
          {!isDownloading && !isTraining && !dataDownloaded && (
            <>
              <div className="log-info">⚠️ Click "Download Training Data" to begin</div>
              <div className="log-debug">Training button will be enabled after download completes</div>
            </>
          )}

          {/* Data Ready State */}
          {!isDownloading && !isTraining && dataDownloaded && (
            <>
              <div className="log-success">✓ Data downloaded and ready for training</div>
              <div className="log-info">Click "Start Training" to begin optimization</div>
            </>
          )}
        </div>
      </Card>

      {/* Training Results Summary - Only show when training is complete and has real results */}
      {trainingProgress === 100 && trainingId && (
        <Card style={{ marginTop: '12px' }}>
          <div className="control-title">Training Results</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
            {/* Best Optuna Trial */}
            <Card style={{ background: '#0d1117', padding: '10px' }}>
              <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '4px' }}>Best Trial</div>
              <div style={{ fontSize: '16px', fontWeight: 600 }}>Waiting for results...</div>
              <div style={{ fontSize: '10px', color: '#6e7681', marginTop: '2px' }}>Check logs for details</div>
            </Card>
            
            {/* Average Reward */}
            <Card style={{ background: '#0d1117', padding: '10px' }}>
              <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '4px' }}>Avg Reward</div>
              <div style={{ fontSize: '16px', fontWeight: 600, color: '#3fb950' }}>N/A</div>
              <div style={{ fontSize: '10px', color: '#6e7681', marginTop: '2px' }}>Last 1000 episodes</div>
            </Card>
            
            {/* Training Time */}
            <Card style={{ background: '#0d1117', padding: '10px' }}>
              <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '4px' }}>Training Time</div>
              <div style={{ fontSize: '16px', fontWeight: 600 }}>N/A</div>
              <div style={{ fontSize: '10px', color: '#6e7681', marginTop: '2px' }}>Completed: N/A</div>
            </Card>
          </div>
        </Card>
      )}
    </>
  );
}

export default TrainingProgress;
