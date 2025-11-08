/**
 * TabTraining.jsx
 * Training Tab - Main Component
 * 
 * Purpose: Configure and train RL agents (PPO for stocks, SAC for leveraged ETFs)
 * 
 * Workflow:
 * 1. Configure hyperparameters (learning rate, episodes, etc.)
 * 2. Select input features (technical indicators, alternative data)
 * 3. Download historical training data
 * 4. Start training with Optuna hyperparameter optimization
 * 5. Monitor progress and view results
 * 
 * Sub-components:
 * - HyperparameterGrid: PPO/SAC hyperparameters and training settings
 * - FeatureSelection: Choose which features to train on
 * - TrainingProgress: Download/training progress, action buttons, results
 */

import React, { useState } from 'react';
import HyperparameterGrid from './training/HyperparameterGrid';
import FeatureSelection from './training/FeatureSelection';
import TrainingProgress from './training/TrainingProgress';

function TabTraining() {
  // LLM Integration State
  const [llmEnabled, setLlmEnabled] = useState(false);
  const [selectedLLM, setSelectedLLM] = useState('Perplexity API');
  
  // Download and Training State Management
  const [isDownloading, setIsDownloading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [dataDownloaded, setDataDownloaded] = useState(false); // Tracks if data is ready for training

  // Download Data Handler (connected to backend API)
  const handleDownloadData = async () => {
    setIsDownloading(true);
    setDownloadProgress(0);
    setDataDownloaded(false);
    
    try {
      // Call backend API to download training data
      const response = await fetch('http://localhost:8000/api/training/download', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: 'IWM',  // Default symbol, can be made configurable later
          period: '5y',   // 5 years of historical data
          enable_sentiment: false,  // Based on feature selection
          enable_multi_asset: false,  // Based on feature selection
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
        alert(`✅ Training Data Ready!\n\n` +
              `Symbol: ${result.symbol}\n` +
              `Rows: ${result.rows}\n` +
              `Features: ${result.features}\n` +
              `Train: ${result.train_size} / Test: ${result.test_size}`);
      } else {
        throw new Error(result.error || 'Download failed');
      }
    } catch (error) {
      console.error('Error downloading training data:', error);
      alert(`❌ Download Failed\n\n${error.message}\n\nPlease check:\n1. Backend API is running (port 8000)\n2. Internet connection is working\n3. Symbol is valid`);
      setDownloadProgress(0);
    } finally {
      setIsDownloading(false);
    }
  };

  // Start Training Handler (will be connected to backend API)
  const handleStartTraining = () => {
    if (!dataDownloaded) return; // Safety check - should not happen due to disabled button
    
    setIsTraining(true);
    setTrainingProgress(0);
    // Backend will handle Optuna trials and training episodes
  };

  return (
    <div>
      {/* Current Model Version Info */}
      <div className="model-version">
        <strong>Current Models:</strong><br />
        PPO: v3.2_20241105 | Trained: 05/11/2024 | Episodes: 50,000<br />
        SAC: v2.8_20241105 | Trained: 05/11/2024 | Episodes: 45,000
      </div>

      {/* Hyperparameter Configuration */}
      <HyperparameterGrid />

      {/* Feature Selection */}
      <FeatureSelection 
        llmEnabled={llmEnabled}
        setLlmEnabled={setLlmEnabled}
        selectedLLM={selectedLLM}
        setSelectedLLM={setSelectedLLM}
      />

      {/* Training Controls and Progress */}
      <TrainingProgress 
        isDownloading={isDownloading}
        isTraining={isTraining}
        downloadProgress={downloadProgress}
        trainingProgress={trainingProgress}
        dataDownloaded={dataDownloaded}
        handleDownloadData={handleDownloadData}
        handleStartTraining={handleStartTraining}
      />
    </div>
  );
}

export default TabTraining;
