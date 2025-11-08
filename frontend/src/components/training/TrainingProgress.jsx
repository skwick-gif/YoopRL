/**
 * TrainingProgress.jsx
 * Part of: Training Tab
 * 
 * Purpose: Displays training progress, logs, and results
 * Components:
 * - Action Bar: Download data, start/stop training, load model, save config
 * - Training Progress: Real-time progress display with download/training status
 * - Training Results: Best trial, average reward, training time
 * 
 * State Management:
 * - Download progress: Tracks data download completion
 * - Training progress: Tracks Optuna trials and episode completion
 * - Data Downloaded flag: Controls when training can start
 */

import React from 'react';
import { Button, Card } from '../common/UIComponents';

function TrainingProgress({ 
  isDownloading, 
  isTraining, 
  downloadProgress, 
  trainingProgress, 
  dataDownloaded,
  handleDownloadData,
  handleStartTraining 
}) {
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
        <Button variant="secondary" disabled={isTraining}>Stop Training</Button>
        
        {/* Load Model Button:
            - Loads a previously saved trained model from disk
            - Allows you to continue training or use for live trading
            - Skip training if you already have a good model
        */}
        <Button variant="secondary">Load Model</Button>
        
        {/* Save Config Button: Saves current hyperparameter configuration */}
        <Button variant="secondary">Save Config</Button>
      </div>

      {/* Training Progress Display */}
      <Card style={{ marginTop: '12px' }}>
        <div className="control-title">Training Progress</div>
        <div className="log-display" style={{ maxHeight: '200px' }}>
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
          {isTraining && (
            <>
              <div className="log-info">[{new Date().toLocaleTimeString()}] Training started: Selected agents</div>
              <div className="log-info" style={{ fontWeight: 600, marginTop: '8px' }}>
                🔄 Training... {trainingProgress}%
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
                  width: `${trainingProgress}%`, 
                  height: '100%', 
                  background: '#3fb950',
                  transition: 'width 0.3s ease'
                }}></div>
              </div>
              <div className="log-info">[{new Date().toLocaleTimeString()}] Running Optuna optimization trials...</div>
              <div className="log-debug">Episodes running, optimizing hyperparameters...</div>
              {trainingProgress === 100 && (
                <div className="log-success">✓ Training complete</div>
              )}
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

          {/* Example Log (shown only when not actively downloading/training) */}
          {!isDownloading && !isTraining && (
            <>
              <div className="log-info" style={{ marginTop: '12px', opacity: 0.5 }}>[2024-11-07 14:30:15] Example: Training started: PPO Agent on AAPL</div>
              <div className="log-info" style={{ opacity: 0.5 }}>[2024-11-07 14:30:20] Optuna Study: Trial 1/100</div>
              <div className="log-info" style={{ opacity: 0.5 }}>[2024-11-07 14:30:45] Episode 100/50000 | Reward: 0.42 | Loss: 0.023</div>
              <div className="log-success" style={{ opacity: 0.5 }}>[2024-11-07 14:31:40] Trial 1 complete | Score: 0.65</div>
            </>
          )}
        </div>
      </Card>

      {/* Training Results Summary */}
      <Card style={{ marginTop: '12px' }}>
        <div className="control-title">Training Results</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
          {/* Best Optuna Trial */}
          <Card style={{ background: '#0d1117', padding: '10px' }}>
            <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '4px' }}>Best Optuna Trial</div>
            <div style={{ fontSize: '16px', fontWeight: 600 }}>Trial #47</div>
            <div style={{ fontSize: '10px', color: '#6e7681', marginTop: '2px' }}>LR: 0.00025 | γ: 0.98</div>
          </Card>
          
          {/* Average Reward */}
          <Card style={{ background: '#0d1117', padding: '10px' }}>
            <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '4px' }}>Avg Reward</div>
            <div style={{ fontSize: '16px', fontWeight: 600, color: '#3fb950' }}>+0.87</div>
            <div style={{ fontSize: '10px', color: '#6e7681', marginTop: '2px' }}>Last 1000 episodes</div>
          </Card>
          
          {/* Training Time */}
          <Card style={{ background: '#0d1117', padding: '10px' }}>
            <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '4px' }}>Training Time</div>
            <div style={{ fontSize: '16px', fontWeight: 600 }}>2h 34m</div>
            <div style={{ fontSize: '10px', color: '#6e7681', marginTop: '2px' }}>Completed: 05/11/2024</div>
          </Card>
        </div>
      </Card>
    </>
  );
}

export default TrainingProgress;
