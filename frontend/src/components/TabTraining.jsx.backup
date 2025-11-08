import React, { useState } from 'react';
import { Button, ParamItem, Card } from './common/UIComponents';

function TabTraining() {
  const [llmEnabled, setLlmEnabled] = useState(false);
  const [selectedLLM, setSelectedLLM] = useState('Perplexity API');
  
  // Download and Training State Management
  const [isDownloading, setIsDownloading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [dataDownloaded, setDataDownloaded] = useState(false); // Tracks if data is ready for training

  // Simulated download function (will be connected to backend API)
  const handleDownloadData = () => {
    setIsDownloading(true);
    setDownloadProgress(0);
    setDataDownloaded(false);
    // Backend will handle actual download and update progress
    // This downloads: Price, Volume, Technical Indicators, Alternative Data (if selected)
    
    // Simulate download completion (backend will actually control this)
    setTimeout(() => {
      setDownloadProgress(100);
      setIsDownloading(false);
      setDataDownloaded(true); // Mark data as ready
    }, 3000);
  };

  // Simulated training function (will be connected to backend API)
  const handleStartTraining = () => {
    if (!dataDownloaded) return; // Safety check - should not happen due to disabled button
    
    setIsTraining(true);
    setTrainingProgress(0);
    // Backend will handle Optuna trials and training episodes
  };

  return (
    <div>
      <div className="model-version">
        <strong>Current Models:</strong><br />
        PPO: v3.2_20241105 | Trained: 05/11/2024 | Episodes: 50,000<br />
        SAC: v2.8_20241105 | Trained: 05/11/2024 | Episodes: 45,000
      </div>

      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">PPO Hyperparameters (Stock)</div>
        <div className="hyperparam-inline" style={{ gridTemplateColumns: 'repeat(7, 1fr)' }}>
          <ParamItem label="Symbol">
            <input 
              type="text" 
              className="param-input" 
              defaultValue="AAPL" 
              title="Stock symbol to trade (e.g., AAPL, TSLA, GOOGL). This determines which stock the agent will learn to trade."
            />
          </ParamItem>
          <ParamItem label="LR">
            <input 
              type="number" 
              className="param-input" 
              defaultValue="0.0003" 
              step="0.0001"
              title="Learning Rate: Controls how quickly the agent learns. Lower values (0.0001-0.0003) = slower but more stable learning. Higher values = faster but potentially unstable."
            />
          </ParamItem>
          <ParamItem label="γ">
            <input 
              type="number" 
              className="param-input" 
              defaultValue="0.99" 
              step="0.01"
              title="Gamma (Discount Factor): How much the agent values future rewards vs immediate rewards. 0.99 = values long-term gains. Lower values = prefers quick profits."
            />
          </ParamItem>
          <ParamItem label="Batch">
            <input 
              type="number" 
              className="param-input" 
              defaultValue="256"
              title="Batch Size: Number of experiences used in each training step. Larger batches (256-512) = more stable learning but slower. Smaller batches = faster updates but noisier."
            />
          </ParamItem>
          <ParamItem label="Risk Penalty">
            <input 
              type="number" 
              className="param-input" 
              defaultValue="-0.5" 
              step="0.1"
              title="Risk Penalty: Negative reward for taking risky actions. More negative (-0.5 to -1.0) = agent becomes more conservative. Less negative = agent takes more risks."
            />
          </ParamItem>
          <ParamItem label="Episodes">
            <input 
              type="number" 
              className="param-input" 
              defaultValue="50000"
              title="Training Episodes: How many times the agent practices trading. More episodes = better learning but longer training time. Typical range: 30,000-100,000."
            />
          </ParamItem>
          <ParamItem label="Retrain">
            <select 
              className="param-input"
              title="Retraining Frequency: How often the agent retrains itself with new data. Weekly = good balance. Daily = adapts faster to market changes. OFF = never retrains."
            >
              <option>OFF</option>
              <option>Daily</option>
              <option selected>Weekly</option>
              <option>Monthly</option>
            </select>
          </ParamItem>
        </div>
      </Card>

      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">SAC Hyperparameters (ETF)</div>
        <div className="hyperparam-inline" style={{ gridTemplateColumns: 'repeat(7, 1fr)' }}>
          <ParamItem label="Symbol">
            <input 
              type="text" 
              className="param-input" 
              defaultValue="TNA"
              title="ETF symbol to trade (e.g., TNA, TQQQ, SPXL). This determines which leveraged ETF the agent will learn to trade."
            />
          </ParamItem>
          <ParamItem label="LR">
            <input 
              type="number" 
              className="param-input" 
              defaultValue="0.0003" 
              step="0.0001"
              title="Learning Rate: Controls how quickly the agent learns. Lower values (0.0001-0.0003) = slower but more stable learning. Higher values = faster but potentially unstable."
            />
          </ParamItem>
          <ParamItem label="Entropy">
            <input 
              type="number" 
              className="param-input" 
              defaultValue="0.2" 
              step="0.01"
              title="Entropy Coefficient: Controls exploration vs exploitation. Higher (0.2-0.5) = more random exploration. Lower = more focused on known strategies."
            />
          </ParamItem>
          <ParamItem label="Batch">
            <input 
              type="number" 
              className="param-input" 
              defaultValue="256"
              title="Batch Size: Number of experiences used in each training step. Larger batches (256-512) = more stable learning but slower. Smaller batches = faster updates but noisier."
            />
          </ParamItem>
          <ParamItem label="Vol Penalty">
            <input 
              type="number" 
              className="param-input" 
              defaultValue="-0.3" 
              step="0.1"
              title="Volatility Penalty: Negative reward for trading during high volatility periods. More negative = agent avoids volatile markets. Less negative = agent is comfortable with volatility."
            />
          </ParamItem>
          <ParamItem label="Episodes">
            <input 
              type="number" 
              className="param-input" 
              defaultValue="45000"
              title="Training Episodes: How many times the agent practices trading. More episodes = better learning but longer training time. Typical range: 30,000-100,000."
            />
          </ParamItem>
          <ParamItem label="Retrain">
            <select 
              className="param-input"
              title="Retraining Frequency: How often the agent retrains itself with new data. Weekly = good balance. Daily = adapts faster to market changes. OFF = never retrains."
            >
              <option>OFF</option>
              <option>Daily</option>
              <option selected>Weekly</option>
              <option>Monthly</option>
            </select>
          </ParamItem>
        </div>
      </Card>

      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">Training Settings</div>
        <div className="hyperparam-inline" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
          <ParamItem label="Start Date">
            <input 
              type="date" 
              className="param-input" 
              defaultValue="2023-01-01"
              title="Start Date: The beginning date of historical data for training. Choose a date with enough market history (typically 1-2 years back)."
            />
          </ParamItem>
          <ParamItem label="End Date">
            <input 
              type="date" 
              className="param-input" 
              defaultValue="2024-11-01"
              title="End Date: The last date of historical data for training. Usually set to today or recent date. More data = better learning."
            />
          </ParamItem>
          <ParamItem label="Commission">
            <input 
              type="number" 
              className="param-input" 
              defaultValue="1.0" 
              step="0.1"
              title="Commission Fee: Cost per trade in dollars. This is deducted from profits during training. Set to match your broker's actual fees (e.g., $1-$5 per trade)."
            />
          </ParamItem>
          <ParamItem label="Optuna Trials">
            <input 
              type="number" 
              className="param-input" 
              defaultValue="100"
              title="Optuna Trials: Number of hyperparameter combinations to test. More trials = better optimization but longer training. 50-200 trials is typical."
            />
          </ParamItem>
        </div>
      </Card>

      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">Input Features Selection</div>
        <div className="feature-grid">
          <div className="feature-group">
            <div className="feature-group-title">Price Data</div>
            <label className="feature-checkbox">
              <input type="checkbox" checked disabled /> Price
            </label>
            <label className="feature-checkbox">
              <input type="checkbox" checked disabled /> Volume
            </label>
            <label className="feature-checkbox">
              <input type="checkbox" defaultChecked /> OHLC
            </label>
          </div>

          <div className="feature-group">
            <div className="feature-group-title">Technical Indicators</div>
            <label className="feature-checkbox" title="RSI (Relative Strength Index): Measures if a stock is overbought (>70) or oversold (<30). Period 14 is standard. Helps identify potential reversals.">
              <input type="checkbox" defaultChecked /> RSI
              <input type="number" defaultValue="14" title="RSI Period: Number of bars to calculate RSI. 14 is standard. Lower values (7-10) = more sensitive. Higher (20-30) = smoother." />
            </label>
            <label className="feature-checkbox" title="MACD (Moving Average Convergence Divergence): Shows trend direction and momentum. Parameters are fast period, slow period, and signal line. Helps identify trend changes.">
              <input type="checkbox" defaultChecked /> MACD
              <input type="text" defaultValue="12,26,9" title="MACD Parameters: Fast period (12), Slow period (26), Signal line (9). Standard settings work well for most stocks." />
            </label>
            <label className="feature-checkbox" title="EMA (Exponential Moving Average): Shows average price over time, giving more weight to recent prices. Helps identify support/resistance levels.">
              <input type="checkbox" defaultChecked /> EMA
              <input type="text" defaultValue="10,50" title="EMA Periods: Short term (10) and long term (50). Crossovers indicate potential buy/sell signals. 10,50 is common for day trading." />
            </label>
            <label className="feature-checkbox" title="VIX (Volatility Index): Measures market fear and volatility. High VIX = more market uncertainty. Helps adjust risk during volatile periods.">
              <input type="checkbox" defaultChecked /> VIX
            </label>
            <label className="feature-checkbox" title="Bollinger Bands: Shows price volatility using upper and lower bands. Price touching upper band = potentially overbought. Touching lower = oversold.">
              <input type="checkbox" /> Bollinger
              <input type="text" defaultValue="20,2" title="Bollinger Parameters: Period (20) and standard deviations (2). Standard settings capture ~95% of price action." />
            </label>
            <label className="feature-checkbox" title="Stochastic Oscillator: Compares closing price to price range over time. Values >80 = overbought, <20 = oversold. Good for ranging markets.">
              <input type="checkbox" /> Stochastic
              <input type="text" defaultValue="14,3" title="Stochastic Parameters: K period (14) and D period (3). Standard settings work for most timeframes." />
            </label>
          </div>

          <div className="feature-group">
            <div className="feature-group-title">Alternative Data</div>
            <label className="feature-checkbox">
              <input type="checkbox" /> Sentiment (News)
            </label>
            <label className="feature-checkbox">
              <input type="checkbox" /> Social Media
            </label>
            <label className="feature-checkbox">
              <input type="checkbox" /> News Headlines
            </label>
            <label className="feature-checkbox">
              <input type="checkbox" /> Market Events
            </label>
            <label className="feature-checkbox">
              <input type="checkbox" /> Fundamental
            </label>
          </div>

          <div className="feature-group">
            <div className="feature-group-title">Agent History</div>
            <label className="feature-checkbox">
              <input type="checkbox" defaultChecked /> Recent Actions
            </label>
            <label className="feature-checkbox">
              <input type="checkbox" defaultChecked /> Performance
              <input type="text" defaultValue="30d" />
            </label>
            <label className="feature-checkbox">
              <input type="checkbox" defaultChecked /> Position History
            </label>
            <label className="feature-checkbox">
              <input type="checkbox" /> Reward History
            </label>

            <div style={{ marginTop: '10px', borderTop: '1px solid #21262d', paddingTop: '8px' }}>
              <div className="feature-group-title">LLM Integration</div>
              <label className="feature-checkbox">
                <input 
                  type="checkbox" 
                  checked={llmEnabled}
                  onChange={(e) => setLlmEnabled(e.target.checked)}
                /> Enable LLM
              </label>
              <select 
                className="param-input" 
                disabled={!llmEnabled} 
                style={{ marginTop: '4px', fontSize: '10px' }}
                value={selectedLLM}
                onChange={(e) => setSelectedLLM(e.target.value)}
              >
                <option>Perplexity API</option>
                <option>Gemini API</option>
              </select>
            </div>
          </div>
        </div>
      </Card>

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
        
        <Button variant="secondary" disabled={isTraining}>Stop Training</Button>
        
        {/* Load Model Button:
            - Loads a previously saved trained model from disk
            - Allows you to continue training or use for live trading
            - Skip training if you already have a good model
        */}
        <Button variant="secondary">Load Model</Button>
        
        <Button variant="secondary">Save Config</Button>
      </div>

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

      <Card style={{ marginTop: '12px' }}>
        <div className="control-title">Training Results</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
          <Card style={{ background: '#0d1117', padding: '10px' }}>
            <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '4px' }}>Best Optuna Trial</div>
            <div style={{ fontSize: '16px', fontWeight: 600 }}>Trial #47</div>
            <div style={{ fontSize: '10px', color: '#6e7681', marginTop: '2px' }}>LR: 0.00025 | γ: 0.98</div>
          </Card>
          <Card style={{ background: '#0d1117', padding: '10px' }}>
            <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '4px' }}>Avg Reward</div>
            <div style={{ fontSize: '16px', fontWeight: 600, color: '#3fb950' }}>+0.87</div>
            <div style={{ fontSize: '10px', color: '#6e7681', marginTop: '2px' }}>Last 1000 episodes</div>
          </Card>
          <Card style={{ background: '#0d1117', padding: '10px' }}>
            <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '4px' }}>Training Time</div>
            <div style={{ fontSize: '16px', fontWeight: 600 }}>2h 34m</div>
            <div style={{ fontSize: '10px', color: '#6e7681', marginTop: '2px' }}>Completed: 05/11/2024</div>
          </Card>
        </div>
      </Card>
    </div>
  );
}

export default TabTraining;
