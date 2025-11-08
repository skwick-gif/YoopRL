import React, { useState } from 'react';
import { Button, Card, InputGroup } from './common/UIComponents';

/**
 * ENHANCED TabSimulation Component - Unified Testing Environment
 * 
 * TWO MODES:
 * 1. Backtest Mode - Run complete backtest with full metrics
 * 2. Live Mode - Watch agent trade step-by-step in real-time
 * 
 * FEATURES:
 * - Load trained model (PPO/SAC)
 * - 3 data sources: CSV (historical), Yahoo Finance (pseudo-live), IBKR (real-time)
 * - Comprehensive backtest results: 8 metrics, charts, trade history, analytics
 * - Export functionality (PDF, CSV)
 */
function TabSimulation() {
  const [dataSource, setDataSource] = useState('csv');
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('PPO_AAPL_v3.2');
  const [simulationMode, setSimulationMode] = useState('backtest'); // 'backtest' or 'live'
  const [showResults, setShowResults] = useState(true); // Demo: show results by default

  // Available trained models (will be loaded from backend)
  const availableModels = [
    { id: 'PPO_AAPL_v3.2', name: 'PPO_AAPL_v3.2', algorithm: 'PPO', symbol: 'AAPL', episodes: '50,000', date: '05/11/2024' },
    { id: 'PPO_MSFT_v2.8', name: 'PPO_MSFT_v2.8', algorithm: 'PPO', symbol: 'MSFT', episodes: '45,000', date: '03/11/2024' },
    { id: 'PPO_TSLA_v1.5', name: 'PPO_TSLA_v1.5', algorithm: 'PPO', symbol: 'TSLA', episodes: '30,000', date: '01/11/2024' },
    { id: 'SAC_TNA_v4.1', name: 'SAC_TNA_v4.1', algorithm: 'SAC', symbol: 'TNA', episodes: '45,000', date: '04/11/2024' },
    { id: 'SAC_TQQQ_v3.0', name: 'SAC_TQQQ_v3.0', algorithm: 'SAC', symbol: 'TQQQ', episodes: '40,000', date: '02/11/2024' },
  ];

  const currentModel = availableModels.find(m => m.id === selectedModel);

  // Mock backtest results (will come from API POST /api/backtest/run)
  const backtestResults = {
    total_return: 34.2,
    sharpe_ratio: 1.82,
    sortino_ratio: 2.15,
    max_drawdown: -12.4,
    win_rate: 58.3,
    calmar_ratio: 2.76,
    profit_factor: 1.85,
    total_trades: 142,
    winning_trades: 83,
    losing_trades: 59,
    avg_win: 2.8,
    avg_loss: -1.4,
    largest_win: 8.5,
    largest_loss: -4.2,
    final_equity: 134200,
    initial_capital: 100000,
    duration_days: 304
  };

  const handleRunBacktest = () => {
    setShowResults(true);
    // Will call: POST /api/backtest/run with { model_id, symbol, start_date, end_date, initial_capital }
  };

  return (
    <div>
      {/* ==================== SIMULATION MODE TOGGLE ==================== */}
      <Card style={{ marginBottom: '12px', padding: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ fontSize: '11px', color: '#8b949e', fontWeight: 600 }}>
            SIMULATION MODE:
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => setSimulationMode('backtest')}
              style={{
                padding: '6px 16px',
                fontSize: '11px',
                fontWeight: 600,
                borderRadius: '6px',
                border: simulationMode === 'backtest' ? '2px solid #58a6ff' : '1px solid #30363d',
                background: simulationMode === 'backtest' ? '#1f6feb' : '#21262d',
                color: simulationMode === 'backtest' ? '#ffffff' : '#8b949e',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              📊 Backtest Mode
            </button>
            <button
              onClick={() => setSimulationMode('live')}
              style={{
                padding: '6px 16px',
                fontSize: '11px',
                fontWeight: 600,
                borderRadius: '6px',
                border: simulationMode === 'live' ? '2px solid #3fb950' : '1px solid #30363d',
                background: simulationMode === 'live' ? '#238636' : '#21262d',
                color: simulationMode === 'live' ? '#ffffff' : '#8b949e',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              🔴 Live Mode
            </button>
          </div>
          <div style={{ flex: 1 }} />
          <div style={{ fontSize: '10px', color: '#6e7681', fontStyle: 'italic' }}>
            {simulationMode === 'backtest' 
              ? 'Run complete backtest with full metrics and analysis' 
              : 'Watch agent trade step-by-step in real-time'}
          </div>
        </div>
      </Card>

      {/* ==================== MODEL SELECTION (UNCHANGED) ==================== */}
      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">Select Trained Model</div>
        <div className="input-wrapper">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{
              padding: '8px 12px',
              fontSize: '12px',
              background: '#0d1117',
              border: '1px solid #30363d',
              borderRadius: '6px',
              color: '#c9d1d9',
              width: '100%',
              marginBottom: '12px'
            }}
          >
            {availableModels.map(model => (
              <option key={model.id} value={model.id}>
                {model.name} - {model.algorithm} ({model.symbol}) - {model.episodes} episodes
              </option>
            ))}
          </select>
          
          {currentModel && (
            <div style={{
              padding: '10px',
              background: '#0d1117',
              border: '1px solid #30363d',
              borderRadius: '6px',
              fontSize: '11px',
              color: '#8b949e',
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: '10px'
            }}>
              <div>
                <div style={{ color: '#6e7681' }}>Algorithm</div>
                <div style={{ fontSize: '12px', color: '#c9d1d9', fontWeight: 600 }}>{currentModel.algorithm}</div>
              </div>
              <div>
                <div style={{ color: '#6e7681' }}>Trained On</div>
                <div style={{ fontSize: '12px', color: '#c9d1d9', fontWeight: 600 }}>{currentModel.symbol}</div>
              </div>
              <div>
                <div style={{ color: '#6e7681' }}>Episodes</div>
                <div style={{ fontSize: '12px', color: '#c9d1d9', fontWeight: 600 }}>{currentModel.episodes}</div>
              </div>
              <div>
                <div style={{ color: '#6e7681' }}>Last Trained</div>
                <div style={{ fontSize: '12px', color: '#c9d1d9', fontWeight: 600 }}>{currentModel.date}</div>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* ==================== DATA SOURCE SELECTION (UNCHANGED) ==================== */}
      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">Data Source</div>
        <div className="tabs">
          <button
            className={`tab ${dataSource === 'csv' ? 'active' : ''}`}
            onClick={() => setDataSource('csv')}
          >
            📂 CSV File
          </button>
          <button
            className={`tab ${dataSource === 'yahoo' ? 'active' : ''}`}
            onClick={() => setDataSource('yahoo')}
          >
            📈 Yahoo Finance
          </button>
          <button
            className={`tab ${dataSource === 'ibkr' ? 'active' : ''}`}
            onClick={() => setDataSource('ibkr')}
          >
            🔗 IBKR
          </button>
        </div>

        {dataSource === 'csv' && (
          <div className="csv-controls">
            <div className="file-upload">
              <input
                type="file"
                accept=".csv"
                id="csvFile"
                style={{ display: 'none' }}
                onChange={(e) => setSelectedFile(e.target.files[0])}
              />
              <label htmlFor="csvFile" className="upload-label">
                {selectedFile ? `📄 ${selectedFile.name}` : '📁 Choose CSV File'}
              </label>
            </div>
          </div>
        )}

        {dataSource === 'yahoo' && (
          <div className="yf-controls">
            <InputGroup>
              <input type="text" placeholder="Symbol (e.g., AAPL)" defaultValue="AAPL" />
              <select>
                <option>1 minute</option>
                <option selected>5 minutes</option>
                <option>15 minutes</option>
              </select>
              <input type="date" defaultValue="2024-10-01" />
              <input type="date" defaultValue="2024-11-01" />
            </InputGroup>
            <div className="btn-group">
              <Button className="start">Start Streaming</Button>
              <Button className="stop">Stop</Button>
            </div>
          </div>
        )}

        {dataSource === 'ibkr' && (
          <div className="yf-controls">
            <InputGroup>
              <input type="text" placeholder="Symbol (e.g., AAPL)" defaultValue="AAPL" />
              <select>
                <option>1 minute</option>
                <option selected>5 minutes</option>
                <option>15 minutes</option>
              </select>
            </InputGroup>
            <div className="btn-group">
              <Button className="start">Start Streaming</Button>
              <Button className="stop">Stop</Button>
            </div>
            <div style={{ fontSize: '10px', color: '#f59e0b', marginTop: '8px' }}>
              ⚠️ Purpose and usage to be determined
            </div>
          </div>
        )}
      </Card>

      {/* ==================== SIMULATION CONTROLS (MODE-DEPENDENT) ==================== */}
      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">
          {simulationMode === 'backtest' ? '📊 BACKTEST CONFIGURATION' : '🎮 LIVE SIMULATION CONTROLS'}
        </div>
        <div style={{ padding: '20px' }}>
          {simulationMode === 'backtest' ? (
            /* ========== BACKTEST MODE ========== */
            <>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '6px', fontWeight: 600 }}>
                    Starting Capital ($)
                  </div>
                  <input
                    type="number"
                    defaultValue="100000"
                    min="1000"
                    step="1000"
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      background: '#0d1117',
                      border: '1px solid #30363d',
                      borderRadius: '6px',
                      color: '#c9d1d9',
                      fontSize: '12px'
                    }}
                  />
                </div>
                <div>
                  <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '6px', fontWeight: 600 }}>
                    Start Date
                  </div>
                  <input
                    type="date"
                    defaultValue="2023-01-01"
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      background: '#0d1117',
                      border: '1px solid #30363d',
                      borderRadius: '6px',
                      color: '#c9d1d9',
                      fontSize: '12px'
                    }}
                  />
                </div>
                <div>
                  <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '6px', fontWeight: 600 }}>
                    End Date
                  </div>
                  <input
                    type="date"
                    defaultValue="2024-11-01"
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      background: '#0d1117',
                      border: '1px solid #30363d',
                      borderRadius: '6px',
                      color: '#c9d1d9',
                      fontSize: '12px'
                    }}
                  />
                </div>
              </div>
              <div style={{ display: 'flex', gap: '12px' }}>
                <Button onClick={handleRunBacktest} style={{ flex: 1 }}>
                  🚀 Run Full Backtest
                </Button>
                <button style={{
                  padding: '10px 20px',
                  background: '#21262d',
                  border: '1px solid #30363d',
                  borderRadius: '6px',
                  color: '#c9d1d9',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}>
                  📥 Export Config
                </button>
              </div>
            </>
          ) : (
            /* ========== LIVE MODE ========== */
            <>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '6px', fontWeight: 600 }}>
                    Starting Capital ($)
                  </div>
                  <input
                    type="number"
                    defaultValue="100000"
                    min="1000"
                    step="1000"
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      background: '#0d1117',
                      border: '1px solid #30363d',
                      borderRadius: '6px',
                      color: '#c9d1d9',
                      fontSize: '12px'
                    }}
                  />
                </div>
                <div>
                  <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '6px', fontWeight: 600 }}>
                    Playback Speed
                  </div>
                  <select style={{
                    width: '100%',
                    padding: '8px 12px',
                    background: '#0d1117',
                    border: '1px solid #30363d',
                    borderRadius: '6px',
                    color: '#c9d1d9',
                    fontSize: '12px'
                  }}>
                    <option>1x (Real-time)</option>
                    <option selected>10x</option>
                    <option>50x</option>
                    <option>100x</option>
                    <option>Max Speed</option>
                  </select>
                </div>
              </div>
              <div style={{ display: 'flex', gap: '12px' }}>
                <Button style={{ flex: 1 }}>▶️ Start Simulation</Button>
                <button style={{
                  padding: '10px 20px',
                  background: '#21262d',
                  border: '1px solid #30363d',
                  borderRadius: '6px',
                  color: '#c9d1d9',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}>
                  ⏸️ Stop
                </button>
                <button style={{
                  padding: '10px 20px',
                  background: '#21262d',
                  border: '1px solid #30363d',
                  borderRadius: '6px',
                  color: '#c9d1d9',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}>
                  🔄 Reset
                </button>
              </div>
            </>
          )}
        </div>
      </Card>

      {/* ==================== BACKTEST RESULTS PANEL (ONLY IN BACKTEST MODE) ==================== */}
      {showResults && simulationMode === 'backtest' && (
        <div style={{ marginTop: '12px' }}>
          {/* ========== PERFORMANCE METRICS CARDS ========== */}
          <Card style={{ marginBottom: '12px' }}>
            <div style={{
              borderBottom: '1px solid #30363d',
              padding: '12px 16px',
              fontSize: '12px',
              fontWeight: 600,
              color: '#c9d1d9',
              letterSpacing: '0.5px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <span>📈 PERFORMANCE METRICS</span>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button style={{
                  padding: '4px 12px',
                  fontSize: '10px',
                  background: '#238636',
                  border: '1px solid #2ea043',
                  borderRadius: '6px',
                  color: '#ffffff',
                  cursor: 'pointer',
                  fontWeight: 600
                }}>
                  📄 Export PDF
                </button>
                <button style={{
                  padding: '4px 12px',
                  fontSize: '10px',
                  background: '#1f6feb',
                  border: '1px solid #388bfd',
                  borderRadius: '6px',
                  color: '#ffffff',
                  cursor: 'pointer',
                  fontWeight: 600
                }}>
                  💾 Export CSV
                </button>
              </div>
            </div>
            <div style={{ padding: '20px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                {/* Total Return */}
                <div style={{
                  background: backtestResults.total_return >= 0 ? 'rgba(35, 134, 54, 0.1)' : 'rgba(248, 81, 73, 0.1)',
                  border: backtestResults.total_return >= 0 ? '1px solid #238636' : '1px solid #da3633',
                  borderRadius: '8px',
                  padding: '16px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '8px', fontWeight: 600 }}>
                    TOTAL RETURN
                  </div>
                  <div style={{
                    fontSize: '24px',
                    fontWeight: 700,
                    color: backtestResults.total_return >= 0 ? '#3fb950' : '#f85149'
                  }}>
                    {backtestResults.total_return >= 0 ? '+' : ''}{backtestResults.total_return}%
                  </div>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginTop: '4px' }}>
                    ${backtestResults.initial_capital.toLocaleString()} → ${backtestResults.final_equity.toLocaleString()}
                  </div>
                </div>

                {/* Sharpe Ratio */}
                <div style={{
                  background: backtestResults.sharpe_ratio >= 1 ? 'rgba(35, 134, 54, 0.1)' : 'rgba(187, 128, 9, 0.1)',
                  border: backtestResults.sharpe_ratio >= 1 ? '1px solid #238636' : '1px solid #bb8009',
                  borderRadius: '8px',
                  padding: '16px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '8px', fontWeight: 600 }}>
                    SHARPE RATIO
                  </div>
                  <div style={{
                    fontSize: '24px',
                    fontWeight: 700,
                    color: backtestResults.sharpe_ratio >= 1 ? '#3fb950' : '#d29922'
                  }}>
                    {backtestResults.sharpe_ratio.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginTop: '4px' }}>
                    Risk-adjusted return
                  </div>
                </div>

                {/* Max Drawdown */}
                <div style={{
                  background: 'rgba(248, 81, 73, 0.1)',
                  border: '1px solid #da3633',
                  borderRadius: '8px',
                  padding: '16px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '8px', fontWeight: 600 }}>
                    MAX DRAWDOWN
                  </div>
                  <div style={{
                    fontSize: '24px',
                    fontWeight: 700,
                    color: '#f85149'
                  }}>
                    {backtestResults.max_drawdown}%
                  </div>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginTop: '4px' }}>
                    Peak-to-trough decline
                  </div>
                </div>

                {/* Win Rate */}
                <div style={{
                  background: backtestResults.win_rate >= 50 ? 'rgba(35, 134, 54, 0.1)' : 'rgba(248, 81, 73, 0.1)',
                  border: backtestResults.win_rate >= 50 ? '1px solid #238636' : '1px solid #da3633',
                  borderRadius: '8px',
                  padding: '16px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '8px', fontWeight: 600 }}>
                    WIN RATE
                  </div>
                  <div style={{
                    fontSize: '24px',
                    fontWeight: 700,
                    color: backtestResults.win_rate >= 50 ? '#3fb950' : '#f85149'
                  }}>
                    {backtestResults.win_rate}%
                  </div>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginTop: '4px' }}>
                    {backtestResults.winning_trades}W / {backtestResults.losing_trades}L
                  </div>
                </div>

                {/* Sortino Ratio */}
                <div style={{
                  background: 'rgba(31, 111, 235, 0.1)',
                  border: '1px solid #1f6feb',
                  borderRadius: '8px',
                  padding: '16px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '8px', fontWeight: 600 }}>
                    SORTINO RATIO
                  </div>
                  <div style={{
                    fontSize: '24px',
                    fontWeight: 700,
                    color: '#58a6ff'
                  }}>
                    {backtestResults.sortino_ratio.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginTop: '4px' }}>
                    Downside risk-adjusted
                  </div>
                </div>

                {/* Calmar Ratio */}
                <div style={{
                  background: 'rgba(31, 111, 235, 0.1)',
                  border: '1px solid #1f6feb',
                  borderRadius: '8px',
                  padding: '16px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '8px', fontWeight: 600 }}>
                    CALMAR RATIO
                  </div>
                  <div style={{
                    fontSize: '24px',
                    fontWeight: 700,
                    color: '#58a6ff'
                  }}>
                    {backtestResults.calmar_ratio.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginTop: '4px' }}>
                    Return / Max DD
                  </div>
                </div>

                {/* Profit Factor */}
                <div style={{
                  background: 'rgba(31, 111, 235, 0.1)',
                  border: '1px solid #1f6feb',
                  borderRadius: '8px',
                  padding: '16px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '8px', fontWeight: 600 }}>
                    PROFIT FACTOR
                  </div>
                  <div style={{
                    fontSize: '24px',
                    fontWeight: 700,
                    color: '#58a6ff'
                  }}>
                    {backtestResults.profit_factor.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginTop: '4px' }}>
                    Gross profit / Gross loss
                  </div>
                </div>

                {/* Total Trades */}
                <div style={{
                  background: 'rgba(139, 148, 158, 0.1)',
                  border: '1px solid #30363d',
                  borderRadius: '8px',
                  padding: '16px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '8px', fontWeight: 600 }}>
                    TOTAL TRADES
                  </div>
                  <div style={{
                    fontSize: '24px',
                    fontWeight: 700,
                    color: '#c9d1d9'
                  }}>
                    {backtestResults.total_trades}
                  </div>
                  <div style={{ fontSize: '10px', color: '#8b949e', marginTop: '4px' }}>
                    Avg Win: {backtestResults.avg_win}% | Avg Loss: {backtestResults.avg_loss}%
                  </div>
                </div>
              </div>

              {/* Additional Stats Row */}
              <div style={{ 
                marginTop: '16px',
                padding: '12px',
                background: 'rgba(139, 148, 158, 0.05)',
                borderRadius: '6px',
                display: 'flex',
                justifyContent: 'space-around',
                fontSize: '11px',
                color: '#8b949e'
              }}>
                <div>
                  <span style={{ fontWeight: 600, color: '#c9d1d9' }}>Test Period:</span> 
                  <span style={{ color: '#58a6ff', marginLeft: '6px' }}>{backtestResults.duration_days} days</span>
                </div>
                <div>
                  <span style={{ fontWeight: 600, color: '#c9d1d9' }}>Largest Win:</span> 
                  <span style={{ color: '#3fb950', marginLeft: '6px' }}>+{backtestResults.largest_win}%</span>
                </div>
                <div>
                  <span style={{ fontWeight: 600, color: '#c9d1d9' }}>Largest Loss:</span> 
                  <span style={{ color: '#f85149', marginLeft: '6px' }}>{backtestResults.largest_loss}%</span>
                </div>
                <div>
                  <span style={{ fontWeight: 600, color: '#c9d1d9' }}>Final Equity:</span> 
                  <span style={{ color: '#58a6ff', marginLeft: '6px' }}>${backtestResults.final_equity.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </Card>

          {/* ========== CHARTS SECTION ========== */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
            {/* Equity Curve Chart */}
            <Card>
              <div style={{
                borderBottom: '1px solid #30363d',
                padding: '12px 16px',
                fontSize: '12px',
                fontWeight: 600,
                color: '#c9d1d9',
                letterSpacing: '0.5px'
              }}>
                📈 EQUITY CURVE
              </div>
              <div style={{ padding: '20px', height: '280px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ textAlign: 'center', color: '#6e7681' }}>
                  <div style={{ fontSize: '48px', marginBottom: '8px' }}>📊</div>
                  <div style={{ fontSize: '12px', lineHeight: '1.6' }}>
                    Portfolio value over time
                    <br />
                    <span style={{ fontSize: '10px', color: '#8b949e' }}>
                      Recharts LineChart with:
                    </span>
                    <br />
                    <span style={{ fontSize: '10px', color: '#58a6ff' }}>
                      • Equity curve (blue line)
                      <br />
                      • Benchmark overlay (gray dashed)
                      <br />
                      • Trade markers (🟢 Buy / 🔴 Sell)
                    </span>
                  </div>
                </div>
              </div>
            </Card>

            {/* Drawdown Chart */}
            <Card>
              <div style={{
                borderBottom: '1px solid #30363d',
                padding: '12px 16px',
                fontSize: '12px',
                fontWeight: 600,
                color: '#c9d1d9',
                letterSpacing: '0.5px'
              }}>
                📉 DRAWDOWN CHART
              </div>
              <div style={{ padding: '20px', height: '280px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ textAlign: 'center', color: '#6e7681' }}>
                  <div style={{ fontSize: '48px', marginBottom: '8px' }}>📉</div>
                  <div style={{ fontSize: '12px', lineHeight: '1.6' }}>
                    Peak-to-trough decline
                    <br />
                    <span style={{ fontSize: '10px', color: '#8b949e' }}>
                      Recharts AreaChart:
                    </span>
                    <br />
                    <span style={{ fontSize: '10px', color: '#f85149' }}>
                      • Underwater equity (%)
                      <br />
                      • Recovery periods highlighted
                      <br />
                      • Max drawdown marker
                    </span>
                  </div>
                </div>
              </div>
            </Card>
          </div>

          {/* ========== TRADE HISTORY TABLE ========== */}
          <Card style={{ marginBottom: '12px' }}>
            <div style={{
              borderBottom: '1px solid #30363d',
              padding: '12px 16px',
              fontSize: '12px',
              fontWeight: 600,
              color: '#c9d1d9',
              letterSpacing: '0.5px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <span>📋 TRADE HISTORY ({backtestResults.total_trades} trades)</span>
              <button style={{
                padding: '4px 12px',
                fontSize: '10px',
                background: '#21262d',
                border: '1px solid #30363d',
                borderRadius: '6px',
                color: '#c9d1d9',
                cursor: 'pointer',
                fontWeight: 600
              }}>
                🔽 Download Full History (CSV)
              </button>
            </div>
            <div style={{ padding: '20px' }}>
              {/* Table Header */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: '80px 80px 100px 80px 80px 100px 100px',
                gap: '12px',
                padding: '8px 12px',
                background: 'rgba(139, 148, 158, 0.05)',
                borderRadius: '6px',
                fontSize: '10px',
                fontWeight: 600,
                color: '#8b949e',
                marginBottom: '8px'
              }}>
                <div>DATE</div>
                <div>ACTION</div>
                <div>PRICE</div>
                <div>SHARES</div>
                <div>VALUE</div>
                <div>P&L</div>
                <div>RETURN %</div>
              </div>

              {/* Sample Trade Rows (first 5 of 142) */}
              {[
                { date: '11/05/24', action: 'BUY', price: 223.45, shares: 100, value: 22345, pnl: null, return: null },
                { date: '11/08/24', action: 'SELL', price: 228.92, shares: 100, value: 22892, pnl: 547, return: 2.45 },
                { date: '11/12/24', action: 'BUY', price: 225.10, shares: 100, value: 22510, pnl: null, return: null },
                { date: '11/15/24', action: 'SELL', price: 221.34, shares: 100, value: 22134, pnl: -376, return: -1.67 },
                { date: '11/18/24', action: 'BUY', price: 219.88, shares: 100, value: 21988, pnl: null, return: null },
              ].map((trade, idx) => (
                <div key={idx} style={{
                  display: 'grid',
                  gridTemplateColumns: '80px 80px 100px 80px 80px 100px 100px',
                  gap: '12px',
                  padding: '8px 12px',
                  borderBottom: '1px solid #21262d',
                  fontSize: '11px',
                  color: '#c9d1d9',
                  alignItems: 'center'
                }}>
                  <div style={{ color: '#8b949e' }}>{trade.date}</div>
                  <div style={{
                    color: trade.action === 'BUY' ? '#3fb950' : '#f85149',
                    fontWeight: 600
                  }}>
                    {trade.action === 'BUY' ? '🟢' : '🔴'} {trade.action}
                  </div>
                  <div>${trade.price.toFixed(2)}</div>
                  <div>{trade.shares}</div>
                  <div>${trade.value.toLocaleString()}</div>
                  <div style={{
                    color: trade.pnl === null ? '#8b949e' : (trade.pnl >= 0 ? '#3fb950' : '#f85149'),
                    fontWeight: trade.pnl !== null ? 600 : 400
                  }}>
                    {trade.pnl === null ? '--' : (trade.pnl >= 0 ? '+' : '') + '$' + trade.pnl.toLocaleString()}
                  </div>
                  <div style={{
                    color: trade.return === null ? '#8b949e' : (trade.return >= 0 ? '#3fb950' : '#f85149'),
                    fontWeight: trade.return !== null ? 600 : 400
                  }}>
                    {trade.return === null ? '--' : (trade.return >= 0 ? '+' : '') + trade.return.toFixed(2) + '%'}
                  </div>
                </div>
              ))}

              {/* Pagination */}
              <div style={{
                marginTop: '16px',
                display: 'flex',
                justifyContent: 'center',
                gap: '8px',
                fontSize: '11px'
              }}>
                <button style={{
                  padding: '6px 12px',
                  background: '#21262d',
                  border: '1px solid #30363d',
                  borderRadius: '6px',
                  color: '#8b949e',
                  cursor: 'not-allowed',
                  fontWeight: 600
                }}>
                  ← Previous
                </button>
                <div style={{
                  padding: '6px 12px',
                  background: '#1f6feb',
                  border: '1px solid #388bfd',
                  borderRadius: '6px',
                  color: '#ffffff',
                  fontWeight: 600
                }}>
                  Page 1 of 29
                </div>
                <button style={{
                  padding: '6px 12px',
                  background: '#21262d',
                  border: '1px solid #30363d',
                  borderRadius: '6px',
                  color: '#c9d1d9',
                  cursor: 'pointer',
                  fontWeight: 600
                }}>
                  Next →
                </button>
              </div>
            </div>
          </Card>

          {/* ========== ANALYTICS GRID ========== */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            {/* Monthly Returns Heatmap */}
            <Card>
              <div style={{
                borderBottom: '1px solid #30363d',
                padding: '12px 16px',
                fontSize: '12px',
                fontWeight: 600,
                color: '#c9d1d9',
                letterSpacing: '0.5px'
              }}>
                🗓️ MONTHLY RETURNS HEATMAP
              </div>
              <div style={{ padding: '20px', height: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ textAlign: 'center', color: '#6e7681' }}>
                  <div style={{ fontSize: '48px', marginBottom: '8px' }}>📅</div>
                  <div style={{ fontSize: '12px', lineHeight: '1.6' }}>
                    Calendar heatmap showing monthly performance
                    <br />
                    <span style={{ fontSize: '10px', color: '#8b949e' }}>
                      (Custom component or react-calendar-heatmap)
                    </span>
                    <br />
                    <span style={{ fontSize: '10px', color: '#58a6ff' }}>
                      🟢 Green = positive | 🔴 Red = negative
                    </span>
                  </div>
                </div>
              </div>
            </Card>

            {/* Returns Distribution */}
            <Card>
              <div style={{
                borderBottom: '1px solid #30363d',
                padding: '12px 16px',
                fontSize: '12px',
                fontWeight: 600,
                color: '#c9d1d9',
                letterSpacing: '0.5px'
              }}>
                📊 RETURNS DISTRIBUTION
              </div>
              <div style={{ padding: '20px', height: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ textAlign: 'center', color: '#6e7681' }}>
                  <div style={{ fontSize: '48px', marginBottom: '8px' }}>📊</div>
                  <div style={{ fontSize: '12px', lineHeight: '1.6' }}>
                    Histogram of trade returns
                    <br />
                    <span style={{ fontSize: '10px', color: '#8b949e' }}>
                      Recharts BarChart:
                    </span>
                    <br />
                    <span style={{ fontSize: '10px', color: '#58a6ff' }}>
                      • Return frequency distribution
                      <br />
                      • Normal curve overlay
                      <br />
                      • Mean/Median/Std markers
                    </span>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* ==================== LIVE MODE CHART (ORIGINAL) ==================== */}
      {simulationMode === 'live' && (
        <div className="chart-area" style={{ minHeight: '350px', marginBottom: '12px' }}>
          <div className="chart-title">Simulated Live Chart</div>
          <div className="chart-placeholder" style={{ height: '310px' }}>
            {dataSource === 'csv' && 'CSV data will stream here (bar by bar playback)'}
            {dataSource === 'yahoo' && 'Yahoo Finance live snapshot (15-20 min delay)'}
            {dataSource === 'ibkr' && 'IBKR real-time data (purpose TBD)'}
          </div>
        </div>
      )}
    </div>
  );
}

export default TabSimulation;
