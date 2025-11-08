import React, { useState } from 'react';
import { Button, Card, InputGroup } from './common/UIComponents';

function TabSimulation() {
  const [dataSource, setDataSource] = useState('csv');
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('PPO_AAPL_v3.2');

  // Available trained models (will be loaded from backend)
  const availableModels = [
    { id: 'PPO_AAPL_v3.2', name: 'PPO_AAPL_v3.2', algorithm: 'PPO', symbol: 'AAPL', episodes: '50,000', date: '05/11/2024' },
    { id: 'PPO_MSFT_v2.8', name: 'PPO_MSFT_v2.8', algorithm: 'PPO', symbol: 'MSFT', episodes: '45,000', date: '03/11/2024' },
    { id: 'PPO_TSLA_v1.5', name: 'PPO_TSLA_v1.5', algorithm: 'PPO', symbol: 'TSLA', episodes: '30,000', date: '01/11/2024' },
    { id: 'SAC_TNA_v4.1', name: 'SAC_TNA_v4.1', algorithm: 'SAC', symbol: 'TNA', episodes: '45,000', date: '04/11/2024' },
    { id: 'SAC_TQQQ_v3.0', name: 'SAC_TQQQ_v3.0', algorithm: 'SAC', symbol: 'TQQQ', episodes: '40,000', date: '02/11/2024' },
  ];

  const currentModel = availableModels.find(m => m.id === selectedModel);

  return (
    <div>
      {/* Model Selection - Choose which trained model to use for simulation */}
      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">Model Selection</div>
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '12px', alignItems: 'start' }}>
          <div>
            {/* Model Dropdown: Select which trained model to test
                - Each model was trained on specific symbol and market conditions
                - Different models perform better on different stocks/ETFs
                - Example: PPO_AAPL trained on AAPL may work differently on MSFT
                - You can test any model on any data to see generalization */}
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Select Trained Model</div>
            <select 
              className="param-input" 
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{ width: '100%', padding: '8px', fontSize: '12px' }}
            >
              {availableModels.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.algorithm} - {model.symbol})
                </option>
              ))}
            </select>
          </div>
          
          {/* Model Info Display - Shows details about selected model */}
          <div style={{ 
            padding: '10px', 
            background: '#0d1117', 
            borderRadius: '6px', 
            border: '1px solid #21262d',
            fontSize: '10px'
          }}>
            <div style={{ color: '#8b949e', marginBottom: '6px' }}>Model Details:</div>
            <div style={{ marginBottom: '3px' }}>
              <span style={{ color: '#6e7681' }}>Algorithm:</span> 
              <span style={{ color: '#58a6ff', marginLeft: '6px' }}>{currentModel?.algorithm}</span>
            </div>
            <div style={{ marginBottom: '3px' }}>
              <span style={{ color: '#6e7681' }}>Trained on:</span> 
              <span style={{ color: '#f59e0b', marginLeft: '6px' }}>{currentModel?.symbol}</span>
            </div>
            <div style={{ marginBottom: '3px' }}>
              <span style={{ color: '#6e7681' }}>Episodes:</span> 
              <span style={{ marginLeft: '6px' }}>{currentModel?.episodes}</span>
            </div>
            <div>
              <span style={{ color: '#6e7681' }}>Date:</span> 
              <span style={{ marginLeft: '6px' }}>{currentModel?.date}</span>
            </div>
          </div>
        </div>
        
        {/* Warning if simulating on different symbol than trained on */}
        {dataSource === 'csv' && (
          <div style={{ 
            marginTop: '10px', 
            padding: '8px', 
            background: '#1c1f26', 
            borderRadius: '4px',
            border: '1px solid #f59e0b',
            fontSize: '10px',
            color: '#f59e0b'
          }}>
            ℹ️ <strong>Note:</strong> This model was trained on <strong>{currentModel?.symbol}</strong>. 
            Performance may vary when testing on different symbols. 
            This helps evaluate model generalization.
          </div>
        )}
      </Card>

      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">Data Source</div>
        <div className="data-source-selector">
          {/* Local CSV: Upload historical data from file
              - Works offline - no internet required
              - Load your own historical price data (OHLCV)
              - Perfect for backtesting on custom date ranges
              - Example: Load 3 years of INTC data and run simulation */}
          <button 
            className={`data-source-btn ${dataSource === 'csv' ? 'active' : ''}`}
            onClick={() => setDataSource('csv')}
          >
            Local CSV
          </button>
          
          {/* Yahoo Finance: Live snapshot with delay
              - Gets live market snapshot from Yahoo Finance (15-20 min delay)
              - Available timeframes: 1min, 5min, 15min
              - Use for testing strategies with recent market data
              - Does NOT interfere with live trading agents */}
          <button 
            className={`data-source-btn ${dataSource === 'yahoo' ? 'active' : ''}`}
            onClick={() => setDataSource('yahoo')}
          >
            Yahoo Finance
          </button>
          
          {/* IBKR Live Data: Real-time market data (Future use)
              - Streams live market data from Interactive Brokers
              - Real-time quotes with millisecond precision
              - Requires IBKR account and active connection
              - Purpose TBD */}
          <button 
            className={`data-source-btn ${dataSource === 'ibkr' ? 'active' : ''}`}
            onClick={() => setDataSource('ibkr')}
          >
            IBKR Live Data
          </button>
        </div>

        {/* CSV File Upload - Show only when Local CSV is selected */}
        {dataSource === 'csv' && (
          <div className="yf-controls">
            <InputGroup>
              {/* CSV File Input: Upload local historical data
                  - Format: Date, Open, High, Low, Close, Volume
                  - Example: INTC_3years.csv with daily or intraday data
                  - After upload, click Start Streaming to load data */}
              <input 
                type="file" 
                accept=".csv"
                onChange={(e) => setSelectedFile(e.target.files[0])}
                style={{ padding: '8px', fontSize: '11px' }}
              />
              
              {/* Symbol: Stock symbol for the uploaded CSV */}
              <input type="text" placeholder="Symbol (e.g., INTC)" defaultValue="" />
            </InputGroup>
            <div className="btn-group">
              {/* Start Streaming Button: Loads and plays back CSV data
                  - Reads CSV file line by line
                  - Plays back historical data bar by bar
                  - Chart updates as simulation progresses
                  - Agent can then trade on this data */}
              <Button className="start" disabled={!selectedFile}>Start Streaming</Button>
              
              {/* Stop Button: Stops CSV playback */}
              <Button className="stop">Stop</Button>
            </div>
            {selectedFile && (
              <div style={{ fontSize: '10px', color: '#8b949e', marginTop: '8px' }}>
                Selected: {selectedFile.name}
              </div>
            )}
          </div>
        )}

        {/* Yahoo Finance Controls - Show only when Yahoo Finance is selected */}
        {dataSource === 'yahoo' && (
          <div className="yf-controls">
            <InputGroup>
              {/* Symbol: Stock/ETF to get live snapshot */}
              <input type="text" placeholder="Symbol (e.g., AAPL)" defaultValue="AAPL" />
              
              {/* Timeframe: Bar interval for live snapshot
                  - 1 minute = fastest updates (15-20 min delayed)
                  - 5 minutes = balanced
                  - 15 minutes = longer bars */}
              <select>
                <option>1 minute</option>
                <option selected>5 minutes</option>
                <option>15 minutes</option>
              </select>
              
              {/* Date Range: Historical range to fetch (optional for snapshot mode) */}
              <input type="date" defaultValue="2024-10-01" />
              <input type="date" defaultValue="2024-11-01" />
            </InputGroup>
            <div className="btn-group">
              {/* Start Streaming Button: Begins Yahoo Finance live snapshot
                  - Fetches recent bars with 15-20 min delay
                  - Updates chart in real-time as new bars arrive
                  - Does NOT interfere with Live Trading tab agents
                  - Separate simulation environment */}
              <Button className="start">Start Streaming</Button>
              
              {/* Stop Button: Stops Yahoo Finance snapshot updates */}
              <Button className="stop">Stop</Button>
            </div>
          </div>
        )}

        {/* IBKR Controls - Show only when IBKR is selected */}
        {dataSource === 'ibkr' && (
          <div className="yf-controls">
            <InputGroup>
              {/* Symbol: Stock/ETF to stream from IBKR */}
              <input type="text" placeholder="Symbol (e.g., AAPL)" defaultValue="AAPL" />
              
              {/* Timeframe: Bar interval */}
              <select>
                <option>1 minute</option>
                <option selected>5 minutes</option>
                <option>15 minutes</option>
              </select>
            </InputGroup>
            <div className="btn-group">
              {/* Start Streaming Button: IBKR real-time data (purpose TBD) */}
              <Button className="start">Start Streaming</Button>
              <Button className="stop">Stop</Button>
            </div>
            <div style={{ fontSize: '10px', color: '#f59e0b', marginTop: '8px' }}>
              ⚠️ Purpose and usage to be determined
            </div>
          </div>
        )}
      </Card>

      <div className="chart-area" style={{ minHeight: '350px', marginBottom: '12px' }}>
        <div className="chart-title">Simulated Live Chart</div>
        <div className="chart-placeholder" style={{ height: '310px' }}>
          {dataSource === 'csv' && 'CSV data will stream here (bar by bar playback)'}
          {dataSource === 'yahoo' && 'Yahoo Finance live snapshot (15-20 min delay)'}
          {dataSource === 'ibkr' && 'IBKR real-time data (purpose TBD)'}
        </div>
      </div>

      <Card>
        <div className="control-title">Agent Simulation</div>
        
        {/* Currently Selected Model Indicator */}
        <div style={{ 
          marginBottom: '12px', 
          padding: '8px', 
          background: '#0d1117', 
          borderRadius: '6px',
          border: '1px solid #30363d',
          fontSize: '11px'
        }}>
          <strong>Active Model:</strong> {currentModel?.name} 
          <span style={{ color: '#8b949e', marginLeft: '8px' }}>
            ({currentModel?.algorithm} trained on {currentModel?.symbol})
          </span>
        </div>
        
        <div className="param-grid">
          <div className="param-item">
            <div className="param-label">Starting Capital</div>
            {/* Virtual starting balance for simulation
                - No real money involved
                - Used to calculate P&L and position sizes
                - Helps estimate real-world performance */}
            <input type="number" className="param-input" defaultValue="100000" />
          </div>
          <div className="param-item">
            <div className="param-label">Playback Speed</div>
            {/* Simulation speed control (for CSV/historical data)
                - 1x = real-time speed
                - 10x = 10 times faster
                - 100x = very fast backtest
                - Only applies to CSV and Yahoo historical data */}
            <select className="param-input">
              <option>1x (Real-time)</option>
              <option selected>10x</option>
              <option>50x</option>
              <option>100x</option>
              <option>Max Speed</option>
            </select>
          </div>
        </div>
        <div className="btn-group" style={{ marginTop: '10px' }}>
          {/* Start Simulation Button: Agent begins trading on streamed data
              - Selected model ({currentModel?.name}) will make trading decisions
              - Agent analyzes incoming bars and decides BUY/SELL/HOLD
              - Simulated orders filled at market prices
              - P&L and equity curve tracked throughout simulation
              - Example Flow:
                1. Select model: PPO_AAPL_v3.2
                2. Load INTC 3-year CSV
                3. Start Streaming → Data loads
                4. Start Simulation → Agent trades on INTC using AAPL-trained model
                5. Watch BUY/SELL decisions + Equity curve
              - Perfect for testing model generalization and performance */}
          <Button className="start">Start Simulation</Button>
          
          {/* Stop Button: Ends simulation and closes positions
              - Final P&L and equity curve displayed
              - Results can be reviewed in Monitoring tab
              - Agent stops making decisions */}
          <Button className="stop">Stop</Button>
          
          {/* Reset Button: Clears simulation results and resets to initial state */}
          <Button className="" style={{ background: '#1f2937' }}>Reset</Button>
        </div>
        
        {/* Simulation Results Summary */}
        <div style={{ marginTop: '12px', padding: '10px', background: '#0d1117', borderRadius: '6px', border: '1px solid #21262d' }}>
          <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '8px' }}>Simulation Results</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '10px', fontSize: '11px' }}>
            <div>
              <div style={{ color: '#6e7681' }}>Total Trades</div>
              <div style={{ fontSize: '14px', fontWeight: 600 }}>--</div>
            </div>
            <div>
              <div style={{ color: '#6e7681' }}>Win Rate</div>
              <div style={{ fontSize: '14px', fontWeight: 600 }}>--</div>
            </div>
            <div>
              <div style={{ color: '#6e7681' }}>Final P&L</div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: '#3fb950' }}>--</div>
            </div>
            <div>
              <div style={{ color: '#6e7681' }}>Final Equity</div>
              <div style={{ fontSize: '14px', fontWeight: 600 }}>$100,000</div>
            </div>
          </div>
          
          {/* Model Performance Comparison */}
          <div style={{ 
            marginTop: '10px', 
            paddingTop: '10px', 
            borderTop: '1px solid #21262d',
            fontSize: '10px',
            color: '#6e7681'
          }}>
            💡 <strong>Tip:</strong> Run simulation with different models on same data to compare performance. 
            Example: Test PPO_AAPL vs PPO_MSFT on INTC data to see which generalizes better.
          </div>
        </div>
      </Card>
    </div>
  );
}

export default TabSimulation;
