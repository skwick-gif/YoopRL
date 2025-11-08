/**
 * FeatureSelection.jsx
 * Part of: Training Tab
 * 
 * Purpose: Allows selection of input features for training the RL agents
 * Features are grouped into categories:
 * - Price Data: Basic OHLCV data (always required)
 * - Technical Indicators: RSI, MACD, EMA, VIX, Bollinger, Stochastic
 * - Alternative Data: Sentiment, Social Media, News, Market Events, Fundamentals
 * - Agent History: Recent actions, performance, positions, rewards
 * - LLM Integration: Optional integration with Perplexity or Gemini API
 * 
 * Selected features will be downloaded and used during training
 */

import React from 'react';
import { Card } from '../common/UIComponents';

function FeatureSelection({ llmEnabled, setLlmEnabled, selectedLLM, setSelectedLLM }) {
  return (
    <Card style={{ marginBottom: '12px' }}>
      <div className="control-title">Input Features Selection</div>
      <div className="feature-grid">
        {/* Price Data - Core features (mandatory) */}
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

        {/* Technical Indicators - Optional indicators with configurable parameters */}
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

        {/* Alternative Data - External data sources (requires API keys/subscriptions) */}
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

        {/* Agent History - Agent's own past performance and actions */}
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

          {/* LLM Integration - Optional AI-powered market analysis */}
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
  );
}

export default FeatureSelection;
