/**
 * FeatureSelection.jsx (Updated - Phase 1)
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
 * Phase 1 Updates:
 * - Converted from uncontrolled (defaultChecked) to controlled components (checked + onChange)
 * - Accepts trainingState props from TabTraining.jsx
 * - All checkboxes and inputs wired to state management
 * 
 * Props:
 * - trainingState: Object with all state values and setters from useTrainingState hook
 * 
 * Wiring:
 * - Each checkbox uses checked={trainingState.X} and onChange
 * - Parameter inputs use value and onChange
 * - Changes immediately reflected in parent state
 */

import React from 'react';
import { Card } from '../common/UIComponents';

function FeatureSelection({ trainingState }) {
  return (
    <Card style={{ marginBottom: '12px' }}>
      <div className="control-title">Input Features Selection</div>
      <div className="feature-grid">
        {/* Price Data - Core features (mandatory) */}
        <div className="feature-group">
          <div className="feature-group-title">Price Data</div>
          <label className="feature-checkbox" title="Price: The current trading price of the asset. This is the most fundamental data point - always required.">
            <input type="checkbox" checked disabled /> Price
          </label>
          <label className="feature-checkbox" title="Volume: Number of shares traded. High volume = strong interest. Low volume = weak interest. Confirms price movements.">
            <input type="checkbox" checked disabled /> Volume
          </label>
          <label className="feature-checkbox" title="OHLC: Open, High, Low, Close prices. Shows the full price range during each time period. Useful for pattern recognition.">
            <input 
              type="checkbox" 
              checked={trainingState.ohlcEnabled}
              onChange={(e) => trainingState.setOhlcEnabled(e.target.checked)}
            /> OHLC
          </label>
        </div>

        {/* Technical Indicators - Optional indicators with configurable parameters */}
        <div className="feature-group">
          <div className="feature-group-title">Technical Indicators</div>
          
          <label className="feature-checkbox" title="RSI (Relative Strength Index): Measures if a stock is overbought (>70) or oversold (<30). Period 14 is standard. Helps identify potential reversals.">
            <input 
              type="checkbox" 
              checked={trainingState.rsiEnabled}
              onChange={(e) => trainingState.setRsiEnabled(e.target.checked)}
            /> RSI
            <input 
              type="number" 
              value={trainingState.rsiPeriod}
              onChange={(e) => trainingState.setRsiPeriod(parseInt(e.target.value))}
              title="RSI Period: Number of bars to calculate RSI. 14 is standard. Lower values (7-10) = more sensitive. Higher (20-30) = smoother." 
            />
          </label>
          
          <label className="feature-checkbox" title="MACD (Moving Average Convergence Divergence): Shows trend direction and momentum. Parameters are fast period, slow period, and signal line. Helps identify trend changes.">
            <input 
              type="checkbox" 
              checked={trainingState.macdEnabled}
              onChange={(e) => trainingState.setMacdEnabled(e.target.checked)}
            /> MACD
            <input 
              type="text" 
              value={trainingState.macdParams}
              onChange={(e) => trainingState.setMacdParams(e.target.value)}
              title="MACD Parameters: Fast period (12), Slow period (26), Signal line (9). Standard settings work well for most stocks." 
            />
          </label>
          
          <label className="feature-checkbox" title="EMA (Exponential Moving Average): Shows average price over time, giving more weight to recent prices. Helps identify support/resistance levels.">
            <input 
              type="checkbox" 
              checked={trainingState.emaEnabled}
              onChange={(e) => trainingState.setEmaEnabled(e.target.checked)}
            /> EMA
            <input 
              type="text" 
              value={trainingState.emaPeriods}
              onChange={(e) => trainingState.setEmaPeriods(e.target.value)}
              title="EMA Periods: Short term (10) and long term (50). Crossovers indicate potential buy/sell signals. 10,50 is common for day trading." 
            />
          </label>
          
          <label className="feature-checkbox" title="VIX (Volatility Index): Measures market fear and volatility. High VIX = more market uncertainty. Helps adjust risk during volatile periods.">
            <input 
              type="checkbox" 
              checked={trainingState.vixEnabled}
              onChange={(e) => trainingState.setVixEnabled(e.target.checked)}
            /> VIX
          </label>
          
          <label className="feature-checkbox" title="Bollinger Bands: Shows price volatility using upper and lower bands. Price touching upper band = potentially overbought. Touching lower = oversold.">
            <input 
              type="checkbox" 
              checked={trainingState.bollingerEnabled}
              onChange={(e) => trainingState.setBollingerEnabled(e.target.checked)}
            /> Bollinger
            <input 
              type="text" 
              value={trainingState.bollingerParams}
              onChange={(e) => trainingState.setBollingerParams(e.target.value)}
              title="Bollinger Parameters: Period (20) and standard deviations (2). Standard settings capture ~95% of price action." 
            />
          </label>
          
          <label className="feature-checkbox" title="Stochastic Oscillator: Compares closing price to price range over time. Values >80 = overbought, <20 = oversold. Good for ranging markets.">
            <input 
              type="checkbox" 
              checked={trainingState.stochasticEnabled}
              onChange={(e) => trainingState.setStochasticEnabled(e.target.checked)}
            /> Stochastic
            <input 
              type="text" 
              value={trainingState.stochasticParams}
              onChange={(e) => trainingState.setStochasticParams(e.target.value)}
              title="Stochastic Parameters: K period (14) and D period (3). Standard settings work for most timeframes." 
            />
          </label>

          <label className="feature-checkbox" title="ADX / DMI: Average Directional Index with +DI / -DI to gauge trend strength. ADX above 25 usually signals a strong trend, below 20 a weak one.">
            <input 
              type="checkbox" 
              checked={trainingState.adxEnabled}
              onChange={(e) => trainingState.setAdxEnabled(e.target.checked)}
            /> ADX / DMI
            <input 
              type="number"
              min={5}
              value={trainingState.adxPeriod}
              onChange={(e) => trainingState.setAdxPeriod(parseInt(e.target.value, 10) || 14)}
              title="ADX Period: Default 14. Lower values respond faster but can be noisy; higher values smooth the trend strength." 
            />
          </label>
        </div>

        {/* Alternative Data - External data sources (requires API keys/subscriptions) */}
        <div className="feature-group">
          <div className="feature-group-title">Alternative Data</div>
          <label className="feature-checkbox" title="Sentiment (News): Analyzes news articles to determine positive/negative market sentiment. Range: -1 (negative) to +1 (positive). Requires API key.">
            <input 
              type="checkbox" 
              checked={trainingState.sentimentEnabled}
              onChange={(e) => trainingState.setSentimentEnabled(e.target.checked)}
            /> Sentiment (News)
          </label>
          <label className="feature-checkbox" title="Social Media: Tracks mentions and sentiment on Twitter, Reddit, StockTwits. Captures retail investor sentiment. Requires API keys.">
            <input 
              type="checkbox" 
              checked={trainingState.socialMediaEnabled}
              onChange={(e) => trainingState.setSocialMediaEnabled(e.target.checked)}
            /> Social Media
          </label>
          <label className="feature-checkbox" title="News Headlines: Counts and analyzes breaking news headlines about the stock. More news = more market attention. Requires NewsAPI key.">
            <input 
              type="checkbox" 
              checked={trainingState.newsHeadlinesEnabled}
              onChange={(e) => trainingState.setNewsHeadlinesEnabled(e.target.checked)}
            /> News Headlines
          </label>
          <label className="feature-checkbox" title="Market Events: Tracks earnings reports, dividends, stock splits, and other corporate events. These often cause price movements.">
            <input 
              type="checkbox" 
              checked={trainingState.marketEventsEnabled}
              onChange={(e) => trainingState.setMarketEventsEnabled(e.target.checked)}
            /> Market Events
          </label>
          <label className="feature-checkbox" title="Fundamental: Company financials like P/E ratio, EPS, revenue growth. Good for long-term investing. Updates quarterly.">
            <input 
              type="checkbox" 
              checked={trainingState.fundamentalEnabled}
              onChange={(e) => trainingState.setFundamentalEnabled(e.target.checked)}
            /> Fundamental
          </label>
          <label className="feature-checkbox" title="Multi-Asset Correlation: Tracks correlation with SPY, QQQ, TLT, GLD. Helps understand market context and portfolio risk.">
            <input 
              type="checkbox" 
              checked={trainingState.multiAssetEnabled}
              onChange={(e) => trainingState.setMultiAssetEnabled(e.target.checked)}
            /> Multi-Asset
            <input 
              type="text" 
              value={trainingState.multiAssetSymbols}
              onChange={(e) => trainingState.setMultiAssetSymbols(e.target.value)}
              title="Multi-Asset Symbols: Comma-separated list of symbols to track (e.g., SPY,QQQ,TLT,GLD). Each adds 6 correlation features."
            />
          </label>
          <label className="feature-checkbox" title="Macro Indicators: Market-wide economic data including VIX (volatility/fear gauge), Treasury Yields (10Y/2Y interest rates), Dollar Index (currency strength), Oil/Gold prices (commodities/inflation). Helps agent understand broader market regime, risk conditions, and economic cycles.">
            <input 
              type="checkbox" 
              checked={trainingState.macroEnabled}
              onChange={(e) => trainingState.setMacroEnabled(e.target.checked)}
            /> Macro Indicators (VIX, Yields, DXY, Commodities)
          </label>
        </div>

        {/* Agent History - Agent's own past performance and actions */}
        <div className="feature-group">
          <div className="feature-group-title">Agent History</div>
          <label className="feature-checkbox" title="Recent Actions: Last few buy/sell/hold decisions the agent made. Helps the agent learn from its own trading patterns.">
            <input 
              type="checkbox" 
              checked={trainingState.recentActionsEnabled}
              onChange={(e) => trainingState.setRecentActionsEnabled(e.target.checked)}
            /> Recent Actions
          </label>
          <label className="feature-checkbox" title="Performance: Agent's profit/loss over time. Helps it understand which strategies worked well in the past.">
            <input 
              type="checkbox" 
              checked={trainingState.performanceEnabled}
              onChange={(e) => trainingState.setPerformanceEnabled(e.target.checked)}
            /> Performance
            <input 
              type="text" 
              value={trainingState.performancePeriod}
              onChange={(e) => trainingState.setPerformancePeriod(e.target.value)}
              title="Performance Period: How many past days to track (e.g., '30d' = 30 days). Longer period = more historical context."
            />
          </label>
          <label className="feature-checkbox" title="Position History: Record of when the agent entered/exited positions. Shows holding patterns and position sizes over time.">
            <input 
              type="checkbox" 
              checked={trainingState.positionHistoryEnabled}
              onChange={(e) => trainingState.setPositionHistoryEnabled(e.target.checked)}
            /> Position History
          </label>
          <label className="feature-checkbox" title="Reward History: Past rewards (profits/losses) from previous trades. Core learning signal - teaches what actions led to profits.">
            <input 
              type="checkbox" 
              checked={trainingState.rewardHistoryEnabled}
              onChange={(e) => trainingState.setRewardHistoryEnabled(e.target.checked)}
            /> Reward History
          </label>

          {/* LLM Integration - Optional AI-powered market analysis */}
          <div style={{ marginTop: '10px', borderTop: '1px solid #21262d', paddingTop: '8px' }}>
            <div className="feature-group-title">LLM Integration</div>
            <label className="feature-checkbox" title="Enable LLM: Use AI language models (Perplexity or Gemini) to analyze market conditions and news. Provides natural language insights.">
              <input 
                type="checkbox" 
                checked={trainingState.llmEnabled}
                onChange={(e) => trainingState.setLlmEnabled(e.target.checked)}
              /> Enable LLM
            </label>
            <select 
              className="param-input" 
              disabled={!trainingState.llmEnabled} 
              style={{ marginTop: '4px', fontSize: '10px' }}
              value={trainingState.selectedLLM}
              onChange={(e) => trainingState.setSelectedLLM(e.target.value)}
              title="LLM Provider: Choose between Perplexity (good for news analysis) or Gemini (Google's AI, good for general market insights)."
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
