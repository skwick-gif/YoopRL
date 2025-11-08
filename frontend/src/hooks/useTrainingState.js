/**
 * useTrainingState.js
 * Custom React Hook for Training State Management
 * 
 * Purpose:
 * - Centralized state management for all training-related data
 * - Manages PPO/SAC hyperparameters, feature selection, training settings
 * - Provides validation and JSON serialization for API calls
 * 
 * Why separate file:
 * - Keeps TabTraining.jsx clean and focused on UI orchestration
 * - Reusable if we need training state in other components
 * - Easier to test state logic independently
 * - Single source of truth for training configuration
 * 
 * State Structure:
 * - ppoParams: {symbol, learning_rate, gamma, batch_size, risk_penalty, episodes, retrain}
 * - sacParams: {symbol, learning_rate, entropy, batch_size, vol_penalty, episodes, retrain}
 * - features: {price, volume, rsi, macd, ema, vix, sentiment, llm}
 * - trainingSettings: {start_date, end_date, commission, optuna_trials}
 * 
 * Returns:
 * - All state values
 * - Setter functions for each state
 * - buildTrainingConfig(): Builds JSON object for API calls
 * - validateConfig(): Validates all inputs before training
 * - resetConfig(): Resets to default values
 */

import { useState } from 'react';

export const useTrainingState = () => {
  // ===== PPO Hyperparameters (Stock Trading) =====
  const [ppoSymbol, setPpoSymbol] = useState('AAPL');
  const [ppoLearningRate, setPpoLearningRate] = useState(0.0003);
  const [ppoGamma, setPpoGamma] = useState(0.99);
  const [ppoBatchSize, setPpoBatchSize] = useState(256);
  const [ppoRiskPenalty, setPpoRiskPenalty] = useState(-0.5);
  const [ppoEpisodes, setPpoEpisodes] = useState(50000);
  const [ppoRetrain, setPpoRetrain] = useState('Weekly');

  // ===== SAC Hyperparameters (Leveraged ETF Trading) =====
  const [sacSymbol, setSacSymbol] = useState('TNA');
  const [sacLearningRate, setSacLearningRate] = useState(0.0003);
  const [sacEntropy, setSacEntropy] = useState(0.2);
  const [sacBatchSize, setSacBatchSize] = useState(256);
  const [sacVolPenalty, setSacVolPenalty] = useState(-0.3);
  const [sacEpisodes, setSacEpisodes] = useState(45000);
  const [sacRetrain, setSacRetrain] = useState('Weekly');

  // ===== Training Settings =====
  // Get current date in YYYY-MM-DD format
  const getCurrentDate = () => {
    const now = new Date();
    return now.toISOString().split('T')[0];
  };
  
  const [startDate, setStartDate] = useState('2020-01-01');  // Always start from 2020
  const [endDate, setEndDate] = useState(getCurrentDate());   // Always current date
  const [commission, setCommission] = useState(1.0);
  const [optunaTrials, setOptunaTrials] = useState(100);

  // ===== Feature Selection =====
  // Price Data (mandatory)
  const [priceEnabled] = useState(true);
  const [volumeEnabled] = useState(true);
  const [ohlcEnabled, setOhlcEnabled] = useState(true);

  // Technical Indicators
  const [rsiEnabled, setRsiEnabled] = useState(true);
  const [rsiPeriod, setRsiPeriod] = useState(14);
  
  const [macdEnabled, setMacdEnabled] = useState(true);
  const [macdParams, setMacdParams] = useState('12,26,9');
  
  const [emaEnabled, setEmaEnabled] = useState(true);
  const [emaPeriods, setEmaPeriods] = useState('10,50');
  
  const [vixEnabled, setVixEnabled] = useState(true);
  
  const [bollingerEnabled, setBollingerEnabled] = useState(false);
  const [bollingerParams, setBollingerParams] = useState('20,2');
  
  const [stochasticEnabled, setStochasticEnabled] = useState(false);
  const [stochasticParams, setStochasticParams] = useState('14,3');

  // Alternative Data
  const [sentimentEnabled, setSentimentEnabled] = useState(false);
  const [socialMediaEnabled, setSocialMediaEnabled] = useState(false);
  const [newsHeadlinesEnabled, setNewsHeadlinesEnabled] = useState(false);
  const [marketEventsEnabled, setMarketEventsEnabled] = useState(false);
  const [fundamentalEnabled, setFundamentalEnabled] = useState(false);
  const [multiAssetEnabled, setMultiAssetEnabled] = useState(false);
  const [multiAssetSymbols, setMultiAssetSymbols] = useState('SPY,QQQ,TLT,GLD');
  const [macroEnabled, setMacroEnabled] = useState(false);

  // Agent History
  const [recentActionsEnabled, setRecentActionsEnabled] = useState(true);
  const [performanceEnabled, setPerformanceEnabled] = useState(true);
  const [performancePeriod, setPerformancePeriod] = useState('30d');
  const [positionHistoryEnabled, setPositionHistoryEnabled] = useState(true);
  const [rewardHistoryEnabled, setRewardHistoryEnabled] = useState(false);

  // LLM Integration
  const [llmEnabled, setLlmEnabled] = useState(false);
  const [selectedLLM, setSelectedLLM] = useState('Perplexity API');

  /**
   * Build Training Configuration JSON
   * 
   * Returns JSON object ready to send to backend API
   */
  const buildTrainingConfig = (agentType) => {
    const config = {
      agent_type: agentType, // 'PPO' or 'SAC'
      symbol: agentType === 'PPO' ? ppoSymbol : sacSymbol,
      hyperparameters: agentType === 'PPO' ? {
        learning_rate: parseFloat(ppoLearningRate),
        gamma: parseFloat(ppoGamma),
        batch_size: parseInt(ppoBatchSize),
        risk_penalty: parseFloat(ppoRiskPenalty),
        episodes: parseInt(ppoEpisodes)
      } : {
        learning_rate: parseFloat(sacLearningRate),
        entropy_coef: parseFloat(sacEntropy),
        batch_size: parseInt(sacBatchSize),
        vol_penalty: parseFloat(sacVolPenalty),
        episodes: parseInt(sacEpisodes)
      },
      features: {
        price: priceEnabled,
        volume: volumeEnabled,
        ohlc: ohlcEnabled,
        rsi: {
          enabled: rsiEnabled,
          period: parseInt(rsiPeriod)
        },
        macd: {
          enabled: macdEnabled,
          params: macdParams
        },
        ema: {
          enabled: emaEnabled,
          periods: emaPeriods
        },
        vix: vixEnabled,
        bollinger: {
          enabled: bollingerEnabled,
          params: bollingerParams
        },
        stochastic: {
          enabled: stochasticEnabled,
          params: stochasticParams
        },
        sentiment: sentimentEnabled,
        social_media: socialMediaEnabled,
        news_headlines: newsHeadlinesEnabled,
        market_events: marketEventsEnabled,
        fundamental: fundamentalEnabled,
        multi_asset: {
          enabled: multiAssetEnabled,
          symbols: multiAssetSymbols
        },
        macro: macroEnabled,
        recent_actions: recentActionsEnabled,
        performance: {
          enabled: performanceEnabled,
          period: performancePeriod
        },
        position_history: positionHistoryEnabled,
        reward_history: rewardHistoryEnabled,
        llm: {
          enabled: llmEnabled,
          provider: selectedLLM
        }
      },
      training_settings: {
        start_date: startDate,
        end_date: endDate,
        commission: parseFloat(commission),
        optuna_trials: parseInt(optunaTrials)
      }
    };

    return config;
  };

  /**
   * Validate Configuration
   * 
   * Returns: { valid: boolean, errors: string[] }
   */
  const validateConfig = (agentType) => {
    const errors = [];

    // Validate hyperparameters
    const lr = agentType === 'PPO' ? ppoLearningRate : sacLearningRate;
    if (lr <= 0 || lr > 0.1) {
      errors.push('Learning rate must be between 0 and 0.1');
    }

    const episodes = agentType === 'PPO' ? ppoEpisodes : sacEpisodes;
    if (episodes < 1000) {
      errors.push('Episodes must be at least 1000');
    }

    // Validate dates
    const start = new Date(startDate);
    const end = new Date(endDate);
    if (start >= end) {
      errors.push('Start date must be before end date');
    }

    // Validate commission
    if (commission < 0) {
      errors.push('Commission cannot be negative');
    }

    // Validate Optuna trials
    if (optunaTrials < 10) {
      errors.push('Optuna trials must be at least 10');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  };

  /**
   * Reset Configuration to Defaults
   */
  const resetConfig = () => {
    // PPO defaults
    setPpoSymbol('AAPL');
    setPpoLearningRate(0.0003);
    setPpoGamma(0.99);
    setPpoBatchSize(256);
    setPpoRiskPenalty(-0.5);
    setPpoEpisodes(50000);
    setPpoRetrain('Weekly');

    // SAC defaults
    setSacSymbol('TNA');
    setSacLearningRate(0.0003);
    setSacEntropy(0.2);
    setSacBatchSize(256);
    setSacVolPenalty(-0.3);
    setSacEpisodes(45000);
    setSacRetrain('Weekly');

    // Training settings defaults
    setStartDate('2023-01-01');
    setEndDate('2024-11-01');
    setCommission(1.0);
    setOptunaTrials(100);

    // Features defaults
    setOhlcEnabled(true);
    setRsiEnabled(true);
    setRsiPeriod(14);
    setMacdEnabled(true);
    setMacdParams('12,26,9');
    setEmaEnabled(true);
    setEmaPeriods('10,50');
    setVixEnabled(true);
    setBollingerEnabled(false);
    setStochasticEnabled(false);
    setSentimentEnabled(false);
    setLlmEnabled(false);
  };

  // Return all state and functions
  return {
    // PPO State
    ppoSymbol, setPpoSymbol,
    ppoLearningRate, setPpoLearningRate,
    ppoGamma, setPpoGamma,
    ppoBatchSize, setPpoBatchSize,
    ppoRiskPenalty, setPpoRiskPenalty,
    ppoEpisodes, setPpoEpisodes,
    ppoRetrain, setPpoRetrain,

    // SAC State
    sacSymbol, setSacSymbol,
    sacLearningRate, setSacLearningRate,
    sacEntropy, setSacEntropy,
    sacBatchSize, setSacBatchSize,
    sacVolPenalty, setSacVolPenalty,
    sacEpisodes, setSacEpisodes,
    sacRetrain, setSacRetrain,

    // Training Settings
    startDate, setStartDate,
    endDate, setEndDate,
    commission, setCommission,
    optunaTrials, setOptunaTrials,

    // Features
    priceEnabled, volumeEnabled, ohlcEnabled, setOhlcEnabled,
    rsiEnabled, setRsiEnabled, rsiPeriod, setRsiPeriod,
    macdEnabled, setMacdEnabled, macdParams, setMacdParams,
    emaEnabled, setEmaEnabled, emaPeriods, setEmaPeriods,
    vixEnabled, setVixEnabled,
    bollingerEnabled, setBollingerEnabled, bollingerParams, setBollingerParams,
    stochasticEnabled, setStochasticEnabled, stochasticParams, setStochasticParams,
    sentimentEnabled, setSentimentEnabled,
    socialMediaEnabled, setSocialMediaEnabled,
    newsHeadlinesEnabled, setNewsHeadlinesEnabled,
    marketEventsEnabled, setMarketEventsEnabled,
    fundamentalEnabled, setFundamentalEnabled,
    multiAssetEnabled, setMultiAssetEnabled,
    multiAssetSymbols, setMultiAssetSymbols,
    macroEnabled, setMacroEnabled,
    recentActionsEnabled, setRecentActionsEnabled,
    performanceEnabled, setPerformanceEnabled, performancePeriod, setPerformancePeriod,
    positionHistoryEnabled, setPositionHistoryEnabled,
    rewardHistoryEnabled, setRewardHistoryEnabled,
    llmEnabled, setLlmEnabled, selectedLLM, setSelectedLLM,

    // Utility Functions
    buildTrainingConfig,
    validateConfig,
    resetConfig
  };
};
