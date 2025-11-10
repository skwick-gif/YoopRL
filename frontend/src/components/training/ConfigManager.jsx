/**
 * ConfigManager.jsx
 * Quick preset selector for training configurations
 *
 * Purpose:
 * - Provide curated presets tailored to PPO (stocks) and SAC (leveraged ETFs)
 * - Notify parent component when a preset is applied
 *
 * Props:
 * - onLoadConfig: callback that receives the selected preset config
 * - agentType: active agent type ('PPO' | 'SAC')
 */

import React, { useState, useRef, useEffect } from 'react';
import { DEFAULT_COMMISSION_CONFIG } from '../../hooks/useTrainingState';

const todayIso = () => new Date().toISOString().split('T')[0];

const ConfigManager = ({ onLoadConfig, agentType, compact = false }) => {
  const [message, setMessage] = useState({ text: '', type: '' });
  const timeoutRef = useRef(null);

  // Preset configurations - different for PPO (stocks) and SAC (ETFs)
  const defaultEndDate = todayIso();
  const buildCommissionConfig = (perShare = DEFAULT_COMMISSION_CONFIG.per_share) => ({
    per_share: perShare,
    min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
    max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
  });

  const presets = {
    PPO: {
      Conservative: {
        name: 'Conservative (Stock)',
        description: 'Low risk, stable returns for stocks',
        config: {
          name: 'Conservative (Stock)',
          agent_type: 'PPO',
          hyperparameters: {
            learning_rate: 0.0001,
            gamma: 0.99,
            batch_size: 512,
            n_steps: 2048,
            ent_coef: 0.01,
            clip_range: 0.2,
            episodes: 60000,
            risk_penalty: -0.8
          },
          features: {
            price: true,
            volume: true,
            ohlc: true,
            rsi: { enabled: true, period: 14 },
            macd: { enabled: true, params: '12,26,9' },
            ema: { enabled: true, periods: '10,50' },
            vix: true,
            bollinger: { enabled: false, params: '20,2' },
            stochastic: { enabled: false, params: '14,3' },
            sentiment: false,
            social_media: false,
            news_headlines: false,
            market_events: false,
            fundamental: false,
            multi_asset: { enabled: false, symbols: ['SPY', 'QQQ', 'TLT', 'GLD'] },
            macro: false,
            recent_actions: true,
            performance: { enabled: true, period: '60d' },
            position_history: true,
            reward_history: false,
            llm: { enabled: false, provider: 'Perplexity API' }
          },
          training_settings: {
            start_date: '2018-01-01',
            end_date: defaultEndDate,
            commission: buildCommissionConfig(DEFAULT_COMMISSION_CONFIG.per_share),
            commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
            commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
            commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
            commission_model: 'ibkr_tiered_us_equities',
            commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
            commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
            commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
            commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
              commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
              commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
              commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
              commission_model: 'ibkr_tiered_us_equities',
            optuna_trials: 120,
            max_position_size: 0.7
          }
        }
      },
      Aggressive: {
        name: 'Aggressive (Stock)',
        description: 'Higher risk, faster learning for stocks',
        config: {
          name: 'Aggressive (Stock)',
          agent_type: 'PPO',
          hyperparameters: {
            learning_rate: 0.0005,
            gamma: 0.95,
            batch_size: 128,
            n_steps: 1024,
            ent_coef: 0.05,
            clip_range: 0.3,
            episodes: 40000,
            risk_penalty: -0.25
          },
          features: {
            price: true,
            volume: true,
            ohlc: true,
            rsi: { enabled: true, period: 9 },
            macd: { enabled: true, params: '8,21,5' },
            ema: { enabled: true, periods: '5,20,50' },
            vix: true,
            bollinger: { enabled: true, params: '20,2' },
            stochastic: { enabled: true, params: '14,3' },
            sentiment: false,
            social_media: false,
            news_headlines: false,
            market_events: false,
            fundamental: false,
            multi_asset: { enabled: false, symbols: ['SPY', 'QQQ', 'TLT', 'GLD'] },
            macro: false,
            recent_actions: true,
            performance: { enabled: true, period: '30d' },
            position_history: true,
            reward_history: true,
            llm: { enabled: false, provider: 'Perplexity API' }
          },
          training_settings: {
            start_date: '2020-01-01',
            end_date: defaultEndDate,
            commission: buildCommissionConfig(DEFAULT_COMMISSION_CONFIG.per_share),
            commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
            commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
            commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
            commission_model: 'ibkr_tiered_us_equities',
            commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
            commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
            commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
            commission_model: 'ibkr_tiered_us_equities',
            commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
            commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
            commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
            commission_model: 'ibkr_tiered_us_equities',
            optuna_trials: 80,
            max_position_size: 1.0
          }
        }
      },
      Balanced: {
        name: 'Balanced (Stock)',
        description: 'Moderate risk-return for stocks',
        config: {
          name: 'Balanced (Stock)',
          agent_type: 'PPO',
          hyperparameters: {
            learning_rate: 0.0003,
            gamma: 0.99,
            batch_size: 256,
            n_steps: 2048,
            ent_coef: 0.02,
            clip_range: 0.2,
            episodes: 50000,
            risk_penalty: -0.5
          },
          features: {
            price: true,
            volume: true,
            ohlc: true,
            rsi: { enabled: true, period: 14 },
            macd: { enabled: true, params: '12,26,9' },
            ema: { enabled: true, periods: '10,50' },
            vix: true,
            bollinger: { enabled: true, params: '20,2' },
            stochastic: { enabled: false, params: '14,3' },
            sentiment: false,
            social_media: false,
            news_headlines: false,
            market_events: false,
            fundamental: false,
            multi_asset: { enabled: false, symbols: ['SPY', 'QQQ', 'TLT', 'GLD'] },
            macro: false,
            recent_actions: true,
            performance: { enabled: true, period: '45d' },
            position_history: true,
            reward_history: false,
            llm: { enabled: false, provider: 'Perplexity API' }
          },
          training_settings: {
            start_date: '2019-01-01',
            end_date: defaultEndDate,
            commission: buildCommissionConfig(DEFAULT_COMMISSION_CONFIG.per_share),
            commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
            commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
            commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
            commission_model: 'ibkr_tiered_us_equities',
            optuna_trials: 100,
            max_position_size: 0.85
          }
        }
      }
    },
    SAC: {
      Conservative: {
        name: 'Conservative (ETF)',
        description: 'Low risk, stable returns for leveraged ETFs',
        config: {
          name: 'Conservative (ETF)',
          agent_type: 'SAC',
          hyperparameters: {
            learning_rate: 0.0001,
            gamma: 0.99,
            batch_size: 512,
            tau: 0.005,
            ent_coef: 0.1,
            buffer_size: 100000,
            episodes: 60000,
            vol_penalty: -0.45,
            leverage_factor: 2.0
          },
          features: {
            price: true,
            volume: true,
            ohlc: true,
            rsi: { enabled: true, period: 14 },
            macd: { enabled: true, params: '12,26,9' },
            ema: { enabled: true, periods: '10,50' },
            vix: true,
            bollinger: { enabled: true, params: '20,2' },
            stochastic: { enabled: true, params: '14,3' },
            sentiment: true,
            social_media: true,
            news_headlines: true,
            market_events: true,
            fundamental: false,
            multi_asset: { enabled: true, symbols: ['SPY', 'QQQ', 'TLT'] },
            macro: true,
            recent_actions: true,
            performance: { enabled: true, period: '30d' },
            position_history: true,
            reward_history: true,
            llm: { enabled: false, provider: 'Perplexity API' }
          },
          training_settings: {
            start_date: '2020-01-01',
            end_date: defaultEndDate,
            commission: buildCommissionConfig(DEFAULT_COMMISSION_CONFIG.per_share),
            commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
            commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
            commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
            commission_model: 'ibkr_tiered_us_equities',
            optuna_trials: 120,
            max_position_size: 0.7
          }
        }
      },
      Aggressive: {
        name: 'Aggressive (ETF)',
        description: 'High risk, high reward for leveraged ETFs',
        config: {
          name: 'Aggressive (ETF)',
          agent_type: 'SAC',
          hyperparameters: {
            learning_rate: 0.0005,
            gamma: 0.95,
            batch_size: 128,
            tau: 0.01,
            ent_coef: 0.3,
            buffer_size: 50000,
            episodes: 40000,
            vol_penalty: -0.2,
            leverage_factor: 3.0
          },
          features: {
            price: true,
            volume: true,
            ohlc: true,
            rsi: { enabled: true, period: 9 },
            macd: { enabled: true, params: '8,21,5' },
            ema: { enabled: true, periods: '5,20,50' },
            vix: true,
            bollinger: { enabled: true, params: '20,2' },
            stochastic: { enabled: true, params: '10,3' },
            sentiment: true,
            social_media: true,
            news_headlines: true,
            market_events: true,
            fundamental: true,
            multi_asset: { enabled: true, symbols: ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'] },
            macro: true,
            recent_actions: { enabled: true, length: 10 },
            performance: { enabled: true, period: '21d' },
            position_history: { enabled: true, length: 10 },
            reward_history: true,
            llm: { enabled: true, provider: 'Perplexity API' }
          },
          training_settings: {
            start_date: '2021-01-01',
            end_date: defaultEndDate,
            commission: buildCommissionConfig(DEFAULT_COMMISSION_CONFIG.per_share),
            commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
            commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
            commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
            commission_model: 'ibkr_tiered_us_equities',
            optuna_trials: 90,
            max_position_size: 1.0
          }
        }
      },
      Balanced: {
        name: 'Balanced (ETF)',
        description: 'Moderate risk-return for leveraged ETFs',
        config: {
          name: 'Balanced (ETF)',
          agent_type: 'SAC',
          hyperparameters: {
            learning_rate: 0.0003,
            gamma: 0.99,
            batch_size: 256,
            tau: 0.005,
            ent_coef: 0.2,
            buffer_size: 100000,
            episodes: 50000,
            vol_penalty: -0.3,
            leverage_factor: 2.5
          },
          features: {
            price: true,
            volume: true,
            ohlc: true,
            rsi: { enabled: true, period: 14 },
            macd: { enabled: true, params: '12,26,9' },
            ema: { enabled: true, periods: '10,50' },
            vix: true,
            bollinger: { enabled: true, params: '20,2' },
            stochastic: { enabled: true, params: '14,3' },
            sentiment: true,
            social_media: true,
            news_headlines: true,
            market_events: true,
            fundamental: false,
            multi_asset: { enabled: true, symbols: ['SPY', 'QQQ', 'TLT', 'GLD'] },
            macro: true,
            recent_actions: true,
            performance: { enabled: true, period: '28d' },
            position_history: true,
            reward_history: true,
            llm: { enabled: false, provider: 'Perplexity API' }
          },
          training_settings: {
            start_date: '2020-06-01',
            end_date: defaultEndDate,
            commission: buildCommissionConfig(DEFAULT_COMMISSION_CONFIG.per_share),
            commission_per_share: DEFAULT_COMMISSION_CONFIG.per_share,
            commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
            commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
            commission_model: 'ibkr_tiered_us_equities',
            optuna_trials: 110,
            max_position_size: 0.85
          }
        }
      }
    }
  };

  // Get presets for current agent type
  const currentPresets = presets[agentType] || presets.PPO;

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const showMessage = (text, type) => {
    setMessage({ text, type });
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = setTimeout(() => {
      setMessage({ text: '', type: '' });
    }, 3000);
  };

  // Load preset configuration
  const handleLoadPreset = (presetName) => {
    const preset = currentPresets[presetName];
    if (preset) {
      onLoadConfig(preset.config);
      showMessage(`${preset.name} preset loaded!`, 'success');
    }
  };

  const containerStyle = compact ? styles.toolbarCompact : styles.toolbar;
  const labelStyle = compact ? styles.labelCompact : styles.label;

  return (
    <div style={containerStyle}>
      <span style={labelStyle}>
        Quick Presets {agentType && `(${agentType === 'PPO' ? 'Stock' : 'ETF'})`}
      </span>
      <div style={styles.buttonRow}>
        {Object.entries(currentPresets).map(([key, preset]) => (
          <button
            key={key}
            style={styles.presetButton}
            onClick={() => handleLoadPreset(key)}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#45a049'}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#4CAF50'}
          >
            {preset.name}
          </button>
        ))}
      </div>
      {message.text && (
        <div
          style={{
            ...styles.inlineMessage,
            color: message.type === 'success' ? '#4CAF50' : '#ff6b6b'
          }}
        >
          {message.text}
        </div>
      )}
    </div>
  );
};

const styles = {
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    flexWrap: 'wrap',
    padding: '12px 16px',
    backgroundColor: '#1e1e1e',
    border: '1px solid #30363d',
    borderRadius: '8px',
    marginBottom: '12px'
  },
  toolbarCompact: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    flexWrap: 'wrap',
    padding: '0',
    marginBottom: '0'
  },
  label: {
    color: '#c9d1d9',
    fontWeight: 'bold',
    fontSize: '13px'
  },
  labelCompact: {
    color: '#8b949e',
    fontWeight: 600,
    fontSize: '12px',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    whiteSpace: 'nowrap'
  },
  buttonRow: {
    display: 'flex',
    gap: '8px',
    flexWrap: 'wrap'
  },
  presetButton: {
    padding: '6px 14px',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '20px',
    fontSize: '12px',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'background-color 0.25s ease-in-out'
  },
  inlineMessage: {
    flexBasis: '100%',
    fontSize: '12px',
    fontWeight: 600,
    marginTop: '4px'
  }
};

export default ConfigManager;
