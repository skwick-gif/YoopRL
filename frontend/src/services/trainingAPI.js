/**
 * trainingAPI.js
 * Training API Service Layer
 * 
 * Purpose:
 * - Centralized service for all training-related API calls
 * - Handles HTTP requests to backend training endpoints
 * - Error handling and response formatting
 * - Isolates network logic from UI components
 * 
 * Why separate file:
 * - Single source of truth for API endpoint definitions
 * - Reusable across multiple components
 * - Easy to mock for testing
 * - Decouples UI from backend implementation
 * 
 * API Endpoints:
 * - POST /api/training/train - Start new training session
 * - POST /api/training/stop - Stop active training
 * - GET /api/training/progress/:id - Get training progress
 * - GET /api/training/models - List all trained models
 * - POST /api/training/load_model - Load specific model
 * - POST /api/training/save_config - Save training config
 * - GET /api/training/load_config/:name - Load saved config
 * - GET /api/training/drift_status - Check for data drift
 * - POST /api/training/backtest - Run backtesting
 * 
 * Usage in components:
 * import { startTraining, getProgress } from '../services/trainingAPI';
 * const result = await startTraining(config);
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const toDateOnly = (value) => {
  if (!value) {
    return null;
  }

  if (typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value.trim())) {
    return value.trim();
  }

  const parsed = new Date(value);
  if (!Number.isNaN(parsed.getTime())) {
    return parsed.toISOString().split('T')[0];
  }

  return null;
};

/**
 * Start Training
 * 
 * @param {Object} config - Training configuration from useTrainingState
 * @returns {Promise<Object>} { training_id, status, message }
 * 
 * Example config:
 * {
 *   agent_type: 'PPO',
 *   symbol: 'AAPL',
 *   hyperparameters: { learning_rate: 0.0003, ... },
 *   features: { price: true, rsi: { enabled: true, period: 14 }, ... },
 *   training_settings: { start_date: '2023-01-01', ... }
 * }
 */
export const startTraining = async (config) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/training/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Failed to start training');
    }

    const data = await response.json();
    return {
      success: true,
      training_id: data.training_id,
      status: data.status,
      message: data.message
    };
  } catch (error) {
    console.error('Error starting training:', error);
    return {
      success: false,
      error: error.message
    };
  }
};

/**
 * Stop Training
 * 
 * @param {string} trainingId - ID of active training session
 * @returns {Promise<Object>} { success, message }
 */
export const stopTraining = async (trainingId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/training/stop`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ training_id: trainingId })
    });

    if (!response.ok) {
      throw new Error('Failed to stop training');
    }

    const data = await response.json();
    return {
      success: true,
      message: data.message
    };
  } catch (error) {
    console.error('Error stopping training:', error);
    return {
      success: false,
      error: error.message
    };
  }
};

/**
 * Get Training Progress
 * 
 * @param {string} trainingId - ID of training session
 * @returns {Promise<Object>} Progress data with episode, reward, loss, etc.
 * 
 * Response format:
 * {
 *   training_id: 'abc123',
 *   status: 'running' | 'completed' | 'failed',
 *   progress: 75,
 *   current_episode: 37500,
 *   total_episodes: 50000,
 *   avg_reward: 1250.5,
 *   recent_loss: 0.0023,
 *   elapsed_time: '02:15:30',
 *   eta: '00:45:00'
 * }
 */
export const getProgress = async (trainingId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/training/progress/${trainingId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });

    if (!response.ok) {
      throw new Error('Failed to get training progress');
    }

    const data = await response.json();
    return {
      success: true,
      data: data
    };
  } catch (error) {
    console.error('Error getting progress:', error);
    return {
      success: false,
      error: error.message
    };
  }
};

/**
 * Load Available Models
 * 
 * @returns {Promise<Array>} List of trained models
 * 
 * Response format:
 * [
 *   {
 *     model_id: 'ppo_aapl_v1.2',
 *     agent_type: 'PPO',
 *     symbol: 'AAPL',
 *     version: 'v1.2',
 *     created_at: '2024-11-01T10:30:00',
 *     episodes: 50000,
 *     sharpe_ratio: 1.85,
 *     total_return: 23.4,
 *     file_path: '/models/ppo_aapl_v1.2.pkl'
 *   },
 *   ...
 * ]
 */
export const loadModels = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/training/models`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });

    if (!response.ok) {
      throw new Error('Failed to load models');
    }

    const data = await response.json();
    return {
      success: true,
      models: data.models || []
    };
  } catch (error) {
    console.error('Error loading models:', error);
    return {
      success: false,
      models: [],
      error: error.message
    };
  }
};

/**
 * Load Specific Model
 * 
 * @param {string} modelPath - Path to model file
 * @returns {Promise<Object>} Success status and message
 */
export const loadModel = async (modelPath) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/training/load_model`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model_path: modelPath })
    });

    if (!response.ok) {
      throw new Error('Failed to load model');
    }

    const data = await response.json();
    return {
      success: true,
      message: data.message,
      model_info: data.model_info
    };
  } catch (error) {
    console.error('Error loading model:', error);
    return {
      success: false,
      error: error.message
    };
  }
};

/**
 * Save Training Configuration
 * 
 * @param {Object} config - Training configuration to save
 * @param {string} name - Name for saved configuration
 * @returns {Promise<Object>} Success status and message
 */
export const saveConfig = async (config, name) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/training/save_config`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ config, name })
    });

    if (!response.ok) {
      throw new Error('Failed to save configuration');
    }

    const data = await response.json();
    return {
      success: true,
      message: data.message,
      config_id: data.config_id
    };
  } catch (error) {
    console.error('Error saving config:', error);
    return {
      success: false,
      error: error.message
    };
  }
};

/**
 * Load Saved Configuration
 * 
 * @param {string} configName - Name of saved configuration
 * @returns {Promise<Object>} Configuration data
 */
export const loadConfig = async (configName) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/training/load_config/${configName}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });

    if (!response.ok) {
      throw new Error('Failed to load configuration');
    }

    const data = await response.json();
    return {
      success: true,
      config: data.config
    };
  } catch (error) {
    console.error('Error loading config:', error);
    return {
      success: false,
      error: error.message
    };
  }
};

/**
 * Check Data Drift Status
 * 
 * @param {string} symbol - Stock symbol (e.g., 'AAPL')
 * @param {string} agentType - 'PPO' or 'SAC' (default: 'PPO')
 * @param {number} days - Days to check (default: 30)
 * @returns {Promise<Object>} Drift detection results
 * 
 * Response format:
 * {
 *   drift_detected: true,
 *   severity: 'medium' | 'high' | 'critical',
 *   affected_features: ['rsi', 'macd'],
 *   drift_scores: { rsi: 0.45, macd: 0.62 },
 *   threshold: 0.4,
 *   recommendation: 'Retrain model with recent data',
 *   last_check: '2024-11-01T14:30:00'
 * }
 */
export const checkDriftStatus = async (symbol = 'AAPL', agentType = 'PPO', days = 30) => {
  try {
    const url = `${API_BASE_URL}/api/training/drift_status?symbol=${symbol}&agent_type=${agentType}&days=${days}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });

    if (!response.ok) {
      throw new Error('Failed to check drift status');
    }

    const data = await response.json();
    return {
      success: true,
      drift_data: data
    };
  } catch (error) {
    console.error('Error checking drift:', error);
    return {
      success: false,
      drift_detected: false,
      error: error.message
    };
  }
};

/**
 * Fetch Cached Training Date Range
 *
 * @param {Object} params - Query parameters
 * @param {string} params.symbol - Trading symbol to inspect
 * @param {string} [params.frequency='daily'] - 'daily' or 'intraday'
 * @param {string} [params.interval] - Interval hint (e.g., '15m' for intraday)
 * @returns {Promise<Object>} { success, start_date, end_date, source }
 */
export const fetchTrainingDateRange = async ({ symbol, frequency = 'daily', interval } = {}) => {
  if (!symbol) {
    return {
      success: false,
      error: 'Symbol is required'
    };
  }

  const params = new URLSearchParams({
    symbol,
    frequency
  });

  if (interval) {
    params.append('interval', interval);
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/training/date-range?${params.toString()}`);
    const data = await response.json().catch(() => ({}));

    if (!response.ok || data.status === 'error') {
      return {
        success: false,
        status: data.status || 'error',
        error: data.error || data.message || `Failed to fetch date range (HTTP ${response.status})`
      };
    }

    return {
      success: true,
      status: data.status,
      start_date: toDateOnly(data.start_date),
      end_date: toDateOnly(data.end_date),
      interval: data.interval,
      frequency: data.frequency,
      source: data.source
    };
  } catch (error) {
    console.error('Error fetching date range:', error);
    return {
      success: false,
      status: 'error',
      error: error.message
    };
  }
};

/**
 * Run Backtesting
 * 
 * @param {Object} backtestRequest - Backtest parameters
 * @returns {Promise<Object>} Backtest results
 * 
 * Request format:
 * {
 *   model_path: '/models/ppo_aapl_v1.2.pkl',
 *   start_date: '2024-01-01',
 *   end_date: '2024-11-01',
 *   initial_capital: 100000,
 *   commission: 1.0
 * }
 * 
 * Response format:
 * {
 *   sharpe_ratio: 1.85,
 *   sortino_ratio: 2.12,
 *   max_drawdown: -12.5,
 *   win_rate: 58.3,
 *   total_return: 23.4,
 *   final_portfolio_value: 123400,
 *   num_trades: 145,
 *   avg_trade_return: 0.85,
 *   equity_curve: [...],
 *   trades: [...]
 * }
 */
export const runBacktest = async (backtestRequest) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/training/backtest`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(backtestRequest)
    });

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const message = data?.error || data?.message || `Failed to run backtest (HTTP ${response.status})`;
      throw new Error(message);
    }

    if (!data) {
      throw new Error('Failed to parse backtest response');
    }

    return {
      success: true,
      results: data
    };
  } catch (error) {
    console.error('Error running backtest:', error);
    return {
      success: false,
      error: error.message
    };
  }
};
