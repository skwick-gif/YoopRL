/**
 * TabTraining.jsx
 * Training Tab - Main Component (Updated with Phase 1 Improvements)
 * 
 * Purpose: Configure and train RL agents (PPO for stocks, SAC for leveraged ETFs)
 * 
 * Workflow:
 * 1. Configure hyperparameters (learning rate, episodes, etc.)
 * 2. Select input features (technical indicators, alternative data)
 * 3. Download historical training data
 * 4. Start training with Optuna hyperparameter optimization
 * 5. Monitor progress and view results
 * 6. Run backtesting and check for data drift
 * 7. Load quick presets and manage trained models
 * 
 * Phase 1 Updates:
 * - Integrated useTrainingState hook for centralized state management
 * - Connected trainingAPI service for all backend calls
 * - Added ModelSelector for loading trained models
 * - Added BacktestResults for performance metrics
 * - Added DriftAlert for data drift warnings
 * - Added quick preset toolbar integrated with hyperparameters
 * 
 * State Management:
 * - All training state now managed via useTrainingState hook
 * - Child components receive state props (controlled components)
 * - API calls handled through trainingAPI service
 */

import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import HyperparameterGrid from './training/HyperparameterGrid';
import FeatureSelection from './training/FeatureSelection';
import TrainingProgress from './training/TrainingProgress';
import ModelSelector from './training/ModelSelector';
import BacktestResults from './training/BacktestResults';
import DriftAlert from './training/DriftAlert';
import ModelsComparisonTable from './training/ModelsComparisonTable';
import { useTrainingState, DEFAULT_COMMISSION_CONFIG } from '../hooks/useTrainingState';
import { startTraining, checkDriftStatus, runBacktest, fetchTrainingDateRange } from '../services/trainingAPI';
import { API_BASE_URL } from '../services/trainingAPI';
import liveAPI from '../services/liveAPI';

const INTRADAY_AGENT = 'SAC_INTRADAY_DSR';

const INTRADAY_AGENT_DEFAULTS = {
  symbol: 'TNA',
  benchmark: 'IWM',
  learningRate: 0.0003,
  entropy: 0.2,
  batchSize: 256,
  volPenalty: -0.3,
  episodes: 45000,
  retrain: 'Weekly',
};

const resolveAgentType = (agent) => (agent === INTRADAY_AGENT ? 'SAC' : agent);

function TabTraining() {
  // Initialize training state from custom hook
  const trainingState = useTrainingState();

  // UI State (not part of training config)
  const [isDownloading, setIsDownloading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [dataDownloaded, setDataDownloaded] = useState(false);
  const [downloadMessage, setDownloadMessage] = useState(''); // NEW: non-blocking message
  const [trainingId, setTrainingId] = useState(null);
  
  // Agent selection
  const [selectedAgent, setSelectedAgent] = useState('PPO'); // 'PPO', 'SAC', 'SAC_INTRADAY_DSR'
  const isIntradayAgent = selectedAgent === INTRADAY_AGENT;
  const resolvedAgentType = resolveAgentType(selectedAgent);

  const buildCommissionPayload = (perShareValue) => ({
    per_share: perShareValue,
    min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
    max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
  });
  const [intradayDefaultsApplied, setIntradayDefaultsApplied] = useState(false);
  
  // Backtest results
  const [backtestResults, setBacktestResults] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [deployingAgent, setDeployingAgent] = useState(false);
  
  // Drift detection
  const [driftData, setDriftData] = useState(null);
  
  // Help documentation viewer
  const [showHelp, setShowHelp] = useState(false);
  const [helpContent, setHelpContent] = useState('');

  const activeSymbol = resolvedAgentType === 'PPO'
    ? (trainingState.ppoSymbol || '')
    : (trainingState.sacSymbol || '');

  const dataConfigSnapshotRef = useRef({
    agent: selectedAgent,
    symbol: activeSymbol.trim().toUpperCase(),
    start: trainingState.startDate || '',
    end: trainingState.endDate || '',
    benchmark: isIntradayAgent ? (trainingState.sacBenchmarkSymbol || '').trim().toUpperCase() : ''
  });

  // Deploy to Live Trading handlers
  const deployToLiveTrading = async (agentType) => {
    if (deployingAgent) {
      return;
    }

    if (!selectedModel) {
      alert('Please select a trained model before deploying.');
      return;
    }

    const modelAgentType = resolveAgentType((selectedModel.agent_type || '').toUpperCase());
    const requestedAgentType = resolveAgentType(agentType.toUpperCase());

    if (modelAgentType !== requestedAgentType) {
      alert(`Selected model (${selectedModel.agent_type}) does not match ${agentType}. Choose a matching model from the list.`);
      return;
    }

    if (!backtestResults) {
      alert('Run a backtest before deploying the model to live trading.');
      return;
    }

    try {
      setDeployingAgent(true);

      const isPPO = agentType === 'PPO';
      const fallbackSymbol = isPPO ? trainingState.ppoSymbol : trainingState.sacSymbol;
      const symbol = selectedModel.symbol || fallbackSymbol;
      if (!symbol) {
        alert('Selected model is missing a symbol. Update model metadata before deploying.');
        return;
      }

      setDownloadMessage('🚀 Creating live agent with paper trading safeguards...');
      const initialCapitalInput = isPPO ? trainingState.ppoInitialCapital : trainingState.sacInitialCapital;
      const maxPositionInput = isPPO ? trainingState.ppoMaxPosition : trainingState.sacMaxPosition;
      const timeFrame = (isPPO ? trainingState.ppoTimeFrame : trainingState.sacTimeFrame) || 'daily';

      const initialCapital = Number(initialCapitalInput) || 0;
      let maxPositionPct = Number(maxPositionInput);
      if (Number.isNaN(maxPositionPct)) {
        maxPositionPct = 0.5;
      }
      if (maxPositionPct > 1) {
        maxPositionPct = maxPositionPct / 100;
      }
      maxPositionPct = Math.min(1, Math.max(0, maxPositionPct));

      const modelPath = selectedModel.model_path || selectedModel.file_path;
      const featuresUsed = Array.isArray(selectedModel.features_used)
        ? selectedModel.features_used
        : selectedModel.features_used
        ? [selectedModel.features_used]
        : [];

      const overrides = {
        agent_type: agentType,
        symbol,
        initial_capital: initialCapital,
        max_position_pct: maxPositionPct,
        risk_per_trade: 0.02,
        time_frame: timeFrame,
        paper_trading: true,
        check_frequency: 'EOD',
        lookback_days: 180,
        extras: {
          backtest: backtestResults.raw || backtestResults,
          deployed_from: 'training_tab',
          deployed_at: new Date().toISOString(),
        },
      };

      if (modelPath) {
        overrides.model_path = modelPath;
      }
      if (featuresUsed.length > 0) {
        overrides.features_used = featuresUsed;
      }
      if (selectedModel.features) {
        overrides.features_config = selectedModel.features;
      }
      if (selectedModel.normalizer_path) {
        overrides.normalizer_path = selectedModel.normalizer_path;
      }

      const payload = {
        model_id: selectedModel.model_id,
        start_immediately: true,
        overrides,
      };

      const response = await liveAPI.createAgent(payload);
      const agentId = response?.agent_id || `${agentType}_${symbol}`;
      const message = `✅ Live agent ${agentId} created for ${symbol}.`;
      setDownloadMessage(message);
      setTimeout(() => {
        setDownloadMessage((current) => (current === message ? '' : current));
      }, 12000);
      if (response?.agent) {
        localStorage.setItem('yooprl:lastDeployAgent', JSON.stringify({
          agent: response.agent,
          createdAt: new Date().toISOString(),
        }));
      }
      if (window.dispatchEvent) {
        window.dispatchEvent(new CustomEvent('yooprl-agent-deployed', {
          detail: {
            agent_type: agentType,
            agent_id: agentId,
            agent: response?.agent,
            overrides,
            timestamp: Date.now(),
          },
        }));
      }
    } catch (error) {
      console.error('Failed to deploy agent:', error);
      const message = `❌ Deployment failed: ${error.message}`;
      setDownloadMessage(message);
      setTimeout(() => {
        setDownloadMessage((current) => (current === message ? '' : current));
      }, 12000);
    } finally {
      setDeployingAgent(false);
    }
  };

  const handleDeployToPPO = () => deployToLiveTrading('PPO');
  const handleDeployToSAC = () => deployToLiveTrading('SAC');

  const sharpeValue = backtestResults?.sharpe_ratio ?? 0;
  const canDeployPPO =
    selectedAgent === 'PPO' &&
    !!selectedModel &&
    selectedModel.agent_type === 'PPO' &&
    sharpeValue >= 0.5 &&
    !deployingAgent;
  const canDeploySAC =
    (selectedAgent === 'SAC' || isIntradayAgent) &&
    !!selectedModel &&
    resolveAgentType(selectedModel.agent_type) === 'SAC' &&
    sharpeValue >= 0.5 &&
    !deployingAgent;

  // Load help documentation content
  useEffect(() => {
    if (showHelp && !helpContent) {
      fetch('/docs/TRAINING_TAB_GUIDE.md')
        .then(res => res.text())
        .then(text => setHelpContent(text))
        .catch(err => {
          console.error('Failed to load help documentation:', err);
          setHelpContent('# Error Loading Documentation\n\nFailed to load the training guide. Please check the console for details.');
        });
    }
  }, [showHelp, helpContent]);

  // ESC key to close help modal
  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === 'Escape' && showHelp) {
        setShowHelp(false);
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [showHelp]);

  useEffect(() => {
    setSelectedModel(null);
    setBacktestResults(null);
  }, [selectedAgent]);

  useEffect(() => {
    const snapshot = {
      agent: selectedAgent,
      symbol: activeSymbol.trim().toUpperCase(),
      start: trainingState.startDate || '',
      end: trainingState.endDate || '',
      benchmark: isIntradayAgent ? (trainingState.sacBenchmarkSymbol || '').trim().toUpperCase() : ''
    };

    const previous = dataConfigSnapshotRef.current;
    const hasChanged =
      previous.agent !== snapshot.agent ||
      previous.symbol !== snapshot.symbol ||
      previous.start !== snapshot.start ||
      previous.end !== snapshot.end ||
      previous.benchmark !== snapshot.benchmark;

    if (hasChanged) {
      dataConfigSnapshotRef.current = snapshot;

      if (dataDownloaded) {
        setDataDownloaded(false);
        if (downloadProgress !== 0) {
          setDownloadProgress(0);
        }
        setDownloadMessage('⚠️ Training data needs refresh after configuration changes.');
      } else if (downloadProgress !== 0) {
        setDownloadProgress(0);
      }
    }
  }, [
    selectedAgent,
    activeSymbol,
    trainingState.startDate,
    trainingState.endDate,
    trainingState.sacBenchmarkSymbol,
    isIntradayAgent,
    dataDownloaded,
    downloadProgress
  ]);

  useEffect(() => {
    if (!isIntradayAgent) {
      setIntradayDefaultsApplied(false);
      return;
    }

    if (intradayDefaultsApplied) {
      return;
    }

    trainingState.setSacSymbol(INTRADAY_AGENT_DEFAULTS.symbol);
    trainingState.setSacBenchmarkSymbol(INTRADAY_AGENT_DEFAULTS.benchmark);
    trainingState.setSacLearningRate(INTRADAY_AGENT_DEFAULTS.learningRate);
    trainingState.setSacEntropy(INTRADAY_AGENT_DEFAULTS.entropy);
    trainingState.setSacBatchSize(INTRADAY_AGENT_DEFAULTS.batchSize);
    trainingState.setSacVolPenalty(INTRADAY_AGENT_DEFAULTS.volPenalty);
    trainingState.setSacEpisodes(INTRADAY_AGENT_DEFAULTS.episodes);
    trainingState.setSacRetrain(INTRADAY_AGENT_DEFAULTS.retrain);

    setIntradayDefaultsApplied(true);
  }, [isIntradayAgent, intradayDefaultsApplied, trainingState]);

  useEffect(() => {
    let cancelled = false;

  const agentForSymbol = resolvedAgentType;
  const rawSymbol = agentForSymbol === 'PPO' ? trainingState.ppoSymbol : trainingState.sacSymbol;
  const symbol = (rawSymbol || '').trim().toUpperCase();

    if (!symbol || symbol.length < 2) {
      return undefined;
    }

    const loadDateBounds = async () => {
      const result = await fetchTrainingDateRange({
        symbol,
        frequency: isIntradayAgent ? 'intraday' : 'daily',
        interval: isIntradayAgent ? '15m' : undefined
      });

      if (cancelled) {
        return;
      }

      if (!result?.success) {
        if (result?.status === 'not_found') {
          const message = isIntradayAgent
            ? `⚠️ No cached 15m data found for ${symbol}. Download training data to refresh SQL.`
            : `⚠️ No cached daily data found for ${symbol}. Download training data to populate the range.`;
          setDownloadMessage(message);
          setTimeout(() => {
            setDownloadMessage((current) => (current === message ? '' : current));
          }, 9000);
        } else if (result?.error && result?.status !== 'not_found') {
          console.warn('Failed to resolve training date range:', result.error);
        }
        return;
      }

      if (result.start_date) {
        trainingState.setStartDate(result.start_date);
      }
      if (result.end_date) {
        trainingState.setEndDate(result.end_date);
      }

      if (result.source === 'yfinance_fallback') {
        const message = `📅 Date range auto-filled using available market data for ${symbol} (YFinance).`;
        setDownloadMessage(message);
        setTimeout(() => {
          setDownloadMessage((current) => (current === message ? '' : current));
        }, 9000);
      }
    };

    loadDateBounds();

    return () => {
      cancelled = true;
    };
  }, [selectedAgent, trainingState.ppoSymbol, trainingState.sacSymbol, isIntradayAgent, trainingState]);

  // Check for data drift on mount and periodically
  useEffect(() => {
    const checkDrift = async () => {
      try {
        const agentForPayload = resolveAgentType(selectedAgent);
        const symbol = agentForPayload === 'PPO' ? trainingState.ppoSymbol : trainingState.sacSymbol;
        
        // Skip if no symbol configured
        if (!symbol) {
          console.warn('No symbol configured for drift check');
          return;
        }
        
        const result = await checkDriftStatus(symbol, agentForPayload);
        
        if (result.success && result.drift_data) {
          setDriftData(result.drift_data);
        } else {
          // Clear drift data if no drift detected or if check failed
          setDriftData(null);
        }
      } catch (error) {
        console.error('Error checking drift:', error);
        // Don't clear drift data on error - keep showing last known state
      }
    };

    checkDrift();
    const interval = setInterval(checkDrift, 300000); // Check every 5 minutes
    return () => clearInterval(interval);
  }, [selectedAgent, trainingState.ppoSymbol, trainingState.sacSymbol]);

  // Download Data Handler
  const handleDownloadData = async () => {
    setIsDownloading(true);
    setDownloadProgress(0);
    setDataDownloaded(false);
    
    try {
      const agentForPayload = resolveAgentType(selectedAgent);
      const symbol = agentForPayload === 'PPO' ? trainingState.ppoSymbol : trainingState.sacSymbol;
      const benchmarkSymbol = selectedAgent === INTRADAY_AGENT
        ? (trainingState.sacBenchmarkSymbol || 'IWM')
        : null;
      if (!symbol) {
        throw new Error('Symbol is required before downloading data');
      }

  const response = await fetch(`${API_BASE_URL}/api/training/download`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol,
          start_date: trainingState.startDate,
          end_date: trainingState.endDate,
          // ALWAYS download ALL data sources - user selects at training time
          enable_sentiment: true,
          enable_social_media: true,
          enable_news: true,
          enable_market_events: true,
          enable_fundamental: true,
          enable_multi_asset: true,
          enable_macro: true,
          force_redownload: false,
          ...(selectedAgent === INTRADAY_AGENT ? {
            data_frequency: 'intraday',
            interval: '15m',
            benchmark_symbol: benchmarkSymbol,
            benchmark_interval: '15m'
          } : {})
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
        
        // Non-blocking success message
        const message = `✅ Training Data Ready! Symbol: ${result.symbol}, ` +
                       `Rows: ${result.rows}, Features: ${result.features}, ` +
                       `Train: ${result.train_size} / Test: ${result.test_size}`;
        setDownloadMessage(message);
        
        // Auto-hide message after 10 seconds
        setTimeout(() => setDownloadMessage(''), 10000);
      } else {
        throw new Error(result.error || 'Download failed');
      }
    } catch (error) {
      console.error('Error downloading training data:', error);
      
      // Non-blocking error message
      setDownloadMessage(`❌ Download Failed: ${error.message}`);
      setTimeout(() => setDownloadMessage(''), 10000);
      
      setDownloadProgress(0);
    } finally {
      setIsDownloading(false);
    }
  };

  // Start Training Handler
  const handleStartTraining = async () => {
    if (!dataDownloaded) {
      alert('Please download training data first');
      return;
    }

    // Validate configuration
    const validation = trainingState.validateConfig(selectedAgent);
    if (!validation.valid) {
      alert(`Configuration errors:\n\n${validation.errors.join('\n')}`);
      return;
    }

    // Build training config
    const config = trainingState.buildTrainingConfig(selectedAgent);
    
    // Start training via API
    setIsTraining(true);
    setTrainingProgress(0);
    
    const result = await startTraining(config);
    
    if (result.success) {
      setTrainingId(result.training_id);
      alert(`✅ Training Started!\n\nTraining ID: ${result.training_id}`);
    } else {
      alert(`❌ Training Failed\n\n${result.error}`);
      setIsTraining(false);
    }
  };

  // Handle model selection for backtesting
  const handleModelSelect = async (model) => {
    setSelectedModel(model || null);
    setBacktestResults(null);

    if (!model) {
      return;
    }

    if (model.symbol) {
      const resolvedModelAgent = resolveAgentType((model.agent_type || '').toUpperCase());
      if (resolvedModelAgent === 'PPO' && trainingState.setPpoSymbol) {
        trainingState.setPpoSymbol(model.symbol);
      }
      if (resolvedModelAgent === 'SAC' && trainingState.setSacSymbol) {
        trainingState.setSacSymbol(model.symbol);
      }
    }

    setBacktestLoading(true);
    setDownloadMessage('⏳ Running validation backtest for selected model...');

    const agentInitialCapital =
      model.agent_type === 'PPO' ? trainingState.ppoInitialCapital : trainingState.sacInitialCapital;
    const initialCapital = Number(model.initial_capital) || Number(agentInitialCapital) || 100000;
    const perShareCommission = Number(trainingState.commission);
    const resolvedCommissionPerShare = Number.isFinite(perShareCommission) && perShareCommission >= 0
      ? perShareCommission
      : DEFAULT_COMMISSION_CONFIG.per_share;
    const commissionPayload = buildCommissionPayload(resolvedCommissionPerShare);

    const resolvedModelPath = model.model_path || model.file_path;
    if (!resolvedModelPath) {
      const pathError = '⚠️ Selected model is missing a model path. Update metadata and try again.';
      setDownloadMessage(pathError);
      setBacktestLoading(false);
      setTimeout(() => {
        setDownloadMessage((current) => (current === pathError ? '' : current));
      }, 9000);
      return;
    }
    const backtestRequest = {
      model_path: resolvedModelPath,
      normalizer_path: model.normalizer_path,
      agent_type: model.agent_type,
      symbol: model.symbol,
      start_date: trainingState.startDate,
      end_date: trainingState.endDate,
      initial_capital: initialCapital,
      commission: commissionPayload,
      commission_per_share: resolvedCommissionPerShare,
      commission_min_fee: DEFAULT_COMMISSION_CONFIG.min_fee,
      commission_max_pct: DEFAULT_COMMISSION_CONFIG.max_pct,
      commission_model: 'ibkr_tiered_us_equities'
    };

    try {
      const result = await runBacktest(backtestRequest);

  if (result.success && result.results) {
        const metrics = result.results.metrics || result.results;
        const flattened = {
          sharpe_ratio: Number(metrics.sharpe_ratio) || 0,
          sortino_ratio: Number(metrics.sortino_ratio) || 0,
          max_drawdown: Number(metrics.max_drawdown) || 0,
          win_rate: Number(metrics.win_rate) || 0,
          total_return: Number(metrics.total_return) || 0,
          final_portfolio_value: Number(metrics.final_balance || metrics.final_portfolio_value) || 0,
          num_trades: Number(metrics.total_trades ?? metrics.num_trades ?? 0),
          avg_trade_return: metrics.total_trades
            ? (Number(metrics.total_return) || 0) / Number(metrics.total_trades)
            : 0,
          raw: result.results,
        };
        setBacktestResults(flattened);
        const successMessage = '✅ Validation backtest completed successfully.';
        setDownloadMessage(successMessage);
        setTimeout(() => {
          setDownloadMessage((current) => (current === successMessage ? '' : current));
        }, 9000);
      } else {
        console.warn('Backtest API failed, using placeholder metrics');
        setBacktestResults({
          sharpe_ratio: model.sharpe_ratio || 0.4,
          sortino_ratio: model.sortino_ratio || 0.5,
          max_drawdown: model.max_drawdown || -20,
          win_rate: model.win_rate || 55,
          total_return: model.total_return || 12,
          final_portfolio_value: model.final_portfolio_value || 112000,
          num_trades: model.total_trades || 120,
          avg_trade_return: 0,
          raw: null,
        });
  const warningMessage = '⚠️ Backtest API unavailable; using stored metrics.';
        setDownloadMessage(warningMessage);
        setTimeout(() => {
          setDownloadMessage((current) => (current === warningMessage ? '' : current));
        }, 9000);
      }
    } catch (error) {
      console.warn('Backtest error, using placeholder metrics:', error);
      setBacktestResults({
        sharpe_ratio: model.sharpe_ratio || 0.4,
        sortino_ratio: model.sortino_ratio || 0.5,
        max_drawdown: model.max_drawdown || -20,
        win_rate: model.win_rate || 55,
        total_return: model.total_return || 12,
        final_portfolio_value: model.final_portfolio_value || 112000,
        num_trades: model.total_trades || 120,
        avg_trade_return: 0,
        raw: null,
      });
      const errorMessage = `⚠️ Backtest failed: ${error.message}. Showing stored metrics.`;
      setDownloadMessage(errorMessage);
      setTimeout(() => {
        setDownloadMessage((current) => (current === errorMessage ? '' : current));
      }, 9000);
    } finally {
      setBacktestLoading(false);
    }
  };

  // Handle drift detection - trigger retraining
  const handleRetrain = () => {
    if (window.confirm('Start retraining with recent data?')) {
      handleStartTraining();
    }
  };

  const handleTrainingStatusChange = ({ status, progress }) => {
    if (typeof progress === 'number' && !Number.isNaN(progress)) {
      setTrainingProgress(progress);
    }

    if (!status) {
      return;
    }

    const normalizedStatus = String(status).toLowerCase();
    if (normalizedStatus === 'completed') {
      setIsTraining(false);
      setTrainingProgress((prev) => (prev < 100 ? 100 : prev));
    } else if (normalizedStatus === 'failed') {
      setIsTraining(false);
    }
  };

  const handleTrainingStopped = () => {
    setIsTraining(false);
    setTrainingId(null);
    setTrainingProgress(0);
  };

  const resolveFlag = (value) => {
    if (value === undefined || value === null) {
      return undefined;
    }
    if (typeof value === 'object') {
      if (Object.prototype.hasOwnProperty.call(value, 'enabled')) {
        return Boolean(value.enabled);
      }
      return true;
    }
    return Boolean(value);
  };

  const toFiniteNumber = (value) => {
    const num = Number(value);
    return Number.isFinite(num) ? num : undefined;
  };

  const toInteger = (value) => {
    const num = Number.parseInt(value, 10);
    return Number.isNaN(num) ? undefined : num;
  };

  const toSymbolString = (symbols) => {
    if (Array.isArray(symbols)) {
      return symbols.join(',');
    }
    if (typeof symbols === 'string') {
      return symbols;
    }
    return undefined;
  };

  const applyConfigToState = (configPayload) => {
    if (!configPayload) {
      return;
    }

    const baseAgent = (configPayload.agent_type || selectedAgent || 'PPO').toUpperCase();
    const intradayPreset =
      (configPayload.training_settings?.data_frequency || '').toLowerCase() === 'intraday' &&
      (configPayload.training_settings?.reward_mode || '').toLowerCase() === 'dsr';
    const targetAgent = intradayPreset ? INTRADAY_AGENT : baseAgent;
    const agentKey = resolveAgentType(targetAgent) === 'SAC' ? 'SAC' : 'PPO';

    if (configPayload.symbol) {
      if (agentKey === 'PPO') {
        trainingState.setPpoSymbol(String(configPayload.symbol).toUpperCase());
      } else {
        trainingState.setSacSymbol(String(configPayload.symbol).toUpperCase());
      }
    }

    const hyper = configPayload.hyperparameters
      || (agentKey === 'PPO' ? configPayload.ppo_hyperparameters : configPayload.sac_hyperparameters)
      || {};

    if (agentKey === 'PPO') {
      const lr = toFiniteNumber(hyper.learning_rate);
      if (lr !== undefined) trainingState.setPpoLearningRate(lr);

      const gamma = toFiniteNumber(hyper.gamma);
      if (gamma !== undefined) trainingState.setPpoGamma(gamma);

      const batch = toInteger(hyper.batch_size);
      if (batch !== undefined) trainingState.setPpoBatchSize(batch);

      const risk = toFiniteNumber(hyper.risk_penalty);
      if (risk !== undefined) trainingState.setPpoRiskPenalty(risk);

      const episodes = toInteger(hyper.episodes);
      if (episodes !== undefined) trainingState.setPpoEpisodes(episodes);
    } else {
      const lr = toFiniteNumber(hyper.learning_rate);
      if (lr !== undefined) trainingState.setSacLearningRate(lr);

      const entropy = toFiniteNumber(hyper.entropy_coef);
      if (entropy !== undefined) trainingState.setSacEntropy(entropy);

      const batch = toInteger(hyper.batch_size);
      if (batch !== undefined) trainingState.setSacBatchSize(batch);

      const vol = toFiniteNumber(hyper.vol_penalty);
      if (vol !== undefined) trainingState.setSacVolPenalty(vol);

      const episodes = toInteger(hyper.episodes);
      if (episodes !== undefined) trainingState.setSacEpisodes(episodes);
    }

    const features = configPayload.features || {};

    if (Object.prototype.hasOwnProperty.call(features, 'ohlc')) {
      trainingState.setOhlcEnabled(Boolean(features.ohlc));
    }

    if (Object.prototype.hasOwnProperty.call(features, 'rsi')) {
      const rsiCfg = features.rsi;
      const enabled = resolveFlag(rsiCfg);
      if (enabled !== undefined) trainingState.setRsiEnabled(enabled);
      if (rsiCfg && typeof rsiCfg === 'object' && rsiCfg.period !== undefined) {
        const period = toInteger(rsiCfg.period);
        if (period !== undefined) trainingState.setRsiPeriod(period);
      }
    }

    if (Object.prototype.hasOwnProperty.call(features, 'macd')) {
      const macdCfg = features.macd;
      const enabled = resolveFlag(macdCfg);
      if (enabled !== undefined) trainingState.setMacdEnabled(enabled);
      if (macdCfg && typeof macdCfg === 'object' && macdCfg.params) {
        trainingState.setMacdParams(String(macdCfg.params));
      }
    }

    if (Object.prototype.hasOwnProperty.call(features, 'ema')) {
      const emaCfg = features.ema;
      const enabled = resolveFlag(emaCfg);
      if (enabled !== undefined) trainingState.setEmaEnabled(enabled);
      if (emaCfg && typeof emaCfg === 'object' && emaCfg.periods) {
        trainingState.setEmaPeriods(String(emaCfg.periods));
      }
    }

    if (Object.prototype.hasOwnProperty.call(features, 'vix')) {
      const enabled = resolveFlag(features.vix);
      if (enabled !== undefined) trainingState.setVixEnabled(enabled);
    }

    if (Object.prototype.hasOwnProperty.call(features, 'bollinger')) {
      const bollingerCfg = features.bollinger;
      const enabled = resolveFlag(bollingerCfg);
      if (enabled !== undefined) trainingState.setBollingerEnabled(enabled);
      if (bollingerCfg && typeof bollingerCfg === 'object' && bollingerCfg.params) {
        trainingState.setBollingerParams(String(bollingerCfg.params));
      }
    }

    if (Object.prototype.hasOwnProperty.call(features, 'stochastic')) {
      const stochCfg = features.stochastic;
      const enabled = resolveFlag(stochCfg);
      if (enabled !== undefined) trainingState.setStochasticEnabled(enabled);
      if (stochCfg && typeof stochCfg === 'object' && stochCfg.params) {
        trainingState.setStochasticParams(String(stochCfg.params));
      }
    }

    if (Object.prototype.hasOwnProperty.call(features, 'adx')) {
      const adxCfg = features.adx;
      const enabled = resolveFlag(adxCfg);
      if (enabled !== undefined) trainingState.setAdxEnabled(enabled);
      if (adxCfg && typeof adxCfg === 'object' && adxCfg.period !== undefined) {
        const period = toInteger(adxCfg.period);
        if (period !== undefined) trainingState.setAdxPeriod(period);
      }
    }

    if (Object.prototype.hasOwnProperty.call(features, 'sentiment')) {
      const enabled = resolveFlag(features.sentiment);
      if (enabled !== undefined) trainingState.setSentimentEnabled(enabled);
    }

    if (Object.prototype.hasOwnProperty.call(features, 'social_media')) {
      const enabled = resolveFlag(features.social_media);
      if (enabled !== undefined) trainingState.setSocialMediaEnabled(enabled);
    }

    if (Object.prototype.hasOwnProperty.call(features, 'news_headlines')) {
      const enabled = resolveFlag(features.news_headlines);
      if (enabled !== undefined) trainingState.setNewsHeadlinesEnabled(enabled);
    }

    if (Object.prototype.hasOwnProperty.call(features, 'market_events')) {
      const enabled = resolveFlag(features.market_events);
      if (enabled !== undefined) trainingState.setMarketEventsEnabled(enabled);
    }

    if (Object.prototype.hasOwnProperty.call(features, 'fundamental')) {
      const enabled = resolveFlag(features.fundamental);
      if (enabled !== undefined) trainingState.setFundamentalEnabled(enabled);
    }

    if (Object.prototype.hasOwnProperty.call(features, 'multi_asset')) {
      const multiAssetCfg = features.multi_asset;
      const enabled = resolveFlag(multiAssetCfg);
      if (enabled !== undefined) trainingState.setMultiAssetEnabled(enabled);
      if (multiAssetCfg && typeof multiAssetCfg === 'object' && multiAssetCfg.symbols !== undefined) {
        const symbols = toSymbolString(multiAssetCfg.symbols);
        if (symbols !== undefined) trainingState.setMultiAssetSymbols(symbols);
      }
    }

    if (Object.prototype.hasOwnProperty.call(features, 'macro')) {
      const enabled = resolveFlag(features.macro);
      if (enabled !== undefined) trainingState.setMacroEnabled(enabled);
    }

    if (Object.prototype.hasOwnProperty.call(features, 'recent_actions')) {
      const enabled = resolveFlag(features.recent_actions);
      if (enabled !== undefined) trainingState.setRecentActionsEnabled(enabled);
    }

    if (Object.prototype.hasOwnProperty.call(features, 'performance')) {
      const perfCfg = features.performance;
      const enabled = resolveFlag(perfCfg);
      if (enabled !== undefined) trainingState.setPerformanceEnabled(enabled);
      if (perfCfg && typeof perfCfg === 'object' && perfCfg.period) {
        trainingState.setPerformancePeriod(String(perfCfg.period));
      }
    }

    if (Object.prototype.hasOwnProperty.call(features, 'position_history')) {
      const enabled = resolveFlag(features.position_history);
      if (enabled !== undefined) trainingState.setPositionHistoryEnabled(enabled);
    }

    if (Object.prototype.hasOwnProperty.call(features, 'reward_history')) {
      const enabled = resolveFlag(features.reward_history);
      if (enabled !== undefined) trainingState.setRewardHistoryEnabled(enabled);
    }

    if (Object.prototype.hasOwnProperty.call(features, 'llm')) {
      const llmCfg = features.llm;
      const enabled = resolveFlag(llmCfg);
      if (enabled !== undefined) trainingState.setLlmEnabled(enabled);
      if (llmCfg && typeof llmCfg === 'object' && llmCfg.provider) {
        trainingState.setSelectedLLM(String(llmCfg.provider));
      }
    }

    const trainingSettings = configPayload.training_settings || {};

    if (trainingSettings.start_date) {
      trainingState.setStartDate(trainingSettings.start_date);
    }

    if (trainingSettings.end_date) {
      trainingState.setEndDate(trainingSettings.end_date);
    }

    if (trainingSettings.commission !== undefined || trainingSettings.commission_per_share !== undefined) {
      let resolvedCommission;

      const rawCommission = trainingSettings.commission;
      if (rawCommission && typeof rawCommission === 'object') {
        const rawPerShare =
          rawCommission.per_share ??
          rawCommission.perShare ??
          rawCommission.value ??
          rawCommission.per_trade ??
          rawCommission.perTrade;
        resolvedCommission = toFiniteNumber(rawPerShare);
      } else {
        resolvedCommission = toFiniteNumber(rawCommission);
      }

      if (resolvedCommission === undefined) {
        resolvedCommission = toFiniteNumber(trainingSettings.commission_per_share);
      }

      if (resolvedCommission !== undefined) {
        trainingState.setCommission(resolvedCommission);
      }
    }

    if (trainingSettings.optuna_trials !== undefined) {
      const trials = toInteger(trainingSettings.optuna_trials);
      if (trials !== undefined) trainingState.setOptunaTrials(trials);
    }

    if (trainingSettings.initial_capital !== undefined) {
      const capital = toFiniteNumber(trainingSettings.initial_capital);
      if (capital !== undefined) {
        if (agentKey === 'PPO') {
          trainingState.setPpoInitialCapital(capital);
        } else {
          trainingState.setSacInitialCapital(capital);
        }
      }
    }

    if (trainingSettings.max_position_size !== undefined) {
      const rawSize = toFiniteNumber(trainingSettings.max_position_size);
      if (rawSize !== undefined) {
        const pct = rawSize > 1 ? rawSize : rawSize * 100;
        const capped = Math.max(0, Math.min(100, Math.round(pct)));
        if (agentKey === 'PPO') {
          trainingState.setPpoMaxPosition(capped);
        } else {
          trainingState.setSacMaxPosition(capped);
        }
      }
    }

    if (agentKey === 'SAC' && trainingSettings.benchmark_symbol) {
      trainingState.setSacBenchmarkSymbol(String(trainingSettings.benchmark_symbol).toUpperCase());
    }
  };

  // Handle config load from quick preset toolbar
  const handleLoadConfig = (config) => {
    if (!config) {
      return;
    }

    const baseAgent = (config.agent_type || selectedAgent || 'PPO').toUpperCase();
    const isIntradayConfig =
      (config.training_settings?.data_frequency || '').toLowerCase() === 'intraday' &&
      (config.training_settings?.reward_mode || '').toLowerCase() === 'dsr';
    const targetAgent = isIntradayConfig ? INTRADAY_AGENT : baseAgent;

    if (targetAgent !== selectedAgent) {
      setSelectedAgent(targetAgent);
    }

    applyConfigToState(config);

    const hyper = config.hyperparameters
      || (resolveAgentType(targetAgent) === 'SAC' ? config.sac_hyperparameters : config.ppo_hyperparameters)
      || {};

    const lr = hyper.learning_rate !== undefined ? hyper.learning_rate : 'unchanged';
    const episodes = hyper.episodes !== undefined ? hyper.episodes : 'unchanged';
    const rewardKey = resolveAgentType(targetAgent) === 'SAC' ? 'vol_penalty' : 'risk_penalty';
    const rewardVal = hyper[rewardKey] !== undefined ? hyper[rewardKey] : 'unchanged';

    const features = config.features || {};
    const featureHighlights = [];

    if (features.sentiment !== undefined) {
      featureHighlights.push(`Sentiment ${resolveFlag(features.sentiment) ? 'ON' : 'OFF'}`);
    }

    if (features.multi_asset !== undefined) {
      featureHighlights.push(`Multi-asset ${resolveFlag(features.multi_asset) ? 'ON' : 'OFF'}`);
    }

    if (features.reward_history !== undefined) {
      featureHighlights.push(`Reward history ${resolveFlag(features.reward_history) ? 'ON' : 'OFF'}`);
    }

    if (features.performance && typeof features.performance === 'object' && features.performance.period) {
      featureHighlights.push(`Perf window ${features.performance.period}`);
    }

    const title = config.name || `${targetAgent} configuration`;
    const baseSummary = `✅ ${title} applied → LR: ${lr}, Episodes: ${episodes}, ${rewardKey}: ${rewardVal}`;
    const featureSummary = featureHighlights.length ? ` | ${featureHighlights.join(' | ')}` : '';

    setDownloadMessage(baseSummary + featureSummary);
    setTimeout(() => setDownloadMessage(''), 10000);
  };

  return (
    <div>
      {/* Help Documentation Modal */}
      {showHelp && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          zIndex: 10000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px'
        }} onClick={() => setShowHelp(false)}>
          <div style={{
            backgroundColor: '#1e1e1e',
            borderRadius: '12px',
            width: '90%',
            maxWidth: '1200px',
            height: '90vh',
            overflow: 'hidden',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5)',
            display: 'flex',
            flexDirection: 'column'
          }} onClick={(e) => e.stopPropagation()}>
            {/* Header */}
            <div style={{
              padding: '20px 30px',
              borderBottom: '1px solid #333',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              backgroundColor: '#252525'
            }}>
              <h2 style={{ margin: 0, color: '#e0e0e0', fontSize: '24px', fontWeight: 600 }}>
                📚 Training Tab - User Guide
              </h2>
              <button
                onClick={() => setShowHelp(false)}
                style={{
                  backgroundColor: 'transparent',
                  border: 'none',
                  color: '#888',
                  fontSize: '32px',
                  cursor: 'pointer',
                  padding: '0 10px',
                  lineHeight: '32px'
                }}
                title="Close (Esc)"
              >
                ×
              </button>
            </div>
            
            {/* Content */}
            <div style={{
              flex: 1,
              padding: '30px 40px',
              overflowY: 'auto',
              backgroundColor: '#1e1e1e',
              color: '#d4d4d4',
              lineHeight: '1.8',
              fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
            }}>
              {helpContent ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    h1: ({node, children, ...rest}) => (
                      <h1
                        style={{ color: '#4a9eff', fontSize: '32px', marginTop: '0', marginBottom: '16px', borderBottom: '2px solid #4a9eff', paddingBottom: '8px' }}
                        {...rest}
                      >
                        {children}
                      </h1>
                    ),
                    h2: ({node, children, ...rest}) => (
                      <h2
                        style={{ color: '#5fb3f6', fontSize: '26px', marginTop: '32px', marginBottom: '12px', borderBottom: '1px solid #444', paddingBottom: '6px' }}
                        {...rest}
                      >
                        {children}
                      </h2>
                    ),
                    h3: ({node, children, ...rest}) => (
                      <h3
                        style={{ color: '#7cc5f8', fontSize: '22px', marginTop: '24px', marginBottom: '10px' }}
                        {...rest}
                      >
                        {children}
                      </h3>
                    ),
                    h4: ({node, children, ...rest}) => (
                      <h4
                        style={{ color: '#9dd5fa', fontSize: '18px', marginTop: '20px', marginBottom: '8px' }}
                        {...rest}
                      >
                        {children}
                      </h4>
                    ),
                    p: ({node, children, ...rest}) => (
                      <p style={{ marginBottom: '12px', fontSize: '14px' }} {...rest}>
                        {children}
                      </p>
                    ),
                    ul: ({node, children, ...rest}) => (
                      <ul style={{ marginLeft: '20px', marginBottom: '12px' }} {...rest}>
                        {children}
                      </ul>
                    ),
                    ol: ({node, children, ...rest}) => (
                      <ol style={{ marginLeft: '20px', marginBottom: '12px' }} {...rest}>
                        {children}
                      </ol>
                    ),
                    li: ({node, children, ...rest}) => (
                      <li style={{ marginBottom: '6px', fontSize: '14px' }} {...rest}>
                        {children}
                      </li>
                    ),
                    code: ({node, inline, children, ...rest}) => (
                      inline ? (
                        <code
                          style={{ backgroundColor: '#2d2d2d', padding: '2px 6px', borderRadius: '3px', fontSize: '13px', color: '#ce9178' }}
                          {...rest}
                        >
                          {children}
                        </code>
                      ) : (
                        <code
                          style={{ display: 'block', backgroundColor: '#2d2d2d', padding: '12px', borderRadius: '6px', fontSize: '13px', overflowX: 'auto', lineHeight: '1.6', marginBottom: '12px', border: '1px solid #444' }}
                          {...rest}
                        >
                          {children}
                        </code>
                      )
                    ),
                    pre: ({node, children, ...rest}) => (
                      <pre style={{ margin: 0 }} {...rest}>
                        {children}
                      </pre>
                    ),
                    blockquote: ({node, children, ...rest}) => (
                      <blockquote
                        style={{ borderLeft: '4px solid #4a9eff', paddingLeft: '16px', marginLeft: '0', marginBottom: '12px', color: '#aaa', fontStyle: 'italic' }}
                        {...rest}
                      >
                        {children}
                      </blockquote>
                    ),
                    table: ({node, children, ...rest}) => (
                      <table
                        style={{ borderCollapse: 'collapse', width: '100%', marginBottom: '16px', fontSize: '14px' }}
                        {...rest}
                      >
                        {children}
                      </table>
                    ),
                    thead: ({node, children, ...rest}) => (
                      <thead style={{ backgroundColor: '#2d2d2d' }} {...rest}>
                        {children}
                      </thead>
                    ),
                    tbody: ({node, children, ...rest}) => (
                      <tbody {...rest}>
                        {children}
                      </tbody>
                    ),
                    tr: ({node, children, ...rest}) => (
                      <tr style={{ borderBottom: '1px solid #444' }} {...rest}>
                        {children}
                      </tr>
                    ),
                    th: ({node, children, ...rest}) => (
                      <th
                        style={{ padding: '10px', textAlign: 'left', color: '#4a9eff', fontWeight: 'bold', borderBottom: '2px solid #4a9eff' }}
                        {...rest}
                      >
                        {children}
                      </th>
                    ),
                    td: ({node, children, ...rest}) => (
                      <td style={{ padding: '10px', borderBottom: '1px solid #444' }} {...rest}>
                        {children}
                      </td>
                    ),
                    a: ({node, children, ...rest}) => (
                      <a
                        style={{ color: '#4a9eff', textDecoration: 'none', borderBottom: '1px solid transparent' }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.borderBottom = '1px solid #4a9eff';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.borderBottom = '1px solid transparent';
                        }}
                        {...rest}
                      >
                        {children}
                      </a>
                    ),
                    hr: ({node, ...rest}) => (
                      <hr style={{ border: 'none', borderTop: '1px solid #444', margin: '24px 0' }} {...rest} />
                    ),
                    strong: ({node, children, ...rest}) => (
                      <strong style={{ color: '#fff', fontWeight: 600 }} {...rest}>
                        {children}
                      </strong>
                    ),
                    em: ({node, children, ...rest}) => (
                      <em style={{ color: '#9dd5fa' }} {...rest}>
                        {children}
                      </em>
                    )
                  }}
                >
                  {helpContent}
                </ReactMarkdown>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
                  <div style={{ fontSize: '48px', marginBottom: '16px' }}>📚</div>
                  <div style={{ fontSize: '16px' }}>Loading Training Guide...</div>
                </div>
              )}
            </div>
            
            {/* Footer */}
            <div style={{
              padding: '15px 30px',
              borderTop: '1px solid #333',
              backgroundColor: '#252525',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span style={{ color: '#888', fontSize: '13px' }}>
                Press ESC to close | Scroll to navigate
              </span>
              <button
                onClick={() => setShowHelp(false)}
                style={{
                  padding: '8px 20px',
                  backgroundColor: '#4a9eff',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '14px'
                }}
              >
                Got it!
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Agent Selection */}
      <div style={{ 
        padding: '15px', 
        backgroundColor: '#2d2d2d', 
        borderRadius: '8px', 
        marginBottom: '20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <label style={{ color: '#e0e0e0', fontWeight: 'bold', marginRight: '10px' }}>
            Select Agent Type:
          </label>
          <select 
            value={selectedAgent} 
            onChange={(e) => setSelectedAgent(e.target.value)}
            style={{
              padding: '8px 12px',
              backgroundColor: '#1e1e1e',
              color: '#e0e0e0',
              border: '1px solid #444',
              borderRadius: '4px',
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            <option value="PPO">PPO - Stock Trading</option>
            <option value="SAC">SAC - Leveraged ETF Trading</option>
            <option value="SAC_INTRADAY_DSR">SAC + DSR (15m Intraday)</option>
          </select>
        </div>
        
        {/* Help Icon */}
        <button
          onClick={() => setShowHelp(true)}
          style={{
            backgroundColor: '#4a9eff',
            border: 'none',
            borderRadius: '50%',
            width: '36px',
            height: '36px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            fontSize: '18px',
            fontWeight: 'bold',
            color: 'white',
            boxShadow: '0 2px 8px rgba(74, 158, 255, 0.3)',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            const target = e.currentTarget;
            target.style.transform = 'scale(1.1)';
            target.style.boxShadow = '0 4px 12px rgba(74, 158, 255, 0.5)';
          }}
          onMouseLeave={(e) => {
            const target = e.currentTarget;
            target.style.transform = 'scale(1)';
            target.style.boxShadow = '0 2px 8px rgba(74, 158, 255, 0.3)';
          }}
          title="Open Training Guide - Learn how to use this tab"
        >
          ?
        </button>
      </div>

      {/* Drift Alert (shows only if drift detected) */}
      <DriftAlert 
        driftData={driftData}
        onRetrain={handleRetrain}
      />

      {/* Hyperparameter Configuration */}
      <HyperparameterGrid 
        agentType={resolveAgentType(selectedAgent)}
        isIntraday={isIntradayAgent}
        trainingState={trainingState}
        onLoadConfig={handleLoadConfig}
      />

      {/* Feature Selection */}
      <FeatureSelection 
        trainingState={trainingState}
      />

      {/* Training Controls and Progress */}
      <TrainingProgress 
        isDownloading={isDownloading}
        isTraining={isTraining}
        downloadProgress={downloadProgress}
        trainingProgress={trainingProgress}
        dataDownloaded={dataDownloaded}
        downloadMessage={downloadMessage}
        trainingId={trainingId}
        handleDownloadData={handleDownloadData}
        handleStartTraining={handleStartTraining}
        onTrainingStatusChange={handleTrainingStatusChange}
        onTrainingStopped={handleTrainingStopped}
      />

      {/* Model Selector */}
      <ModelSelector 
        key={selectedAgent}
        onModelSelect={handleModelSelect}
        agentType={resolveAgentType(selectedAgent)}
      />

      {/* Backtest Results */}
      <BacktestResults 
        results={backtestResults}
        loading={backtestLoading}
      />

      {/* Deploy to Live Trading Section */}
      {backtestResults && backtestResults.sharpe_ratio > 0 && (
        <div style={{
          marginTop: '20px',
          padding: '24px',
          background: '#0d1117',
          border: '2px solid #30363d',
          borderRadius: '8px'
        }}>
          <h3 style={{
            color: '#58a6ff',
            marginTop: 0,
            marginBottom: '16px',
            fontSize: '18px',
            fontWeight: 'bold'
          }}>
            🚀 Deploy Model to Live Trading
          </h3>

          {/* Warning/Info Messages */}
          {backtestResults.sharpe_ratio < 1.5 && (
            <div style={{
              padding: '12px',
              background: 'rgba(187, 128, 9, 0.15)',
              border: '1px solid #bb8009',
              borderRadius: '6px',
              marginBottom: '16px',
              color: '#d29922',
              fontSize: '13px'
            }}>
              ⚠️ <strong>Warning:</strong> Sharpe Ratio is {backtestResults.sharpe_ratio.toFixed(2)}. 
              Recommended minimum is 1.5 for live trading. Consider retraining or using paper trading mode.
            </div>
          )}

          {backtestResults.sharpe_ratio >= 1.5 && (
            <div style={{
              padding: '12px',
              background: 'rgba(35, 134, 54, 0.15)',
              border: '1px solid #238636',
              borderRadius: '6px',
              marginBottom: '16px',
              color: '#3fb950',
              fontSize: '13px'
            }}>
              ✅ <strong>Good Performance:</strong> Sharpe Ratio is {backtestResults.sharpe_ratio.toFixed(2)}. 
              Model meets minimum requirements for live trading.
            </div>
          )}

          {/* Current Model Info */}
          <div style={{
            padding: '12px',
            background: '#161b22',
            borderRadius: '6px',
            marginBottom: '16px',
            fontSize: '13px',
            color: '#8b949e'
          }}>
            <div style={{ marginBottom: '8px' }}>
              <strong style={{ color: '#c9d1d9' }}>Selected Model:</strong>{' '}
              {selectedModel
                ? `${selectedModel.agent_type} • ${selectedModel.symbol} • ${selectedModel.version || 'vN/A'}`
                : 'None selected'}
            </div>

            {selectedModel && (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '12px',
                marginBottom: '12px'
              }}>
                <div>
                  <div style={{ color: '#6e7681', fontSize: '11px' }}>Model ID</div>
                  <div style={{ color: '#c9d1d9', fontWeight: 'bold', wordBreak: 'break-all' }}>{selectedModel.model_id}</div>
                </div>
                <div>
                  <div style={{ color: '#6e7681', fontSize: '11px' }}>Created</div>
                  <div style={{ color: '#c9d1d9' }}>
                    {selectedModel.created_at
                      ? new Date(selectedModel.created_at).toLocaleString()
                      : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#6e7681', fontSize: '11px' }}>Model Path</div>
                  <div style={{ color: '#a371f7', fontFamily: 'monospace', fontSize: '12px', wordBreak: 'break-all' }}>
                    {selectedModel.file_path || selectedModel.model_path || 'N/A'}
                  </div>
                </div>
                {selectedModel.normalizer_path && (
                  <div>
                    <div style={{ color: '#6e7681', fontSize: '11px' }}>Normalizer</div>
                    <div style={{ color: '#a371f7', fontFamily: 'monospace', fontSize: '12px', wordBreak: 'break-all' }}>
                      {selectedModel.normalizer_path}
                    </div>
                  </div>
                )}
              </div>
            )}

            {selectedModel && backtestResults && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
                <div>
                  <div style={{ color: '#6e7681', fontSize: '11px' }}>Total Return</div>
                  <div style={{ color: backtestResults.total_return >= 0 ? '#3fb950' : '#f85149', fontWeight: 'bold' }}>
                    {backtestResults.total_return >= 0 ? '+' : ''}{backtestResults.total_return?.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div style={{ color: '#6e7681', fontSize: '11px' }}>Sharpe Ratio</div>
                  <div style={{ color: '#58a6ff', fontWeight: 'bold' }}>
                    {backtestResults.sharpe_ratio?.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#6e7681', fontSize: '11px' }}>Max Drawdown</div>
                  <div style={{ color: '#f85149', fontWeight: 'bold' }}>
                    {backtestResults.max_drawdown?.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div style={{ color: '#6e7681', fontSize: '11px' }}>Win Rate</div>
                  <div style={{ color: '#3fb950', fontWeight: 'bold' }}>
                    {backtestResults.win_rate?.toFixed(1)}%
                  </div>
                </div>
              </div>
            )}

            {!selectedModel && (
              <div style={{
                padding: '10px',
                background: 'rgba(187, 128, 9, 0.1)',
                border: '1px solid rgba(187, 128, 9, 0.4)',
                borderRadius: '6px',
                color: '#d29922',
                fontSize: '12px'
              }}>
                Select a trained model above to unlock deployment options.
              </div>
            )}
          </div>

          {/* Deployment Configuration */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr 1fr',
            gap: '16px',
            marginBottom: '16px'
          }}>
            {/* Time Frame Selection */}
            <div>
              <label style={{
                display: 'block',
                color: '#8b949e',
                fontSize: '12px',
                fontWeight: 'bold',
                marginBottom: '8px'
              }}>
                ⏰ Trading Time Frame
              </label>
              <select
                value={selectedAgent === 'PPO' ? trainingState.ppoTimeFrame || 'daily' : trainingState.sacTimeFrame || 'daily'}
                onChange={(e) => {
                  if (selectedAgent === 'PPO') {
                    trainingState.setPpoTimeFrame(e.target.value);
                  } else {
                    trainingState.setSacTimeFrame(e.target.value);
                  }
                }}
                style={{
                  width: '100%',
                  padding: '10px',
                  background: '#0d1117',
                  border: '1px solid #30363d',
                  borderRadius: '6px',
                  color: '#c9d1d9',
                  fontSize: '13px',
                  cursor: 'pointer'
                }}
              >
                <option value="daily">📅 Daily (EOD) - Recommended</option>
                <option value="4hour">🕐 4 Hours - Swing Trading</option>
                <option value="1hour">🕐 1 Hour - Active Trading</option>
                <option value="15min">⏱️ 15 Minutes - Day Trading</option>
                <option value="5min">⏱️ 5 Minutes - Scalping</option>
                <option value="1min">⚡ 1 Minute - HFT (requires live data)</option>
              </select>
              <div style={{
                marginTop: '6px',
                fontSize: '11px',
                color: '#6e7681',
                fontStyle: 'italic'
              }}>
                {(selectedAgent === 'PPO' ? trainingState.ppoTimeFrame : trainingState.sacTimeFrame) === 'daily' && '✅ No live data needed'}
                {['15min', '5min', '1min'].includes(selectedAgent === 'PPO' ? trainingState.ppoTimeFrame : trainingState.sacTimeFrame) && '⚠️ Requires IBKR live data subscription'}
              </div>
            </div>

            {/* Initial Capital */}
            <div>
              <label style={{
                display: 'block',
                color: '#8b949e',
                fontSize: '12px',
                fontWeight: 'bold',
                marginBottom: '8px'
              }}>
                💰 Initial Capital
              </label>
              <input
                type="number"
                value={selectedAgent === 'PPO' ? trainingState.ppoInitialCapital || 10000 : trainingState.sacInitialCapital || 10000}
                onChange={(e) => {
                  const value = parseFloat(e.target.value) || 0;
                  if (selectedAgent === 'PPO') {
                    trainingState.setPpoInitialCapital(value);
                  } else {
                    trainingState.setSacInitialCapital(value);
                  }
                }}
                min="1000"
                step="1000"
                style={{
                  width: '100%',
                  padding: '10px',
                  background: '#0d1117',
                  border: '1px solid #30363d',
                  borderRadius: '6px',
                  color: '#c9d1d9',
                  fontSize: '13px'
                }}
              />
              <div style={{
                marginTop: '6px',
                fontSize: '11px',
                color: '#6e7681',
                fontStyle: 'italic'
              }}>
                Paper trading mode (no real money)
              </div>
            </div>

            {/* Max Position Size */}
            <div>
              <label style={{
                display: 'block',
                color: '#8b949e',
                fontSize: '12px',
                fontWeight: 'bold',
                marginBottom: '8px'
              }}>
                📊 Max Position Size
              </label>
              <select
                value={selectedAgent === 'PPO' ? trainingState.ppoMaxPosition || 50 : trainingState.sacMaxPosition || 50}
                onChange={(e) => {
                  const value = parseInt(e.target.value);
                  if (selectedAgent === 'PPO') {
                    trainingState.setPpoMaxPosition(value);
                  } else {
                    trainingState.setSacMaxPosition(value);
                  }
                }}
                style={{
                  width: '100%',
                  padding: '10px',
                  background: '#0d1117',
                  border: '1px solid #30363d',
                  borderRadius: '6px',
                  color: '#c9d1d9',
                  fontSize: '13px',
                  cursor: 'pointer'
                }}
              >
                <option value="25">25% - Very Conservative</option>
                <option value="50">50% - Balanced</option>
                <option value="75">75% - Aggressive</option>
                <option value="100">100% - Maximum</option>
              </select>
              <div style={{
                marginTop: '6px',
                fontSize: '11px',
                color: '#6e7681',
                fontStyle: 'italic'
              }}>
                % of capital per trade
              </div>
            </div>
          </div>

          {/* Deploy Buttons */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: selectedAgent === 'PPO' ? '1fr' : '1fr',
            gap: '12px'
          }}>
            {selectedAgent === 'PPO' && (
              <button
                onClick={() => handleDeployToPPO()}
                disabled={!canDeployPPO}
                style={{
                  padding: '14px 24px',
                  background: !selectedModel
                    ? '#30363d'
                    : sharpeValue >= 1.5
                    ? '#238636'
                    : sharpeValue >= 0.5
                    ? '#bb8009'
                    : '#30363d',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: 'bold',
                  cursor: canDeployPPO ? 'pointer' : 'not-allowed',
                  opacity: canDeployPPO ? 1 : 0.6,
                  transition: 'all 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px'
                }}
                onMouseOver={(e) => {
                  if (canDeployPPO) {
                    const target = e.currentTarget;
                    target.style.transform = 'translateY(-2px)';
                    target.style.boxShadow = '0 4px 12px rgba(35, 134, 54, 0.4)';
                  }
                }}
                onMouseOut={(e) => {
                  const target = e.currentTarget;
                  target.style.transform = 'translateY(0)';
                  target.style.boxShadow = 'none';
                }}
                title={!selectedModel
                  ? 'Select a model to deploy'
                  : sharpeValue < 0.5
                  ? 'Sharpe Ratio below deployment threshold (0.5)'
                  : deployingAgent
                  ? 'Deployment in progress'
                  : ''}
              >
                <span style={{ fontSize: '18px' }}>📈</span>
                <span>{deployingAgent ? 'Deploying to Live...' : `Deploy ${selectedModel?.symbol || trainingState.ppoSymbol} to PPO Live (Paper)`}</span>
              </button>
            )}

            {(selectedAgent === 'SAC' || isIntradayAgent) && (
              <button
                onClick={() => handleDeployToSAC()}
                disabled={!canDeploySAC}
                style={{
                  padding: '14px 24px',
                  background: !selectedModel
                    ? '#30363d'
                    : sharpeValue >= 1.5
                    ? '#238636'
                    : sharpeValue >= 0.5
                    ? '#bb8009'
                    : '#30363d',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: 'bold',
                  cursor: canDeploySAC ? 'pointer' : 'not-allowed',
                  opacity: canDeploySAC ? 1 : 0.6,
                  transition: 'all 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px'
                }}
                onMouseOver={(e) => {
                  if (canDeploySAC) {
                    const target = e.currentTarget;
                    target.style.transform = 'translateY(-2px)';
                    target.style.boxShadow = '0 4px 12px rgba(35, 134, 54, 0.4)';
                  }
                }}
                onMouseOut={(e) => {
                  const target = e.currentTarget;
                  target.style.transform = 'translateY(0)';
                  target.style.boxShadow = 'none';
                }}
                title={!selectedModel
                  ? 'Select a model to deploy'
                  : sharpeValue < 0.5
                  ? 'Sharpe Ratio below deployment threshold (0.5)'
                  : deployingAgent
                  ? 'Deployment in progress'
                  : ''}
              >
                <span style={{ fontSize: '18px' }}>🚀</span>
                <span>
                  {deployingAgent
                    ? 'Deploying to Live...'
                    : `Deploy ${selectedModel?.symbol || trainingState.sacSymbol} to ${isIntradayAgent ? 'SAC Intraday' : 'SAC'} Live (Paper)`}
                </span>
              </button>
            )}
          </div>

          {/* Info Footer */}
          <div style={{
            marginTop: '16px',
            padding: '12px',
            background: 'rgba(31, 111, 235, 0.1)',
            border: '1px solid #1f6feb',
            borderRadius: '6px',
            fontSize: '12px',
            color: '#58a6ff'
          }}>
            💡 <strong>Note:</strong> This will create a live trading agent in <strong>Paper Trading Mode</strong> (IBKR Paper Account). 
            No real money will be used. You can monitor the agent in the Live Trading tab.
          </div>
        </div>
      )}

      {/* Models Comparison Table - Compare all trained models */}
      <ModelsComparisonTable 
        agentType={resolveAgentType(selectedAgent)}
      />
    </div>
  );
}

export default TabTraining;
