/**
 * HyperparameterGrid.jsx (Updated - Phase 1)
 * Part of: Training Tab
 * 
 * Purpose: Displays hyperparameter configuration grids for PPO and SAC agents
 * - PPO (Proximal Policy Optimization) for regular stocks
 * - SAC (Soft Actor-Critic) for leveraged ETFs
 * 
 * Phase 1 Updates:
 * - Converted from uncontrolled (defaultValue) to controlled components (value + onChange)
 * - Accepts trainingState props from TabTraining.jsx
 * - All inputs now wired to state management
 * - Shows only relevant agent based on agentType prop
 * 
 * Props:
 * - agentType: 'PPO' or 'SAC' - determines which form to show
 * - trainingState: Object with all state values and setters from useTrainingState hook
 * - isIntraday: When true, render intraday-specific inputs and hints
 * 
 * Wiring:
 * - Each input uses value={trainingState.X} and onChange={trainingState.setX}
 * - Changes immediately reflected in parent state
 * - State used when building training config for API
 */

import React from 'react';
import { ParamItem, Card } from '../common/UIComponents';
import ConfigManager from './ConfigManager';
import { DEFAULT_COMMISSION_CONFIG } from '../../hooks/useTrainingState';

const headerRowStyle = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: '12px',
  flexWrap: 'wrap',
  marginBottom: '12px'
};

const INTRADAY_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'TNA', 'UPRO', 'TQQQ', 'DIA', 'UDOW'];
const INTRADAY_BENCHMARKS = ['SPY', 'QQQ', 'IWM', 'DIA'];
const SYMBOL_BENCHMARK_MAP = {
  SPY: 'SPY',
  QQQ: 'QQQ',
  IWM: 'IWM',
  DIA: 'DIA',
  TNA: 'IWM',
  UPRO: 'SPY',
  TQQQ: 'QQQ',
  UDOW: 'DIA',
};

function HyperparameterGrid({ agentType, trainingState, onLoadConfig, isIntraday = false }) {
  const allowedIntradaySymbols = INTRADAY_SYMBOLS.join(', ');
  const allowedBenchmarkSymbols = INTRADAY_BENCHMARKS.join(', ');

  const handleSacSymbolChange = (value) => {
    const upper = (value || '').toUpperCase();
    trainingState.setSacSymbol(upper);

    if (!isIntraday) {
      return;
    }

    const suggestedBenchmark = SYMBOL_BENCHMARK_MAP[upper];
    if (!suggestedBenchmark) {
      return;
    }

    const currentBenchmark = (trainingState.sacBenchmarkSymbol || '').toUpperCase();
    if (!currentBenchmark || currentBenchmark === suggestedBenchmark || !SYMBOL_BENCHMARK_MAP[currentBenchmark]) {
      trainingState.setSacBenchmarkSymbol(suggestedBenchmark);
    }
  };

  return (
    <>
      {isIntraday && (
        <>
          <datalist id="intraday-symbol-options">
            {INTRADAY_SYMBOLS.map((symbol) => (
              <option key={symbol} value={symbol} />
            ))}
          </datalist>
          <datalist id="intraday-benchmark-options">
            {INTRADAY_BENCHMARKS.map((symbol) => (
              <option key={symbol} value={symbol} />
            ))}
          </datalist>
        </>
      )}

      {/* Show PPO form when PPO is selected */}
      {agentType === 'PPO' && (
        <Card style={{ marginBottom: '12px' }}>
          <div style={headerRowStyle}>
            <div className="control-title">PPO Hyperparameters (Stock)</div>
            {onLoadConfig && (
              <ConfigManager agentType="PPO" onLoadConfig={onLoadConfig} compact />
            )}
          </div>
          <div className="hyperparam-inline">
            <ParamItem label="Symbol">
              <input 
                type="text" 
                className="param-input" 
                value={trainingState.ppoSymbol}
                onChange={(e) => trainingState.setPpoSymbol(e.target.value.toUpperCase())}
                placeholder="e.g. AAPL, TSLA, GOOGL"
                autoComplete="off"
                title="Stock symbol to trade (e.g., AAPL, TSLA, GOOGL). This determines which stock the agent will learn to trade."
              />
            </ParamItem>
            <ParamItem label="LR">
              <input 
                type="number" 
                className="param-input" 
                value={trainingState.ppoLearningRate}
                onChange={(e) => trainingState.setPpoLearningRate(parseFloat(e.target.value))}
                step="0.0001"
                title="Learning Rate: Controls how quickly the agent learns. Lower values (0.0001-0.0003) = slower but more stable learning. Higher values = faster but potentially unstable."
              />
            </ParamItem>
            <ParamItem label="γ">
              <input 
                type="number" 
                className="param-input" 
                value={trainingState.ppoGamma}
                onChange={(e) => trainingState.setPpoGamma(parseFloat(e.target.value))}
                step="0.01"
                title="Gamma (Discount Factor): How much the agent values future rewards vs immediate rewards. 0.99 = values long-term gains. Lower values = prefers quick profits."
              />
            </ParamItem>
            <ParamItem label="Batch">
              <input 
                type="number" 
                className="param-input" 
                value={trainingState.ppoBatchSize}
                onChange={(e) => trainingState.setPpoBatchSize(parseInt(e.target.value))}
                title="Batch Size: Number of experiences used in each training step. Larger batches (256-512) = more stable learning but slower. Smaller batches = faster updates but noisier."
              />
            </ParamItem>
            <ParamItem label="Risk Penalty">
              <input 
                type="number" 
                className="param-input" 
                value={trainingState.ppoRiskPenalty}
                onChange={(e) => trainingState.setPpoRiskPenalty(parseFloat(e.target.value))}
                step="0.1"
                title="Risk Penalty: Negative reward for taking risky actions. More negative (-0.5 to -1.0) = agent becomes more conservative. Less negative = agent takes more risks."
              />
            </ParamItem>
            <ParamItem label="Episodes">
              <input 
                type="number" 
                className="param-input" 
                value={trainingState.ppoEpisodes}
                onChange={(e) => trainingState.setPpoEpisodes(parseInt(e.target.value))}
                title="Training Episodes: How many times the agent practices trading. More episodes = better learning but longer training time. Typical range: 30,000-100,000."
              />
            </ParamItem>
            <ParamItem label="Retrain">
              <select 
                className="param-input"
                value={trainingState.ppoRetrain}
                onChange={(e) => trainingState.setPpoRetrain(e.target.value)}
                title="Retraining Frequency: How often the agent retrains itself with new data. Weekly = good balance. Daily = adapts faster to market changes. OFF = never retrains."
              >
                <option>OFF</option>
                <option>Daily</option>
                <option>Weekly</option>
                <option>Monthly</option>
              </select>
            </ParamItem>
          </div>
        </Card>
      )}

      {/* Show SAC form when SAC is selected */}
      {agentType === 'SAC' && (
        <Card style={{ marginBottom: '12px' }}>
          <div style={headerRowStyle}>
            <div className="control-title">
              {isIntraday ? 'SAC + DSR Hyperparameters (15m Intraday)' : 'SAC Hyperparameters (ETF)'}
            </div>
            {onLoadConfig && (
              <ConfigManager
                agentType={isIntraday ? 'SAC_INTRADAY_DSR' : 'SAC'}
                onLoadConfig={onLoadConfig}
                compact
              />
            )}
          </div>
          {isIntraday && (
            <div
              style={{
                marginBottom: '12px',
                backgroundColor: '#353535',
                borderRadius: '6px',
                padding: '10px 12px',
                fontSize: '13px',
                color: '#d9e6ff'
              }}
            >
              ⚡ Intraday mode supports symbols: {allowedIntradaySymbols}. Benchmark defaults to the paired index but can be adjusted here.
            </div>
          )}
          <div className="hyperparam-inline">
            <ParamItem label="Symbol">
              <input 
                type="text" 
                className="param-input" 
                value={trainingState.sacSymbol}
                onChange={(e) => handleSacSymbolChange(e.target.value)}
                list={isIntraday ? 'intraday-symbol-options' : undefined}
                placeholder={isIntraday ? 'e.g. SPY, QQQ, IWM' : 'e.g. TNA, TQQQ, SPXL'}
                title={isIntraday
                  ? `Intraday mode currently supports: ${allowedIntradaySymbols}.`
                  : 'ETF symbol to trade (e.g., TNA, TQQQ, SPXL). This determines which leveraged ETF the agent will learn to trade.'
                }
                autoComplete="off"
              />
            </ParamItem>
            {isIntraday && (
              <ParamItem label="Benchmark (15m)">
                <input
                  type="text"
                  className="param-input"
                  value={trainingState.sacBenchmarkSymbol || ''}
                  onChange={(e) => trainingState.setSacBenchmarkSymbol(e.target.value.toUpperCase())}
                  list="intraday-benchmark-options"
                  placeholder="Default derived automatically"
                  title={`Benchmark symbol used for DSR calculations (allowed: ${allowedBenchmarkSymbols}).`}
                  autoComplete="off"
                />
              </ParamItem>
            )}
            <ParamItem label="LR">
              <input 
                type="number" 
                className="param-input" 
                value={trainingState.sacLearningRate}
                onChange={(e) => trainingState.setSacLearningRate(parseFloat(e.target.value))}
                step="0.0001"
                title="Learning Rate: Controls how quickly the agent learns. Lower values (0.0001-0.0003) = slower but more stable learning. Higher values = faster but potentially unstable."
              />
            </ParamItem>
            <ParamItem label="Entropy">
              <input 
                type="number" 
                className="param-input" 
                value={trainingState.sacEntropy}
                onChange={(e) => trainingState.setSacEntropy(parseFloat(e.target.value))}
                step="0.01"
                title="Entropy Coefficient: Controls exploration vs exploitation. Higher (0.2-0.5) = more random exploration. Lower = more focused on known strategies."
              />
            </ParamItem>
            <ParamItem label="Batch">
              <input 
                type="number" 
                className="param-input" 
                value={trainingState.sacBatchSize}
                onChange={(e) => trainingState.setSacBatchSize(parseInt(e.target.value))}
                title="Batch Size: Number of experiences used in each training step. Larger batches (256-512) = more stable learning but slower. Smaller batches = faster updates but noisier."
              />
            </ParamItem>
            <ParamItem label="Vol Penalty">
              <input 
                type="number" 
                className="param-input" 
                value={trainingState.sacVolPenalty}
                onChange={(e) => trainingState.setSacVolPenalty(parseFloat(e.target.value))}
                step="0.1"
                title="Volatility Penalty: Negative reward for trading during high volatility periods. More negative = agent avoids volatile markets. Less negative = agent is comfortable with volatility."
              />
            </ParamItem>
            <ParamItem label="Episodes">
              <input 
                type="number" 
                className="param-input" 
                value={trainingState.sacEpisodes}
                onChange={(e) => trainingState.setSacEpisodes(parseInt(e.target.value))}
                title="Training Episodes: How many times the agent practices trading. More episodes = better learning but longer training time. Typical range: 30,000-100,000."
              />
            </ParamItem>
            <ParamItem label="Retrain">
              <select 
                className="param-input"
                value={trainingState.sacRetrain}
                onChange={(e) => trainingState.setSacRetrain(e.target.value)}
                title="Retraining Frequency: How often the agent retrains itself with new data. Weekly = good balance. Daily = adapts faster to market changes. OFF = never retrains."
              >
                <option>OFF</option>
                <option>Daily</option>
                <option>Weekly</option>
                <option>Monthly</option>
              </select>
            </ParamItem>
          </div>
        </Card>
      )}

      {/* Training Settings - Date Range, Commission, Optuna Trials */}
      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">Training Settings</div>
        <div className="hyperparam-inline">
          <ParamItem label="Start Date">
            <input 
              type="date" 
              className="param-input" 
              value={trainingState.startDate}
              onChange={(e) => trainingState.setStartDate(e.target.value)}
              title="Start Date: The beginning date of historical data for training. Choose a date with enough market history (typically 1-2 years back)."
            />
          </ParamItem>
          <ParamItem label="End Date">
            <input 
              type="date" 
              className="param-input" 
              value={trainingState.endDate}
              onChange={(e) => trainingState.setEndDate(e.target.value)}
              title="End Date: The last date of historical data for training. Usually set to today or recent date. More data = better learning."
            />
          </ParamItem>
          <ParamItem label="Commission ($/share)">
            <input 
              type="number" 
              className="param-input" 
              value={trainingState.commission}
              onChange={(e) => {
                const nextValue = Number.parseFloat(e.target.value);
                trainingState.setCommission(Number.isFinite(nextValue) ? nextValue : DEFAULT_COMMISSION_CONFIG.per_share);
              }}
              step="0.001"
              min="0"
              title="Commission per share in USD. Default aligns with IBKR tiered equities pricing (0.01 $/share, $2.50 minimum, 1% cap)."
            />
          </ParamItem>
          <ParamItem label="Optuna Trials">
            <input 
              type="number" 
              className="param-input" 
              value={trainingState.optunaTrials}
              onChange={(e) => trainingState.setOptunaTrials(parseInt(e.target.value))}
              title="Optuna Trials: Number of hyperparameter combinations to test. More trials = better optimization but longer training. 50-200 trials is typical."
            />
          </ParamItem>
          {isIntraday && (
            <>
              <ParamItem label="Slippage (bps)">
                <input
                  type="number"
                  className="param-input"
                  value={trainingState.intradaySlippageBps}
                  onChange={(e) => trainingState.setIntradaySlippageBps(parseFloat(e.target.value) || 0)}
                  step="0.1"
                  min="0"
                  title="Estimated round-trip slippage expressed in basis points. Applied symmetrically to buys and sells."
                />
              </ParamItem>
              <ParamItem label="Slippage ($/share)">
                <input
                  type="number"
                  className="param-input"
                  value={trainingState.intradaySlippagePerShare}
                  onChange={(e) => trainingState.setIntradaySlippagePerShare(parseFloat(e.target.value) || 0)}
                  step="0.005"
                  min="0"
                  title="Optional absolute slippage per share (in USD). Useful when spreads remain fixed in cents."
                />
              </ParamItem>
              <ParamItem label="Forced Exit (minutes)">
                <input
                  type="number"
                  className="param-input"
                  value={trainingState.forcedExitMinutes}
                  onChange={(e) => trainingState.setForcedExitMinutes(parseFloat(e.target.value) || 0)}
                  step="1"
                  min="0"
                  title="Minute-of-session cutoff for forced liquidation (market open = 0). 375 corresponds to 15:45 New York."
                />
              </ParamItem>
              <ParamItem label="Exit Tolerance (min)">
                <input
                  type="number"
                  className="param-input"
                  value={trainingState.forcedExitTolerance}
                  onChange={(e) => trainingState.setForcedExitTolerance(parseFloat(e.target.value) || 0)}
                  step="0.5"
                  min="0"
                  title="Grace window before the forced exit triggers. Helps avoid premature exits due to clock jitter."
                />
              </ParamItem>
              <ParamItem label="Exit Signal Column">
                <select
                  className="param-input"
                  value={trainingState.forcedExitColumn || ''}
                  onChange={(e) => trainingState.setForcedExitColumn(e.target.value)}
                  title="Which metadata column the environment should inspect for forced exit timing. Leave on minutes_from_open for default behaviour."
                >
                  <option value="minutes_from_open">minutes_from_open</option>
                  <option value="time_fraction">time_fraction</option>
                  <option value="timestamp">timestamp</option>
                  <option value="is_session_end">is_session_end</option>
                  <option value="auto">auto (use heuristics)</option>
                </select>
              </ParamItem>
            </>
          )}
        </div>
      </Card>
    </>
  );
}

export default HyperparameterGrid;
