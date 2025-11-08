/**
 * HyperparameterGrid.jsx
 * Part of: Training Tab
 * 
 * Purpose: Displays hyperparameter configuration grids for PPO and SAC agents
 * - PPO (Proximal Policy Optimization) for regular stocks
 * - SAC (Soft Actor-Critic) for leveraged ETFs
 * 
 * Each agent has 7 configurable parameters:
 * - Symbol, Learning Rate, Gamma/Entropy, Batch Size, Penalties, Episodes, Retrain Frequency
 */

import React from 'react';
import { ParamItem, Card } from '../common/UIComponents';

function HyperparameterGrid() {
  return (
    <>
      {/* PPO Agent Hyperparameters - For Stock Trading */}
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

      {/* SAC Agent Hyperparameters - For Leveraged ETF Trading */}
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

      {/* Training Settings - Date Range, Commission, Optuna Trials */}
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
    </>
  );
}

export default HyperparameterGrid;
