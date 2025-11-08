import React from 'react';
import { Button } from './common/UIComponents';

function TabLiveTrading() {
  const agents = [
    {
      name: 'PPO Agent - Stock (AAPL)',
      status: 'Active',
      indicators: [
        { label: 'RSI', value: '32.1', color: '' },
        { label: 'MACD', value: '0.45', color: '' },
        { label: 'EMA Δ', value: '+2.3%', color: '#3fb950' }
      ],
      lastDecision: 'BUY AAPL 50 shares @ $178.50',
      reason: 'Momentum + High volume + RSI oversold',
      confidence: '85%',
      time: '14:23:15'
    },
    {
      name: 'SAC Agent - Leveraged ETF (TNA)',
      status: 'Active',
      indicators: [
        { label: 'VIX', value: '18.4', color: '#f85149' },
        { label: 'MACD', value: '-0.12', color: '' },
        { label: 'Volatility', value: 'High', color: '#f59e0b' }
      ],
      lastDecision: 'HOLD TNA 100 shares',
      reason: 'Volatility spike + Risk-adjusted waiting',
      confidence: '72%',
      time: '14:22:48'
    }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      {agents.map((agent, index) => (
        <div key={index} className="agent-card">
          <div className="agent-header">
            <div className="agent-name">{agent.name}</div>
            <div className="agent-status">{agent.status}</div>
          </div>

          <div className="chart-area agent-chart-container">
            <div style={{ height: '100%', background: '#0d1117', border: '1px dashed #30363d', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6e7681', fontSize: '11px' }}>
              Price Chart + Buy/Sell Markers + Volume + Indicators
            </div>
          </div>

          <div className="agent-thinking">
            <strong>Last Decision:</strong> {agent.lastDecision}<br />
            <strong>Reason:</strong> {agent.reason}<br />
            <strong>Confidence:</strong> {agent.confidence} | <strong>Time:</strong> {agent.time}
          </div>

          <div className="btn-group" style={{ marginTop: '10px' }}>
            {/* Pause Agent Button: Temporarily stops this specific agent
                - Agent stops making new trading decisions
                - Keeps current positions open
                - Other agents continue running normally
                - Can resume later without restarting system */}
            <Button className="pause">Pause</Button>
            
            {/* Stop Agent Button: Completely stops this agent
                - Agent will not trade anymore until manually restarted
                - Current positions remain open (manual close needed)
                - Requires going back to Training or Dashboard to restart */}
            <Button className="stop">Stop</Button>
            
            {/* Close Position Button: Immediately closes all positions for this agent
                - Sends market order to exit current positions
                - Realizes profit/loss immediately
                - Agent can continue trading after positions are closed
                - Use when you want to lock in gains or cut losses */}
            <Button className="" style={{ background: '#1f2937' }}>Close Position</Button>
          </div>
        </div>
      ))}
    </div>
  );
}

export default TabLiveTrading;
