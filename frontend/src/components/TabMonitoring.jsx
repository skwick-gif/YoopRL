import React from 'react';
import { Card, Switch } from './common/UIComponents';

function TabMonitoring() {
  const metrics = [
    { label: 'IBKR Latency', value: '12ms', sub: 'Avg: 15ms' },
    { label: 'Action Frequency', value: '3.2/hr', sub: 'PPO: 2.1 | SAC: 1.1' },
    { label: 'Reward/Episode', value: '+0.87', sub: 'Last 100 episodes', positive: true },
    { label: 'System Health', value: '98%', sub: 'All systems nominal', positive: true }
  ];

  const alerts = [
    { time: '14:05', type: 'Warning', message: 'Latency spike: 52ms', sent: 'Yes' },
    { time: '11:23', type: 'Info', message: 'Daily P&L target reached', sent: 'Yes' },
    { time: '09:30', type: 'Critical', message: 'IBKR reconnected after 30s', sent: 'Yes' }
  ];

  return (
    <div>
      <div className="grid">
        {metrics.map((metric, index) => (
          <Card
            key={index}
            label={metric.label}
            value={metric.value}
            sub={metric.sub}
            positive={metric.positive}
          />
        ))}
      </div>

      <div className="main-layout">
        <div className="chart-area">
          <div className="chart-title">Performance Metrics</div>
          <div className="chart-placeholder">Reward/Episode & Action Frequency</div>
        </div>

        <div className="controls">
          <div className="control-card">
            <div className="control-title">Alert Settings</div>
            {/* Drawdown > 15%: Triggers when portfolio loses more than 15% from peak
                - Critical alert to prevent excessive losses
                - System may auto-stop trading if enabled in Settings
                - Sends immediate notification */}
            <div style={{ fontSize: '11px', marginBottom: '8px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <input type="checkbox" defaultChecked /> Drawdown &gt; 15%
              </label>
            </div>
            
            {/* IBKR Disconnect: Alerts when connection to Interactive Brokers is lost
                - Critical for live trading - no data or order execution during disconnect
                - System attempts auto-reconnect
                - Immediate notification sent */}
            <div style={{ fontSize: '11px', marginBottom: '8px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <input type="checkbox" defaultChecked /> IBKR Disconnect
              </label>
            </div>
            
            {/* Latency > 50ms: Triggers when order execution delay exceeds threshold
                - High latency can cause slippage and missed opportunities
                - May indicate network or broker issues
                - Consider pausing trading if persistent */}
            <div style={{ fontSize: '11px', marginBottom: '8px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <input type="checkbox" defaultChecked /> Latency &gt; 50ms
              </label>
            </div>
            
            {/* Agent Stall: Detects when agent stops making decisions
                - Could indicate model freeze or data feed issue
                - System health check runs every 5 minutes
                - Helps catch silent failures */}
            <div style={{ fontSize: '11px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <input type="checkbox" /> Agent Stall
              </label>
            </div>
          </div>

          <div className="control-card">
            <div className="control-title">Alert Channels</div>
            {/* WhatsApp: Sends alerts via WhatsApp message
                - Instant mobile notifications
                - Requires WhatsApp Business API setup
                - Best for critical alerts */}
            <Switch label="WhatsApp" checked={true} onChange={() => {}} />
            
            {/* Telegram: Sends alerts via Telegram bot
                - Fast and reliable
                - Requires Telegram bot token (set in Settings)
                - Good for all alert types */}
            <Switch label="Telegram" checked={false} onChange={() => {}} />
          </div>
        </div>
      </div>

      <div className="trades-table">
        <div className="control-title">Alert History</div>
        <div className="table-header">
          <span>Time</span>
          <span>Type</span>
          <span style={{ gridColumn: 'span 2' }}>Message</span>
          <span>Sent</span>
        </div>
        {alerts.map((alert, index) => (
          <div key={index} className="table-row">
            <span>{alert.time}</span>
            <span>{alert.type}</span>
            <span style={{ gridColumn: 'span 2' }}>{alert.message}</span>
            <span style={{ color: '#3fb950' }}>{alert.sent}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default TabMonitoring;
