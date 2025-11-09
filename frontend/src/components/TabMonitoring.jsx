import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Card, Switch, Button, InputGroup } from './common/UIComponents';
import monitoringAPI from '../services/monitoringAPI';

const REFRESH_PRESETS = [
  { label: 'Off', value: 0 },
  { label: '15s', value: 15000 },
  { label: '30s', value: 30000 },
  { label: '60s', value: 60000 },
];

const HOURS_OPTIONS = [6, 12, 24, 48, 168];

const DEFAULT_ACTION_LIMIT = 50;
const DEFAULT_ALERT_LIMIT = 50;

const formatCurrency = (value) => {
  if (value === null || value === undefined) return '—';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: value >= 1 ? 0 : 2,
  }).format(value);
};

const formatPercent = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const formatted = new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
  return `${value >= 0 ? '+' : ''}${formatted}%`;
};

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
};

const formatDateTime = (isoString) => {
  if (!isoString) return '—';
  const date = new Date(isoString);
  return `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
};

const formatTime = (isoString) => {
  if (!isoString) return '—';
  return new Date(isoString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const computeActionRate = (actions, hours) => {
  if (!actions?.length || !hours) return 0;
  const lookback = Math.min(hours, 24);
  return actions.length / lookback;
};

const Sparkline = ({ data, width = 320, height = 140 }) => {
  if (!data?.length) {
    return <div className="chart-placeholder">No equity history available</div>;
  }

  const values = data.map((point) => point.net_liquidation || 0);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const points = values.map((value, index) => {
    const x = (index / (values.length - 1 || 1)) * width;
    const y = height - ((value - min) / range) * height;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Equity sparkline">
      <defs>
        <linearGradient id="sparklineGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#238636" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#238636" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        fill="none"
        stroke="#3fb950"
        strokeWidth="2"
        points={points.join(' ')}
      />
      <polygon
        points={`0,${height} ${points.join(' ')} ${width},${height}`}
        fill="url(#sparklineGradient)"
      />
    </svg>
  );
};

const SeverityBadge = ({ severity }) => {
  const normalized = (severity || '').toLowerCase();
  const color = normalized === 'critical' ? '#f85149' : normalized === 'warning' ? '#d29922' : '#58a6ff';
  return (
    <span style={{ color, fontWeight: 600 }}>
      {severity || 'INFO'}
    </span>
  );
};

function TabMonitoring() {
  const [hours, setHours] = useState(24);
  const [refreshMs, setRefreshMs] = useState(30000);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const [summary, setSummary] = useState(null);
  const [actions, setActions] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [systemLogs, setSystemLogs] = useState([]);

  const refreshData = useCallback(async (initial = false) => {
    if (initial) {
      setIsLoading(true);
    } else {
      setIsRefreshing(true);
    }
    setError(null);

    try {
      const [summaryResp, actionsResp, alertsResp, systemResp] = await Promise.all([
        monitoringAPI.getSummary({ hours }),
        monitoringAPI.getActions({ limit: DEFAULT_ACTION_LIMIT }),
        monitoringAPI.getAlerts({ limit: DEFAULT_ALERT_LIMIT }),
        monitoringAPI.getSystem({ logs_limit: 50 }),
      ]);

      if (summaryResp.status !== 'success') {
        throw new Error(summaryResp.error || 'Failed to load summary');
      }
      if (actionsResp.status !== 'success') {
        throw new Error(actionsResp.error || 'Failed to load recent actions');
      }
      if (alertsResp.status !== 'success') {
        throw new Error(alertsResp.error || 'Failed to load alerts');
      }
      if (systemResp.status !== 'success') {
        throw new Error(systemResp.error || 'Failed to load system diagnostics');
      }

      setSummary(summaryResp.data || null);
      setActions(actionsResp.data?.actions || []);
      setAlerts(alertsResp.data?.alerts || []);
      setSystemLogs(systemResp.data?.logs || []);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err.message || 'Failed to load monitoring data');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [hours]);

  useEffect(() => {
    refreshData(true);
  }, [refreshData]);

  useEffect(() => {
    if (!autoRefresh || refreshMs <= 0) {
      return undefined;
    }

    const interval = setInterval(() => {
      refreshData(false);
    }, refreshMs);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshMs, refreshData]);

  const equity = summary?.equity;
  const agentStats = summary?.agents;
  const actionRate = useMemo(() => computeActionRate(actions, hours), [actions, hours]);

  const cards = useMemo(() => ([
    {
      label: 'Net Liquidation',
      value: formatCurrency(equity?.net_liquidation),
      sub: `Change ${formatPercent(equity?.change_pct)}`,
      positive: equity?.change_pct > 0,
      negative: equity?.change_pct < 0,
    },
    {
      label: 'Max Drawdown',
      value: formatPercent(equity?.max_drawdown_pct),
      sub: `Window: ${hours}h`,
      negative: equity?.max_drawdown_pct > 5,
    },
    {
      label: 'Agents Running',
      value: `${agentStats?.running || 0}/${agentStats?.total || 0}`,
      sub: `${agentStats?.stopped || 0} stopped`,
    },
    {
      label: 'Recent Alerts',
      value: alerts.length,
      sub: `${alerts.filter((item) => (item.severity || '').toUpperCase() === 'CRITICAL').length} critical`,
      negative: alerts.some((item) => (item.severity || '').toUpperCase() === 'CRITICAL'),
    },
  ]), [equity, agentStats, alerts, hours]);

  return (
    <div>
      <div className="grid">
        {cards.map((metric, index) => (
          <Card
            key={metric.label || index}
            label={metric.label}
            value={metric.value}
            sub={metric.sub}
            positive={metric.positive}
            negative={metric.negative}
          />
        ))}
      </div>

      <div className="main-layout" style={{ marginTop: '20px' }}>
        <div className="chart-area">
          <div className="chart-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Equity Curve</span>
            <span style={{ fontSize: '11px', color: '#8b949e' }}>
              {equity?.history?.length ? `${equity.history.length} points` : 'No data'}
            </span>
          </div>
          <Sparkline data={equity?.history} />
          <div style={{ display: 'flex', gap: '20px', marginTop: '16px', fontSize: '12px' }}>
            <div>
              <div style={{ color: '#8b949e' }}>Buying Power</div>
              <div style={{ fontWeight: 600 }}>{formatCurrency(equity?.buying_power)}</div>
            </div>
            <div>
              <div style={{ color: '#8b949e' }}>Cash</div>
              <div style={{ fontWeight: 600 }}>{formatCurrency(equity?.cash)}</div>
            </div>
            <div>
              <div style={{ color: '#8b949e' }}>Unrealized P&amp;L</div>
              <div style={{ fontWeight: 600 }}>{formatCurrency(equity?.unrealized_pnl)}</div>
            </div>
            <div>
              <div style={{ color: '#8b949e' }}>Realized P&amp;L</div>
              <div style={{ fontWeight: 600 }}>{formatCurrency(equity?.realized_pnl)}</div>
            </div>
            <div>
              <div style={{ color: '#8b949e' }}>Action Rate</div>
              <div style={{ fontWeight: 600 }}>{formatNumber(actionRate, 2)} / hr</div>
            </div>
          </div>
        </div>

        <div className="controls">
          <div className="control-card">
            <div className="control-title">Data Controls</div>
            <InputGroup className="monitoring-input-group">
              <label htmlFor="monitoring-hours" style={{ fontSize: '12px', color: '#8b949e' }}>Time Window</label>
              <select
                id="monitoring-hours"
                value={hours}
                onChange={(event) => setHours(Number(event.target.value))}
                style={{ width: '100%', marginTop: '4px' }}
              >
                {HOURS_OPTIONS.map((option) => (
                  <option key={option} value={option}>{option}h</option>
                ))}
              </select>
            </InputGroup>

            <InputGroup className="monitoring-input-group" style={{ marginTop: '12px' }}>
              <label htmlFor="monitoring-refresh" style={{ fontSize: '12px', color: '#8b949e' }}>Auto Refresh</label>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '4px' }}>
                <Switch
                  checked={autoRefresh && refreshMs > 0}
                  onChange={(event) => setAutoRefresh(event.target.checked)}
                />
                <select
                  id="monitoring-refresh"
                  value={refreshMs}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    setRefreshMs(value);
                    if (value === 0) {
                      setAutoRefresh(false);
                    }
                  }}
                  style={{ marginLeft: '12px', flex: 1 }}
                >
                  {REFRESH_PRESETS.map((option) => (
                    <option key={option.label} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>
            </InputGroup>

            <Button
              onClick={() => refreshData(false)}
              variant="primary"
              className="monitoring-refresh-btn"
              style={{ marginTop: '14px' }}
            >
              {isLoading || isRefreshing ? 'Refreshing…' : 'Refresh Now'}
            </Button>
            <div style={{ fontSize: '11px', color: '#8b949e', marginTop: '8px' }}>
              Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : '—'}
            </div>
            {error && (
              <div style={{ color: '#f85149', fontSize: '11px', marginTop: '8px' }}>
                {error}
              </div>
            )}
          </div>

          <div className="control-card">
            <div className="control-title">Agent Status</div>
            <div style={{ fontSize: '12px', color: '#8b949e', marginBottom: '8px' }}>
              {agentStats?.timestamp ? `Last heartbeat ${formatDateTime(agentStats.timestamp)}` : 'Awaiting updates'}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {(agentStats?.statuses || []).map((agent) => (
                <div key={agent.agent_id} className="monitoring-agent-row">
                  <div style={{ fontWeight: 600 }}>{agent.agent_id}</div>
                  <div style={{ fontSize: '11px', color: '#8b949e' }}>
                    {agent.symbol} • {agent.agent_type}
                  </div>
                  <div style={{ fontSize: '11px', marginTop: '4px' }}>
                    Status: {agent.is_running ? <span style={{ color: '#3fb950' }}>RUNNING</span> : <span style={{ color: '#f85149' }}>STOPPED</span>} · Last action {agent.last_action || '—'}
                  </div>
                  <div style={{ fontSize: '11px', color: '#8b949e' }}>
                    Last run {agent.last_run_at ? formatTime(agent.last_run_at) : '—'} · Position {agent.current_position}
                  </div>
                </div>
              ))}
              {(!agentStats?.statuses || agentStats.statuses.length === 0) && (
                <div style={{ fontSize: '12px', color: '#8b949e' }}>No live agents deployed.</div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="trades-table" style={{ marginTop: '24px' }}>
        <div className="control-title">Recent Agent Actions</div>
        <div className="table-header">
          <span>Time</span>
          <span>Agent</span>
          <span>Symbol</span>
          <span>Action</span>
          <span>Qty</span>
          <span>Price</span>
        </div>
        {actions.length === 0 && (
          <div className="table-row" style={{ color: '#8b949e' }}>No actions recorded in this window.</div>
        )}
        {actions.map((action) => (
          <div key={action.id || `${action.agent_name}-${action.timestamp}`} className="table-row">
            <span>{formatTime(action.datetime)}</span>
            <span>{action.agent_name}</span>
            <span>{action.symbol}</span>
            <span>{action.action}</span>
            <span>{formatNumber(action.quantity, 0)}</span>
            <span>{formatCurrency(action.price)}</span>
          </div>
        ))}
      </div>

      <div className="trades-table" style={{ marginTop: '24px' }}>
        <div className="control-title">Alert History</div>
        <div className="table-header">
          <span>Time</span>
          <span>Severity</span>
          <span>Agent</span>
          <span>Symbol</span>
          <span style={{ gridColumn: 'span 2' }}>Description</span>
        </div>
        {alerts.length === 0 && (
          <div className="table-row" style={{ color: '#8b949e' }}>No alerts logged.</div>
        )}
        {alerts.map((alert) => (
          <div key={alert.id || `${alert.event_type}-${alert.timestamp}`} className="table-row">
            <span>{formatTime(alert.datetime)}</span>
            <span><SeverityBadge severity={alert.severity} /></span>
            <span>{alert.agent_name || '—'}</span>
            <span>{alert.symbol || '—'}</span>
            <span style={{ gridColumn: 'span 2' }}>{alert.description}</span>
          </div>
        ))}
      </div>

      <div className="trades-table" style={{ marginTop: '24px' }}>
        <div className="control-title">System Logs</div>
        <div className="table-header">
          <span>Time</span>
          <span>Component</span>
          <span>Level</span>
          <span style={{ gridColumn: 'span 2' }}>Message</span>
        </div>
        {systemLogs.length === 0 && (
          <div className="table-row" style={{ color: '#8b949e' }}>No system events recorded.</div>
        )}
        {systemLogs.map((log) => (
          <div key={log.id || `${log.timestamp}-${log.component}`} className="table-row">
            <span>{formatTime(log.datetime)}</span>
            <span>{log.component}</span>
            <span>{log.level}</span>
            <span style={{ gridColumn: 'span 2' }}>{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default TabMonitoring;
