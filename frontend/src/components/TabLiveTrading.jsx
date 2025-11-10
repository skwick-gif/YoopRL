/**
 * TabLiveTrading.jsx
 * Live Trading Tab - Operational Dashboard
 *
 * Purpose:
 * - Display current status for all live trading agents
 * - Provide operator controls (start, stop, run once, close, remove)
 * - Expose emergency stop for all agents in flight
 * - Surface key portfolio and execution metrics for monitoring
 *
 * Data Flow:
 * - Polls liveAPI.listAgents() every 10s for fresh agent snapshots
 * - Action buttons call corresponding liveAPI endpoints and refresh state
 * - Emergency stop invokes liveAPI.emergencyStop() and reloads agent list
 *
 * Dependencies:
 * - liveAPI: frontend service that wraps backend /api/live endpoints
 * - UIComponents: shared Button and Card primitives for consistent styling
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button, Card } from './common/UIComponents';
import liveAPI from '../services/liveAPI';

function formatCurrency(value) {
  if (value === null || value === undefined) return '-';
  return `$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  return `${Number(value).toFixed(2)}%`;
}

function TabLiveTrading() {
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pending, setPending] = useState(false);
  const [toast, setToast] = useState(null);
  const [hoursUpdating, setHoursUpdating] = useState({});
  const [streamingAgents, setStreamingAgents] = useState({});

  const fetchAgents = useCallback(async () => {
    try {
      setError(null);
      const result = await liveAPI.listAgents();
      const rawAgents = result.agents || [];
      const normalized = rawAgents.map((agent) => ({
        ...agent,
        allow_premarket: Boolean(agent.allow_premarket),
        allow_afterhours: Boolean(agent.allow_afterhours),
      }));
      setAgents(normalized);
      setStreamingAgents((prev) => {
        const activeIds = new Set(normalized.map((agent) => agent.agent_id));
        const prevKeys = Object.keys(prev);
        const next = {};
        prevKeys.forEach((agentId) => {
          if (prev[agentId] && activeIds.has(agentId)) {
            next[agentId] = true;
          }
        });
        if (prevKeys.length === Object.keys(next).length && prevKeys.every((id) => next[id])) {
          return prev;
        }
        return next;
      });
      return normalized;
    } catch (err) {
      setError(err.message || 'Failed to load agents');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const bootstrap = async () => {
      await fetchAgents();

      const stored = localStorage.getItem('yooprl:lastDeployAgent');
      if (stored) {
        try {
          const parsed = JSON.parse(stored);
          setToast({
            type: 'success',
            message: `🚀 ${parsed.agent?.agent_type || 'Agent'} ${parsed.agent?.agent_id || ''} deployed.`,
            createdAt: Date.now(),
          });
        } catch (err) {
          console.warn('Failed to parse stored deployment payload', err);
        } finally {
          localStorage.removeItem('yooprl:lastDeployAgent');
        }
      }
    };

    bootstrap();
    const interval = setInterval(fetchAgents, 10000);

    const handleDeployEvent = async (event) => {
      const detail = event.detail || {};
      setToast({
        type: 'success',
        message: `🚀 ${detail.agent_type || detail.agent?.agent_type || 'Agent'} ${detail.agent_id || detail.agent?.agent_id || ''} deployed. Refreshing…`,
        createdAt: Date.now(),
        agent: detail.agent,
      });
      localStorage.removeItem('yooprl:lastDeployAgent');
      await fetchAgents();
    };

    window.addEventListener('yooprl-agent-deployed', handleDeployEvent);

    return () => {
      clearInterval(interval);
      window.removeEventListener('yooprl-agent-deployed', handleDeployEvent);
    };
  }, [fetchAgents]);

  useEffect(() => {
    if (!toast) {
      return undefined;
    }
    const timer = setTimeout(() => {
      setToast((current) => (current === toast ? null : current));
    }, 6000);
    return () => clearTimeout(timer);
  }, [toast]);

  const handleAction = useCallback(async (action, agentId) => {
    try {
      setPending(true);
      setError(null);
      switch (action) {
        case 'start':
          await liveAPI.startAgent(agentId);
          break;
        case 'stop':
          await liveAPI.stopAgent(agentId);
          break;
        case 'run':
          await liveAPI.runAgent(agentId);
          break;
        case 'close':
          await liveAPI.closePosition(agentId);
          break;
        case 'remove':
          await liveAPI.removeAgent(agentId);
          break;
        default:
          throw new Error(`Unsupported action ${action}`);
      }
      await fetchAgents();
      setToast({
        type: 'success',
        message: `✅ ${action.toUpperCase()} command sent to ${agentId}.`,
        createdAt: Date.now(),
      });
    } catch (err) {
      setToast({
        type: 'warning',
        message: `⚠️ ${action.toUpperCase()} command failed: ${err.message || 'Unknown error'}`,
        createdAt: Date.now(),
      });
      setError(err.message || 'Operation failed');
    } finally {
      setPending(false);
    }
  }, [fetchAgents]);

  const handleToggleTradingHours = useCallback(async (agentId, field, value, symbol) => {
    setHoursUpdating((prev) => ({ ...prev, [agentId]: true }));
    try {
      await liveAPI.updateTradingHours(agentId, { [field]: value });
      setAgents((prev) => prev.map((agent) => (
        agent.agent_id === agentId
          ? { ...agent, [field]: value }
          : agent
      )));
      const label = field === 'allow_premarket' ? 'Pre-market' : 'After-hours';
      setToast({
        type: 'success',
        message: `${value ? '✅' : '⏸️'} ${label} trading ${value ? 'enabled' : 'disabled'} for ${symbol}.`,
        createdAt: Date.now(),
      });
    } catch (err) {
      setToast({
        type: 'warning',
        message: `⚠️ Failed to update trading hours: ${err.message || 'Unknown error'}`,
        createdAt: Date.now(),
      });
    } finally {
      setHoursUpdating((prev) => {
        const next = { ...prev };
        delete next[agentId];
        return next;
      });
    }
  }, [setToast]);

  const handleEmergencyStop = useCallback(async () => {
    try {
      setPending(true);
      await liveAPI.emergencyStop();
      await fetchAgents();
      setToast({
        type: 'warning',
        message: '⛔ Emergency stop executed for all agents.',
        createdAt: Date.now(),
      });
    } catch (err) {
      setToast({
        type: 'warning',
        message: `⚠️ Emergency stop failed: ${err.message || 'Unknown error'}`,
        createdAt: Date.now(),
      });
      setError(err.message || 'Emergency stop failed');
    } finally {
      setPending(false);
    }
  }, [fetchAgents]);

  const handleToggleStreaming = useCallback((agentId, symbol) => {
    const isActive = Boolean(streamingAgents[agentId]);
    setStreamingAgents((prev) => {
      if (isActive) {
        const next = { ...prev };
        delete next[agentId];
        return next;
      }
      return { ...prev, [agentId]: true };
    });
    setToast({
      type: isActive ? 'warning' : 'success',
      message: `${isActive ? '🛑' : '📡'} Live data ${isActive ? 'stopped' : 'started'} for ${symbol}.`,
      createdAt: Date.now(),
    });
  }, [streamingAgents, setToast]);

  const emptyState = useMemo(() => (
    <Card style={{ padding: '32px', textAlign: 'center' }}>
      <p style={{ color: '#8b949e', marginBottom: '20px' }}>
        No live agents are currently active. Create one from the Training tab or via the API.
      </p>
      <Button onClick={fetchAgents} disabled={pending}>Refresh</Button>
    </Card>
  ), [fetchAgents, pending]);

  if (loading) {
    return <div>Loading data...</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>🤖 Agent Status ({agents.length})</h2>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button onClick={fetchAgents} disabled={pending}>Refresh</Button>
          <Button onClick={handleEmergencyStop} disabled={pending} style={{ background: '#da3633' }}>
            🚨 Emergency Stop
          </Button>
        </div>
      </div>

      {error && (
        <div style={{ color: '#f85149' }}>Error: {error}</div>
      )}

      {toast && (
        <div
          style={{
            background: toast.type === 'success' ? 'rgba(35, 134, 54, 0.15)' : 'rgba(187, 128, 9, 0.15)',
            border: toast.type === 'success' ? '1px solid #238636' : '1px solid #bb8009',
            color: toast.type === 'success' ? '#3fb950' : '#d29922',
            padding: '12px 16px',
            borderRadius: '6px',
            fontSize: '13px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: '12px',
          }}
        >
          <span>{toast.message}</span>
          <button
            type="button"
            onClick={() => setToast(null)}
            style={{
              background: 'transparent',
              color: 'inherit',
              border: 'none',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: 'bold',
            }}
          >
            ×
          </button>
        </div>
      )}

      {agents.length === 0 ? (
        emptyState
      ) : (
        agents.map((agent) => {
          const statusColor = agent.is_running ? '#3fb950' : '#8b949e';
          const manualStreaming = Boolean(streamingAgents[agent.agent_id]);
          return (
            <Card key={agent.agent_id} style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontWeight: 'bold', fontSize: '16px' }}>
                    {agent.symbol} · {agent.agent_type}
                  </span>
                  <span style={{ color: '#8b949e', fontSize: '12px' }}>{agent.agent_id}</span>
                </div>
                <span style={{ color: '#ffffff', background: statusColor, padding: '4px 12px', borderRadius: '999px', fontSize: '12px' }}>
                  {agent.is_running ? 'Active' : 'Stopped'}
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '12px' }}>
                <Metric label="Portfolio Value" value={formatCurrency(agent.portfolio_value)} />
                <Metric label="Total PnL" value={`${agent.total_pnl >= 0 ? '+' : ''}${formatCurrency(agent.total_pnl)}`} />
                <Metric label="PnL %" value={formatPercent(agent.total_pnl_pct)} />
                <Metric label="Position" value={`${agent.current_position} @ ${formatCurrency(agent.entry_price)}`} />
                <Metric label="Last Price" value={formatCurrency(agent.current_price)} />
                <Metric label="Last Action" value={agent.last_action || '-'} />
                <Metric label="Last Check" value={agent.last_run_at ? new Date(agent.last_run_at).toLocaleString() : 'Unknown'} />
              </div>

              {agent.last_error && (
                <div style={{
                  background: 'rgba(248, 81, 73, 0.1)',
                  border: '1px solid rgba(248, 81, 73, 0.4)',
                  color: '#f85149',
                  padding: '10px 12px',
                  borderRadius: '6px',
                  fontSize: '12px',
                }}>
                  ⚠️ Last error: {agent.last_error}
                </div>
              )}

              <TradingWindowControls
                agent={agent}
                isUpdating={Boolean(hoursUpdating[agent.agent_id])}
                onToggle={(field, value) => handleToggleTradingHours(agent.agent_id, field, value, agent.symbol)}
              />

              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <Button disabled={pending} onClick={() => handleAction('run', agent.agent_id)}>Run Check</Button>
                {agent.is_running ? (
                  <Button disabled={pending} onClick={() => handleAction('stop', agent.agent_id)} style={{ background: '#f59e0b' }}>
                    Pause
                  </Button>
                ) : (
                  <Button disabled={pending} onClick={() => handleAction('start', agent.agent_id)} style={{ background: '#3fb950' }}>
                    Start Agent
                  </Button>
                )}
                <Button
                  disabled={pending || agent.is_running}
                  onClick={() => handleToggleStreaming(agent.agent_id, agent.symbol)}
                  style={{ background: agent.is_running ? '#3fb950' : manualStreaming ? '#9e6cff' : '#1f6feb' }}
                >
                  {agent.is_running
                    ? 'Live Data Active'
                    : manualStreaming
                      ? 'Stop Live Data'
                      : 'Start Live Data'}
                </Button>
                <Button disabled={pending || agent.current_position === 0} onClick={() => handleAction('close', agent.agent_id)}>
                  Close Position
                </Button>
                <Button disabled={pending} onClick={() => handleAction('remove', agent.agent_id)} style={{ background: '#da3633' }}>
                  Remove Agent
                </Button>
              </div>

              <LiveTickChart
                agentId={agent.agent_id}
                symbol={agent.symbol}
                isRunning={agent.is_running}
                forceStreaming={manualStreaming}
              />
            </Card>
          );
        })
      )}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', background: '#161b22', padding: '12px', borderRadius: '6px' }}>
      <span style={{ color: '#8b949e', fontSize: '12px' }}>{label}</span>
      <span style={{ color: '#c9d1d9', fontWeight: 'bold' }}>{value}</span>
    </div>
  );
}

function TradingWindowControls({ agent, isUpdating, onToggle }) {
  if (!agent) {
    return null;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', background: '#161b22', padding: '12px', borderRadius: '6px' }}>
      <span style={{ color: '#8b949e', fontSize: '12px', letterSpacing: '0.05em', textTransform: 'uppercase' }}>Trading Session</span>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#c9d1d9', fontSize: '13px', cursor: isUpdating ? 'not-allowed' : 'pointer' }}>
          <input
            type="checkbox"
            checked={Boolean(agent.allow_premarket)}
            disabled={isUpdating}
            onChange={(e) => onToggle('allow_premarket', e.target.checked)}
            style={{ cursor: isUpdating ? 'not-allowed' : 'pointer' }}
          />
          <span>Pre-market (07:00-09:30 ET)</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#c9d1d9', fontSize: '13px', cursor: isUpdating ? 'not-allowed' : 'pointer' }}>
          <input
            type="checkbox"
            checked={Boolean(agent.allow_afterhours)}
            disabled={isUpdating}
            onChange={(e) => onToggle('allow_afterhours', e.target.checked)}
            style={{ cursor: isUpdating ? 'not-allowed' : 'pointer' }}
          />
          <span>After-hours (16:00-20:00 ET)</span>
        </label>
      </div>
      <span style={{ color: '#6e7681', fontSize: '11px' }}>
        Outside the selected windows automatic runs are skipped for intraday agents.
      </span>
    </div>
  );
}

function LiveTickChart({ agentId, symbol, isRunning, forceStreaming }) {
  const [series, setSeries] = useState([]);
  const [updatedAt, setUpdatedAt] = useState(null);
  const [error, setError] = useState(null);
  const isActive = isRunning || forceStreaming;

  const fetchTicks = useCallback(async () => {
    const payload = await liveAPI.fetchTicks(agentId, { duration: 20 });
    const container = Array.isArray(payload?.ticks)
      ? payload.ticks
      : Array.isArray(payload)
        ? payload
        : [];

    const points = [];
    container.forEach((tick) => {
      const rawPrice = tick?.Price ?? tick?.price ?? tick?.lastPrice ?? tick?.close;
      const price = Number(rawPrice);
      if (!Number.isFinite(price)) {
        return;
      }
      const rawTime = tick?.Time ?? tick?.time ?? tick?.timestamp;
      const timestamp = rawTime ? new Date(rawTime) : new Date();
      points.push({ price, time: timestamp });
    });

    const maxPoints = 80;
    return points.slice(-maxPoints);
  }, [agentId]);

  useEffect(() => {
    let timer;
    let active = true;

    const poll = async () => {
      try {
        const points = await fetchTicks();
        if (!active) {
          return;
        }
        setSeries(points);
        setUpdatedAt(new Date());
        setError(null);
      } catch (err) {
        if (!active) {
          return;
        }
        setError(err.message || 'Failed to load ticks');
      }
    };

    if (!isActive) {
      setSeries([]);
      setUpdatedAt(null);
      setError(null);
      return undefined;
    }

    poll();
    timer = setInterval(poll, 5000);

    return () => {
      active = false;
      if (timer) {
        clearInterval(timer);
      }
    };
  }, [fetchTicks, isActive]);

  const WIDTH = 320;
  const HEIGHT = 120;
  const PADDING = 12;

  const chartData = useMemo(() => {
    if (series.length < 2) {
      return { path: '', min: null, max: null };
    }

    const prices = series.map((point) => point.price);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const range = maxPrice - minPrice || Math.max(maxPrice * 0.001, 0.01);

    const commands = series
      .map((point, index) => {
        const x = (index / (series.length - 1)) * (WIDTH - PADDING * 2) + PADDING;
        const normalized = (point.price - minPrice) / range;
        const y = HEIGHT - PADDING - normalized * (HEIGHT - PADDING * 2);
        const op = index === 0 ? 'M' : 'L';
        return `${op}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(' ');

    return { path: commands, min: minPrice, max: maxPrice };
  }, [series]);

  const lastPoint = series.length ? series[series.length - 1] : null;

  return (
    <div style={{ background: '#161b22', borderRadius: '6px', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ color: '#8b949e', fontSize: '12px', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
          {symbol} Live Ticks
        </span>
        {updatedAt && isActive && (
          <span style={{ color: '#6e7681', fontSize: '11px' }}>Updated {updatedAt.toLocaleTimeString()}</span>
        )}
      </div>

      {!isActive && (
        <div style={{ color: '#6e7681', fontSize: '12px' }}>Start the agent or live data streaming to view ticks.</div>
      )}

      {error && (
        <div style={{ color: '#f85149', fontSize: '12px' }}>{error}</div>
      )}

      {chartData.path && isActive && !error && (
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} width="100%" height="120" style={{ background: 'transparent' }}>
          <rect x="0" y="0" width={WIDTH} height={HEIGHT} fill="rgba(88, 166, 255, 0.06)" rx="8" ry="8" />
          <path d={chartData.path} fill="none" stroke="#58a6ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}

      {(!chartData.path || series.length < 2) && isActive && !error && (
        <div style={{ color: '#6e7681', fontSize: '12px' }}>Collecting ticks...</div>
      )}

      {lastPoint && (
        <div style={{ display: 'flex', gap: '12px', color: '#c9d1d9', fontSize: '13px' }}>
          <span><strong>Last:</strong> ${lastPoint.price.toFixed(2)}</span>
          {chartData.min !== null && chartData.max !== null && (
            <span>
              <strong>Range:</strong> ${chartData.min.toFixed(2)} — ${chartData.max.toFixed(2)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default TabLiveTrading;
