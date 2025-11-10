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

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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

const TIMEFRAMES_MINUTES = [1, 5, 10, 15];
const BAR_SIZE_LABEL = {
  1: '1 min',
  5: '5 mins',
  10: '10 mins',
  15: '15 mins',
};
const MAX_TIMEFRAME_MINUTES = 15;
const MAX_CANDLES = 200;

function LiveTickChart({ agentId, symbol, isRunning, forceStreaming }) {
  const [timeframe, setTimeframe] = useState(15);
  const [historicalCandles, setHistoricalCandles] = useState([]);
  const [candles, setCandles] = useState([]);
  const [ticks, setTicks] = useState([]);
  const [updatedAt, setUpdatedAt] = useState(null);
  const [error, setError] = useState(null);
  const [tickError, setTickError] = useState(null);
  const [candleWarning, setCandleWarning] = useState(null);
  const [tickWarning, setTickWarning] = useState(null);
  const [tickLatencyMs, setTickLatencyMs] = useState(null);
  const isActive = isRunning || forceStreaming;

  const newestTickRef = useRef(0);
  const pollingRef = useRef(false);

  const parseHistorical = useCallback((bars, minutes) => {
    const bucketMs = minutes * 60 * 1000;
    const parsed = [];
    bars.forEach((bar) => {
      if (!bar) {
        return;
      }
      const open = Number(bar.open ?? bar.Open);
      const high = Number(bar.high ?? bar.High);
      const low = Number(bar.low ?? bar.Low);
      const close = Number(bar.close ?? bar.Close);
      if (![open, high, low, close].every(Number.isFinite)) {
        return;
      }
      const rawTime = bar.time ?? bar.Time ?? bar.timestamp ?? bar.Timestamp;
      const ts = rawTime ? new Date(rawTime) : null;
      if (!ts || Number.isNaN(ts.getTime())) {
        return;
      }
      const bucket = Math.floor(ts.getTime() / bucketMs) * bucketMs;
      const volumeRaw = Number(bar.volume ?? bar.Volume);
      parsed.push({
        time: new Date(bucket),
        open,
        high,
        low,
        close,
        volume: Number.isFinite(volumeRaw) ? volumeRaw : null,
      });
    });
    return parsed.slice(-MAX_CANDLES);
  }, []);

  const loadHistorical = useCallback(async (minutes) => {
    try {
      const payload = await liveAPI.fetchCandles(agentId, {
        durationDays: minutes <= 1 ? 1 : 3,
        barSize: BAR_SIZE_LABEL[minutes] || '15 mins',
        limit: MAX_CANDLES,
      });
      const bars = Array.isArray(payload?.bars)
        ? payload.bars
        : Array.isArray(payload)
          ? payload
          : [];
      const parsed = parseHistorical(bars, minutes);
      setHistoricalCandles(parsed);
      setError(null);
      if (payload?.status === 'warning') {
        const note = buildBridgeWarning('candles', payload);
        setCandleWarning(note);
      } else {
        setCandleWarning(null);
      }
    } catch (err) {
      setError(err.message || 'Failed to load historical candles');
      setHistoricalCandles([]);
      setCandleWarning(null);
    }
  }, [agentId, parseHistorical]);

  const aggregateTicks = useCallback((tickList, minutes) => {
    if (tickList.length === 0) {
      return [];
    }
    const bucketMs = minutes * 60 * 1000;
    const buckets = new Map();

    tickList.forEach((tick) => {
      const ms = tick.time.getTime();
      const bucket = Math.floor(ms / bucketMs) * bucketMs;
      const existing = buckets.get(bucket);
      if (!existing) {
        buckets.set(bucket, {
          time: new Date(bucket),
          open: tick.price,
          high: tick.price,
          low: tick.price,
          close: tick.price,
          volume: tick.volume ?? null,
        });
      } else {
        existing.high = Math.max(existing.high, tick.price);
        existing.low = Math.min(existing.low, tick.price);
        existing.close = tick.price;
        if (typeof tick.volume === 'number') {
          existing.volume = (existing.volume ?? 0) + tick.volume;
        }
      }
    });

    return Array.from(buckets.values()).sort((a, b) => a.time.getTime() - b.time.getTime());
  }, []);

  const mergeCandles = useCallback((historical, liveCandles, minutes) => {
    const bucketMs = minutes * 60 * 1000;
    const map = new Map();

    historical.forEach((bar) => {
      const bucket = Math.floor(bar.time.getTime() / bucketMs) * bucketMs;
      if (!map.has(bucket)) {
        map.set(bucket, { ...bar, time: new Date(bucket) });
      }
    });

    liveCandles.forEach((bar) => {
      const bucket = Math.floor(bar.time.getTime() / bucketMs) * bucketMs;
      map.set(bucket, { ...bar, time: new Date(bucket) });
    });

    return Array.from(map.values())
      .sort((a, b) => a.time.getTime() - b.time.getTime())
      .slice(-MAX_CANDLES);
  }, []);

  useEffect(() => {
    if (!isActive) {
      setHistoricalCandles([]);
      setCandles([]);
      setTicks([]);
      setUpdatedAt(null);
      newestTickRef.current = 0;
      return undefined;
    }

    loadHistorical(timeframe);
    return undefined;
  }, [isActive, timeframe, loadHistorical]);

  useEffect(() => {
    if (!isActive) {
      return undefined;
    }

    let cancelled = false;

    const poll = async () => {
      if (pollingRef.current || cancelled) {
        return;
      }
      pollingRef.current = true;
      try {
        const payload = await liveAPI.fetchTicks(agentId, { duration: 5 });
        const container = Array.isArray(payload?.ticks)
          ? payload.ticks
          : Array.isArray(payload)
            ? payload
            : [];

        const fresh = [];
        let newest = newestTickRef.current;
        container.forEach((tick) => {
          const price = Number(tick?.Price ?? tick?.price ?? tick?.lastPrice ?? tick?.close);
          if (!Number.isFinite(price)) {
            return;
          }
          const rawTime = tick?.Time ?? tick?.time ?? tick?.timestamp;
          const ts = rawTime ? new Date(rawTime) : null;
          if (!ts || Number.isNaN(ts.getTime())) {
            return;
          }
          const millis = ts.getTime();
          if (millis <= newest) {
            return;
          }
          const volume = Number(tick?.size ?? tick?.Size ?? tick?.volume ?? tick?.Volume);
          fresh.push({
            time: ts,
            price,
            volume: Number.isFinite(volume) ? volume : null,
          });
          newest = Math.max(newest, millis);
        });

        if (fresh.length) {
          newestTickRef.current = newest;
          const maxWindow = MAX_TIMEFRAME_MINUTES * 60 * 1000;
          setTicks((prev) => {
            const merged = [...prev, ...fresh];
            const cutoff = newest - maxWindow;
            return merged
              .filter((tick) => tick.time.getTime() >= cutoff)
              .sort((a, b) => a.time.getTime() - b.time.getTime());
          });
        }
        setTickError(null);
        if (payload?.status === 'warning') {
          const note = buildBridgeWarning('ticks', payload);
          setTickWarning(note);
        } else {
          setTickWarning(null);
        }
        if (payload?.latency_ms) {
          setTickLatencyMs(Number(payload.latency_ms) || null);
        } else if (payload?.cached && payload?.cached_at) {
          const cachedTs = new Date(payload.cached_at).getTime();
          if (!Number.isNaN(cachedTs)) {
            setTickLatencyMs(Date.now() - cachedTs);
          }
        } else {
          setTickLatencyMs(null);
        }
      } catch (err) {
        setTickError(err.message || 'Failed to stream live ticks');
        setTickWarning(null);
        setTickLatencyMs(null);
      } finally {
        pollingRef.current = false;
      }
    };

    poll();
    const interval = setInterval(poll, 6000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [agentId, isActive]);

  useEffect(() => {
    if (!isActive) {
      return;
    }
    const liveCandles = aggregateTicks(ticks, timeframe);
    const combined = mergeCandles(historicalCandles, liveCandles, timeframe);
    setCandles(combined);

    if (liveCandles.length) {
      setUpdatedAt(liveCandles[liveCandles.length - 1].time);
    } else if (combined.length) {
      setUpdatedAt(combined[combined.length - 1].time);
    }
  }, [ticks, historicalCandles, timeframe, isActive, aggregateTicks, mergeCandles]);

  const CHART_WIDTH = 360;
  const CHART_HEIGHT = 180;
  const LEFT_AXIS = 52;
  const RIGHT_PADDING = 12;
  const TOP_PADDING = 12;
  const BOTTOM_AXIS = 28;

  const metrics = useMemo(() => {
    if (candles.length === 0) {
      return null;
    }
    const highs = candles.map((bar) => bar.high);
    const lows = candles.map((bar) => bar.low);
    const maxHigh = Math.max(...highs);
    const minLow = Math.min(...lows);
    const rawRange = maxHigh - minLow;
    const range = rawRange > 0 ? rawRange : Math.max(Math.abs(maxHigh) * 0.001, 0.01);
    const plotWidth = CHART_WIDTH - LEFT_AXIS - RIGHT_PADDING;
    const plotHeight = CHART_HEIGHT - TOP_PADDING - BOTTOM_AXIS;
    const segments = Math.max(1, candles.length - 1);
    const spacing = candles.length === 1 ? 0 : plotWidth / segments;
    const bodyWidth = candles.length === 1
      ? Math.min(22, Math.max(6, plotWidth * 0.3))
      : Math.min(18, Math.max(4, spacing * 0.6));
    const priceTickCount = 4;
    const priceTicks = Array.from({ length: priceTickCount + 1 }, (_, idx) => {
      const value = maxHigh - (range * idx) / priceTickCount;
      const y = TOP_PADDING + (plotHeight * idx) / priceTickCount;
      return { value, y };
    });
    const timeTickCount = Math.min(5, candles.length);
    const timeTickIndices = [];
    if (timeTickCount === 1) {
      timeTickIndices.push(0);
    } else {
      const step = Math.max(1, Math.floor((candles.length - 1) / (timeTickCount - 1)));
      for (let i = 0; i < candles.length; i += step) {
        timeTickIndices.push(i);
      }
      const lastIndex = candles.length - 1;
      if (timeTickIndices[timeTickIndices.length - 1] !== lastIndex) {
        timeTickIndices.push(lastIndex);
      }
    }
    const timeTicks = timeTickIndices.map((idx, position) => {
      const bar = candles[idx];
      const label = position === 0
        ? bar.time.toLocaleString([], { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
        : bar.time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      return { index: idx, label };
    });
    return {
      maxHigh,
      minLow,
      range,
      spacing,
      bodyWidth,
      plotWidth,
      plotHeight,
      priceTicks,
      timeTicks,
      count: candles.length,
    };
  }, [candles]);

  const priceToY = useCallback((price) => {
    if (!metrics) {
      return CHART_HEIGHT / 2;
    }
    const normalized = (metrics.maxHigh - price) / metrics.range;
    const clamped = Math.max(0, Math.min(1, normalized));
    return TOP_PADDING + clamped * metrics.plotHeight;
  }, [metrics]);

  const xForIndex = useCallback((index) => {
    if (!metrics) {
      return LEFT_AXIS;
    }
    if (metrics.count === 1) {
      return LEFT_AXIS + metrics.plotWidth / 2;
    }
    return LEFT_AXIS + index * metrics.spacing;
  }, [metrics]);

  const lastCandle = candles.length ? candles[candles.length - 1] : null;
  const combinedError = error || tickError;

  return (
    <div style={{ background: '#161b22', borderRadius: '6px', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ color: '#8b949e', fontSize: '12px', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
            {symbol} · {timeframe}m Candles
          </span>
          {updatedAt && isActive && (
            <span style={{ color: '#6e7681', fontSize: '11px' }}>Updated {updatedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
          )}
        </div>
        <div style={{ display: 'flex', gap: '6px' }}>
          {TIMEFRAMES_MINUTES.map((value) => {
            const active = value === timeframe;
            return (
              <button
                key={value}
                type="button"
                onClick={() => setTimeframe(value)}
                style={{
                  padding: '4px 10px',
                  borderRadius: '4px',
                  border: active ? '1px solid #58a6ff' : '1px solid #30363d',
                  background: active ? 'rgba(88, 166, 255, 0.15)' : 'transparent',
                  color: active ? '#58a6ff' : '#c9d1d9',
                  fontSize: '12px',
                  cursor: 'pointer',
                }}
              >
                {value}m
              </button>
            );
          })}
        </div>
      </div>

      {!isActive && (
        <div style={{ color: '#6e7681', fontSize: '12px' }}>Start the agent or live data streaming to view candles.</div>
      )}

      {combinedError && (
        <div style={{ color: '#f85149', fontSize: '12px' }}>{combinedError}</div>
      )}

      {!combinedError && (candleWarning || tickWarning || Number.isFinite(tickLatencyMs)) && (
        <div style={{ color: '#d29922', fontSize: '12px', lineHeight: 1.4 }}>
          {candleWarning && <div>{candleWarning}</div>}
          {tickWarning && <div>{tickWarning}</div>}
          {Number.isFinite(tickLatencyMs) && tickLatencyMs > 3000 && (
            <div>⚠️ אחרוני הטיקים התקבלו לפני {(tickLatencyMs / 1000).toFixed(1)} שניות.</div>
          )}
        </div>
      )}

      {metrics && candles.length > 0 && isActive && !combinedError && (
        <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} width="100%" height="180" style={{ background: 'transparent' }}>
          <rect x="0" y="0" width={CHART_WIDTH} height={CHART_HEIGHT} fill="rgba(88, 166, 255, 0.06)" rx="8" ry="8" />
          <line
            x1={LEFT_AXIS}
            x2={LEFT_AXIS}
            y1={TOP_PADDING}
            y2={CHART_HEIGHT - BOTTOM_AXIS}
            stroke="rgba(201, 209, 217, 0.3)"
          />
          <line
            x1={LEFT_AXIS}
            x2={CHART_WIDTH - RIGHT_PADDING}
            y1={CHART_HEIGHT - BOTTOM_AXIS}
            y2={CHART_HEIGHT - BOTTOM_AXIS}
            stroke="rgba(201, 209, 217, 0.3)"
          />
          {metrics.priceTicks.map((tick, idx) => (
            <g key={`price-${idx}`}>
              <line
                x1={LEFT_AXIS}
                x2={CHART_WIDTH - RIGHT_PADDING}
                y1={tick.y}
                y2={tick.y}
                stroke="rgba(201, 209, 217, 0.1)"
              />
              <text
                x={LEFT_AXIS - 6}
                y={tick.y + 4}
                fontSize="10"
                fill="#8b949e"
                textAnchor="end"
              >
                {tick.value.toFixed(2)}
              </text>
            </g>
          ))}
          {metrics.timeTicks.map((tick, idx) => {
            const x = xForIndex(tick.index);
            return (
              <g key={`time-${tick.index}`}>
                <line
                  x1={x}
                  x2={x}
                  y1={CHART_HEIGHT - BOTTOM_AXIS}
                  y2={CHART_HEIGHT - BOTTOM_AXIS + 6}
                  stroke="rgba(201, 209, 217, 0.3)"
                />
                <text
                  x={x}
                  y={CHART_HEIGHT - 6}
                  fontSize="10"
                  fill="#8b949e"
                  textAnchor="middle"
                >
                  {tick.label}
                </text>
              </g>
            );
          })}
          {candles.map((bar, index) => {
            const xCenter = xForIndex(index);
            const bodyTop = priceToY(Math.max(bar.open, bar.close));
            const bodyBottom = priceToY(Math.min(bar.open, bar.close));
            const bodyHeight = Math.max(1, bodyBottom - bodyTop);
            const bodyX = xCenter - metrics.bodyWidth / 2;
            const color = bar.close >= bar.open ? '#3fb950' : '#f85149';

            return (
              <rect
                key={`${bar.time.toISOString()}_${index}`}
                x={bodyX}
                y={bodyTop}
                width={metrics.bodyWidth}
                height={bodyHeight}
                fill={color}
                stroke={color}
                rx="2"
              />
            );
          })}
        </svg>
      )}

      {(!metrics || candles.length === 0) && isActive && !combinedError && (
        <div style={{ color: '#6e7681', fontSize: '12px' }}>Collecting live candles...</div>
      )}

      {lastCandle && (
        <div style={{ display: 'flex', gap: '12px', color: '#c9d1d9', fontSize: '13px', flexWrap: 'wrap' }}>
          <span><strong>Open:</strong> ${lastCandle.open.toFixed(2)}</span>
          <span><strong>High:</strong> ${lastCandle.high.toFixed(2)}</span>
          <span><strong>Low:</strong> ${lastCandle.low.toFixed(2)}</span>
          <span><strong>Close:</strong> ${lastCandle.close.toFixed(2)}</span>
          <span><strong>Time:</strong> {lastCandle.time.toLocaleString()}</span>
        </div>
      )}
    </div>
  );
}

function buildBridgeWarning(source, payload) {
  const message = payload?.message || 'Bridge returned a warning';
  let cachedStr = null;
  if (payload?.cached) {
    if (payload?.cached_at) {
      const parsed = new Date(payload.cached_at);
      cachedStr = Number.isNaN(parsed?.getTime()) ? 'recent cached data' : parsed.toLocaleString();
    } else {
      cachedStr = 'recent cached data';
    }
  }
  if (cachedStr) {
    return source === 'candles'
      ? `⚠️ ${message}. Using cached candles from ${cachedStr}.`
      : `⚠️ ${message}. Using cached ticks from ${cachedStr}.`;
  }
  return source === 'candles'
    ? `⚠️ ${message}. No cached candles available yet.`
    : `⚠️ ${message}. Waiting on fresh ticks…`;
}

export default TabLiveTrading;
