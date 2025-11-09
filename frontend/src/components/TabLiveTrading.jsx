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

  const fetchAgents = useCallback(async () => {
    try {
      setError(null);
      const result = await liveAPI.listAgents();
      setAgents(result.agents || []);
      return result.agents || [];
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

              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <Button disabled={pending} onClick={() => handleAction('run', agent.agent_id)}>Run Check</Button>
                {agent.is_running ? (
                  <Button disabled={pending} onClick={() => handleAction('stop', agent.agent_id)} style={{ background: '#f59e0b' }}>
                    Pause
                  </Button>
                ) : (
                  <Button disabled={pending} onClick={() => handleAction('start', agent.agent_id)} style={{ background: '#3fb950' }}>
                    Start
                  </Button>
                )}
                <Button disabled={pending || agent.current_position === 0} onClick={() => handleAction('close', agent.agent_id)}>
                  Close Position
                </Button>
                <Button disabled={pending} onClick={() => handleAction('remove', agent.agent_id)} style={{ background: '#da3633' }}>
                  Remove Agent
                </Button>
              </div>
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

export default TabLiveTrading;
