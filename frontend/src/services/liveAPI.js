// Live Trading API Client
// Provides typed wrappers around the backend live trading endpoints.

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });

  const contentType = response.headers.get('content-type');
  let payload = null;
  if (contentType && contentType.includes('application/json')) {
    payload = await response.json();
  } else {
    payload = await response.text();
  }

  if (!response.ok) {
    const errorMessage = payload?.error || payload || response.statusText;
    throw new Error(errorMessage);
  }

  return payload;
}

const liveAPI = {
  async listAgents() {
    return request('/api/live/agents');
  },

  async createAgent(body) {
    return request('/api/live/agents', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },

  async startAgent(agentId) {
    return request(`/api/live/agents/${agentId}/start`, { method: 'POST' });
  },

  async stopAgent(agentId) {
    return request(`/api/live/agents/${agentId}/stop`, { method: 'POST' });
  },

  async runAgent(agentId) {
    return request(`/api/live/agents/${agentId}/run`, { method: 'POST' });
  },

  async closePosition(agentId) {
    return request(`/api/live/agents/${agentId}/position/close`, { method: 'POST' });
  },

  async removeAgent(agentId) {
    return request(`/api/live/agents/${agentId}`, { method: 'DELETE' });
  },

  async emergencyStop() {
    return request('/api/live/emergency-stop', { method: 'POST' });
  },

  async updateTradingHours(agentId, payload) {
    return request(`/api/live/agents/${agentId}/trading-hours`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    });
  },

  async fetchTicks(agentId, { duration = 20, secType = 'STK', exchange = 'SMART' } = {}) {
    const params = new URLSearchParams({
      duration: String(duration),
      secType,
      exchange,
    });
    return request(`/api/live/agents/${agentId}/ticks?${params.toString()}`);
  },

  async fetchCandles(
    agentId,
    {
      durationDays = 3,
      barSize = '15 mins',
      secType = 'STK',
      exchange = 'SMART',
      limit = 160,
    } = {},
  ) {
    const params = new URLSearchParams({
      durationDays: String(durationDays),
      barSize,
      secType,
      exchange,
      limit: String(limit),
    });
    return request(`/api/live/agents/${agentId}/candles?${params.toString()}`);
  },
};

export default liveAPI;
