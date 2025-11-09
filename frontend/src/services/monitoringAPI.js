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

const monitoringAPI = {
  getSummary(params = {}) {
    const search = new URLSearchParams(params);
    return request(`/api/monitoring/summary${search.toString() ? `?${search}` : ''}`);
  },

  getActions(params = {}) {
    const search = new URLSearchParams(params);
    return request(`/api/monitoring/actions${search.toString() ? `?${search}` : ''}`);
  },

  getAlerts(params = {}) {
    const search = new URLSearchParams(params);
    return request(`/api/monitoring/alerts${search.toString() ? `?${search}` : ''}`);
  },

  getSystem(params = {}) {
    const search = new URLSearchParams(params);
    return request(`/api/monitoring/system${search.toString() ? `?${search}` : ''}`);
  },
};

export default monitoringAPI;
