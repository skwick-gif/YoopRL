/**
 * ModelSelector.jsx
 * Trained Model Selection Component
 * 
 * Purpose:
 * - Dropdown selector for previously trained models
 * - Displays model metadata: version, date, performance metrics
 * - Allows loading specific model for further training or backtesting
 * 
 * Why separate file:
 * - Reusable in multiple contexts (training, backtesting, live trading)
 * - Isolated model selection logic from parent components
 * - Easier to add features like filtering, sorting, searching
 * 
 * State:
 * - models: Array of available models from API
 * - selectedModel: Currently selected model object
 * - loading: API call in progress
 * 
 * Props:
 * - onModelSelect(model): Callback when user selects a model
 * - agentType: Filter models by 'PPO' or 'SAC' (optional)
 * 
 * Wiring:
 * - Fetches models on mount via trainingAPI.loadModels()
 * - Calls parent's onModelSelect() when user picks a model
 * - Parent can use selected model for training or backtesting
 */

import React, { useState, useEffect } from 'react';
import { loadModels } from '../../services/trainingAPI';

const normalizeAgentType = (value) => {
  if (!value) {
    return value;
  }

  const upper = value.toUpperCase();
  return upper === 'SAC_INTRADAY_DSR' ? 'SAC' : upper;
};

const requireString = (value, message) => {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(message);
  }
  return value.trim();
};

const requireFiniteNumber = (value, message) => {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    throw new Error(message);
  }
  return num;
};

const formatDecimal = (value, digits = 2, message = 'ModelSelector: expected numeric value') => {
  const num = requireFiniteNumber(value, message);
  return num.toFixed(digits);
};

const formatPercent = (value, digits = 2, { showPlus = false, message = 'ModelSelector: expected numeric percent' } = {}) => {
  const num = requireFiniteNumber(value, message);
  const prefix = showPlus && num >= 0 ? '+' : '';
  return `${prefix}${num.toFixed(digits)}%`;
};

const formatInteger = (value, message = 'ModelSelector: expected integer value') => {
  const num = requireFiniteNumber(value, message);
  return Math.round(num).toLocaleString();
};

const formatDateTime = (dateString, message = 'ModelSelector: expected valid datetime') => {
  const dateValue = requireString(dateString, message);
  const date = new Date(dateValue);
  if (Number.isNaN(date.getTime())) {
    throw new Error(message);
  }

  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

const ModelSelector = ({ onModelSelect, agentType }) => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const normalizedAgentType = normalizeAgentType(agentType);

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      setLoading(true);
      setError(null);

      const result = await loadModels();
      if (result.success) {
        // Filter by agent type if specified
        const filteredModels = normalizedAgentType
          ? result.models.filter(model => (model.agent_type || '').toUpperCase() === normalizedAgentType)
          : result.models;

        setModels(filteredModels);
      } else {
        setError(result.error);
      }

      setLoading(false);
    };

    fetchModels();
  }, [normalizedAgentType]);

  // Handle model selection
  const handleSelectChange = (event) => {
    const modelId = event.target.value;
    const model = models.find(m => m.model_id === modelId);
    
    setSelectedModel(model);
    
    // Notify parent component
    if (onModelSelect) {
      onModelSelect(model);
    }
  };

  // Format date for display
  return (
    <div style={styles.container}>
      <label style={styles.label}>Select Trained Model:</label>
      
      {loading && <p style={styles.loading}>Loading models...</p>}
      
      {error && <p style={styles.error}>Error: {error}</p>}
      
      {!loading && !error && models.length === 0 && (
        <p style={styles.noModels}>No trained models available. Train a model first.</p>
      )}
      
      {!loading && !error && models.length > 0 && (
        <select
          value={selectedModel?.model_id || ''}
          onChange={handleSelectChange}
          style={styles.dropdown}
        >
          <option value="">-- Select Model --</option>
          {models.map(model => {
            const optionId = requireString(model.model_id, 'ModelSelector: missing model_id');
            const optionAgent = requireString(model.agent_type, 'ModelSelector: missing agent_type');
            const optionSymbol = requireString(model.symbol, 'ModelSelector: missing symbol');
            const optionVersion = requireString(model.version, 'ModelSelector: missing version');
            const optionSharpe = formatDecimal(model.sharpe_ratio, 2, 'ModelSelector: missing Sharpe ratio');
            const createdSource = model.created_at ?? model.created;
            const optionDate = formatDateTime(createdSource, 'ModelSelector: missing created date');

            return (
              <option key={optionId} value={optionId}>
                {`${optionAgent} - ${optionSymbol} - ${optionVersion} - Sharpe: ${optionSharpe} - ${optionDate}`}
              </option>
            );
          })}
        </select>
      )}
      
      {selectedModel && (
        <div style={styles.details}>
          <h4 style={styles.detailsTitle}>Model Details:</h4>
          <div style={styles.detailsGrid}>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Agent Type:</span>
              <span style={styles.detailValue}>{requireString(selectedModel.agent_type, 'ModelSelector: missing agent_type')}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Symbol:</span>
              <span style={styles.detailValue}>{requireString(selectedModel.symbol, 'ModelSelector: missing symbol')}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Version:</span>
              <span style={styles.detailValue}>{requireString(selectedModel.version, 'ModelSelector: missing version')}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Episodes:</span>
              <span style={styles.detailValue}>{formatInteger(selectedModel.episodes, 'ModelSelector: missing episodes')}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Sharpe Ratio:</span>
              <span style={styles.detailValue}>{formatDecimal(selectedModel.sharpe_ratio, 2, 'ModelSelector: missing Sharpe ratio')}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Total Return:</span>
              <span style={styles.detailValue}>{formatPercent(selectedModel.total_return, 2, { showPlus: true, message: 'ModelSelector: missing total return' })}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Created:</span>
              <span style={styles.detailValue}>{formatDateTime(selectedModel.created_at ?? selectedModel.created, 'ModelSelector: missing created date')}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>File Path:</span>
              <span style={styles.detailValueMono}>{requireString(selectedModel.file_path, 'ModelSelector: missing file path')}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Inline styles
const styles = {
  container: {
    padding: '15px',
    backgroundColor: '#1e1e1e',
    borderRadius: '8px',
    marginBottom: '20px'
  },
  label: {
    display: 'block',
    color: '#e0e0e0',
    fontWeight: 'bold',
    marginBottom: '10px',
    fontSize: '14px'
  },
  dropdown: {
    width: '100%',
    padding: '10px',
    backgroundColor: '#2d2d2d',
    color: '#e0e0e0',
    border: '1px solid #444',
    borderRadius: '4px',
    fontSize: '14px',
    cursor: 'pointer'
  },
  loading: {
    color: '#888',
    fontStyle: 'italic'
  },
  error: {
    color: '#ff6b6b',
    fontWeight: 'bold'
  },
  noModels: {
    color: '#888',
    fontStyle: 'italic'
  },
  details: {
    marginTop: '20px',
    padding: '15px',
    backgroundColor: '#2d2d2d',
    borderRadius: '6px',
    border: '1px solid #444'
  },
  detailsTitle: {
    color: '#4CAF50',
    marginTop: 0,
    marginBottom: '15px',
    fontSize: '16px'
  },
  detailsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
    gap: '12px'
  },
  detailItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px'
  },
  detailLabel: {
    color: '#888',
    fontSize: '12px',
    fontWeight: 'bold'
  },
  detailValue: {
    color: '#e0e0e0',
    fontSize: '14px'
  },
  detailValueMono: {
    color: '#e0e0e0',
    fontSize: '12px',
    fontFamily: 'monospace',
    wordBreak: 'break-all'
  }
};

export default ModelSelector;
