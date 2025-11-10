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
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

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
          {models.map(model => (
            <option key={model.model_id} value={model.model_id}>
              {`${model.agent_type} - ${model.symbol} - ${model.version} - Sharpe: ${model.sharpe_ratio.toFixed(2)} - ${formatDate(model.created_at)}`}
            </option>
          ))}
        </select>
      )}
      
      {selectedModel && (
        <div style={styles.details}>
          <h4 style={styles.detailsTitle}>Model Details:</h4>
          <div style={styles.detailsGrid}>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Agent Type:</span>
              <span style={styles.detailValue}>{selectedModel.agent_type}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Symbol:</span>
              <span style={styles.detailValue}>{selectedModel.symbol}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Version:</span>
              <span style={styles.detailValue}>{selectedModel.version}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Episodes:</span>
              <span style={styles.detailValue}>{selectedModel.episodes.toLocaleString()}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Sharpe Ratio:</span>
              <span style={styles.detailValue}>{selectedModel.sharpe_ratio.toFixed(2)}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Total Return:</span>
              <span style={styles.detailValue}>{selectedModel.total_return.toFixed(2)}%</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>Created:</span>
              <span style={styles.detailValue}>{formatDate(selectedModel.created_at)}</span>
            </div>
            <div style={styles.detailItem}>
              <span style={styles.detailLabel}>File Path:</span>
              <span style={styles.detailValueMono}>{selectedModel.file_path}</span>
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
