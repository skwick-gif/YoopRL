/**
 * ModelsComparisonTable.jsx
 * Models Comparison Table Component
 * 
 * Purpose:
 * - Display all trained models in a sortable table
 * - Compare performance metrics across multiple training runs
 * - Show which features were used in each training
 * - Filter by symbol, date range, or agent type
 * - Highlight best performing models
 * 
 * Props:
 * - agentType: Filter by 'PPO' or 'SAC' (optional)
 * - symbol: Filter by specific symbol (optional)
 * 
 * Features Displayed:
 * - Model metadata (version, date, symbol)
 * - Performance metrics (Sharpe, Return, Drawdown, Win Rate)
 * - Training features used (OHLCV, Technical, Fundamentals, Macro, etc.)
 * - Training duration and episodes
 * 
 * Interactions:
 * - Click column header to sort
 * - Click row to see full details
 * - Filter by symbol or date
 * - Export to CSV
 */

import React, { useState, useEffect } from 'react';
import { loadModels } from '../../services/trainingAPI';

const ModelsComparisonTable = ({ agentType, symbol: filterSymbol }) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortBy, setSortBy] = useState('created'); // Default sort by date
  const [sortDirection, setSortDirection] = useState('desc');
  const [selectedModel, setSelectedModel] = useState(null);
  const [symbolFilter, setSymbolFilter] = useState(filterSymbol || '');

  // Fetch models on mount
  useEffect(() => {
    fetchModels();
  }, [agentType]);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);

    const result = await loadModels();
    if (result.success) {
      let filteredModels = result.models || [];
      
      // Filter by agent type if specified
      if (agentType) {
        filteredModels = filteredModels.filter(m => m.agent_type === agentType);
      }

      setModels(filteredModels);
    } else {
      setError(result.error || 'Failed to load models');
    }

    setLoading(false);
  };

  // Sort handler
  const handleSort = (field) => {
    if (sortBy === field) {
      // Toggle direction
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortDirection('desc'); // Default to descending for new field
    }
  };

  // Get sorted and filtered models
  const getSortedModels = () => {
    let filtered = [...models];

    // Apply symbol filter
    if (symbolFilter) {
      filtered = filtered.filter(m => 
        m.symbol.toLowerCase().includes(symbolFilter.toLowerCase())
      );
    }

    // Sort
    filtered.sort((a, b) => {
      let aVal = a[sortBy];
      let bVal = b[sortBy];

      // Handle nested properties (e.g., metrics.sharpe_ratio)
      if (sortBy === 'sharpe_ratio') {
        aVal = a.sharpe_ratio || 0;
        bVal = b.sharpe_ratio || 0;
      } else if (sortBy === 'total_return') {
        aVal = a.total_return || 0;
        bVal = b.total_return || 0;
      } else if (sortBy === 'max_drawdown') {
        aVal = a.max_drawdown || 0;
        bVal = b.max_drawdown || 0;
      } else if (sortBy === 'win_rate') {
        aVal = a.win_rate || 0;
        bVal = b.win_rate || 0;
      } else if (sortBy === 'created') {
        aVal = new Date(a.created || a.created_at);
        bVal = new Date(b.created || b.created_at);
      }

      if (sortDirection === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });

    return filtered;
  };

  // Get features summary from model metadata
  const getFeaturesSummary = (model) => {
    const features = model.features || {};
    const featuresList = [];

    if (features.ohlcv) featuresList.push('OHLCV');
    if (features.technical) featuresList.push('Technical');
    if (features.fundamentals) featuresList.push('Fundamentals');
    if (features.macro) featuresList.push('Macro');
    if (features.sentiment) featuresList.push('Sentiment');
    if (features.events) featuresList.push('Events');
    if (features.multi_asset) featuresList.push('Multi-Asset');

    return featuresList.length > 0 ? featuresList.join(', ') : 'Basic';
  };

  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Get color for metric (green=good, yellow=ok, red=bad)
  const getMetricColor = (metric, value) => {
    if (metric === 'sharpe_ratio') {
      if (value >= 2.0) return '#4CAF50';
      if (value >= 1.5) return '#8BC34A';
      if (value >= 1.0) return '#FFC107';
      return '#ff6b6b';
    } else if (metric === 'total_return') {
      if (value >= 20) return '#4CAF50';
      if (value >= 10) return '#8BC34A';
      if (value >= 5) return '#FFC107';
      return '#ff6b6b';
    } else if (metric === 'max_drawdown') {
      if (value > -10) return '#4CAF50';
      if (value > -20) return '#8BC34A';
      if (value > -30) return '#FFC107';
      return '#ff6b6b';
    } else if (metric === 'win_rate') {
      if (value >= 60) return '#4CAF50';
      if (value >= 55) return '#8BC34A';
      if (value >= 50) return '#FFC107';
      return '#ff6b6b';
    }
    return '#e0e0e0';
  };

  // Find best model by Sharpe Ratio
  const getBestModelId = () => {
    const sorted = getSortedModels();
    if (sorted.length === 0) return null;
    
    let best = sorted[0];
    sorted.forEach(m => {
      if ((m.sharpe_ratio || 0) > (best.sharpe_ratio || 0)) {
        best = m;
      }
    });
    
    return best.model_id || best.version;
  };

  const sortedModels = getSortedModels();
  const bestModelId = getBestModelId();

  // Export to CSV
  const exportToCSV = () => {
    const headers = ['Symbol', 'Version', 'Date', 'Sharpe', 'Return %', 'Drawdown %', 'Win Rate %', 'Trades', 'Features'];
    const rows = sortedModels.map(m => [
      m.symbol,
      m.version,
      formatDate(m.created || m.created_at),
      (m.sharpe_ratio || 0).toFixed(2),
      (m.total_return || 0).toFixed(2),
      (m.max_drawdown || 0).toFixed(2),
      (m.win_rate || 0).toFixed(2),
      m.total_trades || 0,
      getFeaturesSummary(m)
    ]);

    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `models_comparison_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
  };

  if (loading) {
    return (
      <div style={styles.container}>
        <p style={styles.loading}>Loading models...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.container}>
        <p style={styles.error}>Error: {error}</p>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div style={styles.container}>
        <p style={styles.noData}>No models found. Train a model first.</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      {/* Header with filters and export */}
      <div style={styles.header}>
        <h3 style={styles.title}>
          📊 Models Comparison ({sortedModels.length} {sortedModels.length === 1 ? 'model' : 'models'})
        </h3>
        
        <div style={styles.controls}>
          {/* Symbol filter */}
          <input
            type="text"
            placeholder="Filter by symbol..."
            value={symbolFilter}
            onChange={(e) => setSymbolFilter(e.target.value)}
            style={styles.filterInput}
          />
          
          {/* Refresh button */}
          <button onClick={fetchModels} style={styles.button}>
            🔄 Refresh
          </button>
          
          {/* Export button */}
          <button onClick={exportToCSV} style={styles.button}>
            📥 Export CSV
          </button>
        </div>
      </div>

      {/* Table */}
      <div style={styles.tableWrapper}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th} onClick={() => handleSort('symbol')}>
                Symbol {sortBy === 'symbol' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th style={styles.th} onClick={() => handleSort('version')}>
                Version {sortBy === 'version' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th style={styles.th} onClick={() => handleSort('created')}>
                Date {sortBy === 'created' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th style={styles.th} onClick={() => handleSort('sharpe_ratio')}>
                Sharpe {sortBy === 'sharpe_ratio' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th style={styles.th} onClick={() => handleSort('total_return')}>
                Return % {sortBy === 'total_return' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th style={styles.th} onClick={() => handleSort('max_drawdown')}>
                Drawdown % {sortBy === 'max_drawdown' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th style={styles.th} onClick={() => handleSort('win_rate')}>
                Win Rate % {sortBy === 'win_rate' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th style={styles.th}>
                Trades
              </th>
              <th style={styles.th}>
                Features Used
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedModels.map((model, index) => {
              const isBest = (model.model_id || model.version) === bestModelId;
              const rowStyle = {
                ...styles.tr,
                backgroundColor: isBest ? 'rgba(76, 175, 80, 0.1)' : (index % 2 === 0 ? '#1e1e1e' : '#252525'),
                borderLeft: isBest ? '4px solid #4CAF50' : 'none'
              };

              return (
                <tr 
                  key={model.model_id || model.version} 
                  style={rowStyle}
                  onClick={() => setSelectedModel(selectedModel?.model_id === model.model_id ? null : model)}
                >
                  <td style={styles.td}>
                    {isBest && <span style={styles.bestBadge}>⭐ BEST</span>}
                    <strong>{model.symbol}</strong>
                  </td>
                  <td style={styles.td}>
                    <span style={styles.versionText}>{model.version}</span>
                  </td>
                  <td style={styles.td}>
                    {formatDate(model.created || model.created_at)}
                  </td>
                  <td style={{...styles.td, color: getMetricColor('sharpe_ratio', model.sharpe_ratio || 0), fontWeight: 'bold'}}>
                    {(model.sharpe_ratio || 0).toFixed(2)}
                  </td>
                  <td style={{...styles.td, color: getMetricColor('total_return', model.total_return || 0), fontWeight: 'bold'}}>
                    {(model.total_return || 0) >= 0 ? '+' : ''}{(model.total_return || 0).toFixed(2)}%
                  </td>
                  <td style={{...styles.td, color: getMetricColor('max_drawdown', model.max_drawdown || 0), fontWeight: 'bold'}}>
                    {(model.max_drawdown || 0).toFixed(2)}%
                  </td>
                  <td style={{...styles.td, color: getMetricColor('win_rate', model.win_rate || 0), fontWeight: 'bold'}}>
                    {(model.win_rate || 0).toFixed(1)}%
                  </td>
                  <td style={styles.td}>
                    {model.total_trades || 0}
                  </td>
                  <td style={styles.td}>
                    <span style={styles.featuresText}>{getFeaturesSummary(model)}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Selected model details */}
      {selectedModel && (
        <div style={styles.detailsPanel}>
          <h4 style={styles.detailsTitle}>
            📋 Model Details: {selectedModel.symbol} - {selectedModel.version}
          </h4>
          
          <div style={styles.detailsGrid}>
            <div style={styles.detailSection}>
              <h5 style={styles.sectionTitle}>Performance Metrics</h5>
              <div style={styles.detailItem}>
                <span>Sharpe Ratio:</span>
                <span style={{color: getMetricColor('sharpe_ratio', selectedModel.sharpe_ratio)}}>{(selectedModel.sharpe_ratio || 0).toFixed(2)}</span>
              </div>
              <div style={styles.detailItem}>
                <span>Sortino Ratio:</span>
                <span>{(selectedModel.sortino_ratio || 0).toFixed(2)}</span>
              </div>
              <div style={styles.detailItem}>
                <span>Calmar Ratio:</span>
                <span>{(selectedModel.calmar_ratio || 0).toFixed(2)}</span>
              </div>
              <div style={styles.detailItem}>
                <span>Total Return:</span>
                <span style={{color: getMetricColor('total_return', selectedModel.total_return)}}>{(selectedModel.total_return || 0).toFixed(2)}%</span>
              </div>
              <div style={styles.detailItem}>
                <span>Max Drawdown:</span>
                <span style={{color: getMetricColor('max_drawdown', selectedModel.max_drawdown)}}>{(selectedModel.max_drawdown || 0).toFixed(2)}%</span>
              </div>
              <div style={styles.detailItem}>
                <span>Win Rate:</span>
                <span style={{color: getMetricColor('win_rate', selectedModel.win_rate)}}>{(selectedModel.win_rate || 0).toFixed(1)}%</span>
              </div>
              <div style={styles.detailItem}>
                <span>Profit Factor:</span>
                <span>{(selectedModel.profit_factor || 0).toFixed(2)}</span>
              </div>
            </div>

            <div style={styles.detailSection}>
              <h5 style={styles.sectionTitle}>Training Info</h5>
              <div style={styles.detailItem}>
                <span>Agent Type:</span>
                <span>{selectedModel.agent_type}</span>
              </div>
              <div style={styles.detailItem}>
                <span>Episodes:</span>
                <span>{(selectedModel.episodes || 0).toLocaleString()}</span>
              </div>
              <div style={styles.detailItem}>
                <span>Duration:</span>
                <span>{(selectedModel.training_duration_seconds || 0).toFixed(1)}s</span>
              </div>
              <div style={styles.detailItem}>
                <span>Total Trades:</span>
                <span>{selectedModel.total_trades || 0} ({selectedModel.winning_trades || 0}W / {selectedModel.losing_trades || 0}L)</span>
              </div>
              <div style={styles.detailItem}>
                <span>Train Samples:</span>
                <span>{selectedModel.train_samples || 0}</span>
              </div>
              <div style={styles.detailItem}>
                <span>Test Samples:</span>
                <span>{selectedModel.test_samples || 0}</span>
              </div>
            </div>

            <div style={styles.detailSection}>
              <h5 style={styles.sectionTitle}>Features Configuration</h5>
              {selectedModel.features && (
                <div style={styles.featuresDetail}>
                  <div style={styles.featureItem}>
                    <span>OHLCV:</span>
                    <span style={{color: selectedModel.features.ohlcv ? '#4CAF50' : '#888'}}>
                      {selectedModel.features.ohlcv ? '✓ Enabled' : '✗ Disabled'}
                    </span>
                  </div>
                  <div style={styles.featureItem}>
                    <span>Technical Indicators:</span>
                    <span style={{color: selectedModel.features.technical ? '#4CAF50' : '#888'}}>
                      {selectedModel.features.technical ? '✓ Enabled' : '✗ Disabled'}
                    </span>
                  </div>
                  <div style={styles.featureItem}>
                    <span>Fundamentals:</span>
                    <span style={{color: selectedModel.features.fundamentals ? '#4CAF50' : '#888'}}>
                      {selectedModel.features.fundamentals ? '✓ Enabled' : '✗ Disabled'}
                    </span>
                  </div>
                  <div style={styles.featureItem}>
                    <span>Macro Indicators:</span>
                    <span style={{color: selectedModel.features.macro ? '#4CAF50' : '#888'}}>
                      {selectedModel.features.macro ? '✓ Enabled' : '✗ Disabled'}
                    </span>
                  </div>
                  <div style={styles.featureItem}>
                    <span>Sentiment Analysis:</span>
                    <span style={{color: selectedModel.features.sentiment ? '#4CAF50' : '#888'}}>
                      {selectedModel.features.sentiment ? '✓ Enabled' : '✗ Disabled'}
                    </span>
                  </div>
                  <div style={styles.featureItem}>
                    <span>Market Events:</span>
                    <span style={{color: selectedModel.features.events ? '#4CAF50' : '#888'}}>
                      {selectedModel.features.events ? '✓ Enabled' : '✗ Disabled'}
                    </span>
                  </div>
                  <div style={styles.featureItem}>
                    <span>Multi-Asset:</span>
                    <span style={{color: selectedModel.features.multi_asset ? '#4CAF50' : '#888'}}>
                      {selectedModel.features.multi_asset ? '✓ Enabled' : '✗ Disabled'}
                    </span>
                  </div>
                </div>
              )}
              
              {selectedModel.features_used && (
                <div style={styles.featuresUsedList}>
                  <strong>Actual Features ({selectedModel.features_used.length}):</strong>
                  <p style={{fontSize: '11px', color: '#888', margin: '5px 0'}}>
                    {selectedModel.features_used.join(', ')}
                  </p>
                </div>
              )}
            </div>
          </div>
          
          <button onClick={() => setSelectedModel(null)} style={{...styles.button, marginTop: '15px'}}>
            Close Details
          </button>
        </div>
      )}
    </div>
  );
};

// Styles
const styles = {
  container: {
    padding: '20px',
    backgroundColor: '#1e1e1e',
    borderRadius: '8px',
    marginTop: '20px'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
    flexWrap: 'wrap',
    gap: '10px'
  },
  title: {
    color: '#e0e0e0',
    margin: 0,
    fontSize: '20px',
    fontWeight: 'bold'
  },
  controls: {
    display: 'flex',
    gap: '10px',
    alignItems: 'center'
  },
  filterInput: {
    padding: '8px 12px',
    backgroundColor: '#2d2d2d',
    color: '#e0e0e0',
    border: '1px solid #444',
    borderRadius: '4px',
    fontSize: '14px',
    outline: 'none'
  },
  button: {
    padding: '8px 16px',
    backgroundColor: '#4a9eff',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '600',
    transition: 'background 0.2s'
  },
  tableWrapper: {
    overflowX: 'auto',
    overflowY: 'auto',
    maxHeight: '600px',
    border: '1px solid #444',
    borderRadius: '4px'
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '13px'
  },
  th: {
    backgroundColor: '#2d2d2d',
    color: '#4a9eff',
    padding: '12px 8px',
    textAlign: 'left',
    fontWeight: 'bold',
    cursor: 'pointer',
    userSelect: 'none',
    position: 'sticky',
    top: 0,
    borderBottom: '2px solid #4a9eff',
    whiteSpace: 'nowrap'
  },
  tr: {
    transition: 'background 0.2s',
    cursor: 'pointer'
  },
  td: {
    padding: '10px 8px',
    color: '#e0e0e0',
    borderBottom: '1px solid #333'
  },
  bestBadge: {
    backgroundColor: '#4CAF50',
    color: 'white',
    padding: '2px 6px',
    borderRadius: '4px',
    fontSize: '10px',
    fontWeight: 'bold',
    marginRight: '8px'
  },
  versionText: {
    fontFamily: 'monospace',
    fontSize: '12px',
    color: '#888'
  },
  featuresText: {
    fontSize: '11px',
    color: '#888'
  },
  loading: {
    color: '#888',
    fontStyle: 'italic',
    textAlign: 'center',
    padding: '40px'
  },
  error: {
    color: '#ff6b6b',
    fontWeight: 'bold',
    textAlign: 'center',
    padding: '40px'
  },
  noData: {
    color: '#888',
    fontStyle: 'italic',
    textAlign: 'center',
    padding: '40px'
  },
  detailsPanel: {
    marginTop: '20px',
    padding: '20px',
    backgroundColor: '#2d2d2d',
    borderRadius: '8px',
    border: '2px solid #4a9eff'
  },
  detailsTitle: {
    color: '#4a9eff',
    marginTop: 0,
    marginBottom: '15px',
    fontSize: '16px'
  },
  detailsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '20px'
  },
  detailSection: {
    backgroundColor: '#1e1e1e',
    padding: '15px',
    borderRadius: '6px'
  },
  sectionTitle: {
    color: '#4CAF50',
    fontSize: '14px',
    marginTop: 0,
    marginBottom: '10px',
    borderBottom: '1px solid #444',
    paddingBottom: '5px'
  },
  detailItem: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '6px 0',
    fontSize: '13px',
    borderBottom: '1px solid #333'
  },
  featuresDetail: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5px'
  },
  featureItem: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '5px 0',
    fontSize: '12px'
  },
  featuresUsedList: {
    marginTop: '15px',
    padding: '10px',
    backgroundColor: '#252525',
    borderRadius: '4px',
    border: '1px solid #444'
  }
};

export default ModelsComparisonTable;
