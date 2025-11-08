/**
 * BacktestResults.jsx
 * Backtesting Results Display Component
 * 
 * Purpose:
 * - Display key performance metrics from backtesting
 * - Visual representation of trading strategy performance
 * - Help users evaluate model quality before live deployment
 * 
 * Why separate file:
 * - Reusable across training and evaluation tabs
 * - Focused component for metrics display
 * - Easy to extend with charts (equity curve, drawdown chart)
 * 
 * Metrics Displayed:
 * - Sharpe Ratio: Risk-adjusted returns
 * - Sortino Ratio: Downside risk-adjusted returns
 * - Max Drawdown: Worst peak-to-trough decline
 * - Win Rate: Percentage of profitable trades
 * - Total Return: Overall portfolio performance
 * 
 * Props:
 * - results: Object containing backtest results from API
 * - loading: Boolean for loading state
 * 
 * Wiring:
 * - Receives backtest results from TabTraining or TabEvaluation
 * - Parent calls trainingAPI.runBacktest() and passes results
 * - Displays metrics in grid of colored cards
 */

import React from 'react';

const BacktestResults = ({ results, loading }) => {
  // If no results yet, show placeholder
  if (!results && !loading) {
    return (
      <div style={styles.container}>
        <p style={styles.noData}>No backtest results available. Run a backtest to see metrics.</p>
      </div>
    );
  }

  // Loading state
  if (loading) {
    return (
      <div style={styles.container}>
        <p style={styles.loading}>Running backtest...</p>
      </div>
    );
  }

  // Extract metrics from results
  const {
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    win_rate,
    total_return,
    final_portfolio_value,
    num_trades,
    avg_trade_return
  } = results;

  // Metric card data with colors and icons
  const metrics = [
    {
      label: 'Sharpe Ratio',
      value: sharpe_ratio?.toFixed(2) || 'N/A',
      color: sharpe_ratio >= 1.5 ? '#4CAF50' : sharpe_ratio >= 1.0 ? '#FFC107' : '#ff6b6b',
      description: 'Risk-adjusted return (>1.5 is excellent)',
      icon: '📊'
    },
    {
      label: 'Sortino Ratio',
      value: sortino_ratio?.toFixed(2) || 'N/A',
      color: sortino_ratio >= 2.0 ? '#4CAF50' : sortino_ratio >= 1.5 ? '#FFC107' : '#ff6b6b',
      description: 'Downside risk-adjusted return',
      icon: '📈'
    },
    {
      label: 'Max Drawdown',
      value: max_drawdown ? `${max_drawdown.toFixed(2)}%` : 'N/A',
      color: max_drawdown > -15 ? '#4CAF50' : max_drawdown > -25 ? '#FFC107' : '#ff6b6b',
      description: 'Worst peak-to-trough decline',
      icon: '📉'
    },
    {
      label: 'Win Rate',
      value: win_rate ? `${win_rate.toFixed(1)}%` : 'N/A',
      color: win_rate >= 55 ? '#4CAF50' : win_rate >= 50 ? '#FFC107' : '#ff6b6b',
      description: 'Percentage of profitable trades',
      icon: '🎯'
    },
    {
      label: 'Total Return',
      value: total_return ? `${total_return.toFixed(2)}%` : 'N/A',
      color: total_return >= 20 ? '#4CAF50' : total_return >= 10 ? '#FFC107' : '#ff6b6b',
      description: 'Overall portfolio performance',
      icon: '💰'
    }
  ];

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>📊 Backtest Results</h3>
      
      {/* Primary Metrics Grid */}
      <div style={styles.metricsGrid}>
        {metrics.map((metric, index) => (
          <div key={index} style={{...styles.metricCard, borderLeftColor: metric.color}}>
            <div style={styles.metricIcon}>{metric.icon}</div>
            <div style={styles.metricContent}>
              <p style={styles.metricLabel}>{metric.label}</p>
              <p style={{...styles.metricValue, color: metric.color}}>
                {metric.value}
              </p>
              <p style={styles.metricDescription}>{metric.description}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Additional Statistics */}
      <div style={styles.additionalStats}>
        <h4 style={styles.additionalTitle}>Additional Statistics</h4>
        <div style={styles.statsGrid}>
          <div style={styles.statItem}>
            <span style={styles.statLabel}>Final Portfolio Value:</span>
            <span style={styles.statValue}>
              ${final_portfolio_value?.toLocaleString() || 'N/A'}
            </span>
          </div>
          <div style={styles.statItem}>
            <span style={styles.statLabel}>Number of Trades:</span>
            <span style={styles.statValue}>
              {num_trades?.toLocaleString() || 'N/A'}
            </span>
          </div>
          <div style={styles.statItem}>
            <span style={styles.statLabel}>Avg Trade Return:</span>
            <span style={{
              ...styles.statValue,
              color: avg_trade_return >= 0 ? '#4CAF50' : '#ff6b6b'
            }}>
              {avg_trade_return ? `${avg_trade_return.toFixed(2)}%` : 'N/A'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Inline styles
const styles = {
  container: {
    padding: '20px',
    backgroundColor: '#1e1e1e',
    borderRadius: '8px',
    marginTop: '20px'
  },
  title: {
    color: '#e0e0e0',
    marginTop: 0,
    marginBottom: '20px',
    fontSize: '18px',
    fontWeight: 'bold'
  },
  noData: {
    color: '#888',
    fontStyle: 'italic',
    textAlign: 'center'
  },
  loading: {
    color: '#888',
    fontStyle: 'italic',
    textAlign: 'center'
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '15px',
    marginBottom: '20px'
  },
  metricCard: {
    display: 'flex',
    alignItems: 'center',
    gap: '15px',
    padding: '20px',
    backgroundColor: '#2d2d2d',
    borderRadius: '8px',
    borderLeft: '4px solid',
    transition: 'transform 0.2s',
    cursor: 'default'
  },
  metricIcon: {
    fontSize: '32px',
    flexShrink: 0
  },
  metricContent: {
    flex: 1
  },
  metricLabel: {
    color: '#888',
    fontSize: '12px',
    fontWeight: 'bold',
    marginBottom: '5px',
    textTransform: 'uppercase'
  },
  metricValue: {
    fontSize: '28px',
    fontWeight: 'bold',
    marginBottom: '5px'
  },
  metricDescription: {
    color: '#888',
    fontSize: '11px',
    fontStyle: 'italic'
  },
  additionalStats: {
    padding: '15px',
    backgroundColor: '#2d2d2d',
    borderRadius: '8px',
    border: '1px solid #444'
  },
  additionalTitle: {
    color: '#4CAF50',
    marginTop: 0,
    marginBottom: '15px',
    fontSize: '14px',
    fontWeight: 'bold'
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '15px'
  },
  statItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '10px',
    backgroundColor: '#1e1e1e',
    borderRadius: '4px'
  },
  statLabel: {
    color: '#888',
    fontSize: '13px',
    fontWeight: 'bold'
  },
  statValue: {
    color: '#e0e0e0',
    fontSize: '14px',
    fontWeight: 'bold'
  }
};

export default BacktestResults;
