/**
 * DriftAlert.jsx
 * Data Drift Detection Alert Component
 * 
 * Purpose:
 * - Display warning when data drift is detected
 * - Show which features have drifted and severity
 * - Provide quick action to retrain model with updated data
 * 
 * Why separate file:
 * - Reusable across different monitoring contexts
 * - Isolated drift visualization logic
 * - Can be enhanced with drift charts later
 * 
 * Data Drift:
 * - Statistical distribution changes in input features
 * - Indicates model may be operating outside training conditions
 * - Requires model retraining for continued accuracy
 * 
 * Props:
 * - driftData: Object with drift detection results from API
 * - onRetrain: Callback function when "Retrain Now" button clicked
 * 
 * Wiring:
 * - Parent calls trainingAPI.checkDriftStatus() periodically
 * - Passes drift results as props
 * - onRetrain triggers training workflow with recent data
 */

import React from 'react';

const DriftAlert = ({ driftData, onRetrain }) => {
  // Don't render if no drift detected
  if (!driftData || !driftData.drift_detected) {
    return null;
  }

  // Extract drift data
  const {
    severity,
    affected_features,
    drift_scores,
    threshold,
    recommendation,
    last_check
  } = driftData;

  if (typeof severity !== 'string' || !severity.trim()) {
    throw new Error('DriftAlert: severity is required');
  }

  if (!Array.isArray(affected_features) || affected_features.length === 0) {
    throw new Error('DriftAlert: affected_features must be a non-empty array');
  }

  if (!drift_scores || typeof drift_scores !== 'object') {
    throw new Error('DriftAlert: drift_scores object is required');
  }

  const thresholdValue = Number(threshold);
  if (!Number.isFinite(thresholdValue)) {
    throw new Error('DriftAlert: numeric threshold is required');
  }

  if (typeof recommendation !== 'string' || !recommendation.trim()) {
    throw new Error('DriftAlert: recommendation text is required');
  }

  // Map severity to colors
  const severityColors = {
    low: '#4CAF50',
    medium: '#FFC107',
    high: '#FF9800',
    critical: '#ff6b6b'
  };
  const normalizedSeverity = severity.trim().toLowerCase();
  const severityColor = severityColors[normalizedSeverity];
  if (!severityColor) {
    throw new Error(`DriftAlert: unsupported severity level "${severity}"`);
  }
  const severityLabel = normalizedSeverity.toUpperCase();
  const featureList = affected_features;

  // Format last check date
  const formatDate = (dateString) => {
    if (typeof dateString !== 'string' || !dateString.trim()) {
      throw new Error('DriftAlert: last_check timestamp is required');
    }

    const date = new Date(dateString);
    if (Number.isNaN(date.getTime())) {
      throw new Error('DriftAlert: invalid last_check timestamp');
    }

    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div style={{...styles.container, borderColor: severityColor}}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <span style={styles.icon}>⚠️</span>
          <h3 style={styles.title}>Data Drift Detected</h3>
        </div>
        <span style={{...styles.severityBadge, backgroundColor: severityColor}}>
          {severityLabel}
        </span>
      </div>

      {/* Recommendation */}
      <p style={styles.recommendation}>
        {recommendation}
      </p>

      {/* Affected Features */}
      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>Affected Features ({featureList.length}):</h4>
        <div style={styles.featuresGrid}>
          {featureList.map((feature, index) => {
            if (typeof feature !== 'string' || !feature.trim()) {
              throw new Error('DriftAlert: feature identifiers must be non-empty strings');
            }

            const rawScore = drift_scores[feature];
            const numericScore = Number(rawScore);
            if (!Number.isFinite(numericScore)) {
              throw new Error(`DriftAlert: drift score missing for feature "${feature}"`);
            }

            const isHighDrift = numericScore > thresholdValue * 1.5;
            const featureLabel = feature.trim().toUpperCase();

            return (
              <div 
                key={index} 
                style={{
                  ...styles.featureChip,
                  backgroundColor: isHighDrift ? '#ff6b6b' : '#FFC107'
                }}
              >
                <span style={styles.featureName}>{featureLabel}</span>
                <span style={styles.featureScore}>
                  {numericScore.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Drift Details */}
      <div style={styles.detailsSection}>
        <div style={styles.detailItem}>
          <span style={styles.detailLabel}>Drift Threshold:</span>
          <span style={styles.detailValue}>{thresholdValue.toFixed(2)}</span>
        </div>
        <div style={styles.detailItem}>
          <span style={styles.detailLabel}>Last Check:</span>
          <span style={styles.detailValue}>
            {formatDate(last_check)}
          </span>
        </div>
      </div>

      {/* Action Button */}
      <button 
        style={styles.retrainButton}
        onClick={onRetrain}
        onMouseOver={(e) => e.target.style.backgroundColor = '#45a049'}
        onMouseOut={(e) => e.target.style.backgroundColor = '#4CAF50'}
      >
        🔄 Retrain Model Now
      </button>

      {/* Info Footer */}
      <div style={styles.footer}>
        <p style={styles.footerText}>
          💡 <strong>Tip:</strong> Retraining with recent data will adapt the model to current market conditions.
        </p>
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
    border: '3px solid',
    marginBottom: '20px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '15px'
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px'
  },
  icon: {
    fontSize: '28px'
  },
  title: {
    color: '#e0e0e0',
    margin: 0,
    fontSize: '20px',
    fontWeight: 'bold'
  },
  severityBadge: {
    padding: '6px 12px',
    borderRadius: '4px',
    color: '#1e1e1e',
    fontWeight: 'bold',
    fontSize: '12px'
  },
  recommendation: {
    color: '#e0e0e0',
    fontSize: '14px',
    lineHeight: '1.6',
    marginBottom: '20px',
    padding: '10px',
    backgroundColor: '#2d2d2d',
    borderRadius: '4px',
    borderLeft: '3px solid #FFC107'
  },
  section: {
    marginBottom: '20px'
  },
  sectionTitle: {
    color: '#4CAF50',
    fontSize: '14px',
    fontWeight: 'bold',
    marginBottom: '10px'
  },
  featuresGrid: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '10px'
  },
  featureChip: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 12px',
    borderRadius: '20px',
    color: '#1e1e1e',
    fontWeight: 'bold',
    fontSize: '13px'
  },
  featureName: {
    textTransform: 'uppercase'
  },
  featureScore: {
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    padding: '2px 6px',
    borderRadius: '10px',
    fontSize: '11px'
  },
  detailsSection: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '15px',
    padding: '15px',
    backgroundColor: '#2d2d2d',
    borderRadius: '6px',
    marginBottom: '20px'
  },
  detailItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5px'
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
  retrainButton: {
    width: '100%',
    padding: '15px',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    fontSize: '16px',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'background-color 0.3s',
    marginBottom: '15px'
  },
  footer: {
    padding: '10px',
    backgroundColor: '#2d2d2d',
    borderRadius: '4px',
    borderLeft: '3px solid #4CAF50'
  },
  footerText: {
    color: '#888',
    fontSize: '12px',
    margin: 0,
    lineHeight: '1.5'
  }
};

export default DriftAlert;
