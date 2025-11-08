/**
 * ConfigManager.jsx
 * Training Configuration Management Component
 * 
 * Purpose:
 * - Save and load training configurations
 * - Provide preset configurations (Conservative, Aggressive, Balanced)
 * - Dynamic presets based on agent type (PPO for stocks, SAC for ETFs)
 * - Export/import configurations as JSON files
 * 
 * Why separate file:
 * - Focused component for config management
 * - Reusable across different training contexts
 * - Separates config I/O from training logic
 * - Easy to extend with more presets or validation
 * 
 * Features:
 * - Save current config with custom name
 * - Load previously saved configs
 * - Quick-select presets optimized for different strategies AND agent types
 * - Export config as downloadable JSON
 * - Import config from JSON file
 * 
 * Props:
 * - currentConfig: Current training configuration object
 * - onLoadConfig: Callback when config is loaded (applies to state)
 * - agentType: 'PPO' or 'SAC' - determines which presets to show
 * 
 * Wiring:
 * - Calls trainingAPI.saveConfig() to persist configs
 * - Calls trainingAPI.loadConfig() to retrieve saved configs
 * - Parent (TabTraining) applies loaded config to useTrainingState
 */

import React, { useState, useEffect } from 'react';
import { saveConfig, loadConfig } from '../../services/trainingAPI';

const ConfigManager = ({ currentConfig, onLoadConfig, agentType }) => {
  const [savedConfigs, setSavedConfigs] = useState([]);
  const [configName, setConfigName] = useState('');
  const [selectedConfig, setSelectedConfig] = useState('');
  const [message, setMessage] = useState({ text: '', type: '' });

  // Preset configurations - different for PPO (stocks) and SAC (ETFs)
  const presets = {
    PPO: {
      Conservative: {
        name: 'Conservative (Stock)',
        description: 'Low risk, stable returns for stocks',
        config: {
          agent_type: 'PPO',
          hyperparameters: {
            learning_rate: 0.0001,
            gamma: 0.99,
            batch_size: 512,
            n_steps: 2048,
            ent_coef: 0.01,
            clip_range: 0.2,
            episodes: 60000
          }
        }
      },
      Aggressive: {
        name: 'Aggressive (Stock)',
        description: 'Higher risk, faster learning for stocks',
        config: {
          agent_type: 'PPO',
          hyperparameters: {
            learning_rate: 0.0005,
            gamma: 0.95,
            batch_size: 128,
            n_steps: 1024,
            ent_coef: 0.05,
            clip_range: 0.3,
            episodes: 40000
          }
        }
      },
      Balanced: {
        name: 'Balanced (Stock)',
        description: 'Moderate risk-return for stocks',
        config: {
          agent_type: 'PPO',
          hyperparameters: {
            learning_rate: 0.0003,
            gamma: 0.99,
            batch_size: 256,
            n_steps: 2048,
            ent_coef: 0.02,
            clip_range: 0.2,
            episodes: 50000
          }
        }
      }
    },
    SAC: {
      Conservative: {
        name: 'Conservative (ETF)',
        description: 'Low risk, stable returns for leveraged ETFs',
        config: {
          agent_type: 'SAC',
          hyperparameters: {
            learning_rate: 0.0001,
            gamma: 0.99,
            batch_size: 512,
            tau: 0.005,
            ent_coef: 0.1,
            buffer_size: 100000,
            episodes: 60000
          }
        }
      },
      Aggressive: {
        name: 'Aggressive (ETF)',
        description: 'High risk, high reward for leveraged ETFs',
        config: {
          agent_type: 'SAC',
          hyperparameters: {
            learning_rate: 0.0005,
            gamma: 0.95,
            batch_size: 128,
            tau: 0.01,
            ent_coef: 0.3,
            buffer_size: 50000,
            episodes: 40000
          }
        }
      },
      Balanced: {
        name: 'Balanced (ETF)',
        description: 'Moderate risk-return for leveraged ETFs',
        config: {
          agent_type: 'SAC',
          hyperparameters: {
            learning_rate: 0.0003,
            gamma: 0.99,
            batch_size: 256,
            tau: 0.005,
            ent_coef: 0.2,
            buffer_size: 100000,
            episodes: 50000
          }
        }
      }
    }
  };

  // Get presets for current agent type
  const currentPresets = presets[agentType] || presets.PPO;

  // Show message with auto-hide
  const showMessage = (text, type) => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: '', type: '' }), 3000);
  };

  // Save current configuration
  const handleSaveConfig = async () => {
    if (!configName.trim()) {
      showMessage('Please enter a configuration name', 'error');
      return;
    }

    const result = await saveConfig(currentConfig, configName);
    if (result.success) {
      showMessage('Configuration saved successfully!', 'success');
      setConfigName('');
      // Refresh saved configs list (in real app, fetch from API)
      setSavedConfigs([...savedConfigs, configName]);
    } else {
      showMessage(`Failed to save: ${result.error}`, 'error');
    }
  };

  // Load saved configuration
  const handleLoadConfig = async () => {
    if (!selectedConfig) {
      showMessage('Please select a configuration', 'error');
      return;
    }

    const result = await loadConfig(selectedConfig);
    if (result.success) {
      onLoadConfig(result.config);
      showMessage('Configuration loaded successfully!', 'success');
    } else {
      showMessage(`Failed to load: ${result.error}`, 'error');
    }
  };

  // Load preset configuration
  const handleLoadPreset = (presetName) => {
    const preset = currentPresets[presetName];
    if (preset) {
      onLoadConfig(preset.config);
      showMessage(`${preset.name} preset loaded!`, 'success');
    }
  };

  // Export config as JSON file
  const handleExportConfig = () => {
    const dataStr = JSON.stringify(currentConfig, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `training_config_${Date.now()}.json`;
    link.click();
    
    URL.revokeObjectURL(url);
    showMessage('Configuration exported!', 'success');
  };

  // Import config from JSON file
  const handleImportConfig = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const config = JSON.parse(e.target.result);
        onLoadConfig(config);
        showMessage('Configuration imported successfully!', 'success');
      } catch (error) {
        showMessage('Invalid JSON file', 'error');
      }
    };
    reader.readAsText(file);
  };

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>⚙️ Configuration Manager</h3>

      {/* Message Display */}
      {message.text && (
        <div style={{
          ...styles.message,
          backgroundColor: message.type === 'success' ? '#4CAF50' : '#ff6b6b'
        }}>
          {message.text}
        </div>
      )}

      {/* Preset Configurations */}
      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>
          Quick Presets {agentType && `(${agentType} - ${agentType === 'PPO' ? 'Stocks' : 'Leveraged ETFs'})`}
        </h4>
        <div style={styles.presetsGrid}>
          {Object.entries(currentPresets).map(([key, preset]) => (
            <div key={key} style={styles.presetCard}>
              <h5 style={styles.presetName}>{preset.name}</h5>
              <p style={styles.presetDescription}>{preset.description}</p>
              <button
                style={styles.presetButton}
                onClick={() => handleLoadPreset(key)}
                onMouseOver={(e) => e.target.style.backgroundColor = '#45a049'}
                onMouseOut={(e) => e.target.style.backgroundColor = '#4CAF50'}
              >
                Load Preset
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Save Configuration */}
      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>Save Current Configuration</h4>
        <div style={styles.saveRow}>
          <input
            type="text"
            placeholder="Enter configuration name..."
            value={configName}
            onChange={(e) => setConfigName(e.target.value)}
            style={styles.input}
          />
          <button
            style={styles.button}
            onClick={handleSaveConfig}
          >
            💾 Save
          </button>
        </div>
      </div>

      {/* Load Saved Configuration */}
      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>Load Saved Configuration</h4>
        <div style={styles.loadRow}>
          <select
            value={selectedConfig}
            onChange={(e) => setSelectedConfig(e.target.value)}
            style={styles.select}
          >
            <option value="">-- Select Saved Config --</option>
            {savedConfigs.map((name, index) => (
              <option key={index} value={name}>{name}</option>
            ))}
          </select>
          <button
            style={styles.button}
            onClick={handleLoadConfig}
          >
            📂 Load
          </button>
        </div>
      </div>

      {/* Export/Import */}
      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>Export / Import</h4>
        <div style={styles.exportImportRow}>
          <button
            style={styles.exportButton}
            onClick={handleExportConfig}
          >
            📥 Export as JSON
          </button>
          <label style={styles.importLabel}>
            📤 Import JSON
            <input
              type="file"
              accept=".json"
              onChange={handleImportConfig}
              style={styles.fileInput}
            />
          </label>
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
  message: {
    padding: '12px',
    borderRadius: '6px',
    color: 'white',
    fontWeight: 'bold',
    marginBottom: '20px',
    textAlign: 'center'
  },
  section: {
    marginBottom: '25px'
  },
  sectionTitle: {
    color: '#4CAF50',
    fontSize: '14px',
    fontWeight: 'bold',
    marginBottom: '12px'
  },
  presetsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '15px'
  },
  presetCard: {
    padding: '15px',
    backgroundColor: '#2d2d2d',
    borderRadius: '6px',
    border: '1px solid #444'
  },
  presetName: {
    color: '#e0e0e0',
    marginTop: 0,
    marginBottom: '8px',
    fontSize: '16px'
  },
  presetDescription: {
    color: '#888',
    fontSize: '12px',
    marginBottom: '12px'
  },
  presetButton: {
    width: '100%',
    padding: '8px',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    fontSize: '13px',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'background-color 0.3s'
  },
  saveRow: {
    display: 'flex',
    gap: '10px'
  },
  loadRow: {
    display: 'flex',
    gap: '10px'
  },
  input: {
    flex: 1,
    padding: '10px',
    backgroundColor: '#2d2d2d',
    color: '#e0e0e0',
    border: '1px solid #444',
    borderRadius: '4px',
    fontSize: '14px'
  },
  select: {
    flex: 1,
    padding: '10px',
    backgroundColor: '#2d2d2d',
    color: '#e0e0e0',
    border: '1px solid #444',
    borderRadius: '4px',
    fontSize: '14px',
    cursor: 'pointer'
  },
  button: {
    padding: '10px 20px',
    backgroundColor: '#2196F3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    fontSize: '14px',
    fontWeight: 'bold',
    cursor: 'pointer',
    whiteSpace: 'nowrap'
  },
  exportImportRow: {
    display: 'flex',
    gap: '15px'
  },
  exportButton: {
    flex: 1,
    padding: '12px',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    fontSize: '14px',
    fontWeight: 'bold',
    cursor: 'pointer'
  },
  importLabel: {
    flex: 1,
    padding: '12px',
    backgroundColor: '#FF9800',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    fontSize: '14px',
    fontWeight: 'bold',
    textAlign: 'center',
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center'
  },
  fileInput: {
    display: 'none'
  }
};

export default ConfigManager;
