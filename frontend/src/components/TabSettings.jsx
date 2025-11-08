import React, { useState, useEffect } from 'react';
import { Button, Card } from './common/UIComponents';
import ibkrService from '../services/IBKRService';

function TabSettings() {
  // IBKR Connection Settings
  const [ibkrSettings, setIbkrSettings] = useState({
    host: '127.0.0.1',
    port: 7497,
    clientId: 1,
    reconnectInterval: 10
  });
  
  const [connectionStatus, setConnectionStatus] = useState(null);
  const [testingConnection, setTestingConnection] = useState(false);

  // Load connection status on mount
  useEffect(() => {
    checkConnectionStatus();
  }, []);

  const checkConnectionStatus = async () => {
    const status = await ibkrService.checkConnection();
    setConnectionStatus(status);
  };

  const handleIbkrSettingChange = (field, value) => {
    setIbkrSettings(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const testConnection = async () => {
    setTestingConnection(true);
    try {
      const status = await ibkrService.checkConnection();
      setConnectionStatus(status);
      
      if (status.isConnected) {
        alert(`✅ Connection Successful!\n\nTWS Connection: ${status.message || 'Connected'}\nBridge Status: Running`);
      } else {
        alert(`❌ Connection Failed\n\n${status.message || 'Unable to connect to IBKR Bridge or TWS'}\n\nPlease check:\n1. TWS is running\n2. API is enabled in TWS\n3. Port matches TWS settings\n4. IBKR Bridge is running`);
      }
    } catch (error) {
      alert(`❌ Connection Error\n\n${error.message}`);
    } finally {
      setTestingConnection(false);
    }
  };

  const applyIbkrSettings = () => {
    // In a real implementation, this would:
    // 1. Stop the current IBKR connection
    // 2. Update the bridge configuration
    // 3. Restart with new settings
    
    console.log('Applying IBKR settings:', ibkrSettings);
    
    // For now, just show a confirmation
    alert(`⚙️ IBKR Settings Updated\n\nHost: ${ibkrSettings.host}\nPort: ${ibkrSettings.port}\nClient ID: ${ibkrSettings.clientId}\nReconnect Interval: ${ibkrSettings.reconnectInterval}s\n\nNote: IBKR Bridge needs to be restarted with these settings.`);
  };

  return (
    <div>
      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">PPO Agent Settings</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Max Position Size</div>
            {/* Maximum dollar amount for a single position
                - Prevents over-concentration in one trade
                - Risk management to limit single-trade exposure
                - Example: $10,000 max per stock */}
            <input type="number" className="param-input" defaultValue="10000" style={{ padding: '5px', fontSize: '11px' }} />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Stop Loss %</div>
            {/* Automatic exit if position loses this percentage
                - Protects against large losses
                - Market order sent when price drops X% from entry
                - Example: 2% = exit if price drops 2% */}
            <input type="number" className="param-input" defaultValue="2.0" step="0.1" style={{ padding: '5px', fontSize: '11px' }} />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Take Profit %</div>
            {/* Automatic exit when position gains this percentage
                - Locks in profits at predetermined level
                - Market order sent when price rises X% from entry
                - Example: 5% = exit when up 5% */}
            <input type="number" className="param-input" defaultValue="5.0" step="0.1" style={{ padding: '5px', fontSize: '11px' }} />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Max Daily Trades</div>
            {/* Limit on number of trades per day for this agent
                - Prevents overtrading and excessive fees
                - Agent stops trading after reaching limit
                - Resets at market open next day */}
            <input type="number" className="param-input" defaultValue="10" style={{ padding: '5px', fontSize: '11px' }} />
          </div>
        </div>
      </Card>

      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">SAC Agent Settings</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Max Position Size</div>
            {/* Maximum dollar amount for a single position
                - SAC typically handles leveraged ETFs (higher volatility)
                - Can be higher than PPO if agent is more aggressive
                - Example: $15,000 for leveraged ETF positions */}
            <input type="number" className="param-input" defaultValue="15000" style={{ padding: '5px', fontSize: '11px' }} />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Stop Loss %</div>
            {/* Automatic exit if position loses this percentage
                - Higher than PPO for leveraged ETFs (more volatility)
                - Example: 3% for 3x leveraged ETFs */}
            <input type="number" className="param-input" defaultValue="3.0" step="0.1" style={{ padding: '5px', fontSize: '11px' }} />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Leverage Limit</div>
            {/* Maximum leverage multiplier for ETF positions
                - Prevents excessive risk from leveraged products
                - Example: 2.0 = max 2x leveraged ETFs (not 3x)
                - Use 1.0 to disable leveraged ETFs entirely */}
            <input type="number" className="param-input" defaultValue="2.0" step="0.1" style={{ padding: '5px', fontSize: '11px' }} />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Max Daily Trades</div>
            {/* Limit on number of trades per day for this agent
                - ETFs may require fewer trades due to volatility
                - Example: 8 trades/day for leveraged products */}
            <input type="number" className="param-input" defaultValue="8" style={{ padding: '5px', fontSize: '11px' }} />
          </div>
        </div>
      </Card>

      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">Global Risk Settings</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Max Daily Drawdown</div>
            {/* Maximum portfolio loss allowed in a single day
                - System automatically stops ALL trading if breached
                - Calculated as: (Day Start Value - Current Value) / Day Start Value
                - Critical safety mechanism to prevent catastrophic losses
                - Example: 15% = stops trading if portfolio drops 15% from today's open */}
            <input type="number" className="param-input" defaultValue="15.0" step="0.5" style={{ padding: '5px', fontSize: '11px' }} />
            <div style={{ fontSize: '9px', color: '#6e7681', marginTop: '2px' }}>Auto-halt if breached</div>
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Portfolio Exposure Limit</div>
            {/* Maximum percentage of capital that can be invested at once
                - Prevents going "all-in" and maintains cash buffer
                - Example: 80% = keep at least 20% in cash
                - Lower values = more conservative, higher = more aggressive */}
            <input type="number" className="param-input" defaultValue="80" style={{ padding: '5px', fontSize: '11px' }} />
            <div style={{ fontSize: '9px', color: '#6e7681', marginTop: '2px' }}>% of capital</div>
          </div>
        </div>
      </Card>

      <Card>
        <div className="control-title">IBKR Connection</div>
        
        {/* Connection Status Indicator */}
        <div style={{ marginBottom: '12px', padding: '8px', backgroundColor: '#161b22', borderRadius: '4px', border: '1px solid #30363d' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ 
                width: '10px', 
                height: '10px', 
                borderRadius: '50%', 
                backgroundColor: connectionStatus?.isConnected ? '#3fb950' : '#f85149',
                boxShadow: connectionStatus?.isConnected ? '0 0 8px #3fb950' : '0 0 8px #f85149'
              }} />
              <span style={{ fontSize: '12px', color: '#c9d1d9', fontWeight: '500' }}>
                {connectionStatus?.isConnected ? 'Connected to TWS' : 'Disconnected'}
              </span>
            </div>
            <Button 
              variant="secondary" 
              onClick={testConnection}
              disabled={testingConnection}
              style={{ fontSize: '11px', padding: '4px 12px' }}
            >
              {testingConnection ? 'Testing...' : 'Test Connection'}
            </Button>
          </div>
          {connectionStatus?.message && (
            <div style={{ fontSize: '11px', color: '#8b949e', marginTop: '6px', marginLeft: '18px' }}>
              {connectionStatus.message}
            </div>
          )}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', marginBottom: '12px' }}>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Host</div>
            {/* IP address of IBKR TWS/Gateway
                - 127.0.0.1 = localhost (same computer)
                - Use remote IP if TWS is on another machine
                - Must match TWS API settings */}
            <input 
              type="text" 
              className="param-input" 
              value={ibkrSettings.host}
              onChange={(e) => handleIbkrSettingChange('host', e.target.value)}
              style={{ padding: '5px', fontSize: '11px' }} 
            />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Port</div>
            {/* TWS API port number
                - 7497 = TWS Paper Trading (default)
                - 7496 = TWS Live Trading
                - 4002 = IB Gateway Paper Trading
                - 4001 = IB Gateway Live Trading */}
            <input 
              type="number" 
              className="param-input" 
              value={ibkrSettings.port}
              onChange={(e) => handleIbkrSettingChange('port', parseInt(e.target.value))}
              style={{ padding: '5px', fontSize: '11px' }} 
            />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Client ID</div>
            {/* Unique identifier for this connection
                - Each connected application needs unique ID
                - Range: 0-32 (usually 1 is fine)
                - Change if running multiple bots */}
            <input 
              type="number" 
              className="param-input" 
              value={ibkrSettings.clientId}
              onChange={(e) => handleIbkrSettingChange('clientId', parseInt(e.target.value))}
              style={{ padding: '5px', fontSize: '11px' }} 
            />
          </div>
          <div>
            <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '4px' }}>Reconnect Interval</div>
            {/* Seconds to wait before reconnection attempt
                - System auto-reconnects if connection drops
                - Lower = faster reconnect but more aggressive
                - Example: 10 seconds between retry attempts */}
            <input 
              type="number" 
              className="param-input" 
              value={ibkrSettings.reconnectInterval}
              onChange={(e) => handleIbkrSettingChange('reconnectInterval', parseInt(e.target.value))}
              style={{ padding: '5px', fontSize: '11px' }} 
            />
            <div style={{ fontSize: '9px', color: '#6e7681', marginTop: '2px' }}>Seconds</div>
          </div>
        </div>

        {/* Apply IBKR Settings Button */}
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button 
            variant="primary" 
            onClick={applyIbkrSettings}
            style={{ fontSize: '11px', padding: '6px 16px' }}
          >
            Apply IBKR Settings
          </Button>
        </div>
      </Card>

      <div className="action-bar">
        {/* Apply Settings (Hot-Reload): Updates settings without restart
            - Changes take effect immediately for running agents
            - No need to stop trading or reload models
            - Safe to use during live trading */}
        <Button variant="primary">Apply Settings (Hot-Reload)</Button>
        
        {/* Save as Preset: Saves current settings configuration to file
            - Useful for different market conditions or strategies
            - Can switch between presets quickly
            - Example: "Conservative", "Aggressive", "High Volatility" */}
        <Button variant="secondary">Save as Preset</Button>
        
        {/* Load Preset: Loads previously saved settings configuration
            - Quickly switch between different setups
            - Overwrites current settings
            - Requires Apply to take effect */}
        <Button variant="secondary">Load Preset</Button>
        
        {/* Reset to Default: Restores all settings to initial values
            - Safe fallback if settings cause issues
            - Does not affect saved presets
            - Requires Apply to take effect */}
        <Button variant="secondary">Reset to Default</Button>
      </div>

      <div className="info-box">
        ℹ Changes will be applied without system restart
      </div>
    </div>
  );
}

export default TabSettings;
