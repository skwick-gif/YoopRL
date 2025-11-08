import React, { useState, useEffect } from 'react';
import './App.css';
import Tabs from './components/Tabs';
import TabDashboard from './components/TabDashboard';
import TabLiveTrading from './components/TabLiveTrading';
import TabTraining from './components/TabTraining';
import TabSimulation from './components/TabSimulation';
import TabMonitoring from './components/TabMonitoring';
import TabLogs from './components/TabLogs';
import TabSettings from './components/TabSettings';
import ibkrService from './services/IBKRService';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [ibkrConnected, setIbkrConnected] = useState(false);
  const [accountData, setAccountData] = useState(null);
  const [portfolioData, setPortfolioData] = useState([]);

  // Initialize IBKR service on mount
  useEffect(() => {
    // Subscribe to connection changes
    const unsubConnection = ibkrService.onConnectionChange((isConnected) => {
      setIbkrConnected(isConnected);
    });

    // Subscribe to account updates
    const unsubAccount = ibkrService.onAccountUpdate((data) => {
      setAccountData(data);
    });

    // Subscribe to portfolio updates
    const unsubPortfolio = ibkrService.onPortfolioUpdate((data) => {
      setPortfolioData(data);
    });

    // Start monitoring
    ibkrService.startMonitoring(5000); // Check every 5 seconds

    // Cleanup on unmount
    return () => {
      unsubConnection();
      unsubAccount();
      unsubPortfolio();
      ibkrService.stopMonitoring();
    };
  }, []);

  const handleKillSwitch = () => {
    if (window.confirm('Are you sure you want to KILL the system?')) {
      console.log('System killed!');
      // Add kill logic here
    }
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <TabDashboard accountData={accountData} portfolioData={portfolioData} />;
      case 'live':
        return <TabLiveTrading />;
      case 'training':
        return <TabTraining />;
      case 'simulation':
        return <TabSimulation />;
      case 'monitoring':
        return <TabMonitoring />;
      case 'logs':
        return <TabLogs />;
      case 'settings':
        return <TabSettings />;
      default:
        return <TabDashboard accountData={accountData} portfolioData={portfolioData} />;
    }
  };

  return (
    <div className="container">
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px', background: '#161b22', padding: '10px 16px', borderBottom: '1px solid #21262d', marginBottom: '12px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.2, alignItems: 'center' }}>
          <h1 style={{ fontSize: '16px', fontWeight: 700, color: '#c9d1d9', margin: 0 }}>YoopRL</h1>
          <span style={{ fontSize: '9px', color: '#8b949e', marginTop: '1px' }}>Trading System</span>
        </div>
        <div style={{ flex: 1 }}>
          <Tabs activeTab={activeTab} onTabChange={setActiveTab} />
        </div>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          {[
            { name: 'IBKR', active: ibkrConnected },
            { name: 'PPO', active: false },
            { name: 'SAC', active: false },
            { name: 'Watchdog', active: false }
          ].map((item) => (
            <div key={item.name} className="status-item">
              <div className={`dot ${!item.active ? 'off' : ''}`}></div>
              <span>{item.name}</span>
            </div>
          ))}
          <button className="kill-switch" onClick={handleKillSwitch}>
            KILL
          </button>
        </div>
      </div>
      <div className="tab-content active">
        {renderTabContent()}
      </div>
    </div>
  );
}

export default App;
