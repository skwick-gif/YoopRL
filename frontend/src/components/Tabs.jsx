import React from 'react';

const Tabs = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'live', label: 'Live Trading' },
    { id: 'training', label: 'Training' },
    { id: 'simulation', label: 'Simulation' },
    { id: 'monitoring', label: 'Monitoring' },
    { id: 'logs', label: 'Logs' },
    { id: 'settings', label: 'Settings' }
  ];

  return (
    <div className="tabs">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={`tab ${activeTab === tab.id ? 'active' : ''}`}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
};

export default Tabs;
