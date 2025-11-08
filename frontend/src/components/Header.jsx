import React from 'react';

const Header = ({ onKillSwitch }) => {
  const statusItems = [
    { name: 'IBKR', active: true },
    { name: 'PPO', active: true },
    { name: 'SAC', active: true },
    { name: 'Watchdog', active: true }
  ];

  return (
    <div className="header">
      <h1>RL Trading System</h1>
      <div className="header-right">
        {statusItems.map((item) => (
          <div key={item.name} className="status-item">
            <div className={`dot ${!item.active ? 'off' : ''}`}></div>
            <span>{item.name}</span>
          </div>
        ))}
        <button className="kill-switch" onClick={onKillSwitch}>
          KILL
        </button>
      </div>
    </div>
  );
};

export default Header;
