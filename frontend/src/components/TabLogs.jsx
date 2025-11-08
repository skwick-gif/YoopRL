import React from 'react';
import { Button, Card, InputGroup } from './common/UIComponents';

function TabLogs() {
  const logs = [
    { time: '2024-11-07 14:23:15', component: 'PPO', level: 'info', message: 'Action executed: BUY AAPL 50 shares @ $178.50' },
    { time: '2024-11-07 14:23:14', component: 'PPO', level: 'debug', message: 'Confidence score: 0.85' },
    { time: '2024-11-07 14:23:14', component: 'PPO', level: 'debug', message: 'Features: RSI=32.1, MACD=0.45, Volume=high' },
    { time: '2024-11-07 14:23:13', component: 'IBKR', level: 'success', message: 'Order confirmed: BUY 50 AAPL' },
    { time: '2024-11-07 14:23:12', component: 'API', level: 'info', message: 'Request received: execute_trade' },
    { time: '2024-11-07 14:05:32', component: 'IBKR', level: 'warning', message: 'Latency spike detected: 52ms' },
    { time: '2024-11-07 13:45:20', component: 'SAC', level: 'info', message: 'Action executed: SELL TNA 100 shares @ $42.80' },
    { time: '2024-11-07 13:45:19', component: 'SAC', level: 'debug', message: 'Risk-adjusted decision: SELL' },
    { time: '2024-11-07 11:23:45', component: 'IBKR', level: 'error', message: 'Connection lost, attempting reconnect...' },
    { time: '2024-11-07 11:23:55', component: 'IBKR', level: 'success', message: 'Connection restored' },
    { time: '2024-11-07 09:30:01', component: 'Watchdog', level: 'info', message: 'All systems healthy' },
    { time: '2024-11-07 09:30:00', component: 'API', level: 'info', message: 'System startup complete' }
  ];

  const getLevelClass = (level) => {
    switch (level) {
      case 'success': return 'log-success';
      case 'error': return 'log-error';
      case 'warning': return 'log-warning';
      case 'debug': return 'log-debug';
      default: return 'log-info';
    }
  };

  return (
    <div>
      <Card style={{ marginBottom: '12px' }}>
        <div className="control-title">Log Filter</div>
        <InputGroup>
          {/* Component Filter: Show logs from specific system components
              - All Components: Shows everything
              - API: Backend API requests/responses
              - IBKR Bridge: Broker connection and order execution
              - PPO/SAC Agent: Agent decisions and reasoning
              - Watchdog: System health monitoring */}
          <select className="param-input">
            <option>All Components</option>
            <option>API</option>
            <option>IBKR Bridge</option>
            <option>PPO Agent</option>
            <option>SAC Agent</option>
            <option>Watchdog</option>
          </select>
          
          {/* Log Level Filter: Filter by severity
              - Error: Critical issues requiring attention
              - Warning: Potential problems (latency, reconnects)
              - Info: Normal operations (trades, system events)
              - Debug: Detailed diagnostic information */}
          <select className="param-input">
            <option>All Levels</option>
            <option>Error</option>
            <option>Warning</option>
            <option>Info</option>
            <option>Debug</option>
          </select>
          
          {/* Search Box: Find specific log messages by text */}
          <input type="text" className="param-input" placeholder="Search logs..." />
          
          {/* Filter Button: Applies selected filters to log display */}
          <Button className="">Filter</Button>
        </InputGroup>
      </Card>

      <Card>
        <div className="log-display">
          {logs.map((log, index) => (
            <div key={index} className={getLevelClass(log.level)}>
              [{log.time}] [{log.component}] {log.level.toUpperCase()}: {log.message}
            </div>
          ))}
        </div>
      </Card>

      <div className="action-bar">
        {/* Download Logs Button: Exports all logs to a text file
            - Saves complete log history to .txt or .csv file
            - Useful for debugging, auditing, or sharing with support
            - Includes all filtered or unfiltered logs */}
        <Button variant="secondary">Download Logs</Button>
        
        {/* Clear Display Button: Removes all logs from screen
            - Only clears the visual display, logs still saved in backend
            - Useful to reduce clutter and focus on new events
            - Does not affect log files on disk */}
        <Button variant="secondary">Clear Display</Button>
        
        {/* Auto-Scroll Button: Automatically scrolls to newest logs
            - Toggle on: Display follows newest log entries in real-time
            - Toggle off: Manual scrolling to review older logs
            - Helpful during active trading to monitor live events */}
        <Button variant="secondary">Auto-Scroll</Button>
      </div>
    </div>
  );
}

export default TabLogs;
