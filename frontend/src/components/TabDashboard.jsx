import React, { useMemo, useState, useEffect } from 'react';
import { Card, Button, Switch } from './common/UIComponents';
import ibkrService from '../services/IBKRService';

function TabDashboard({ accountData, portfolioData }) {
  const [equityHistory, setEquityHistory] = useState([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);

  // Load historical equity data from database on mount
  // This ensures we don't lose data when restarting the application
  useEffect(() => {
    const loadHistory = async () => {
      try {
        const history = await ibkrService.getEquityHistory(24 * 7); // Last 7 days
        if (history && history.length > 0) {
          const formatted = history.map(point => ({
            time: point.timestamp * 1000, // Convert to milliseconds
            value: point.net_liquidation,
            label: new Date(point.timestamp * 1000).toLocaleTimeString('en-US', { 
              hour: '2-digit', 
              minute: '2-digit' 
            })
          }));
          setEquityHistory(formatted);
          console.log(`Loaded ${formatted.length} historical equity points from database`);
        }
      } catch (error) {
        console.error('Error loading equity history:', error);
      } finally {
        setHistoryLoaded(true);
      }
    };

    loadHistory();
  }, []);

  // Track equity over time - builds historical equity curve from account Net Liquidation
  // This creates a live chart showing portfolio value changes
  // Note: Data is automatically saved to database by IBKRService
  useEffect(() => {
    if (!accountData || !historyLoaded) return;
    
    const metrics = ibkrService.parseAccountMetrics(accountData);
    if (!metrics) return;

    const now = new Date();
    const newPoint = {
      time: now.getTime(),  // Timestamp for x-axis
      value: metrics.netLiquidation,  // Portfolio total value (cash + positions)
      label: now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
    };

    setEquityHistory(prev => {
      // Check if this is a duplicate (same minute)
      if (prev.length > 0) {
        const lastPoint = prev[prev.length - 1];
        const timeDiff = (now.getTime() - lastPoint.time) / 1000; // seconds
        if (timeDiff < 4) {
          // Skip if less than 4 seconds (prevent duplicates)
          return prev;
        }
      }

      const updated = [...prev, newPoint];
      
      // Keep last 365 days of data (assuming 5-second updates from IBKR service)
      // 365 days * 24 hours * 60 minutes * 60 seconds / 5 second interval = ~6.3M points
      // For memory efficiency, we'll keep 1 year at 5-second intervals
      const maxPoints = (365 * 24 * 60 * 60) / 5; // 1 year worth of 5-second intervals
      
      if (updated.length > maxPoints) {
        // Remove oldest points to stay within limit
        return updated.slice(-maxPoints);
      }
      return updated;
    });
  }, [accountData, historyLoaded]);
  // Parse account metrics
  const accountMetrics = useMemo(() => {
    return ibkrService.parseAccountMetrics(accountData);
  }, [accountData]);

  // Format portfolio
  const formattedPortfolio = useMemo(() => {
    return ibkrService.formatPortfolio(portfolioData);
  }, [portfolioData]);

  // Calculate metrics for display
  const metrics = useMemo(() => {
    if (!accountMetrics) {
      return [
        { label: 'P&L Today', value: '--', sub: 'No data' },
        { label: 'Portfolio Value', value: '--', sub: 'No data' },
        { label: 'Win Rate', value: '--', sub: 'No data' },
        { label: 'Sharpe Ratio', value: '--', sub: 'No data' },
        { label: 'Max Drawdown', value: '--', sub: 'No data' }
      ];
    }

    const unrealizedPnL = accountMetrics.unrealizedPnL || 0;
    const realizedPnL = accountMetrics.realizedPnL || 0;
    const totalPnL = unrealizedPnL + realizedPnL;
    const netLiq = accountMetrics.netLiquidation || 0;
    const startValue = netLiq - totalPnL;
    const pnlPercent = startValue !== 0 ? ((totalPnL / startValue) * 100).toFixed(2) : '0.00';
    const isPositive = totalPnL >= 0;

    return [
      { 
        label: 'P&L Today', 
        value: `${isPositive ? '+' : ''}$${totalPnL.toFixed(2)}`, 
        sub: `${isPositive ? '+' : ''}${pnlPercent}%`, 
        positive: isPositive,
        negative: !isPositive
      },
      { 
        label: 'Portfolio Value', 
        value: `$${netLiq.toLocaleString('en-US', { maximumFractionDigits: 2 })}`, 
        sub: `Cash: $${(accountMetrics.totalCashValue || 0).toLocaleString('en-US', { maximumFractionDigits: 2 })}`
      },
      { label: 'Win Rate', value: '--', sub: 'Coming soon' },
      { label: 'Sharpe Ratio', value: '--', sub: 'Coming soon' },
      { label: 'Max Drawdown', value: '--', sub: 'Coming soon' }
    ];
  }, [accountMetrics]);

  // Format positions for display
  const positions = useMemo(() => {
    if (!formattedPortfolio || formattedPortfolio.length === 0) {
      return [];
    }

    return formattedPortfolio.map(pos => {
      const costBasis = pos.averageCost * pos.position;
      const currentValue = pos.marketValue;
      const pnlDollar = currentValue - costBasis;
      const pnlPercent = costBasis !== 0 ? ((pnlDollar / costBasis) * 100) : 0;
      const isPositive = pnlDollar >= 0;

      return {
        symbol: pos.symbol,
        shares: pos.position,
        avgCost: pos.averageCost,
        currentPrice: pos.marketPrice,
        costBasis: costBasis,
        currentValue: currentValue,
        pnlDollar: pnlDollar,
        pnlPercent: pnlPercent,
        positive: isPositive
      };
    });
  }, [formattedPortfolio]);

  const recentTrades = [];

  return (
    <div>
      <div className="grid">
        {metrics.map((metric, index) => (
          <Card
            key={index}
            label={metric.label}
            value={metric.value}
            sub={metric.sub}
            positive={metric.positive}
            negative={metric.negative}
          />
        ))}
      </div>

      <div className="main-layout">
        <div className="chart-area">
          {/* Equity Curve: Real-time portfolio value chart
              - Updates every 5 seconds from IBKR account data
              - Shows Net Liquidation (total portfolio value)
              - Stores up to 1 year of historical data
              - Blue line represents equity, dots show current value */}
          <div className="chart-title">Equity Curve - Live (1 Year History)</div>
          <div className="chart-placeholder" style={{ position: 'relative', padding: '20px' }}>
            {equityHistory.length === 0 ? (
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center', 
                height: '100%',
                color: '#8b949e',
                fontSize: '12px'
              }}>
                Collecting equity data...
              </div>
            ) : (
              <svg width="100%" height="100%" viewBox="0 0 800 200" preserveAspectRatio="none">
                {/* Grid lines - horizontal reference lines for better readability */}
                {[0, 1, 2, 3, 4].map(i => (
                  <line
                    key={i}
                    x1="0"
                    y1={i * 50}
                    x2="800"
                    y2={i * 50}
                    stroke="#21262d"
                    strokeWidth="1"
                  />
                ))}
                
                {/* Equity line - main chart showing portfolio value over time
                    X-axis: time progression (left=oldest, right=newest)
                    Y-axis: portfolio value (auto-scaled to min/max range) */}
                <polyline
                  fill="none"
                  stroke="#58a6ff"
                  strokeWidth="2"
                  points={equityHistory.map((point, index) => {
                    // Calculate X position: spread points evenly across width
                    const x = (index / Math.max(equityHistory.length - 1, 1)) * 800;
                    
                    // Calculate Y position: scale value to fit chart height
                    const minValue = Math.min(...equityHistory.map(p => p.value));
                    const maxValue = Math.max(...equityHistory.map(p => p.value));
                    const range = maxValue - minValue || 1;
                    // Invert Y (SVG y=0 is top) and add padding
                    const y = 200 - ((point.value - minValue) / range) * 180 - 10;
                    return `${x},${y}`;
                  }).join(' ')}
                />
                
                {/* Current value marker - dot at the end of the line showing latest value */}
                {equityHistory.length > 0 && (() => {
                  const lastPoint = equityHistory[equityHistory.length - 1];
                  const minValue = Math.min(...equityHistory.map(p => p.value));
                  const maxValue = Math.max(...equityHistory.map(p => p.value));
                  const range = maxValue - minValue || 1;
                  const y = 200 - ((lastPoint.value - minValue) / range) * 180 - 10;
                  return (
                    <circle
                      cx="800"
                      cy={y}
                      r="4"
                      fill="#58a6ff"
                    />
                  );
                })()}
              </svg>
            )}
            
            {/* Stats overlay - shows key metrics about the equity curve
                Current: Latest portfolio value
                Start: First recorded value (session start or oldest stored)
                Points: Number of data points collected (useful for debugging) */}
            {equityHistory.length > 0 && (
              <div style={{
                position: 'absolute',
                top: '10px',
                left: '10px',
                background: 'rgba(13, 17, 23, 0.8)',
                padding: '8px 12px',
                borderRadius: '4px',
                fontSize: '11px',
                color: '#c9d1d9'
              }}>
                <div>Current: ${equityHistory[equityHistory.length - 1].value.toLocaleString('en-US', { maximumFractionDigits: 2 })}</div>
                <div style={{ marginTop: '4px', color: '#8b949e' }}>
                  Start: ${equityHistory[0].value.toLocaleString('en-US', { maximumFractionDigits: 2 })}
                </div>
                <div style={{ marginTop: '4px', color: '#8b949e' }}>
                  Points: {equityHistory.length}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="controls">
          <div className="control-card">
            <div className="control-title">System Control</div>
            <div className="btn-group">
              {/* Start Button: Activates all trained agents to begin live trading
                  - Agents start analyzing market data and making trade decisions
                  - Requires trained models to be loaded
                  - Connects to IBKR for live market data and order execution */}
              <Button className="start">Start</Button>
              
              {/* Pause Button: Temporarily stops trading without closing positions
                  - Agents stop making new trades but keep existing positions open
                  - Can resume later without reloading models
                  - Useful for market volatility or news events */}
              <Button className="pause">Pause</Button>
              
              {/* Stop Button: Completely stops all trading activity
                  - Closes all open positions (if configured)
                  - Stops all agents
                  - Requires restart to resume trading */}
              <Button className="stop">Stop</Button>
            </div>
            
            {/* Paper Trading Toggle: Simulates trading without real money
                - Uses live market data but no real orders sent to broker
                - Perfect for testing strategies before going live
                - All P&L and positions are simulated */}
            <Switch label="Paper Trading" checked={false} onChange={() => {}} />
          </div>

          <div className="control-card">
            <div className="control-title">Risk Limits</div>
            {/* Max Drawdown: Maximum allowed portfolio loss from peak value
                - System automatically stops trading if exceeded
                - Calculated as: (Peak Value - Current Value) / Peak Value
                - Set in Settings tab */}
            <div style={{ fontSize: '11px', marginBottom: '6px' }}>
              <span style={{ color: '#8b949e' }}>Max Drawdown:</span>
              <span style={{ float: 'right' }}>15%</span>
            </div>
            {/* Max Position: Maximum dollar amount allowed per single position
                - Prevents over-concentration in one stock
                - Risk management to limit exposure
                - Set in Settings tab */}
            <div style={{ fontSize: '11px' }}>
              <span style={{ color: '#8b949e' }}>Max Position:</span>
              <span style={{ float: 'right' }}>$10,000</span>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '12px' }}>
        <div className="positions">
          <div className="control-title" style={{ marginBottom: '8px' }}>Open Positions ({positions.length})</div>
          {positions.length === 0 ? (
            <div style={{ padding: '20px', textAlign: 'center', color: '#8b949e', fontSize: '12px' }}>
              No open positions
            </div>
          ) : (
            <>
              {/* Header */}
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: '80px 60px 90px 90px 90px 80px',
                gap: '8px',
                padding: '6px 8px',
                background: '#161b22',
                borderRadius: '4px',
                fontSize: '10px',
                fontWeight: 600,
                color: '#c9d1d9',
                marginBottom: '4px'
              }}>
                <div>Symbol</div>
                <div style={{ textAlign: 'right' }}>Qty</div>
                <div style={{ textAlign: 'right' }}>Avg Cost</div>
                <div style={{ textAlign: 'right' }}>Current</div>
                <div style={{ textAlign: 'right' }}>Value</div>
                <div style={{ textAlign: 'right' }}>P&L</div>
              </div>
              
              {/* Rows */}
              {positions.map((position, index) => (
                <div key={index} style={{ 
                  display: 'grid', 
                  gridTemplateColumns: '80px 60px 90px 90px 90px 80px',
                  gap: '8px',
                  padding: '8px',
                  background: '#0d1117',
                  borderRadius: '4px',
                  marginBottom: '4px',
                  fontSize: '11px',
                  alignItems: 'center'
                }}>
                  <div style={{ fontWeight: 600, color: '#58a6ff' }}>{position.symbol}</div>
                  <div style={{ textAlign: 'right', color: '#c9d1d9' }}>{position.shares}</div>
                  <div style={{ textAlign: 'right', color: '#c9d1d9' }}>${position.avgCost.toFixed(2)}</div>
                  <div style={{ textAlign: 'right', color: '#c9d1d9' }}>${position.currentPrice.toFixed(2)}</div>
                  <div style={{ textAlign: 'right', color: '#c9d1d9' }}>${position.currentValue.toFixed(2)}</div>
                  <div style={{ 
                    textAlign: 'right',
                    background: position.positive ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                    padding: '6px 8px',
                    borderRadius: '4px',
                    border: position.positive ? '1px solid rgba(63, 185, 80, 0.3)' : '1px solid rgba(248, 81, 73, 0.3)'
                  }}>
                    <div className={position.positive ? 'pos' : 'neg'} style={{ fontWeight: 600 }}>
                      {position.positive ? '+' : ''}{position.pnlPercent.toFixed(2)}%
                    </div>
                    <div className={position.positive ? 'pos' : 'neg'} style={{ fontSize: '10px', marginTop: '1px' }}>
                      {position.positive ? '+$' : '-$'}{Math.abs(position.pnlDollar).toFixed(2)}
                    </div>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>

        <div className="trades-table">
          <div className="control-title">Recent Trades</div>
          {recentTrades.length === 0 ? (
            <div style={{ padding: '20px', textAlign: 'center', color: '#8b949e', fontSize: '12px' }}>
              No recent trades
            </div>
          ) : (
            <>
              <div className="table-header">
                <span>Time</span>
                <span>Symbol</span>
                <span>Action</span>
                <span>Price</span>
                <span>P&L</span>
              </div>
              {recentTrades.map((trade, index) => (
                <div key={index} className="table-row">
                  <span>{trade.time}</span>
                  <span>{trade.symbol}</span>
                  <span>{trade.action}</span>
                  <span>{trade.price}</span>
                  <span style={{
                    color: trade.status === 'profit' ? '#3fb950' :
                           trade.status === 'loss' ? '#f85149' : '#6e7681'
                  }}>
                    {trade.pnl}
                  </span>
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default TabDashboard;
