/**
 * IBKR Bridge Service
 * Handles all communication with the IBKR Bridge backend
 */

const BRIDGE_URL = 'http://localhost:5080';
const BACKEND_URL = 'http://localhost:8000';  // Backend API for database operations

class IBKRService {
  constructor() {
    this.isConnected = false;
    this.connectionListeners = [];
    this.accountListeners = [];
    this.portfolioListeners = [];
    this.checkInterval = null;
  }

  /**
   * Start monitoring connection and data updates
   */
  startMonitoring(intervalMs = 5000) {
    // Initial check
    this.checkConnection();
    
    // Periodic checks
    this.checkInterval = setInterval(() => {
      this.checkConnection();
      if (this.isConnected) {
        this.fetchAccountData();
        this.fetchPortfolio();
      }
    }, intervalMs);
  }

  /**
   * Stop monitoring
   */
  stopMonitoring() {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
  }

  /**
   * Check bridge and TWS connection status
   */
  async checkConnection() {
    try {
      const response = await fetch(`${BRIDGE_URL}/connection-status`, {
        method: 'GET',
        timeout: 2000
      });
      
      if (response.ok) {
        const data = await response.json();
        const wasConnected = this.isConnected;
        this.isConnected = data.isConnected === true;
        
        // Notify listeners if status changed
        if (wasConnected !== this.isConnected) {
          this.notifyConnectionListeners(this.isConnected);
        }
        
        return data;
      } else {
        this.setDisconnected();
        return { isConnected: false, message: 'Bridge not responding' };
      }
    } catch (error) {
      this.setDisconnected();
      return { isConnected: false, message: error.message };
    }
  }

  /**
   * Fetch account summary from IBKR
   */
  async fetchAccountData() {
    try {
      const response = await fetch(`${BRIDGE_URL}/account`, {
        method: 'GET',
        timeout: 10000
      });
      
      if (response.ok) {
        const data = await response.json();
        this.notifyAccountListeners(data);
        
        // Save equity point to database (via backend API)
        this.saveEquityPoint(data);
        
        return data;
      }
      return null;
    } catch (error) {
      console.error('Error fetching account data:', error);
      return null;
    }
  }

  /**
   * Save equity point to database
   * Sends account data to backend for persistence
   */
  async saveEquityPoint(accountData) {
    try {
      const metrics = this.parseAccountMetrics(accountData);
      if (!metrics) return;
      
      // Send to backend API to save in database
      await fetch(`${BACKEND_URL}/api/equity/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          net_liquidation: metrics.netLiquidation,
          buying_power: metrics.buyingPower,
          cash: metrics.totalCashValue,
          unrealized_pnl: metrics.unrealizedPnL,
          realized_pnl: metrics.realizedPnL,
          gross_position_value: metrics.grossPositionValue
        })
      });
    } catch (error) {
      // Silent fail - don't block UI if database save fails
      console.debug('Could not save equity point to database:', error);
    }
  }

  /**
   * Get equity history from database
   * Used to populate equity curve chart with historical data
   */
  async getEquityHistory(hours = null) {
    try {
      let url = `${BACKEND_URL}/api/equity/history`;
      if (hours) {
        url += `?hours=${hours}`;
      }
      
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        return data;
      }
      return [];
    } catch (error) {
      console.error('Error fetching equity history:', error);
      return [];
    }
  }

  /**
   * Fetch portfolio positions from IBKR
   */
  async fetchPortfolio() {
    try {
      const response = await fetch(`${BRIDGE_URL}/portfolio`, {
        method: 'GET',
        timeout: 10000
      });
      
      if (response.ok) {
        const data = await response.json();
        this.notifyPortfolioListeners(data);
        return data;
      }
      return [];
    } catch (error) {
      console.error('Error fetching portfolio:', error);
      return [];
    }
  }

  /**
   * Set disconnected state and notify listeners
   */
  setDisconnected() {
    if (this.isConnected) {
      this.isConnected = false;
      this.notifyConnectionListeners(false);
    }
  }

  /**
   * Subscribe to connection status changes
   */
  onConnectionChange(callback) {
    this.connectionListeners.push(callback);
    // Immediately notify with current status
    callback(this.isConnected);
    return () => {
      this.connectionListeners = this.connectionListeners.filter(cb => cb !== callback);
    };
  }

  /**
   * Subscribe to account data updates
   */
  onAccountUpdate(callback) {
    this.accountListeners.push(callback);
    return () => {
      this.accountListeners = this.accountListeners.filter(cb => cb !== callback);
    };
  }

  /**
   * Subscribe to portfolio updates
   */
  onPortfolioUpdate(callback) {
    this.portfolioListeners.push(callback);
    return () => {
      this.portfolioListeners = this.portfolioListeners.filter(cb => cb !== callback);
    };
  }

  /**
   * Notify connection listeners
   */
  notifyConnectionListeners(isConnected) {
    this.connectionListeners.forEach(callback => {
      try {
        callback(isConnected);
      } catch (error) {
        console.error('Error in connection listener:', error);
      }
    });
  }

  /**
   * Notify account listeners
   */
  notifyAccountListeners(data) {
    this.accountListeners.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error('Error in account listener:', error);
      }
    });
  }

  /**
   * Notify portfolio listeners
   */
  notifyPortfolioListeners(data) {
    this.portfolioListeners.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error('Error in portfolio listener:', error);
      }
    });
  }

  /**
   * Parse account data to extract key metrics
   */
  parseAccountMetrics(accountData) {
    if (!accountData) return null;

    const getValue = (key) => {
      return accountData[key]?.value || '0';
    };

    return {
      netLiquidation: parseFloat(getValue('NetLiquidation')),
      buyingPower: parseFloat(getValue('BuyingPower')),
      totalCashValue: parseFloat(getValue('TotalCashValue')),
      grossPositionValue: parseFloat(getValue('GrossPositionValue')),
      unrealizedPnL: parseFloat(getValue('UnrealizedPnL')),
      realizedPnL: parseFloat(getValue('RealizedPnL')),
      availableFunds: parseFloat(getValue('AvailableFunds')),
      excess: parseFloat(getValue('ExcessLiquidity')),
      currency: accountData['NetLiquidation']?.currency || 'USD',
      account: accountData['NetLiquidation']?.account || ''
    };
  }

  /**
   * Format portfolio data for UI
   */
  formatPortfolio(portfolioData) {
    if (!portfolioData || !Array.isArray(portfolioData)) return [];

    // Group by symbol and aggregate quantities
    const symbolMap = new Map();
    
    portfolioData.forEach(pos => {
      const symbol = pos.symbol;
      if (symbolMap.has(symbol)) {
        // Aggregate existing position
        const existing = symbolMap.get(symbol);
        const totalPosition = existing.position + pos.position;
        const totalCost = (existing.averageCost * existing.position) + (pos.average_cost * pos.position);
        const newAvgCost = totalPosition !== 0 ? totalCost / totalPosition : 0;
        
        symbolMap.set(symbol, {
          symbol: pos.symbol,
          position: totalPosition,
          averageCost: newAvgCost,
          marketPrice: pos.market_price,
          marketValue: existing.marketValue + pos.market_value,
          unrealizedPnL: existing.unrealizedPnL + pos.unrealized_pnl,
          currency: pos.currency,
          exchange: pos.exchange
        });
      } else {
        // New position
        symbolMap.set(symbol, {
          symbol: pos.symbol,
          position: pos.position,
          averageCost: pos.average_cost,
          marketPrice: pos.market_price,
          marketValue: pos.market_value,
          unrealizedPnL: pos.unrealized_pnl,
          currency: pos.currency,
          exchange: pos.exchange
        });
      }
    });

    return Array.from(symbolMap.values());
  }
}

// Create singleton instance
const ibkrService = new IBKRService();

export default ibkrService;
