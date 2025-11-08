using InterReact;
using Microsoft.Extensions.Logging;
using System.Reactive.Linq;
using System.Reactive.Threading.Tasks;
using System.Text.Json;
using System.IO;
using InterReactBridge.Models;

namespace InterReactBridge.Services;

public class IbService
{
    private readonly ILogger<IbService> _logger;
    private readonly TwsConnectionService _twsConnection;
    private IInterReactClient? _client;
    private string? _accountCode;

    public IbService(ILogger<IbService> logger, TwsConnectionService twsConnection)
    {
        _logger = logger;
        _twsConnection = twsConnection;
    }

    public async Task<bool> ConnectAsync(string host, int port, int clientId)
    {
        try
        {
            _logger.LogInformation("=== Starting IBKR connection attempt ===");
            _logger.LogInformation("Host: {Host}, Port: {Port}, ClientId: {ClientId}", host, port, clientId);
            
            _logger.LogInformation("Creating InterReactClient...");
            _client = await InterReactClient.ConnectAsync(options =>
            {
                options.TwsIpAddress = System.Net.IPAddress.Parse(host);
                options.IBPortAddresses = new[] { port };
                options.TwsClientId = clientId;
                _logger.LogInformation("Options set: IP={IP}, Port={Port}, ClientId={ClientId}", 
                    options.TwsIpAddress, port, clientId);
            });
            
            _logger.LogInformation("InterReactClient created successfully");

            _logger.LogInformation("InterReactClient created successfully");

            // Try to get managed accounts with timeout
            try
            {
                _logger.LogInformation("Waiting for ManagedAccounts response...");
                var cts = new CancellationTokenSource(5000);
                var managedAccounts = await _client.Response.OfType<ManagedAccounts>().FirstAsync().ToTask(cts.Token);
                _accountCode = managedAccounts.Accounts.Split(',')[0];
                _logger.LogInformation("Connected to IBKR at {Host}:{Port}, Account: {Account}", host, port, _accountCode);
            }
            catch (Exception accountEx)
            {
                _accountCode = null;
                _logger.LogWarning(accountEx, "Could not retrieve account code within timeout");
                _logger.LogInformation("Connected to IBKR at {Host}:{Port}, no account code received", host, port);
            }

            await WriteConnectionStatusAsync(new { connected = true, host, port, clientId, account = _accountCode, time = DateTime.UtcNow });
            _logger.LogInformation("=== Connection successful ===");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "=== Failed to connect to IBKR ===");
            _logger.LogError("Error Type: {Type}", ex.GetType().Name);
            _logger.LogError("Error Message: {Message}", ex.Message);
            if (ex.InnerException != null)
            {
                _logger.LogError("Inner Exception: {InnerMessage}", ex.InnerException.Message);
            }
            await WriteConnectionStatusAsync(new { connected = false, host, port, clientId, error = ex.Message, time = DateTime.UtcNow });
            return false;
        }
    }

    public async Task<object> GetAccountSummary(string? account = null)
    {
        var client = _twsConnection.GetClient();
        if (client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            _logger.LogInformation("Requesting account summary using RequestAccountSummary...");

            // Debounce to avoid frequent account summary requests
            // If TwsConnectionService is already broadcasting account summary periodically, cancel it first
            try { _twsConnection.CancelActiveAccountSummary(); } catch { }

            // Use direct RequestAccountSummary approach
            var summaries = new List<AccountSummary>();

            // Use client's next id to avoid collisions
            var requestId = client.Request.GetNextId();

            var sub = client.Response.OfType<AccountSummary>()
                .Where(s => s.RequestId == requestId)
                .Subscribe(summary =>
                {
                    _logger.LogInformation("Received AccountSummary: Account={Account}, Tag={Tag}, Value={Value}, Currency={Currency}",
                        summary.Account, summary.Tag, summary.Value, summary.Currency);
                    summaries.Add(summary);
                });

            // request for all accounts then filter locally if an account was specified
            client.Request.RequestAccountSummary(requestId, "All");

            // Wait for summaries (debounced)
            await Task.Delay(5000);

            sub.Dispose();

            try { client.Request.CancelAccountSummary(requestId); } catch { }

            _logger.LogInformation("Received {Count} account summary items", summaries.Count);
            // persist a small audit file with count
            await WriteConnectionStatusAsync(new { accountSummaryCount = summaries.Count, time = DateTime.UtcNow });

            // Convert to dictionary grouped by Tag for UI compatibility
            // UI expects: { "NetLiquidation": { value: "123", currency: "USD", account: "U123" } }
            var dictionary = new Dictionary<string, object>();
            foreach (var item in summaries)
            {
                // If caller requested a specific account, skip other accounts
                if (!string.IsNullOrEmpty(account) && !string.Equals(item.Account, account, StringComparison.OrdinalIgnoreCase)) continue;
                var tag = string.IsNullOrEmpty(item.Tag) ? "Unknown" : item.Tag;
                
                // If tag already exists (multiple accounts), keep the first one
                // TODO: In future, support multiple accounts by returning array per tag
                if (!dictionary.ContainsKey(tag))
                {
                    dictionary[tag] = new
                    {
                        value = string.IsNullOrEmpty(item.Value) ? "0" : item.Value,
                        currency = string.IsNullOrEmpty(item.Currency) ? "USD" : item.Currency,
                        account = string.IsNullOrEmpty(item.Account) ? "Unknown" : item.Account
                    };
                }
            }
            
            _logger.LogInformation("Converted to dictionary with {Count} unique tags", dictionary.Count);
            return dictionary;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting account summary");
            throw;
        }
    }

    public async Task<object> GetPortfolio(string? account = null)
    {
        var client = _twsConnection.GetClient();
        if (client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            _logger.LogInformation("Requesting portfolio using PortfolioValue messages...");
            
            var portfolioItems = new List<PortfolioValue>();
            
            // Subscribe to PortfolioValue messages which contain market prices
            var sub = client.Response.OfType<PortfolioValue>()
                .Subscribe(pv => 
                {
                    _logger.LogInformation("Received PortfolioValue: Account={Account}, Symbol={Symbol}, Position={Position}, AverageCost={AverageCost}, MarketPrice={MarketPrice}, MarketValue={MarketValue}, UnrealizedPnl={UnrealizedPnl}", 
                        pv.Account, pv.Contract.Symbol, pv.Position, pv.AverageCost, pv.MarketPrice, pv.MarketValue, pv.UnrealizedPnl);
                    
                    // Filter by account if specified
                    if (!string.IsNullOrEmpty(account) && !string.Equals(pv.Account, account, StringComparison.OrdinalIgnoreCase)) return;
                    
                    portfolioItems.Add(pv);
                });

            try
            {
                // Request account updates which triggers PortfolioValue messages
                if (!string.IsNullOrEmpty(account))
                {
                    client.Request.RequestAccountUpdates(true, account);
                }
                else
                {
                    client.Request.RequestAccountUpdates(true, "");
                }
                
                // Wait for portfolio data
                await Task.Delay(3000);
                
                // Stop updates
                client.Request.RequestAccountUpdates(false, account ?? "");
                sub.Dispose();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error requesting account updates");
                try { sub.Dispose(); } catch { }
                throw;
            }

            _logger.LogInformation("Received {Count} portfolio items", portfolioItems.Count);

            // Return with real market data
            return portfolioItems.Select(p => new
            {
                account = string.IsNullOrEmpty(p.Account) ? "Unknown" : p.Account,
                symbol = string.IsNullOrEmpty(p.Contract.Symbol) ? "Unknown" : p.Contract.Symbol,
                security_type = string.IsNullOrEmpty(p.Contract.SecurityType) ? "Unknown" : p.Contract.SecurityType,
                exchange = string.IsNullOrEmpty(p.Contract.Exchange) ? "Unknown" : p.Contract.Exchange,
                currency = string.IsNullOrEmpty(p.Contract.Currency) ? "USD" : p.Contract.Currency,
                position = p.Position,
                average_cost = p.AverageCost,
                market_price = p.MarketPrice,
                market_value = p.MarketValue,
                unrealized_pnl = p.UnrealizedPnl
            }).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting portfolio");
            throw;
        }
    }

    public async Task<object> GetMarketData(string symbol, string secType, string exchange, TimeSpan duration)
    {
        var client = _twsConnection.GetClient();
        if (client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            var contract = new Contract()
            {
                Symbol = symbol,
                SecurityType = secType,
                Exchange = exchange,
                Currency = "USD"
            };

            var ticks = new List<object>();

            var sub = client.Service
                .CreateMarketDataObservable(contract)
                .OfTickClass(selector => selector.PriceTick)
                .Subscribe(pt =>
                {
                    // collect basic tick info
                    ticks.Add(new {
                        RequestId = pt.RequestId,
                        TickType = pt.TickType.ToString(),
                        Price = pt.Price,
                        Time = DateTime.UtcNow
                    });
                });

            await Task.Delay(duration);

            sub.Dispose();

            return ticks;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting market data");
            throw;
        }
    }

    /// <summary>
    /// Place a simple order (BUY/SELL) for stocks. Market by default, supports Limit/Stop with prices.
    /// </summary>
    public async Task<object> PlaceOrder(
        string symbol,
        string action,
        int quantity,
        string orderType = "MKT",
        decimal? limitPrice = null,
        decimal? stopPrice = null,
        string secType = "STK",
        string exchange = "SMART")
    {
        var client = _twsConnection.GetClient();
        if (client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            // Build contract
            var contract = new Contract()
            {
                Symbol = symbol,
                SecurityType = secType,
                Exchange = exchange,
                Currency = "USD"
            };

            // Build order
            var order = new Order()
            {
                Action = action?.ToUpperInvariant() == "SELL" ? "SELL" : "BUY",
                OrderType = string.IsNullOrWhiteSpace(orderType) ? "MKT" : orderType.ToUpperInvariant(),
                TotalQuantity = quantity,
            };
            // Best-effort mapping for price fields via reflection (InterReact Order may expose different names)
            var otype = order.OrderType;
            var oprops = order.GetType().GetProperties();
            if (otype == "LMT" && limitPrice.HasValue)
            {
                var priceProp = order.GetType().GetProperty("LmtPrice")
                               ?? order.GetType().GetProperty("LimitPrice")
                               ?? order.GetType().GetProperty("Price");
                if (priceProp != null && priceProp.CanWrite)
                {
                    priceProp.SetValue(order, Convert.ToDouble(limitPrice.Value));
                }
            }
            if ((otype == "STP" || otype == "STP LMT" || otype == "STOP") && stopPrice.HasValue)
            {
                var stopProp = order.GetType().GetProperty("AuxPrice")
                              ?? order.GetType().GetProperty("StopPrice")
                              ?? order.GetType().GetProperty("TriggerPrice");
                if (stopProp != null && stopProp.CanWrite)
                {
                    stopProp.SetValue(order, Convert.ToDouble(stopPrice.Value));
                }
            }

            // Place the order
            var orderId = client.Request.GetNextId();
            client.Request.PlaceOrder(orderId, order, contract);
            _logger.LogInformation("Placed order: Id={OrderId}, {Action} {Qty} {Symbol} {Type}", orderId, order.Action, quantity, symbol, order.OrderType);

            // Optionally wait briefly to let IBKR acknowledge
            await Task.Delay(200);
            return new { success = true, orderId = orderId };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing order {Action} {Qty} {Symbol}", action, quantity, symbol);
            return new { success = false, error = ex.Message };
        }
    }

    /// <summary>
    /// Get delayed market prices for calculating indicators
    /// IBKR provides delayed data (15-20 minutes delay for free accounts)
    /// We'll collect price ticks over time to build a price series
    /// </summary>
    /// <param name="symbol">Stock symbol (e.g., "AAPL")</param>
    /// <param name="secType">Security type (e.g., "STK" for stock)</param>
    /// <param name="exchange">Exchange (e.g., "SMART")</param>
    /// <param name="sampleSeconds">How many seconds to collect ticks (default 30)</param>
    /// <returns>List of price points sampled over time</returns>
    public async Task<PriceSeriesResponse> GetDelayedPriceSeries(
        string symbol, 
        string secType, 
        string exchange, 
        int sampleSeconds = 30)
    {
        var client = _twsConnection.GetClient();
        if (client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            _logger.LogInformation("Requesting delayed price series: {Symbol}, Duration={Duration}s", 
                symbol, sampleSeconds);

            var contract = new Contract()
            {
                Symbol = symbol,
                SecurityType = secType,
                Exchange = exchange,
                Currency = "USD"
            };

            var prices = new List<PricePoint>();
            var lastPrice = 0.0;

            var sub = client.Service
                .CreateMarketDataObservable(contract)
                .OfTickClass(selector => selector.PriceTick)
                .Subscribe(pt =>
                {
                    // Collect price ticks
                    if (pt.Price > 0 && pt.Price != lastPrice)
                    {
                        lastPrice = pt.Price;
                        prices.Add(new PricePoint
                        {
                            Price = pt.Price,
                            TickType = pt.TickType.ToString(),
                            Time = DateTime.UtcNow
                        });
                        _logger.LogDebug("Price tick: {Price} at {Time}", pt.Price, DateTime.UtcNow);
                    }
                });

            // Collect ticks for the specified duration
            await Task.Delay(TimeSpan.FromSeconds(sampleSeconds));

            sub.Dispose();

            _logger.LogInformation("Collected {Count} price points for {Symbol}", prices.Count, symbol);

            return new PriceSeriesResponse
            {
                Symbol = symbol,
                SampleSeconds = sampleSeconds,
                PricesCount = prices.Count,
                Prices = prices,
                Note = "Delayed market data from IBKR (15-20 minutes delay for free accounts). Limited to ticks received during sample period."
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting delayed price series for {Symbol}", symbol);
            throw;
        }
    }

    /// <summary>
    /// Request historical bars (OHLC) from TWS for a given symbol and duration.
    /// </summary>
    public async Task<object> GetHistoricalBars(string symbol, string secType, string exchange, int durationDays = 7, string barSizeSetting = "1 day")
    {
        var client = _twsConnection.GetClient();
        if (client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            _logger.LogInformation("Requesting historical bars: {Symbol}, DurationDays={Days}, BarSize={BarSize}", symbol, durationDays, barSizeSetting);

            var contract = new Contract()
            {
                Symbol = symbol,
                SecurityType = secType,
                Exchange = exchange,
                Currency = "USD"
            };

            var bars = new List<object>();

            // Try to resolve contract details first to get ConId (improves matching)
            try
            {
                var contractDetail = await GetContractDetails(symbol, secType, exchange);
                if (contractDetail != null)
                {
                    // if we found ConId, set it on contract
                    if (contractDetail.TryGetValue("conId", out var conIdObj) && conIdObj is int conId)
                    {
                        // Use reflection in case the Contract type does not expose ConId directly
                        var prop = contract.GetType().GetProperty("ConId") ?? contract.GetType().GetProperty("conId");
                        if (prop != null && prop.CanWrite)
                        {
                            prop.SetValue(contract, Convert.ChangeType(conId, prop.PropertyType));
                        }
                        _logger.LogInformation("Resolved ConId={ConId} for {Symbol}", conId, symbol);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Contract details lookup failed for {Symbol}, continuing without ConId", symbol);
            }

            // Try multiple combinations to improve chance of receiving data
            var whatToShowOptions = new[] { "TRADES", "MIDPOINT", "BID", "ASK", "BID_ASK" };
            var useRthOptions = new[] { false, true };

            var durationStr = $"{durationDays} D";

            var diagnosticMessages = new List<object>();

            foreach (var whatToShow in whatToShowOptions)
            {
                foreach (var useRth in useRthOptions)
                {
                    bars.Clear();
                    var requestId = client.Request.GetNextId();

                    _logger.LogInformation("Historical attempt: RequestId={RequestId}, WhatToShow={WhatToShow}, UseRth={UseRth}", requestId, whatToShow, useRth);

                    var sub = client.Response.Subscribe(msg =>
                    {
                        try
                        {
                            var msgType = msg.GetType();
                            var reqProp = msgType.GetProperty("RequestId");
                            if (reqProp != null)
                            {
                                var reqVal = reqProp.GetValue(msg);
                                if (reqVal is int rid && rid != requestId) return; // not our request
                            }

                            var timeProp = msgType.GetProperty("Time") ?? msgType.GetProperty("Date") ?? msgType.GetProperty("Timestamp");
                            var openProp = msgType.GetProperty("Open");
                            var highProp = msgType.GetProperty("High");
                            var lowProp = msgType.GetProperty("Low");
                            var closeProp = msgType.GetProperty("Close") ?? msgType.GetProperty("Price");
                            var volProp = msgType.GetProperty("Volume");

                            if (timeProp != null && (openProp != null || closeProp != null))
                            {
                                var timeVal = timeProp.GetValue(msg);
                                var openVal = openProp != null ? openProp.GetValue(msg) : null;
                                var highVal = highProp != null ? highProp.GetValue(msg) : null;
                                var lowVal = lowProp != null ? lowProp.GetValue(msg) : null;
                                var closeVal = closeProp != null ? closeProp.GetValue(msg) : null;
                                var volVal = volProp != null ? volProp.GetValue(msg) : null;

                                bars.Add(new
                                {
                                    Time = timeVal?.ToString(),
                                    Open = openVal,
                                    High = highVal,
                                    Low = lowVal,
                                    Close = closeVal,
                                    Volume = volVal
                                });
                            }
                        }
                        catch
                        {
                            // ignore
                        }
                        // Also collect a compact diagnostic snapshot of the message
                        try
                        {
                            var t = msg.GetType();
                            var props = t.GetProperties().Where(p => p.GetValue(msg) != null).Select(p => new { p.Name, Value = p.GetValue(msg)?.ToString() });
                            diagnosticMessages.Add(new { Type = t.Name, Props = props });
                        }
                        catch { }
                    });

                    try
                    {
                        // Per IBKR documentation, using an empty endDateTime ("") requests data up to the current time
                        // This avoids possible format mismatches. keepUpToDate not used here (one-shot request).
                        client.Request.RequestHistoricalData(requestId, contract, "", durationStr, barSizeSetting, whatToShow, useRth, 1);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "RequestHistoricalData threw for attempt {RequestId}", requestId);
                    }

                    // Wait a bit to collect bars
                    await Task.Delay(TimeSpan.FromSeconds(8));

                    try { sub.Dispose(); } catch { }
                    try { client.Request.CancelHistoricalData(requestId); } catch { }

                    _logger.LogInformation("Attempt completed: WhatToShow={WhatToShow}, UseRth={UseRth}, BarsCollected={Count}", whatToShow, useRth, bars.Count);

                    if (bars.Count > 0)
                    {
                        _logger.LogInformation("Collected {Count} historical bars for {Symbol} (whatToShow={WhatToShow}, useRth={UseRth})", bars.Count, symbol, whatToShow, useRth);
                        return new
                        {
                            Symbol = symbol,
                            Bars = bars,
                            Count = bars.Count,
                            WhatToShow = whatToShow,
                            UseRth = useRth,
                            Note = "Historical bars requested from TWS"
                        };
                    }
                }
            }

            _logger.LogInformation("Collected {Count} historical bars for {Symbol} (all attempts)", bars.Count, symbol);
            return new
            {
                Symbol = symbol,
                Bars = bars,
                Count = bars.Count,
                Note = "Historical bars requested from TWS (may be delayed or limited by IBKR account/data permissions)",
                Diagnostics = diagnosticMessages
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error requesting historical bars for {Symbol}", symbol);
            throw;
        }
    }
    private async Task WriteConnectionStatusAsync(object obj)
    {
        try
        {
            // prefer the current working directory (content root) so the status file is adjacent to the app files
            var folder = Directory.GetCurrentDirectory();
            var path = Path.Combine(folder, "interreact_status.json");
            var opts = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(obj, opts);
            await File.WriteAllTextAsync(path, json);
        }
        catch
        {
            // ignore write failures to avoid throwing during normal operation
        }
    }

    /// <summary>
    /// Request contract details for a symbol to get ConId and primary exchange information.
    /// Returns a simple dictionary with keys like conId and primaryExchange if available.
    /// </summary>
    private async Task<Dictionary<string, object>?> GetContractDetails(string symbol, string secType, string exchange)
    {
        var client = _twsConnection.GetClient();
        if (client == null) return null;

        var contract = new Contract
        {
            Symbol = symbol,
            SecurityType = secType,
            Exchange = exchange,
            Currency = "USD"
        };

        var tcs = new TaskCompletionSource<Dictionary<string, object>?>(TaskCreationOptions.RunContinuationsAsynchronously);
        var details = new List<Dictionary<string, object>>();

        var sub = client.Response.Subscribe(msg =>
        {
            try
            {
                var msgType = msg.GetType();
                var conIdProp = msgType.GetProperty("ConId") ?? msgType.GetProperty("conId") ?? null;
                var exchangeProp = msgType.GetProperty("PrimaryExchange") ?? msgType.GetProperty("primaryExchange");
                if (conIdProp != null)
                {
                    var d = new Dictionary<string, object>();
                    var cid = conIdProp.GetValue(msg);
                    if (cid != null) d["conId"] = (int)cid;
                    if (exchangeProp != null) d["primaryExchange"] = exchangeProp.GetValue(msg)?.ToString() ?? "";
                    details.Add(d);
                }
            }
            catch { }
        });

        try
        {
            var requestId = client.Request.GetNextId();
            client.Request.RequestContractDetails(requestId, contract);
            // wait briefly
            await Task.Delay(3000);
            sub.Dispose();
            if (details.Count > 0) return details[0];
            return null;
        }
        catch
        {
            try { sub.Dispose(); } catch { }
            return null;
        }
    }
}
