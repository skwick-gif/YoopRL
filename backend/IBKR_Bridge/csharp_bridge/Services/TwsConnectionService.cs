using InterReact;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using System.Reactive.Threading.Tasks;
using System.Threading;
using System.Threading.Tasks;
using InterReactBridge.Hubs;

namespace InterReactBridge.Services;

/// <summary>
/// Background service that maintains persistent connection to TWS/IB Gateway
/// and broadcasts real-time updates via SignalR
/// </summary>
public class TwsConnectionService : BackgroundService
{
    private readonly ILogger<TwsConnectionService> _logger;
    private readonly IHubContext<AccountHub> _accountHub;
    private readonly IHubContext<PortfolioHub> _portfolioHub;
    private readonly IHubContext<MarketDataHub> _marketDataHub;
    private readonly IConfiguration _configuration;

    private IInterReactClient? _client;
    // Managed accounts as returned by TWS (comma-separated list originally)
    private readonly List<string> _managedAccounts = new();
    // Currently selected account for on-demand requests (nullable)
    private string? _selectedAccount;
    private string? _accountCode;
    private bool _isConnected;
    private readonly object _lockObject = new();
    // Track active AccountSummary request to allow safe cancellation
    private int? _activeAccountSummaryRequestId;

    // Connection settings
    private string _host = "127.0.0.1";
    private int _port = 7497;
    private int _clientId = 100;

    // Reconnection settings
    private readonly TimeSpan _reconnectDelay = TimeSpan.FromSeconds(5);
    private readonly int _maxReconnectAttempts = 10;

    // Subscriptions tracking
    private readonly Dictionary<string, IDisposable> _marketDataSubscriptions = new();

    public TwsConnectionService(
        ILogger<TwsConnectionService> logger,
        IHubContext<AccountHub> accountHub,
        IHubContext<PortfolioHub> portfolioHub,
        IHubContext<MarketDataHub> marketDataHub,
        IConfiguration configuration)
    {
        _logger = logger;
        _accountHub = accountHub;
        _portfolioHub = portfolioHub;
        _marketDataHub = marketDataHub;
        _configuration = configuration;

        // Load configuration
        LoadConfiguration();
    }

    private void LoadConfiguration()
    {
        _host = _configuration.GetValue<string>("IBKR:Host") ?? "127.0.0.1";
        _port = _configuration.GetValue<int>("IBKR:Port", 7497);
        _clientId = _configuration.GetValue<int>("IBKR:ClientId", 100);

        _logger.LogInformation("TWS Connection Configuration: Host={Host}, Port={Port}, ClientId={ClientId}",
            _host, _port, _clientId);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("=== TwsConnectionService Starting ===");

        var attempt = 0;
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // If not connected, try to connect
                if (!_isConnected)
                {
                    await ConnectToTwsAsync(stoppingToken);

                    if (_isConnected)
                    {
                        attempt = 0; // reset attempts on success

                        // Attempt to fetch managed accounts (non-fatal if it fails)
                        if (_client != null)
                        {
                            try
                            {
                                using var cts = new CancellationTokenSource(5000);
                                var managed = await _client.Response
                                    .OfType<ManagedAccounts>()
                                    .FirstAsync()
                                    .ToTask(cts.Token);

                                var accounts = managed.Accounts
                                    .Split(',')
                                    .Select(a => a.Trim())
                                    .Where(a => !string.IsNullOrEmpty(a))
                                    .ToList();

                                lock (_lockObject)
                                {
                                    _managedAccounts.Clear();
                                    _managedAccounts.AddRange(accounts);
                                    if (_managedAccounts.Count > 0 && string.IsNullOrEmpty(_selectedAccount))
                                    {
                                        _selectedAccount = _managedAccounts[0];
                                    }
                                }
                                _logger.LogInformation("\u2713 Managed accounts received: {Accounts}. Selected: {Selected}", string.Join(',', _managedAccounts), _selectedAccount);
                            }
                            catch (Exception ex)
                            {
                                _logger.LogWarning(ex, "Could not retrieve managed accounts at startup");
                            }
                        }

                        // Start listeners (fire-and-forget tasks)
                        await StartListeningAsync(stoppingToken);
                    }
                    else
                    {
                        attempt++;
                        if (attempt >= _maxReconnectAttempts)
                        {
                            _logger.LogError("Failed to connect after {MaxAttempts} attempts. Will keep trying...", _maxReconnectAttempts);
                            attempt = 0; // reset and keep trying
                            await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);
                        }
                        else
                        {
                            await Task.Delay(_reconnectDelay, stoppingToken);
                        }
                    }
                }
                else
                {
                    // Connected - wait and periodically loop
                    await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);
                }
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("Service is stopping...");
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in TwsConnectionService main loop");
                lock (_lockObject)
                {
                    _isConnected = false;
                }
                await Task.Delay(_reconnectDelay, stoppingToken);
            }
        }

        await DisconnectAsync();
        _logger.LogInformation("=== TwsConnectionService Stopped ===");
    }

    private async Task ConnectToTwsAsync(CancellationToken cancellationToken)
    {
        try
        {
            _logger.LogInformation("Connecting to TWS at {Host}:{Port}...", _host, _port);

            _client = await InterReactClient.ConnectAsync(options =>
            {
                options.TwsIpAddress = System.Net.IPAddress.Parse(_host);
                options.IBPortAddresses = new[] { _port };
                options.TwsClientId = _clientId;
            });

            _logger.LogInformation("InterReactClient created, waiting for ManagedAccounts...");

            // Try to get account code
            try
            {
                var cts = new CancellationTokenSource(5000);
                var managedAccounts = await _client.Response
                    .OfType<ManagedAccounts>()
                    .FirstAsync()
                    .ToTask(cts.Token);

                _accountCode = managedAccounts.Accounts.Split(',')[0];
                _logger.LogInformation("✓ Account received: {Account}", _accountCode);
            }
            catch
            {
                _logger.LogWarning("Could not retrieve account code, will continue without it");
                _accountCode = null;
            }

            lock (_lockObject)
            {
                _isConnected = true;
            }

            // Broadcast connection status
            await _accountHub.Clients.All.SendAsync("ConnectionStatus", 
                new { connected = true, account = _accountCode, timestamp = DateTime.UtcNow },
                cancellationToken);

            await _portfolioHub.Clients.All.SendAsync("ConnectionStatus",
                new { connected = true, account = _accountCode, timestamp = DateTime.UtcNow },
                cancellationToken);

            await _marketDataHub.Clients.All.SendAsync("ConnectionStatus",
                new { connected = true, timestamp = DateTime.UtcNow },
                cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to connect to TWS");
            lock (_lockObject)
            {
                _isConnected = false;
            }
        }
    }

    private async Task StartListeningAsync(CancellationToken cancellationToken)
    {
        if (_client == null || !_isConnected)
        {
            _logger.LogWarning("Cannot start listening - not connected");
            return;
        }

        _logger.LogInformation("Starting to listen for TWS updates...");

        // Start Account Summary updates (every 10 seconds)
        _ = Task.Run(async () =>
        {
            while (_isConnected && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    await BroadcastAccountSummaryAsync(cancellationToken);
                    await Task.Delay(TimeSpan.FromSeconds(10), cancellationToken);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error broadcasting account summary");
                }
            }
        }, cancellationToken);

        // Start Portfolio updates (every 10 seconds)
        _ = Task.Run(async () =>
        {
            while (_isConnected && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    await BroadcastPortfolioAsync(cancellationToken);
                    await Task.Delay(TimeSpan.FromSeconds(10), cancellationToken);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error broadcasting portfolio");
                }
            }
        }, cancellationToken);

        _logger.LogInformation("✓ Started listening for updates");
    }

    private async Task BroadcastAccountSummaryAsync(CancellationToken cancellationToken)
    {
        if (_client == null || !_isConnected) return;

        try
        {
            var summaries = new List<object>();
            int requestId;
            // Create a new request id and cancel any previous active account summary
            lock (_lockObject)
            {
                try
                {
                    if (_activeAccountSummaryRequestId.HasValue && _client != null)
                    {
                        try { _client.Request.CancelAccountSummary(_activeAccountSummaryRequestId.Value); } catch { }
                        _activeAccountSummaryRequestId = null;
                    }
                }
                catch { }

                // Use client's next id to avoid collisions
                requestId = _client.Request.GetNextId();
                _activeAccountSummaryRequestId = requestId;
            }

            var sub = _client.Response
                .OfType<AccountSummary>()
                .Where(s => s.RequestId == requestId)
                .Subscribe(summary =>
                {
                    summaries.Add(new
                    {
                        tag = summary.Tag,
                        value = summary.Value,
                        account = summary.Account,
                        currency = summary.Currency
                    });
                });

            _client.Request.RequestAccountSummary(requestId, "All");
            await Task.Delay(5000, cancellationToken);

            // Dispose and clear active request id
            sub.Dispose();
            lock (_lockObject)
            {
                try { if (_client != null && _activeAccountSummaryRequestId.HasValue) { _client.Request.CancelAccountSummary(_activeAccountSummaryRequestId.Value); } } catch { }
                _activeAccountSummaryRequestId = null;
            }

            if (summaries.Count > 0)
            {
                _logger.LogDebug("Broadcasting {Count} account summary items", summaries.Count);
                await _accountHub.Clients.All.SendAsync("AccountUpdate",
                    new { items = summaries, timestamp = DateTime.UtcNow },
                    cancellationToken);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in BroadcastAccountSummaryAsync");
        }
    }

    /// <summary>
    /// Cancel any active AccountSummary request created by this service.
    /// Called by other components before initiating a new AccountSummary request.
    /// </summary>
    public void CancelActiveAccountSummary()
    {
        lock (_lockObject)
        {
            if (_client == null || !_activeAccountSummaryRequestId.HasValue) return;
            try
            {
                _client.Request.CancelAccountSummary(_activeAccountSummaryRequestId.Value);
            }
            catch { }
            _activeAccountSummaryRequestId = null;
        }
    }

    private async Task BroadcastPortfolioAsync(CancellationToken cancellationToken)
    {
        if (_client == null || !_isConnected) return;

        try
        {
            var positions = new List<object>();

            var sub = _client.Response
                .OfType<AccountPosition>()
                .Subscribe(pos =>
                {
                    positions.Add(new
                    {
                        account = pos.Account,
                        symbol = pos.Contract.Symbol,
                        securityType = pos.Contract.SecurityType,
                        exchange = pos.Contract.Exchange,
                        currency = pos.Contract.Currency,
                        position = pos.Position,
                        averageCost = pos.AverageCost,
                        marketValue = pos.Position * (decimal)pos.AverageCost
                    });
                });

            _client.Request.RequestPositions();
            await Task.Delay(5000, cancellationToken);
            sub.Dispose();

            if (positions.Count > 0)
            {
                _logger.LogDebug("Broadcasting {Count} portfolio positions", positions.Count);
                await _portfolioHub.Clients.All.SendAsync("PortfolioUpdate",
                    new { positions, timestamp = DateTime.UtcNow },
                    cancellationToken);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in BroadcastPortfolioAsync");
        }
    }

    private async Task DisconnectAsync()
    {
        _logger.LogInformation("Disconnecting from TWS...");

        lock (_lockObject)
        {
            _isConnected = false;
        }

        // Clean up subscriptions
        foreach (var sub in _marketDataSubscriptions.Values)
        {
            sub.Dispose();
        }
        _marketDataSubscriptions.Clear();

        // Notify clients
        try
        {
            await _accountHub.Clients.All.SendAsync("ConnectionStatus",
                new { connected = false, timestamp = DateTime.UtcNow });

            await _portfolioHub.Clients.All.SendAsync("ConnectionStatus",
                new { connected = false, timestamp = DateTime.UtcNow });

            await _marketDataHub.Clients.All.SendAsync("ConnectionStatus",
                new { connected = false, timestamp = DateTime.UtcNow });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error notifying clients of disconnection");
        }

        _client = null;
        _logger.LogInformation("Disconnected from TWS");
    }

    public bool IsConnected()
    {
        lock (_lockObject)
        {
            return _isConnected;
        }
    }

    public string? GetAccountCode()
    {
        return _accountCode;
    }

    public IReadOnlyList<string> GetManagedAccounts()
    {
        lock (_lockObject)
        {
            return _managedAccounts.ToList();
        }
    }

    public void SetSelectedAccount(string account)
    {
        if (string.IsNullOrWhiteSpace(account)) return;
        lock (_lockObject)
        {
            _selectedAccount = account.Trim();
            // Also set as primary account code for convenience
            _accountCode = _selectedAccount;
        }
        _logger.LogInformation("Selected account set to {Account}", _selectedAccount);
    }

    public IInterReactClient? GetClient()
    {
        lock (_lockObject)
        {
            return _isConnected ? _client : null;
        }
    }

    public override async Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("TwsConnectionService stop requested");
        await base.StopAsync(cancellationToken);
    }
}
