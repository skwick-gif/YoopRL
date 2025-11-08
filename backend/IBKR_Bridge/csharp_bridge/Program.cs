using InterReactBridge.Services;
using InterReactBridge.Hubs;
using InterReactBridge.Services.Indicators;
using InterReactBridge.Models;

var builder = WebApplication.CreateBuilder(args);

// הזרקת תלות
builder.Services.AddLogging();

// TwsConnectionService - רשום כ-Singleton וגם כ-HostedService
builder.Services.AddSingleton<TwsConnectionService>();
builder.Services.AddSingleton<IbService>();

// Background Service - The heart of production!
// Maintains persistent connection to TWS and broadcasts real-time updates
builder.Services.AddHostedService(sp => sp.GetRequiredService<TwsConnectionService>());

// SignalR for real-time streaming
builder.Services.AddSignalR();

// CORS policy for web clients
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

// Allow overriding listen URL via ASPNETCORE_URLS env var; default to 5080
var urls = Environment.GetEnvironmentVariable("ASPNETCORE_URLS") ?? "http://localhost:5080";
builder.WebHost.UseUrls(urls);

var app = builder.Build();

// -----------------------------
// Root
// -----------------------------
app.MapGet("/", () => "InterReactBridge is running");

// -----------------------------
// Health check
// -----------------------------
app.MapGet("/health", () =>
{
    return Results.Ok(new { status = "ok" });
});

// -----------------------------
// Connection Status
// -----------------------------
app.MapGet("/connection-status", (TwsConnectionService twsService) =>
{
    try
    {
        var isConnected = twsService.IsConnected();
        var status = new
        {
            isConnected = isConnected,
            accountCode = twsService.GetAccountCode(),
            host = "127.0.0.1", // From configuration
            port = 7497,
            message = isConnected ? "Connected to TWS" : "Not connected to TWS"
        };
        return Results.Ok(status);
    }
    catch (Exception ex)
    {
        return Results.Ok(new { 
            isConnected = false, 
            error = ex.Message 
        });
    }
});

// -----------------------------
// Test SMA Indicator
// Example: GET /test/sma
// -----------------------------
app.MapGet("/test/sma", () =>
{
    try
    {
        var sma3 = new SimpleMovingAverage(3);
        var results = new List<object>();

        var prices = new[] { 10m, 20m, 30m, 40m, 50m };
        foreach (var price in prices)
        {
            sma3.AddPrice(price, DateTime.Now);
            var value = sma3.Calculate();
            results.Add(new
            {
                Price = price,
                IsReady = sma3.IsReady,
                Count = sma3.Count,
                SMA = value
            });
        }

        return Results.Ok(new
        {
            indicator = sma3.Name,
            period = 3,
            results,
            note = "Expected: 20.00, 30.00, 40.00 for last 3 values"
        });
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
});

// -----------------------------
// Delayed Price Series
// Example: GET /delayed-prices?symbol=AAPL&secType=STK&exchange=SMART&sampleSeconds=30
// -----------------------------
app.MapGet("/delayed-prices", async (IbService ib, string symbol, string secType, string exchange, 
    int sampleSeconds = 30) =>
{
    try
    {
        var prices = await ib.GetDelayedPriceSeries(symbol, secType, exchange, sampleSeconds);
        return Results.Ok(prices);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
});

// -----------------------------
// SMA on Delayed Market Data
// Example: GET /indicators/sma?symbol=AAPL&period=5&sampleSeconds=30
// -----------------------------
app.MapGet("/indicators/sma", async (IbService ib, string symbol, int period = 5, 
    string secType = "STK", string exchange = "SMART", int sampleSeconds = 30) =>
{
    try
    {
        // Get delayed price series from IBKR
        var priceData = await ib.GetDelayedPriceSeries(symbol, secType, exchange, sampleSeconds);
        
        if (priceData.Prices == null || priceData.Prices.Count == 0)
        {
            return Results.BadRequest(new { error = "No price data received" });
        }

        // Create SMA indicator
        var sma = new SimpleMovingAverage(period);
        var results = new List<object>();

        // Process each price point
        foreach (var pricePoint in priceData.Prices)
        {
            var price = Convert.ToDecimal(pricePoint.Price);
            var time = pricePoint.Time;

            sma.AddPrice(price, time);
            var smaValue = sma.Calculate();

            results.Add(new
            {
                Time = time,
                Price = price,
                SMA = smaValue,
                IsReady = sma.IsReady
            });
        }

        return Results.Ok(new
        {
            Symbol = symbol,
            Indicator = sma.Name,
            Period = period,
            SampleSeconds = sampleSeconds,
            PricesCount = priceData.Prices.Count,
            Results = results,
            Note = priceData.Note
        });
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message, stack = ex.StackTrace });
    }
});

// -----------------------------
// Historical bars
// Example: GET /historical?symbol=INTC&secType=STK&exchange=SMART&durationDays=7&barSize=1 day
// -----------------------------
app.MapGet("/historical", async (IbService ib, string symbol, string secType = "STK", string exchange = "SMART", int durationDays = 7, string barSize = "1 day") =>
{
    try
    {
        var result = await ib.GetHistoricalBars(symbol, secType, exchange, durationDays, barSize);
        return Results.Ok(result);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
});

// -----------------------------
// Market data (one-shot)
// Example: GET /marketdata?symbol=EUR&secType=CASH&exchange=IDEALPRO&durationSeconds=5
// -----------------------------
app.MapGet("/marketdata", async (IbService ib, string symbol, string secType, string exchange, int durationSeconds) =>
{
    try
    {
        var ticks = await ib.GetMarketData(symbol, secType, exchange, TimeSpan.FromSeconds(durationSeconds));
        return Results.Ok(ticks);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
});

// -----------------------------
// Connect to IBKR
// דוגמה: POST /connect?host=127.0.0.1&port=7496&clientId=1
// -----------------------------
app.MapPost("/connect", async (IbService ib, string host, int port, int clientId) =>
{
    var ok = await ib.ConnectAsync(host, port, clientId);
    return ok
        ? Results.Ok(new { connected = true })
        : Results.BadRequest(new { connected = false });
});

// -----------------------------
// Place order (simple BUY/SELL)
// Accepts JSON body per client expectations
// Example JSON: {"symbol":"AAPL","action":"BUY","quantity":1,"orderType":"MKT"}
// -----------------------------
// PlaceOrderRequest moved to Models/PlaceOrderRequest.cs to avoid top-level statement ordering issues

app.MapPost("/place-order", async (IbService ib, PlaceOrderRequest req) =>
{
    try
    {
        var res = await ib.PlaceOrder(req.Symbol, req.Action, req.Quantity, req.OrderType, req.LimitPrice, req.StopPrice, req.SecType, req.Exchange);
        return Results.Ok(res);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { success = false, error = ex.Message });
    }
});

// -----------------------------
// Account summary
// דוגמה: GET /account
// -----------------------------
app.MapGet("/account", async (IbService ib, TwsConnectionService twsService, string? account = null) =>
{
    try
    {
        // If caller didn't specify an account, use the TwsConnectionService selected account if available
        var acctToUse = account;
        if (string.IsNullOrEmpty(acctToUse)) acctToUse = twsService.GetAccountCode();

        var accountSummary = await ib.GetAccountSummary(acctToUse);
        return Results.Ok(accountSummary);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
});

// -----------------------------
// Portfolio positions
// דוגמה: GET /portfolio
// -----------------------------
app.MapGet("/portfolio", async (IbService ib, TwsConnectionService twsService, string? account = null) =>
{
    try
    {
        var acctToUse = account;
        if (string.IsNullOrEmpty(acctToUse)) acctToUse = twsService.GetAccountCode();

        var portfolio = await ib.GetPortfolio(acctToUse);
        return Results.Ok(portfolio);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
});

// -----------------------------
// Managed accounts listing
// Example: GET /accounts
// -----------------------------
app.MapGet("/accounts", (TwsConnectionService twsService) =>
{
    try
    {
        var accounts = twsService.GetManagedAccounts();
        return Results.Ok(new { accounts, selected = twsService.GetAccountCode() });
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
});

// -----------------------------
// Select default account for on-demand requests
// Example: POST /accounts/select?account=U123
// -----------------------------
app.MapPost("/accounts/select", (TwsConnectionService twsService, string account) =>
{
    try
    {
        twsService.SetSelectedAccount(account);
        return Results.Ok(new { selected = twsService.GetAccountCode() });
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
});

// -----------------------------
// SignalR Hubs for real-time streaming
// -----------------------------
app.UseCors("AllowAll");
app.MapHub<AccountHub>("/hubs/account");
app.MapHub<PortfolioHub>("/hubs/portfolio");
app.MapHub<MarketDataHub>("/hubs/marketdata");

// -----------------------------
// Start the server
// -----------------------------
Console.WriteLine("Starting InterReactBridge...");
Console.WriteLine("SignalR Hubs:");
Console.WriteLine("  - /hubs/account");
Console.WriteLine("  - /hubs/portfolio");
Console.WriteLine("  - /hubs/marketdata");
try
{
    app.Run();
}
catch (Exception ex)
{
    Console.WriteLine($"Application failed to start: {ex.Message}");
    Console.WriteLine(ex.StackTrace);
}
