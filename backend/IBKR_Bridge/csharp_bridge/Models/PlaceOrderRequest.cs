namespace InterReactBridge.Models;

public record PlaceOrderRequest(
    string Symbol,
    string Action,
    int Quantity,
    string OrderType = "MKT",
    decimal? LimitPrice = null,
    decimal? StopPrice = null,
    string SecType = "STK",
    string Exchange = "SMART"
);
