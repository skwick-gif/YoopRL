<#
connect_to_ib.ps1 - request the running InterReactBridge to connect to IBKR TWS/Gateway
Usage:
  .\connect_to_ib.ps1 -Host 127.0.0.1 -Port 7497 -ClientId 100
#>
param(
    [string]$Host = '127.0.0.1',
    [int]$Port = 7497,
    [int]$ClientId = 100,
    [string]$BridgeUrl = 'http://localhost:5080'
)

$uri = "$BridgeUrl/connect?host=$Host&port=$Port&clientId=$ClientId"
Write-Host "Requesting bridge to connect to IBKR: $uri" -ForegroundColor Cyan
try {
    $resp = Invoke-RestMethod -Uri $uri -Method Post -TimeoutSec 30
    Write-Host "Response:`n$(ConvertTo-Json $resp -Depth 5)" -ForegroundColor Green
} catch {
    Write-Host "Error connecting to bridge: $_" -ForegroundColor Red
}
