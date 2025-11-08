# check_all.ps1 - connect bridge to IBKR and fetch status, portfolio, historical bars for INTC
param(
    [string]$Host = '127.0.0.1',
    [int]$Port = 7497,
    [int]$ClientId = 101,
    [string]$BridgeUrl = 'http://localhost:5080'
)

Write-Host "Running connect_to_ib.ps1 with Host=$Host Port=$Port ClientId=$ClientId BridgeUrl=$BridgeUrl" -ForegroundColor Cyan
& "$PSScriptRoot\connect_to_ib.ps1" -Host $Host -Port $Port -ClientId $ClientId -BridgeUrl $BridgeUrl

Start-Sleep -Seconds 4

Write-Host '--- connection-status ---' -ForegroundColor Yellow
try { Invoke-RestMethod "$BridgeUrl/connection-status" | ConvertTo-Json -Depth 5 | Write-Host } catch { Write-Host "Connection-status error: $_" -ForegroundColor Red }

Write-Host '--- portfolio ---' -ForegroundColor Yellow
try { Invoke-RestMethod "$BridgeUrl/portfolio" | ConvertTo-Json -Depth 5 | Write-Host } catch { Write-Host "Portfolio error: $_" -ForegroundColor Red }

Write-Host '--- historical INTC ---' -ForegroundColor Yellow
try { Invoke-RestMethod "$BridgeUrl/historical?symbol=INTC&secType=STK&exchange=SMART&durationDays=7&barSize=1%20day" | ConvertTo-Json -Depth 10 | Write-Host } catch { Write-Host "Historical error: $_" -ForegroundColor Red }
