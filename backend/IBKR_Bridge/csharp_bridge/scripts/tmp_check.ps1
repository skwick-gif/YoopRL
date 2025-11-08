$ErrorActionPreference = 'Stop'
function Get-Json { param($url) try { $r = Invoke-RestMethod -Uri $url -Method Get -UseBasicParsing; $r | ConvertTo-Json -Depth 10 } catch { Write-Host "ERROR: $url -> $_" } }

try {
    Write-Host "POST connect..."
    Invoke-RestMethod -Uri 'http://localhost:5080/connect?host=127.0.0.1&port=7497&clientId=101' -Method Post | ConvertTo-Json -Depth 5 | Write-Host
    Start-Sleep -Seconds 1
    Write-Host "connection-status:"
    Get-Json 'http://localhost:5080/connection-status' | Write-Host
    Start-Sleep -Seconds 1
    Write-Host "portfolio:"
    Get-Json 'http://localhost:5080/portfolio' | Write-Host
    Start-Sleep -Seconds 1
    Write-Host "historical INTC:"
    Get-Json 'http://localhost:5080/historical?symbol=INTC&secType=STK&exchange=SMART&durationDays=7&barSize=1 day' | Write-Host
} catch {
    Write-Host "Script failed: $_"
}
