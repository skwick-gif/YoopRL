<#
Start script for InterReactBridge (PowerShell)
Usage: Open PowerShell and run this script. It will build and run the bridge in the current console.
#>
param(
    [int]$Port = 5080
)

$projRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
n$projRoot = Join-Path $projRoot ".."
Set-Location $projRoot

Write-Host "Project root: $pwd" -ForegroundColor Cyan

Write-Host "Cleaning..." -ForegroundColor Yellow
dotnet clean

Write-Host "Building..." -ForegroundColor Yellow
dotnet build

Write-Host "Starting InterReactBridge (dotnet run) on http://localhost:$Port ..." -ForegroundColor Green
# Ensure the WebHost URL matches Program.cs (UseUrls). We set ASPNETCORE_URLS env so dotnet run binds to desired port.
$env:ASPNETCORE_URLS = "http://localhost:$Port"

# Run the app (this blocks the console)
dotnet run --project "InterReactBridge.csproj"
