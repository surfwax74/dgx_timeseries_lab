# setup_dev_windows.ps1 — first-time setup on a Windows dev box.
#
# Idempotent: re-running is safe.
# Run from the repo root: .\scripts\setup_dev_windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "==> Checking for uv..." -ForegroundColor Cyan
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Host "    uv not found. Installing via winget..."
    winget install --id astral-sh.uv --accept-source-agreements --accept-package-agreements --silent
    # Refresh PATH from registry into this session
    $env:PATH = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}
Write-Host "    uv: $(uv --version)" -ForegroundColor Green

Write-Host "==> Syncing workspace..." -ForegroundColor Cyan
uv sync

Write-Host "==> Running smoke tests..." -ForegroundColor Cyan
uv run pytest packages/ -q

Write-Host "==> Done." -ForegroundColor Green
Write-Host ""
Write-Host "Try:" -ForegroundColor Yellow
Write-Host "    uv run dgx-ts train experiment=phase0_smoke"
Write-Host "    uv run dgx-ts train experiment=phase1_layered"
