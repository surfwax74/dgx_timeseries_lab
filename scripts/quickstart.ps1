# scripts/quickstart.ps1 — one-command demo on Windows.
#
# What this does:
#   1. Verify the venv exists (creates one if missing via uv)
#   2. Run a tiny bake-off (3 detectors x 1 synth dataset x 2 seeds) on CPU
#   3. Render presentation-grade ROC / PR / AUC plots from the result
#   4. Print where everything landed
#
# Usage (from repo root):
#     pwsh scripts/quickstart.ps1
#     pwsh scripts/quickstart.ps1 -SkipBenchmark   # plots only, reuse last run

[CmdletBinding()]
param(
    [switch]$SkipBenchmark,
    [string]$OutputDir = "benchmark_reports/quickstart_viz",
    [string]$FiguresDir = "benchmark_reports/quickstart_viz/figures"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
Set-Location $RepoRoot
Write-Host "[quickstart] repo root: $RepoRoot" -ForegroundColor Cyan

# 1. Locate Python — prefer the in-repo .venv, fall back to PATH
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "[quickstart] .venv not found; trying `"uv sync`" to bootstrap..." -ForegroundColor Yellow
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv sync
    } else {
        throw "No .venv and no 'uv' on PATH. Install uv (https://docs.astral.sh/uv/) or create a venv manually."
    }
    if (-not (Test-Path $Python)) {
        throw "Bootstrap failed: $Python still missing."
    }
}
Write-Host "[quickstart] python: $Python" -ForegroundColor Cyan

$env:UV_NO_SYNC = "1"

# 2. Run the bake-off
if (-not $SkipBenchmark) {
    Write-Host ""
    Write-Host "[quickstart] Step 1/2: running bake-off (experiment=quickstart_viz)..." -ForegroundColor Green
    & $Python -m dgx_ts_lab.cli.main benchmark experiment=quickstart_viz
    if ($LASTEXITCODE -ne 0) { throw "benchmark step failed (exit=$LASTEXITCODE)" }
} else {
    Write-Host "[quickstart] -SkipBenchmark: reusing existing $OutputDir" -ForegroundColor Yellow
}

# 3. Render the figures
Write-Host ""
Write-Host "[quickstart] Step 2/2: rendering figures..." -ForegroundColor Green
& $Python -m dgx_ts_lab.cli.main viz `
    --benchmark-dir $OutputDir `
    --output-dir $FiguresDir `
    --format png,svg `
    --splits val,test
if ($LASTEXITCODE -ne 0) { throw "viz step failed (exit=$LASTEXITCODE)" }

Write-Host ""
Write-Host "[quickstart] DONE." -ForegroundColor Green
Write-Host "  Benchmark report: $OutputDir\benchmark_report.md" -ForegroundColor White
Write-Host "  Figures (PNG/SVG): $FiguresDir" -ForegroundColor White
Write-Host ""
Write-Host "  Drop the PNGs straight into your slide deck." -ForegroundColor White
