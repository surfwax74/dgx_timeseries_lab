# launch_rtx3080.ps1 — single RTX 3080 workstation tier.
# Pre-req: scripts\install_cuda_torch.ps1 has been run + $env:UV_NO_SYNC = '1'
#
# Demonstrates:
#   - Phase 2 bake-off (rolling_mean + 3 from-scratch transformers)
#     against the 83-channel LEO EPS preset
#   - Tier-appropriate batch / window / precision settings

param(
    [string]$Experiment = "phase2_bakeoff",
    [int]$Epochs = 5
)

$ErrorActionPreference = "Stop"
$env:PATH = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

if (-not $env:UV_NO_SYNC) {
    Write-Host "WARNING: UV_NO_SYNC is not set. uv run will re-sync and revert torch to CPU." -ForegroundColor Yellow
    Write-Host "  Run: `$env:UV_NO_SYNC = '1'  (or run install_cuda_torch.ps1 first)" -ForegroundColor Yellow
}

Write-Host "==> GPU preflight" -ForegroundColor Cyan
uv run --no-sync python scripts/check_gpu.py --require=1 --min-vram-gb=8 --recommended-tier=rtx3080

Write-Host ""
Write-Host "==> Synth dataset if missing" -ForegroundColor Cyan
if (-not (Test-Path "data\synth\leo_eps_full_24h\data.parquet")) {
    uv run --no-sync dgx-ts synth dataset=presets/leo_eps_full_24h
}

Write-Host ""
Write-Host "==> Run experiment=$Experiment on RTX 3080 (max_epochs=$Epochs)" -ForegroundColor Cyan
uv run --no-sync dgx-ts benchmark `
    experiment=$Experiment `
    trainer=rtx3080 `
    trainer.max_epochs=$Epochs
