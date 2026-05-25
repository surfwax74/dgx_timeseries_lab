# install_cuda_torch.ps1 — install CUDA-enabled PyTorch on Windows.
#
# uv sync pulls the CPU-only torch wheel by default. Run this AFTER uv sync
# (and after each `uv run` that re-syncs) to swap in the CUDA wheel.
#
# Workaround for the re-sync: set UV_NO_SYNC=1 in your shell, then `uv run`
# will skip the sync pass. See docs/deployment/rtx3080_workstation.md.
#
# Usage:
#     .\scripts\install_cuda_torch.ps1            # cu126 (default)
#     .\scripts\install_cuda_torch.ps1 -Cuda 128  # cu128 for newer drivers

param(
    [string]$Cuda = "126"
)

$ErrorActionPreference = "Stop"
$env:PATH = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

$indexUrl = "https://download.pytorch.org/whl/cu$Cuda"

Write-Host "==> Installing torch with CUDA $Cuda from $indexUrl" -ForegroundColor Cyan
uv pip install --reinstall torch --index-url $indexUrl

Write-Host "==> Verifying CUDA..." -ForegroundColor Cyan
uv run --no-sync python scripts/check_gpu.py

Write-Host ""
Write-Host "TIP: set `$env:UV_NO_SYNC = '1'` for this shell so `uv run` skips" -ForegroundColor Yellow
Write-Host "    its sync pass and keeps the CUDA wheel installed." -ForegroundColor Yellow
