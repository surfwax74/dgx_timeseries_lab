# launch_cpu_smoke.ps1 — fastest possible end-to-end smoke on CPU.
# Use for: dev iteration, CI, "does it run at all?" checks.
# Runtime: <30 seconds.

$ErrorActionPreference = "Stop"
$env:PATH = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "==> CPU smoke test (phase0_smoke + rolling_mean)" -ForegroundColor Cyan
uv run dgx-ts train experiment=phase0_smoke trainer=cpu

Write-Host ""
Write-Host "==> Optional: Phase 1 layered synth smoke (6-ch preset)" -ForegroundColor Cyan
uv run dgx-ts train experiment=phase1_layered trainer=cpu
