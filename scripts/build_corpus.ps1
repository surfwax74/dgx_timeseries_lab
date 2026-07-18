# scripts/build_corpus.ps1 - batch-materialize a pretraining corpus.
#
# Runs build_dataset.ps1 for every entry in the corpus manifest.
# Skips datasets that already exist (unless -Force). Ideal for the
# "materialize once, use in dgx_pretrain_corpus.yaml many times" flow.
#
# Usage:
#     pwsh scripts/build_corpus.ps1                                # build every default entry
#     pwsh scripts/build_corpus.ps1 -Manifest custom.yaml          # different manifest
#     pwsh scripts/build_corpus.ps1 -Force                         # rebuild all
#     pwsh scripts/build_corpus.ps1 -Only leo_eps_v1,leo_eps_v3    # subset
#     pwsh scripts/build_corpus.ps1 -DryRun                        # list only
#
# The default manifest is a hardcoded list matching Phase A of
# docs/pretraining_corpus_roadmap.md - the 5 EPS mission variants plus
# the pre-existing base presets. Edit the $DEFAULT_MEMBERS array below
# to add more variants as the corpus grows.
#
# ASCII-only output so this parses under Windows PowerShell 5.1 as well
# as PowerShell 7+. Do NOT reintroduce box-drawing or Unicode arrows here.

[CmdletBinding()]
param(
    [string]$Manifest,
    [switch]$Force,
    [string[]]$Only,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
Set-Location $RepoRoot

# Phase A default corpus members. Add rows as the corpus grows.
# Each entry: (name, description, estimated_size_MB, estimated_build_time_min).
$DEFAULT_MEMBERS = @(
    @{ name = 'leo_eps_24h';      desc = 'Base LEO EPS preset (6 ch x 24 h)';         size_mb =  10; build_min =  2 },
    @{ name = 'leo_eps_full_24h'; desc = 'Full 83-channel LEO EPS';                   size_mb = 200; build_min = 15 },
    @{ name = 'leo_eps_v1';       desc = 'Quiet mission (low fault rate)';            size_mb =  10; build_min =  2 },
    @{ name = 'leo_eps_v2';       desc = 'Stormy mission (high noise + faults)';      size_mb =  10; build_min =  2 },
    @{ name = 'leo_eps_v3';       desc = 'Sun-sync orbit (6000 s period)';            size_mb =  10; build_min =  2 },
    @{ name = 'leo_eps_v4';       desc = 'Aging spacecraft (heavy drift)';            size_mb =  10; build_min =  2 },
    @{ name = 'leo_eps_v5';       desc = 'Payload-heavy load profile';                size_mb =  10; build_min =  2 }
)

if ($Manifest) {
    if (-not (Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }
    $Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    $members_json = & $Python -c "import yaml, sys, json; print(json.dumps(yaml.safe_load(open(sys.argv[1]))['members']))" $Manifest
    $members = $members_json | ConvertFrom-Json
} else {
    $members = $DEFAULT_MEMBERS
}

if ($Only) {
    $wanted = @($Only)
    $members = $members | Where-Object { $wanted -contains $_.name }
    if ($members.Count -eq 0) { throw "No members matched -Only $($wanted -join ',')" }
}

Write-Host "[build_corpus] $($members.Count) member(s) selected." -ForegroundColor Cyan
# Sum by hand - hashtable keys aren't visible as properties to Measure-Object
# under Windows PowerShell 5.1 (works on 7+ but we support both).
$total_size = 0
$total_time = 0
foreach ($m in $members) { $total_size += [int]$m.size_mb; $total_time += [int]$m.build_min }
Write-Host "  Estimated total: ~$total_size MB / ~$total_time min (if none cached)" -ForegroundColor White

if ($DryRun) {
    Write-Host ""
    Write-Host "-- Dry-run - the following builds would run: --" -ForegroundColor Yellow
    foreach ($m in $members) {
        Write-Host ("  {0,-25}  {1}" -f $m.name, $m.desc)
    }
    Write-Host ""
    Write-Host "Re-run without -DryRun to execute."
    exit 0
}

$failed = @()
$hit = @()
$missed = @()
foreach ($m in $members) {
    Write-Host ""
    Write-Host "==============================================================" -ForegroundColor DarkGray
    Write-Host " Building: $($m.name)  ($($m.desc))" -ForegroundColor White
    Write-Host "==============================================================" -ForegroundColor DarkGray
    $args_to_pass = @($m.name)
    if ($Force) { $args_to_pass += '-Force' }
    try {
        & powershell -NoProfile -File (Join-Path $PSScriptRoot 'build_dataset.ps1') @args_to_pass
        $parquet = Join-Path $RepoRoot "data\synth\$($m.name)\data.parquet"
        if (Test-Path $parquet) {
            $age_min = ((Get-Date) - (Get-Item $parquet).LastWriteTime).TotalMinutes
            if ($age_min -lt 5) {
                $missed += $m.name
            } else {
                $hit += $m.name
            }
        }
    } catch {
        Write-Host "  FAILED: $_" -ForegroundColor Red
        $failed += $m.name
    }
}

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " Corpus build summary" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ("  Cache hits (already present):  {0}" -f $hit.Count) -ForegroundColor Green
Write-Host ("  Newly built:                   {0}" -f $missed.Count) -ForegroundColor Green
$fail_color = if ($failed.Count -gt 0) { 'Red' } else { 'White' }
Write-Host ("  Failed:                        {0}" -f $failed.Count) -ForegroundColor $fail_color
if ($failed.Count -gt 0) {
    Write-Host "  Failed members: $($failed -join ', ')" -ForegroundColor Red
    exit 2
}
Write-Host ""
Write-Host "  Use in an experiment via: experiment=dgx_pretrain_corpus" -ForegroundColor Cyan
