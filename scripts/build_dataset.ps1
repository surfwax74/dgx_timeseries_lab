# scripts/build_dataset.ps1 — smart materializer for cached datasets.
#
# What it does:
#   1. Looks for data/synth/<name>/data.parquet
#   2. If present + not --force: prints where it lives, exits 0
#   3. If missing (or --force): resolves the preset name and runs
#      `dgx-ts synth dataset=<resolved>` to materialize it
#   4. On success, prints the parquet path + suggests the next command
#
# Usage:
#     pwsh scripts/build_dataset.ps1 <name>                     # build if missing
#     pwsh scripts/build_dataset.ps1 <name> -Force              # rebuild regardless
#     pwsh scripts/build_dataset.ps1 <name> -PresetGroup foo    # explicit group override
#
# Supported <name>s out of the box (add more by creating both a
# configs/dataset/presets/<name>.yaml and a
# configs/dataset/cached/<name>.yaml):
#     trivial_synth       (top-level, no preset group)
#     leo_eps_24h         (via presets/leo_eps_24h)
#     leo_eps_full_24h    (via presets/leo_eps_full_24h)
#
# After materialization, use in an experiment via:
#     dataset=cached/<name>
# or in a benchmark suite:
#     - {key: parquet_telemetry, params: {data_path: data/synth/<name>}}

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Name,

    [switch]$Force,

    [string]$PresetGroup,

    [string]$OutputDir = "data/synth"
)

$ErrorActionPreference = "Stop"

# Resolve repo root
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "No .venv at $Python — run 'uv sync' first."
}
$env:UV_NO_SYNC = "1"

$CachePath = Join-Path $RepoRoot "$OutputDir\$Name"
$ParquetFile = Join-Path $CachePath "data.parquet"

# Cache-hit path — skip unless -Force
if ((Test-Path $ParquetFile) -and (-not $Force)) {
    Write-Host "[build_dataset] Cache HIT: $CachePath" -ForegroundColor Green
    $manifest = Join-Path $CachePath "manifest.yaml"
    if (Test-Path $manifest) {
        Write-Host "  Manifest:" -ForegroundColor White
        Get-Content $manifest | Select-Object -First 8 | ForEach-Object { Write-Host "    $_" }
    }
    Write-Host ""
    Write-Host "  Use it via:" -ForegroundColor White
    Write-Host "    dataset=cached/$Name" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Force rebuild with: pwsh scripts/build_dataset.ps1 $Name -Force" -ForegroundColor DarkGray
    exit 0
}

# Cache-miss (or --force) path — resolve the Hydra dataset selector
# and invoke `dgx-ts synth`
if (-not $PresetGroup) {
    # Auto-detect: if configs/dataset/presets/<name>.yaml exists, use presets/
    $PresetYaml = Join-Path $RepoRoot "configs/dataset/presets/$Name.yaml"
    $TopLevelYaml = Join-Path $RepoRoot "configs/dataset/$Name.yaml"
    if (Test-Path $PresetYaml) {
        $DatasetSelector = "presets/$Name"
    } elseif (Test-Path $TopLevelYaml) {
        $DatasetSelector = $Name
    } else {
        throw "No source YAML found for '$Name'. Expected one of:`n" +
              "  $PresetYaml`n" +
              "  $TopLevelYaml`n" +
              "Create one, or pass -PresetGroup <group> to override."
    }
} else {
    $DatasetSelector = "$PresetGroup/$Name"
}

if ($Force -and (Test-Path $CachePath)) {
    Write-Host "[build_dataset] -Force set: removing existing cache at $CachePath" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $CachePath
}

Write-Host "[build_dataset] Cache MISS. Materializing '$Name' from dataset=$DatasetSelector..." -ForegroundColor Yellow
Write-Host "  Output dir: $CachePath" -ForegroundColor White
Write-Host ""

& $Python -m dgx_ts_lab.cli.main synth dataset=$DatasetSelector
if ($LASTEXITCODE -ne 0) {
    throw "synth failed (exit=$LASTEXITCODE)"
}

if (-not (Test-Path $ParquetFile)) {
    throw "synth completed but no parquet at $ParquetFile — check the dataset config's 'name' field matches '$Name'"
}

Write-Host ""
Write-Host "[build_dataset] DONE." -ForegroundColor Green
Write-Host "  Cache path: $CachePath" -ForegroundColor White
Write-Host "  Use it via:" -ForegroundColor White
Write-Host "    dataset=cached/$Name" -ForegroundColor Cyan
