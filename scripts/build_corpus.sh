#!/usr/bin/env bash
# scripts/build_corpus.sh — batch-materialize a pretraining corpus.
#
# Runs build_dataset.sh for every entry in the corpus manifest.
# Skips datasets that already exist (unless --force). See
# docs/pretraining_corpus_roadmap.md for the corpus design.
#
# Usage:
#     bash scripts/build_corpus.sh                          # build every default member
#     bash scripts/build_corpus.sh --force                  # rebuild all
#     bash scripts/build_corpus.sh --only leo_eps_v1,leo_eps_v3
#     bash scripts/build_corpus.sh --dry-run
set -euo pipefail

FORCE=""
ONLY=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)    FORCE="--force"; shift ;;
        --only)     ONLY="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  sed -n '2,15p' "$0"; exit 0 ;;
        *)          echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Phase A default corpus members. Format: name|desc|size_mb|build_min
MEMBERS=(
    "leo_eps_24h|Base LEO EPS preset (6 ch × 24 h)|10|2"
    "leo_eps_full_24h|Full 83-channel LEO EPS|200|15"
    "leo_eps_v1|Quiet mission (low fault rate)|10|2"
    "leo_eps_v2|Stormy mission (high noise + faults)|10|2"
    "leo_eps_v3|Sun-sync orbit (6000 s period)|10|2"
    "leo_eps_v4|Aging spacecraft (heavy drift)|10|2"
    "leo_eps_v5|Payload-heavy load profile|10|2"
)

# Filter to --only subset
if [[ -n "$ONLY" ]]; then
    IFS=',' read -ra WANTED <<< "$ONLY"
    FILTERED=()
    for m in "${MEMBERS[@]}"; do
        name="${m%%|*}"
        for w in "${WANTED[@]}"; do
            [[ "$w" == "$name" ]] && FILTERED+=("$m")
        done
    done
    MEMBERS=("${FILTERED[@]}")
    if [[ ${#MEMBERS[@]} -eq 0 ]]; then
        echo "No members matched --only $ONLY" >&2; exit 2
    fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[build_corpus] ${#MEMBERS[@]} member(s) selected."
total_size=0; total_time=0
for m in "${MEMBERS[@]}"; do
    IFS='|' read -r _ _ sz t <<< "$m"
    total_size=$((total_size + sz))
    total_time=$((total_time + t))
done
echo "  Estimated total: ~${total_size} MB / ~${total_time} min (if none cached)"

if [[ $DRY_RUN -eq 1 ]]; then
    echo
    echo "-- Dry-run — the following builds would run: --"
    for m in "${MEMBERS[@]}"; do
        IFS='|' read -r name desc _ _ <<< "$m"
        printf "  %-25s  %s\n" "$name" "$desc"
    done
    echo
    echo "Re-run without --dry-run to execute."
    exit 0
fi

failed=()
hit=0
missed=0
for m in "${MEMBERS[@]}"; do
    IFS='|' read -r name desc _ _ <<< "$m"
    echo
    echo "══════════════════════════════════════════════════════════════"
    echo " Building: $name  ($desc)"
    echo "══════════════════════════════════════════════════════════════"

    parquet="$REPO_ROOT/data/synth/$name/data.parquet"
    was_present=0
    [[ -f "$parquet" ]] && was_present=1

    if bash "$REPO_ROOT/scripts/build_dataset.sh" "$name" $FORCE; then
        if [[ -f "$parquet" ]]; then
            if [[ $was_present -eq 1 && -z "$FORCE" ]]; then
                hit=$((hit + 1))
            else
                missed=$((missed + 1))
            fi
        fi
    else
        echo "  ✗ FAILED: $name"
        failed+=("$name")
    fi
done

echo
echo "══════════════════════════════════════════════════════════════"
echo " Corpus build summary"
echo "══════════════════════════════════════════════════════════════"
echo "  Cache hits (already present):  $hit"
echo "  Newly built:                   $missed"
echo "  Failed:                        ${#failed[@]}"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "  Failed members: ${failed[*]}"
    exit 2
fi
echo
echo "  Use in an experiment via: experiment=dgx_pretrain_corpus"
