"""Download the ESA OPS-SAT anomaly-detection dataset (connected machine only).

CONNECTED MACHINE ONLY — do not run on the air-gapped DGX. After running,
sneakernet the resulting data/ops_sat_raw/ directory to the target host.

OPS-SAT is ESA's flying software-defined satellite testbed. The
"OPS-SAT-AD" dataset published by ESA + collaborators contains labeled
housekeeping-telemetry anomalies from real orbit operations — hundreds
of channels, weeks of contiguous coverage, an order of magnitude more
data than NASA Telemanom.

--- URL VERIFICATION REQUIRED ---

The dataset URL below is the best I know of at time of writing. Before
kicking off a large download, VERIFY it against:

    https://opssat.esa.int/                           (ESA project page)
    https://zenodo.org/search?q=OPS-SAT               (Zenodo mirrors)
    https://www.esa.int/Enabling_Support/Space_Engineering_Technology/OPS-SAT

If the URL has changed (Zenodo rotates records), find the current one,
update DATASET_URL below, and re-run. The file layout is documented
below the URL — if the released format changes, you'll also need to
adjust `scripts/convert_ops_sat_to_parquet.py`.

Usage:
    python scripts/download_ops_sat.py
    python scripts/download_ops_sat.py --url https://new.url.here/file.zip
    python scripts/download_ops_sat.py --output-dir /mnt/nas/spacecraft
"""

from __future__ import annotations

import argparse
import shutil
import socket
import sys
import urllib.request
import zipfile
from pathlib import Path

# ── VERIFY BEFORE PRODUCTION USE ────────────────────────────────────────
# Best-known public source at time of writing. Update if superseded.
DATASET_URL = "https://zenodo.org/records/12588359/files/ops-sat-ad.zip"
# ─────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "ops_sat_raw"


# Expected layout after unzip (as of dataset v1.x — verify manually):
#
#   ops_sat_raw/
#   ├── telemetry.csv         timestamp + one column per parameter (~few hundred)
#   ├── anomalies.csv         start_time, end_time, label per anomaly interval
#   ├── channels_metadata.csv (optional) channel id, units, subsystem
#   └── README.txt / LICENSE  dataset provenance
#
# If reality differs, adjust scripts/convert_ops_sat_to_parquet.py's
# `_read_ops_sat_raw()` function.


def _has_network(host: str, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DATASET_URL,
                        help=f"OPS-SAT download URL (default: {DATASET_URL})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Where to unpack (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--zip-only", action="store_true",
                        help="Download the zip but skip unpacking")
    args = parser.parse_args()

    host = args.url.split("//", 1)[-1].split("/", 1)[0]
    if not _has_network(host):
        print(
            f"ERROR: No network connection to {host}. This script must run on "
            f"a connected machine. Transfer the output to the air-gapped DGX.\n"
            "See docs/ops_sat_provisioning.md for the manual sneakernet steps.",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.output_dir / "ops_sat.zip"

    print(f"Downloading {args.url}\n         -> {zip_path}")
    try:
        with urllib.request.urlopen(args.url) as resp, open(zip_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            size_gb = total / 1e9 if total > 0 else None
            if size_gb:
                print(f"  Expected size: ~{size_gb:.2f} GB")
            shutil.copyfileobj(resp, f)
    except urllib.error.HTTPError as e:
        print(f"ERROR: {e.code} {e.reason} from {args.url}", file=sys.stderr)
        print(
            "The URL may have moved — check https://zenodo.org/search?q=OPS-SAT "
            "and re-run with --url <new-url>.", file=sys.stderr,
        )
        return 2

    print(f"  Downloaded {zip_path.stat().st_size / 1e9:.2f} GB")

    if args.zip_only:
        print("--zip-only set; skipping unpack.")
        return 0

    print(f"Unpacking -> {args.output_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.output_dir)
    print(f"  Unpacked. Layout:")
    for p in sorted(args.output_dir.rglob("*"))[:20]:
        rel = p.relative_to(args.output_dir)
        marker = "/" if p.is_dir() else ""
        print(f"    {rel}{marker}")

    print("\nNext steps:")
    print(f"  1. Convert to our parquet layout:")
    print(f"       python scripts/convert_ops_sat_to_parquet.py \\")
    print(f"           --raw {args.output_dir} \\")
    print(f"           --output data/ops_sat")
    print(f"  2. Verify: dgx-ts train dataset=cached/ops_sat model=rolling_mean --cfg job")
    print(f"  3. If air-gapping to the DGX, tar the converted dir:")
    print(f"       tar czf ops_sat.tar.gz data/ops_sat/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
