"""Download NASA Telemanom (SMAP + MSL) datasets to data/.

CONNECTED MACHINE ONLY — do not run on the air-gapped DGX. After running,
sneakernet the resulting data/nasa_smap/ and data/nasa_msl/ directories.

Usage:
    python scripts/download_datasets.py
"""

from __future__ import annotations

import shutil
import socket
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ZIP_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
ZIP_PATH = DATA_DIR / "telemanom.zip"


def _has_network(host: str = "s3-us-west-2.amazonaws.com", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    if not _has_network():
        print(
            "ERROR: No network connection. This script must run on a "
            "connected machine. Transfer the output to the air-gapped DGX.\n"
            "See docs/air_gapped_setup.md for the manual layout.",
            file=sys.stderr,
        )
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ZIP_URL} → {ZIP_PATH} (~600 MB)...")
    with urllib.request.urlopen(ZIP_URL) as resp, open(ZIP_PATH, "wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"  done. Extracting...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)
    print(f"  extracted to {DATA_DIR}.")
    print(
        "\nNext: sneakernet data/nasa_smap/ and data/nasa_msl/ to the DGX. "
        "See docs/air_gapped_setup.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
