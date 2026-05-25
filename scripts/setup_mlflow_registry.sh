#!/usr/bin/env bash
# setup_mlflow_registry.sh — stand up an MLflow Registry on the DGX.
#
# Idempotent. Air-gap safe (uses sqlite + local artifact store; no remote DB).
#
# After running:
#   - Tracking server listens on http://127.0.0.1:5000
#   - Backend store: ./mlflow_registry/registry.db (SQLite)
#   - Artifact store: ./mlflow_registry/artifacts/
#
# Export `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` to make the Python
# client (including foundation-model loader) use the Registry.
#
# Run from repo root: ./scripts/setup_mlflow_registry.sh

set -euo pipefail

REG_DIR="${PWD}/mlflow_registry"
mkdir -p "${REG_DIR}/artifacts"

DB_URI="sqlite:///${REG_DIR}/registry.db"
ARTIFACT_ROOT="${REG_DIR}/artifacts"

echo "==> Starting MLflow Registry server..."
echo "    backend:   ${DB_URI}"
echo "    artifacts: ${ARTIFACT_ROOT}"
echo "    URL:       http://127.0.0.1:5000"
echo

# Foreground by default; users can backgound with nohup or systemd.
exec uv run mlflow server \
    --backend-store-uri "${DB_URI}" \
    --default-artifact-root "${ARTIFACT_ROOT}" \
    --host 127.0.0.1 \
    --port 5000
