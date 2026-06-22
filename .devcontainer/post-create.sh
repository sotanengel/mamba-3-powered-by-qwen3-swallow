#!/usr/bin/env bash
# Runs once after the dev container is created.
set -euo pipefail

echo "[post-create] installing project in editable mode..."
pip install --user -e ".[dev]" || pip install -e ".[dev]"

echo "[post-create] verifying mamba-ssm import..."
python - <<'PY'
try:
    from mamba_ssm import Mamba3
    print("[post-create] mamba_ssm.Mamba3 OK")
except Exception as e:  # pragma: no cover
    print(f"[post-create] WARN: cannot import Mamba3: {e}")
PY

echo "[post-create] pytest collect..."
pytest --collect-only -q || true

echo "[post-create] done."
