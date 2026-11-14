#!/usr/bin/env bash
# Run the Phase 7 kernel tests and correctness check under the Triton CPU interpreter in Docker
# (Triton has no macOS build). Works on Apple Silicon (linux/arm64) and x86.
#
#   ./scripts/check_int8_kernel.sh
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src scripts tests || echo -dirty)
HOST="$(uname -m) host, Docker Desktop"
[[ "$(uname)" == "Darwin" ]] && HOST="$(sysctl -n machdep.cpu.brand_string) host, Docker Desktop"
docker run --rm -v "$PWD":/work -w /work -e TRITON_INTERPRET=1 -e GIT_COMMIT="$COMMIT" -e HOST_DESC="$HOST" \
  -e PIP_ROOT_USER_ACTION=ignore -e PIP_DISABLE_PIP_VERSION_CHECK=1 python:3.12-slim sh -c '
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
    pip install -q triton numpy pytest &&
    PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_int8_kernel.py &&
    python scripts/check_int8_kernel.py'
