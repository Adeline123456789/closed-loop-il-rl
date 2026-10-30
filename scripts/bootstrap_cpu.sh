#!/usr/bin/env bash
# Portable CPU-only verification environment; no claim this launches Isaac.
set -euo pipefail
python3.11 -m venv .venv-cpu
PIP_CONFIG_FILE=/dev/null .venv-cpu/bin/python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
PIP_CONFIG_FILE=/dev/null .venv-cpu/bin/python -m pip install -e '.[test]'
.venv-cpu/bin/python -m pytest -q
