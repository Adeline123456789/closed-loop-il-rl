#!/usr/bin/env bash
# Candidate environment install, distinct from CPU test environment. Does not accept EULA.
set -euo pipefail
python3.11 -m venv .venv
PIP_CONFIG_FILE=/dev/null .venv/bin/python -m pip install 'isaacsim[all]==5.0.0' --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com
PIP_CONFIG_FILE=/dev/null .venv/bin/python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
if [[ ! -d .cache/IsaacLab ]]; then git clone --branch v2.2.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git .cache/IsaacLab; fi
actual=$(git -C .cache/IsaacLab rev-parse HEAD)
[[ "$actual" == 0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20 ]] || { echo 'Isaac Lab SHA mismatch' >&2; exit 2; }
for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
  PIP_CONFIG_FILE=/dev/null .venv/bin/python -m pip install -e ".cache/IsaacLab/source/$pkg"
done
.venv/bin/python -m pip install -e '.[test]'
