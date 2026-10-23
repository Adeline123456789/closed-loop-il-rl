#!/usr/bin/env bash
# Candidate environment install, distinct from CPU test environment. Does not accept EULA.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -x .venv/bin/python ]] || python3.11 -m venv .venv
if command -v uv >/dev/null 2>&1; then
  UV=(uv)
else
  PIP_CONFIG_FILE=/dev/null .venv/bin/python -m pip install uv==0.12.10 --index-url https://pypi.org/simple
  UV=(.venv/bin/python -m uv)
fi
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
# Install the audited CUDA build first, avoiding a redundant default CUDA wheel set.
"${UV[@]}" pip install --python .venv/bin/python torch==2.7.0 torchvision==0.22.0 --default-index https://download.pytorch.org/whl/cu128
if [[ ! -d .cache/IsaacLab ]]; then git clone --branch v2.2.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git .cache/IsaacLab; fi
actual=$(git -C .cache/IsaacLab rev-parse HEAD)
[[ "$actual" == 0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20 ]] || { echo 'Isaac Lab SHA mismatch' >&2; exit 2; }
cache=()
[[ ! -d .cache/wheelhouse ]] || cache=(--find-links .cache/wheelhouse)
# Resolve the full candidate together, preserving Isaac's pinned transitive requirements.
"${UV[@]}" pip install --python .venv/bin/python "${cache[@]}" \
  --default-index https://pypi.org/simple --index https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --constraint configs/isaac_constraints.txt \
  --build-constraints configs/isaac_build_constraints.txt 'isaacsim[all]==5.0.0' \
  -e .cache/IsaacLab/source/isaaclab -e .cache/IsaacLab/source/isaaclab_assets \
  -e .cache/IsaacLab/source/isaaclab_tasks -e .cache/IsaacLab/source/isaaclab_rl -e '.[test]'
