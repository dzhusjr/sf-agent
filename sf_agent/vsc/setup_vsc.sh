#!/bin/bash -l
# One-time setup on the VSC login node (run from the repo root):
#   bash vsc/setup_vsc.sh /path/to/folder/with/roms
# The ROM folder must contain the SF2 Genesis ROM (and/or sfiii3n.zip for DIAMBRA).
set -euo pipefail
ROMS=${1:-$VSC_DATA/roms}
module purge
module load PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl
python -m venv --system-site-packages "$VSC_DATA/sf-venv"     # reuse torch from the module
source "$VSC_DATA/sf-venv/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
python -m stable_retro.import "$ROMS" || echo "WARN: no SF2 ROM imported (needed for backend=retro)"
python -m pytest -q tests/                                     # ~2-3 min, no ROM needed
echo "Setup OK. Next: sbatch slurm/bench.slurm"
