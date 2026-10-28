#!/bin/bash -l
# DIAMBRA (SF3) on the VSC through Apptainer. Requires vsc/engine.env to be filled in
# from tools/inspect_diambra_engine.sh output (Step 0, done once on a laptop with Docker).
#   sbatch slurm/train_diambra.slurm configs/lstm_aux.yaml 0
#SBATCH --job-name=sf-diambra
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
CONFIG=${1:?config}; SEED=${2:-0}
shift; shift || true
source vsc/env.sh
N_ENVS=16
source vsc/start_engines_apptainer.sh $((N_ENVS + 1))     # +1 engine for the evaluation env; exports DIAMBRA_ENVS
trap stop_engines EXIT
python train.py -c "$CONFIG" --seed "$SEED" --set env.backend=diambra train.n_envs=$N_ENVS train.device=cpu "$@"
