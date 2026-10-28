#!/bin/bash -l
# Same experiment, 3 seeds in parallel (one job per seed):
#   sbatch slurm/seeds.slurm configs/lstm_aux.yaml
#   sbatch slurm/seeds.slurm configs/lstm_aux.yaml aux.coef=0.1 run_name=lstm_aux_c0.1
#SBATCH --job-name=sf-seeds
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
CONFIG=${1:?usage: sbatch slurm/seeds.slurm <config> [key=value ...]}
shift
source vsc/env.sh
N_ENVS=$((SLURM_CPUS_PER_TASK - 4))
python train.py -c "$CONFIG" --seed "$SLURM_ARRAY_TASK_ID" --set train.n_envs=$N_ENVS train.device=cpu "$@"
