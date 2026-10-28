#!/bin/bash -l
# CPU training (emulator-bound; the small MLP/LSTM policy does not need a GPU).
#   sbatch slurm/train_cpu.slurm configs/lstm_aux.yaml 0
#   sbatch slurm/train_cpu.slurm configs/lstm_aux.yaml 0 aux.coef=0.5 run_name=lstm_aux_c0.5
# Resume after the time limit:
#   sbatch slurm/train_cpu.slurm configs/lstm_aux.yaml 0 train.resume=runs/lstm_aux/seed0/checkpoints/rl_model_2000000_steps.zip
#SBATCH --job-name=sf-train
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
CONFIG=${1:?usage: sbatch slurm/train_cpu.slurm <config> <seed> [key=value ...]}
SEED=${2:-0}
shift; shift || true
source vsc/env.sh
N_ENVS=$((SLURM_CPUS_PER_TASK - 4))          # leave a few cores for the learner + eval env
python train.py -c "$CONFIG" --seed "$SEED" --set train.n_envs=$N_ENVS train.device=cpu "$@"
