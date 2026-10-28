#!/bin/bash -l
# Only for the frame ablation (CNN on 84x84 frames):  sbatch slurm/train_gpu.slurm configs/lstm_frame.yaml 0
#SBATCH --job-name=sf-train-gpu
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --clusters=wice
# for quick (<=1 h) tests use --partition=gpu_a100_debug
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
CONFIG=${1:?config}; SEED=${2:-0}
shift; shift || true
source vsc/env.sh
python train.py -c "$CONFIG" --seed "$SEED" --set train.n_envs=16 train.device=cuda "$@"
