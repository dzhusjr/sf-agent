#!/bin/bash -l
# Throughput test -- run this first. Output: logs/sf-bench-<jobid>.out
#SBATCH --job-name=sf-bench
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=36
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
source vsc/env.sh
python bench_fps.py --backend retro --n 1 8 16 32 --seconds 30
