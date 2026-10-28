# Source this in every job / interactive session on the VSC:   source vsc/env.sh
module purge
module load PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl
source "$VSC_DATA/sf-venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1          # many env processes: keep each one single-threaded
