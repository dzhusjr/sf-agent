#!/bin/bash
# Start N DIAMBRA engines with Apptainer (no Docker on the VSC) and export DIAMBRA_ENVS,
# exactly like `diambra run -s=N` does on a laptop.  Usage (sourced!):
#   source vsc/start_engines_apptainer.sh 17      ...train...      stop_engines
# Needs vsc/engine.env filled in (Step 0). Test interactively first:
#   srun -A lp_deeplearning_gt_2026b -M wice -p interactive -c 8 -t 1:00:00 --pty bash -l   (or a VS Code tunnel terminal)
#   source vsc/env.sh && source vsc/start_engines_apptainer.sh 2 && python bench_fps.py --backend diambra --n 2
N=${1:?number of engines}
source "$(dirname "${BASH_SOURCE[0]}")/engine.env"
if [ ! -f "$ENGINE_SIF" ]; then
  apptainer pull "$ENGINE_SIF" "$ENGINE_IMAGE"
fi
ENGINE_PIDS=()
DIAMBRA_ENVS=""
mkdir -p logs
for i in $(seq 0 $((N - 1))); do
  PORT=$((BASE_PORT + i))
  CMD=${ENGINE_CMD//\{PORT\}/$PORT}
  apptainer exec --writable-tmpfs --cleanenv \
      --bind "$ROMS_DIR:$ROMS_MOUNT:ro" --bind "$CREDS_FILE:$CREDS_MOUNT:ro" \
      "$ENGINE_SIF" $CMD > "logs/engine_${SLURM_JOB_ID:-local}_$i.log" 2>&1 &
  ENGINE_PIDS+=($!)
  DIAMBRA_ENVS+="127.0.0.1:$PORT "
done
export DIAMBRA_ENVS
sleep 10
for i in "${!ENGINE_PIDS[@]}"; do
  if ! kill -0 "${ENGINE_PIDS[$i]}" 2>/dev/null; then
    echo "engine $i died -- see logs/engine_${SLURM_JOB_ID:-local}_$i.log"; tail -20 "logs/engine_${SLURM_JOB_ID:-local}_$i.log"
  fi
done
echo "DIAMBRA_ENVS=$DIAMBRA_ENVS"
stop_engines() { kill "${ENGINE_PIDS[@]}" 2>/dev/null || true; }
