#!/usr/bin/env bash
# STEP 0 (laptop with Docker, after `pip install diambra diambra-arena` and a DIAMBRA account):
# records exactly how `diambra run` starts the engine, so it can be replicated with Apptainer on the VSC.
#   export DIAMBRAROMSPATH=~/roms        # folder with sfiii3n.zip
#   bash tools/inspect_diambra_engine.sh
# Then fill vsc/engine.env from diambra_engine_info.txt.
# Check the file for tokens/secrets before sharing it with anyone.
set -uo pipefail
: "${DIAMBRAROMSPATH:?export DIAMBRAROMSPATH=/folder/with/sfiii3n.zip first}"
OUT=diambra_engine_info.txt
{
  echo "=== check-roms"; diambra arena check-roms "$DIAMBRAROMSPATH/sfiii3n.zip"
  echo "=== uname"; uname -a
} > "$OUT" 2>&1
diambra -d run -s=2 python tools/_print_envs.py > diambra_run.log 2>&1 &
RUN_PID=$!
sleep 45
{
  echo "=== docker ps"; docker ps --no-trunc
  for id in $(docker ps -q); do
    echo "=== docker inspect $id"
    docker inspect --format '{{json .Config.Image}} {{json .Path}} {{json .Args}} {{json .Config.Env}} {{json .Config.User}} {{json .Mounts}} {{json .HostConfig.PortBindings}} {{json .Config.ExposedPorts}}' "$id"
  done
  echo "=== diambra run log (debug)"; cat diambra_run.log
} >> "$OUT" 2>&1
wait $RUN_PID
echo "wrote $OUT -- send it to Claude / use it to fill vsc/engine.env"
