"""Helper for inspect_diambra_engine.sh: show what `diambra run` passes to the agent."""
import os
import time

print("DIAMBRA_ENVS =", os.environ.get("DIAMBRA_ENVS"))
print({k: v for k, v in os.environ.items() if "DIAMBRA" in k})
time.sleep(90)   # keep engines alive while the script runs docker inspect
