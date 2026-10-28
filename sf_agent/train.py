"""Train an agent.

  python train.py -c configs/smoke.yaml                          # 1-min toy test, no ROM
  python train.py -c configs/lstm_aux.yaml --seed 1              # real run (retro backend)
  diambra run -s=17 python train.py -c configs/lstm.yaml --set env.backend=diambra train.n_envs=16
      (DIAMBRA needs n_envs + 1 engines: the extra one is the evaluation env)

Outputs in runs/<run_name>/seed<seed>/:
  model.zip, vecnormalize.pkl, best_model.zip, config.yaml, progress.csv, tb/, checkpoints/
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import yaml
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from sf.agent import ALGOS, build_model, is_recurrent
from sf.callbacks import RoundStatsLogger, WinRateEval
from sf.config import load_config
from sf.envs import make_vec_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.n_envs=32")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    run_dir = os.path.join(cfg["out_dir"], cfg["run_name"], f"seed{cfg['seed']}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    t, env_cfg = cfg["train"], cfg["env"]
    venv = make_vec_env(env_cfg, t["n_envs"], start_rank=0, monitor_dir=os.path.join(run_dir, "monitor"))
    eval_env = make_vec_env(env_cfg, 1, start_rank=t["n_envs"])

    resume = t.get("resume")
    if t.get("normalize_reward", True):
        pkl = None
        if resume:  # CheckpointCallback writes rl_model_vecnormalize_<N>_steps.pkl next to rl_model_<N>_steps.zip
            m = re.search(r"rl_model_(\d+)_steps", resume)
            cand = glob.glob(os.path.join(os.path.dirname(resume), f"*vecnormalize_{m.group(1)}_steps.pkl")) if m else []
            pkl = cand[0] if cand else None
        venv = VecNormalize.load(pkl, venv) if pkl else VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=t["gamma"])

    if resume:
        model = ALGOS[cfg["algo"]].load(resume, env=venv, device=t["device"])
        print(f"resumed from {resume} at {model.num_timesteps} steps")
    else:
        model = build_model(cfg, venv, tb_dir=None)
    model.set_logger(configure(run_dir, ["stdout", "csv", "tensorboard"]))

    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(t["checkpoint_every"] // t["n_envs"], 1),
                           save_path=os.path.join(run_dir, "checkpoints"),
                           save_vecnormalize=True),
        WinRateEval(eval_env, cfg["eval"]["every"], cfg["eval"]["rounds"],
                    is_recurrent(cfg["algo"]), run_dir,
                    deterministic=cfg["eval"].get("deterministic", False)),
        RoundStatsLogger(),
    ])
    try:
        model.learn(total_timesteps=t["total_timesteps"], callback=callbacks,
                    reset_num_timesteps=not bool(resume))
    finally:
        model.save(os.path.join(run_dir, "model"))
        if isinstance(venv, VecNormalize):
            venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
        venv.close()
        eval_env.close()
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
