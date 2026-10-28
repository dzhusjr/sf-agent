"""Model construction / loading and a backend-independent "play N rounds" loop
(used by the eval callback, eval.py and demo.py)."""
from __future__ import annotations

from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from .algos.aux_recurrent_ppo import AuxRecurrentPPO

ALGOS = {"ppo": PPO, "recurrent_ppo": RecurrentPPO, "aux_recurrent_ppo": AuxRecurrentPPO}


def is_recurrent(algo: str) -> bool:
    return algo in ("recurrent_ppo", "aux_recurrent_ppo")


def build_model(cfg: dict, venv, tb_dir: str | None):
    algo, t = cfg["algo"], cfg["train"]
    common = dict(
        n_steps=t["n_steps"], batch_size=t["batch_size"], n_epochs=t["n_epochs"],
        learning_rate=t["learning_rate"], gamma=t["gamma"], gae_lambda=t["gae_lambda"],
        clip_range=t["clip_range"], ent_coef=t["ent_coef"], vf_coef=t["vf_coef"],
        max_grad_norm=t["max_grad_norm"], seed=cfg["seed"], device=t["device"],
        tensorboard_log=tb_dir, verbose=1,
    )
    if algo == "ppo":
        return PPO("MultiInputPolicy", venv, policy_kwargs=dict(net_arch=t["net_arch"]), **common)
    pk = dict(net_arch=t["net_arch"], lstm_hidden_size=t["lstm_hidden"], n_lstm_layers=1,
              enable_critic_lstm=True)
    if algo == "recurrent_ppo":
        return RecurrentPPO("MultiInputLstmPolicy", venv, policy_kwargs=pk, **common)
    a = cfg["aux"]
    return AuxRecurrentPPO("MultiInputLstmPolicy", venv, policy_kwargs=pk,
                           aux_k=a["k"], aux_coef=a["coef"], aux_epochs=a["epochs"],
                           aux_seq_len=a["seq_len"], aux_batch_seqs=a["batch_seqs"],
                           aux_lr=a["lr"], aux_clone_coef=a["clone_coef"], **common)


def load_model(path: str, algo: str, venv=None, device: str = "cpu"):
    return ALGOS[algo].load(path, env=venv, device=device)


def play_rounds(model, venv, n_rounds: int, recurrent: bool, deterministic: bool = True,
                on_step: Callable | None = None, max_steps: int = 200_000) -> dict:
    """Plays until n_rounds rounds are finished on a single-env VecEnv.
    Returns win/loss counts, mean health difference at round end, and action counts."""
    assert venv.num_envs == 1
    obs = venv.reset()
    state, start = None, np.ones((1,), dtype=bool)
    won = lost = steps = 0
    hp_diffs, actions = [], []
    while won + lost < n_rounds and steps < max_steps:
        if recurrent:
            action, state = model.predict(obs, state=state, episode_start=start, deterministic=deterministic)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, infos = venv.step(action)
        start = done
        steps += 1
        actions.append(int(action[0]))
        info = infos[0]
        if info.get("round_done"):
            won += int(bool(info.get("round_won")))
            lost += int(not info.get("round_won"))
            hp_diffs.append(info.get("own_hp", 0.0) - info.get("opp_hp", 0.0))
        if on_step is not None:
            on_step(info)
    n = won + lost
    return {"won": won, "lost": lost, "rounds": n, "win_rate": won / max(n, 1),
            "mean_hp_diff": float(np.mean(hp_diffs)) if hp_diffs else 0.0,
            "action_counts": np.bincount(actions, minlength=venv.action_space.n).tolist(),
            "steps": steps}
