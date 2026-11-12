from __future__ import annotations

import os
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .agent import play_rounds
from .envs.common import wilson_ci


class WinRateEval(BaseCallback):
    """Every `every` env steps: play `rounds` rounds with the policy on a
    separate env, log eval/win_rate (+95% Wilson CI) and keep the best model."""

    def __init__(self, eval_env, every: int, rounds: int, recurrent: bool, save_dir: str,
                 deterministic: bool = False):
        super().__init__()
        self.deterministic = deterministic
        self.eval_env, self.every, self.rounds = eval_env, every, rounds
        self.recurrent, self.save_dir = recurrent, save_dir
        self.best = -1.0
        self._next = every

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        r = play_rounds(self.model, self.eval_env, self.rounds, self.recurrent, self.deterministic)
        lo, hi = wilson_ci(r["won"], r["rounds"])
        self.logger.record("eval/win_rate", r["win_rate"])
        self.logger.record("eval/win_rate_ci_low", lo)
        self.logger.record("eval/win_rate_ci_high", hi)
        self.logger.record("eval/mean_hp_diff", r["mean_hp_diff"])
        counts = np.asarray(r["action_counts"], float)
        p = counts / max(counts.sum(), 1)
        # low entropy of the action distribution = the agent spams one move
        self.logger.record("eval/action_entropy", float(-(p[p > 0] * np.log(p[p > 0])).sum()))
        self.logger.record("eval/top_action_share", float(p.max()))
        self.logger.dump(self.num_timesteps)
        if r["win_rate"] > self.best:
            self.best = r["win_rate"]
            self.model.save(os.path.join(self.save_dir, "best_model"))
        return True


class RoundStatsLogger(BaseCallback):
    """Logs training-time round win rate (all opponents, per rollout) and, for multi-opponent
    training, a rolling win rate per start state as opp/<state_name>."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.won = self.lost = 0
        self.window = window
        self.per_state: dict[str, deque] = {}

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if info.get("round_done"):
                w = bool(info.get("round_won"))
                self.won += int(w)
                self.lost += int(not w)
                name = info.get("state_name")
                if name:
                    self.per_state.setdefault(name, deque(maxlen=self.window)).append(int(w))
        return True

    def _on_rollout_end(self) -> None:
        n = self.won + self.lost
        if n:
            self.logger.record("rollout/round_win_rate", self.won / n)
        for name, d in self.per_state.items():
            if d:
                self.logger.record(f"opp/{name}", sum(d) / len(d))
        self.won = self.lost = 0


class Curriculum(BaseCallback):
    """Opponent curriculum: start with the first `start` states of env.retro.state_paths and add the
    next one whenever the rolling win rate over the last `window` rounds reaches `threshold`.
    All active opponents stay in the pool (no forgetting). Logs curriculum/active."""

    def __init__(self, n_states: int, start: int = 1, threshold: float = 0.5, window: int = 100):
        super().__init__()
        self.n, self.active = n_states, start
        self.threshold, self.window = threshold, window
        self.recent: deque = deque(maxlen=window)

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_active", self.active)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if info.get("round_done"):
                self.recent.append(int(bool(info.get("round_won"))))
        return True

    def _on_rollout_end(self) -> None:
        if self.active < self.n and len(self.recent) == self.window and sum(self.recent) / self.window >= self.threshold:
            self.active += 1
            self.recent.clear()
            self.training_env.env_method("set_active", self.active)
        self.logger.record("curriculum/active", self.active)
