"""End-to-end smoke tests (no ROM, no account, no Docker). ~2-3 min on a laptop CPU.

  python -m pytest -q tests/
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SMALL = ["train.total_timesteps=2048", "train.n_steps=128", "train.batch_size=256",
         "train.n_envs=2", "eval.every=1024", "eval.rounds=2", "train.checkpoint_every=1024"]


def run(args, tmp):
    cmd = [sys.executable, "train.py", "-c", "configs/smoke.yaml", "--set", f"out_dir={tmp}", *SMALL, *args]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)


@pytest.mark.parametrize("algo,extra", [
    ("ppo", ["env.frame_stack=4"]),
    ("recurrent_ppo", []),
    ("aux_recurrent_ppo", []),
    ("recurrent_ppo", ["env.use_frame=true"]),
    ("aux_recurrent_ppo", ["env.backend=diambra_mock"]),
])
def test_train(tmp_path, algo, extra):
    run([f"algo={algo}", f"run_name={algo}", *extra], str(tmp_path))
    d = tmp_path / algo / "seed0"
    assert (d / "model.zip").exists() and (d / "vecnormalize.pkl").exists()
    assert (d / "progress.csv").read_text().count("\n") > 2


def test_resume_and_eval(tmp_path):
    run(["algo=aux_recurrent_ppo", "run_name=r"], str(tmp_path))
    d = tmp_path / "r" / "seed0"
    ckpt = d / "checkpoints" / "rl_model_1024_steps.zip"
    assert ckpt.exists()
    run(["algo=aux_recurrent_ppo", "run_name=r", f"train.resume={ckpt}", "train.total_timesteps=1024"], str(tmp_path))
    subprocess.run([sys.executable, "eval.py", str(d), "--rounds", "3"], cwd=ROOT, check=True, capture_output=True)
    assert (d / "eval_default.json").exists()


def test_threat_labels():
    from sf.algos.aux_recurrent_ppo import threat_labels
    hit = np.array([[0], [0], [1], [0], [0], [0]], np.float32)
    done = np.zeros_like(hit)
    y, m = threat_labels(hit, done, k=2)
    assert y[:, 0].tolist() == [0, 1, 1, 0, 0, 0]
    assert m[:, 0].tolist() == [1, 1, 1, 1, 1, 0]   # last step: window runs past the rollout
