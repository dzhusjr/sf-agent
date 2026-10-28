"""Milestone / poster figure: eval win rate vs env steps, mean +- 95% CI over seeds.

  python tools/plot_runs.py runs/ppo_mlp runs/lstm runs/lstm_aux -o figures/win_rate.png
  python tools/plot_runs.py runs/lstm runs/lstm_aux --metric aux/accuracy -o figures/aux_acc.png
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help="runs/<run_name> folders containing seed*/progress.csv")
ap.add_argument("--metric", default="eval/win_rate")
ap.add_argument("-o", "--out", default="figures/win_rate.png")
args = ap.parse_args()

fig, ax = plt.subplots(figsize=(6, 4))
for run in args.runs:
    curves = []
    for csv in sorted(glob.glob(os.path.join(run, "seed*", "progress.csv"))):
        df = pd.read_csv(csv)
        if args.metric not in df:
            continue
        df = df[["time/total_timesteps", args.metric]].dropna()
        curves.append(df.set_index("time/total_timesteps")[args.metric])
    if not curves:
        print(f"no {args.metric} in {run}")
        continue
    grid = np.linspace(0, min(c.index.max() for c in curves), 50)
    ys = np.stack([np.interp(grid, c.index.values, c.values) for c in curves])
    mean = ys.mean(0)
    half = 1.96 * ys.std(0, ddof=1) / np.sqrt(len(ys)) if len(ys) > 1 else np.zeros_like(mean)
    ax.plot(grid, mean, label=f"{os.path.basename(run.rstrip('/'))} (n={len(ys)})")
    ax.fill_between(grid, mean - half, mean + half, alpha=0.2)
ax.set_xlabel("environment steps")
ax.set_ylabel(args.metric)
ax.grid(alpha=0.3)
ax.legend()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.tight_layout()
fig.savefig(args.out, dpi=200)
print("->", args.out)
