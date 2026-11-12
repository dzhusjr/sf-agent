"""Clean comparison plot: curves cut to a common budget, smoothed, mean over seeds with a 95% interval.

  python tools/plot_compare.py runs_vsc/runs -o figures/win_rate.png --max-steps 3.2e6
  python tools/plot_compare.py runs_vsc/runs --metric aux/bce -o figures/aux_bce.png --algos lstm_aux
Every sub-folder of the runs directory with seed*/progress.csv is one curve.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}   # two-sided t critical value, df = n - 1


def load(path: str, metric: str):
    xs, ys = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get(metric) not in (None, "") and r.get("time/total_timesteps") not in (None, ""):
                xs.append(float(r["time/total_timesteps"]))
                ys.append(float(r[metric]))
    return np.asarray(xs), np.asarray(ys)


def smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(y) < k:
        return y
    pad = np.pad(y, (k - 1, 0), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("-o", "--out", default="win_rate.png")
    ap.add_argument("--metric", default="rollout/round_win_rate")
    ap.add_argument("--max-steps", type=float, default=None, help="cut every curve here (default: shortest run)")
    ap.add_argument("--smooth", type=int, default=8, help="moving-average window in logged points")
    ap.add_argument("--algos", nargs="*", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    curves = {}
    for d in sorted(glob.glob(os.path.join(args.runs, "*"))):
        name = os.path.basename(d)
        if args.algos and name not in args.algos:
            continue
        seeds = [load(p, args.metric) for p in sorted(glob.glob(os.path.join(d, "seed*", "progress.csv")))]
        seeds = [s for s in seeds if len(s[0]) > 3]
        if seeds:
            curves[name] = seeds
    if not curves:
        raise SystemExit("no curves found")
    limit = args.max_steps or min(max(s[0][-1] for s in seeds) for seeds in curves.values())
    grid = np.linspace(0, limit, 60)

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    for i, (name, seeds) in enumerate(curves.items()):
        ys = np.stack([np.interp(grid, x, smooth(y, args.smooth)) for x, y in seeds])
        m = ys.mean(0)
        half = T95.get(len(ys) - 1, 2.0) * ys.std(0, ddof=1) / np.sqrt(len(ys)) if len(ys) > 1 else np.zeros_like(m)
        c = f"C{i}"
        ax.plot(grid / 1e6, m, color=c, lw=1.8, label=f"{name} (n={len(ys)})")
        ax.fill_between(grid / 1e6, m - half, m + half, color=c, alpha=0.15, lw=0)
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel(args.metric)
    ax.set_xlim(0, limit / 1e6)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(args.title or f"{args.metric}: mean over seeds, 95% interval, smoothed ({args.smooth} pts)", fontsize=10)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print("->", args.out, f"(cut at {limit / 1e6:.2f}M steps)")


if __name__ == "__main__":
    main()
