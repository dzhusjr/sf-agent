"""Evaluate every finished run (runs/<algo>/seed<N>/model.zip) on the training mix and on unseen opponents, in parallel.

  python tools/eval_all.py --runs runs_vsc/runs --rounds 100 --workers 6
  python tools/eval_all.py --runs runs --rounds 100 --all-opponents      # also each of the 12 opponents alone

Writes <runs>/eval_summary.csv (run, seed, set, won, rounds, win_rate, ci_low, ci_high) and prints a per-agent summary.
Sets: train_mix (the run's own config), held-out blanka / dhalsim / balrog, hard_guile. Run it from sf_agent/ (needs the ROM).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HELD_OUT = ["blanka", "dhalsim", "balrog"]
ALL = ["ryu", "honda", "chunli", "zangief", "guile", "ken", "vega", "sagat", "bison", "blanka", "dhalsim", "balrog"]


def one(job):
    run_dir, set_name, state, rounds = job
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from sf.agent import is_recurrent, load_model, play_rounds
    from sf.config import load_config
    from sf.envs import make_vec_env
    from sf.envs.common import wilson_ci

    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    if state:
        cfg["env"]["retro"]["state_paths"] = [state]
    venv = make_vec_env(cfg["env"], 1, start_rank=0)
    model = load_model(os.path.join(run_dir, "model.zip"), cfg["algo"])
    r = play_rounds(model, venv, rounds, is_recurrent(cfg["algo"]), False)
    venv.close()
    lo, hi = wilson_ci(r["won"], r["rounds"])
    algo, seed = os.path.normpath(run_dir).split(os.sep)[-2:]
    return {"run": algo, "seed": seed, "set": set_name, "won": r["won"], "rounds": r["rounds"],
            "win_rate": round(r["win_rate"], 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--all-opponents", action="store_true")
    args = ap.parse_args()
    runs = os.path.abspath(args.runs)

    sets = [("train_mix", None)] + [(o, f"states/ryu_vs_{o}.state") for o in (ALL if args.all_opponents else HELD_OUT)]
    sets.append(("hard_guile", "states/hard_guile.state"))
    sets = [(n, st) for n, st in sets if st is None or os.path.exists(os.path.join(ROOT, st))]   # skip missing state files
    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(runs, "*", "seed*", "model.zip")))
    jobs = [(d, n, s, args.rounds) for d in dirs for n, s in sets]
    print(f"{len(dirs)} models x {len(sets)} sets = {len(jobs)} evaluations, {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['run']}/{r['seed']} {r['set']}: {r['win_rate']:.2f} ({r['won']}/{r['rounds']})", flush=True)

    rows.sort(key=lambda r: (r["run"], r["seed"], r["set"]))
    out = os.path.join(runs, "eval_summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}\n\nmean win rate over seeds:")
    for algo in sorted({r["run"] for r in rows}):
        line = []
        for n, _ in sets:
            v = [r["win_rate"] for r in rows if r["run"] == algo and r["set"] == n]
            if v:
                line.append(f"{n} {sum(v) / len(v):.2f}")
        print(f"  {algo:11s} " + "  ".join(line))


if __name__ == "__main__":
    main()
