# sf_agent: Street Fighter RL agent with threat prediction

Deep Learning project, KU Leuven 2026–27. Oleksandr Zahdai and Roman Dzhus.

**Research question:** Does an LSTM policy trained with an *auxiliary threat-prediction head*
(predicting "will I be hit in the next k steps?") win more often and learn faster than
(a) memory-less PPO and (b) a plain LSTM policy, when fighting the built-in CPU?

The agent reads game state from RAM, not pixels (SF II' Special Champion Edition, Genesis, via
stable-retro; needs your own ROM, no account or servers). Backends: `retro` (the game) and `dummy`
(a tiny toy fighting game for smoke tests, no ROM). SF3 / DIAMBRA support was removed to keep the repo
small (SF3 RAM has no player positions); it is in the git history before the "Minimal" commit.

## Layout

```
train.py  eval.py  demo.py  bench_fps.py
configs/   base.yaml (all defaults, commented), ppo_mlp, ppo_cnn, lstm, lstm_aux, lstm_frame, smoke
sf/envs/   retro_backend.py, dummy_backend.py, common.py, __init__.py (factory)
sf/algos/  aux_recurrent_ppo.py   <- RecurrentPPO + PPG-style auxiliary phase (the method)
sf/        agent.py (build/load/play), callbacks.py (win-rate eval), config.py
slurm/     bench, seeds (3-seed array), train_gpu
vsc/       setup_vsc.sh, env.sh
tools/     probe_ram.py, make_state.py, plot_runs.py
tests/     end-to-end smoke tests (no ROM needed)
```

## 1. Laptop quickstart (5 min, no ROM)

```bash
python3.11 -m venv .venv && source .venv/bin/activate      # macOS arm64: use Python 3.11 or 3.12
pip install torch && pip install -r requirements.txt
python -m pytest -q tests/                                 # 7 tests, about 2 min
python train.py -c configs/smoke.yaml                      # toy game, writes runs/smoke/seed0
python eval.py runs/smoke/seed0 --rounds 20
```

## 2. SF2 backend (`retro`)

```bash
python -m stable_retro.import ~/roms          # folder containing the SF2 SCE Genesis ROM (sha1 a5aad1d1...)
python tools/probe_ram.py                     # VERIFY the added x/y/status RAM addresses (see below)
python train.py -c configs/lstm.yaml --set train.n_envs=8 train.total_timesteps=200000   # local test run
python demo.py runs/lstm/seed0                # watch it (q quits)
```

**Verify before trusting it:** `agent_x`, `enemy_x`, `agent_y`, `enemy_y` and `*_status` in
`sf/envs/sf2_genesis_data.json` come from community RAM maps, not from the official integration.
`probe_ram.py` walks right, walks left and jumps. `agent_x` must go up, then down, and `agent_y`
must change during the jump. If they don't, fix the addresses before running experiments.

## 3. VSC (KU Leuven HPC)

Development (per Lab 0): start a **VS Code Tunnel** in VSC OnDemand (cluster wice, account
`lp_deeplearning_gt_2026b`, partition `interactive`), open `$VSC_DATA/sf_agent` in VS Code and use its terminal
for setup, `tools/probe_ram.py`, short test runs and editing. Long runs go through `sbatch` (below), always
submitted **from the repo root** (paths are relative, and `logs/` must exist).
Storage: keep the repo, venv, ROMs and `runs/` in `$VSC_DATA` (backed up, 75 GiB). Never use `$VSC_HOME`
(3 GiB). `$VSC_SCRATCH` is faster but not backed up and cleaned periodically, so if you move `out_dir` there,
copy finished runs back.


```bash
# once, on a login node, from the repo root in $VSC_DATA:
bash vsc/setup_vsc.sh $VSC_DATA/roms          # venv on top of the PyTorch module, imports ROM, runs tests
sbatch slurm/bench.slurm                      # FIRST: steps/s for 1/8/16/32 envs -> budget
sbatch slurm/seeds.slurm configs/ppo_mlp.yaml # baseline, 3 seeds
sbatch slurm/seeds.slurm configs/lstm.yaml
sbatch slurm/seeds.slurm configs/lstm_aux.yaml
squeue --clusters=wice --user=$USER   # PD = waiting, R = running
sacct --clusters=wice -j <jobid>      # state after it finished (TIMEOUT -> resume from checkpoint)
tail -f logs/*.out ; cat logs/*.err    # .err holds the Python traceback
tensorboard --logdir runs                     # via OnDemand, or copy runs/ to the laptop
python tools/plot_runs.py runs/ppo_mlp runs/lstm runs/lstm_aux -o figures/win_rate.png
```

- Training is **CPU-bound** (emulator), so use wice `batch` nodes. GPUs only matter for the frame ablation (`slurm/train_gpu.slurm`).
- 12 h limit: checkpoints are saved every 250k steps. To resume, add `train.resume=runs/.../checkpoints/rl_model_<N>_steps.zip`.
- Check the credit balance of `lp_deeplearning_gt_2026b` before launching big sweeps.

## 4. Demo on the laptop

Copy a run folder from the VSC (`scp -r` or OnDemand download). It contains `model.zip`, `config.yaml`
and `vecnormalize.pkl`. Then run `python demo.py runs/<name>/seed0` (retro).
The demo builds the environment from the run's own `config.yaml`, so it is identical to training.

## Experiments

| id | config | question |
|---|---|---|
| E1 | `ppo_mlp` | baseline: no memory (4 stacked observations + last 8 actions) |
| E2 | `lstm` | does memory help? |
| E3 | `lstm_aux`, `aux.coef` ∈ {0, 0.1, 0.5, 1.0} | does threat prediction help? (`coef=0` is the same code path as a control) |
| E4 | best of E2/E3, `--set env.difficulty=...` or `env.retro.state_path=...` at **eval** | generalisation to stronger / unseen opponents |
| E5 | `lstm_frame` | RAM only vs RAM + 84×84 frame |
| E6 | `ppo_cnn` | pixels only (CNN, 4 stacked frames) vs RAM: what does reading RAM buy? |

Several start states (opponents) can be mixed in training: `--set env.retro.state_paths='[states/a.state,states/b.state]'` (one drawn at random per episode).

Metrics (logged automatically):
- `eval/win_rate` with a 95% Wilson CI;
- `eval/mean_hp_diff`;
- `eval/action_entropy` and `eval/top_action_share`, which reveal "spam one move" exploits;
- `aux/accuracy` and `aux/bce`;
- `rollout/round_win_rate`.

Run 3 seeds per setting. Report mean ± 95% CI across seeds (`tools/plot_runs.py`).

## Method in one paragraph

`AuxRecurrentPPO` (sf/algos) records `info["hp_lost"]` during each rollout. It builds labels
$y_t = \mathbb{1}[\text{hit during } t..t+k-1]$ and masks windows that run past the rollout end.
After the normal PPO update, it runs an auxiliary phase as in Phasic Policy Gradient
(Cobbe et al., 2021). This phase minimises $\text{BCE}(\sigma(w^\top h_t), y_t) + \beta\,\mathrm{KL}(\pi_{\text{old}}\,\|\,\pi)$
over 32-step sequences replayed through the actor LSTM from their stored hidden states. The KL term keeps the policy
from drifting while the shared LSTM is shaped by the threat-prediction task. The label needs no access to the
opponent's actions, so it works against the built-in CPU.
