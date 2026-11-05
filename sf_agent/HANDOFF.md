# HANDOFF (Roman -> Oleksandr), 5 Nov 2026

Goal: train the agents on the VSC, produce win-rate curves for the milestone (**upload 13/11**, feedback 18/11),
poster 15/12, report 04/01/2027. Background and method: `README.md`.

## Where things stand

Done and verified by Roman on a Mac (SF2 `retro` backend, real game):
- Full pipeline works: train, eval, demo, resume, 7 smoke tests.
- RAM addresses (x/y/status) verified with `tools/probe_ram.py`.
- Round-intro skip (`retro.intro_frames: 120`) fixed: input is ignored for ~115 frames after reset.
- Difficulty calibrated. Stock state was too easy (random agent wins). Start states are committed in `states/` (details: `states/README.md`):
  - **12 states `ryu_vs_<opponent>.state`**: Ryu vs every CPU opponent (ryu, honda, chunli, zangief, blanka, dhalsim, guile, ken, balrog, vega, sagat, bison),
    same difficulty LEVEL as `medium_guile`. Opponent IDs were verified from RAM (`enemy_character`), file names match contents.
  - **Default training set = 9 opponents** (`env.retro.state_paths` in `configs/base.yaml`); one is drawn at random per episode.
    **Held out for E4 (never trained on): blanka, dhalsim, balrog** (rushdown, zoner, boxer).
  - `medium_guile.state` (single opponent, 200k-step LSTM test: round_win_rate ~0.3) is the fallback: `--set env.retro.state_paths=null`.
  - `hard_guile.state`: round_win_rate = 0 for 200k steps. Extra E4 test.
- Multi-opponent smoke run (LSTM, 100k steps, 9 opponents, Mac CPU, 8 envs): works; eval win_rate 0.30 (CI 0.15-0.52), round_win_rate still ~0 early in training.
  Training on 9 opponents is harder than on one, expect slower learning; check curves on the VSC before committing to 5M steps.
  If after ~1M steps nothing is learned, fall back to `state_paths=null` (Guile only) and report multi-opponent as a later experiment.
- DIAMBRA (SF3) support was removed to keep the repo minimal (their servers were down, and SF3 RAM has no positions). It is in git history.

NOT done (your part):
1. Windows setup (WSL2), see below.
2. VSC setup and all real training runs. **Nothing has been trained on the VSC yet.**
3. Baselines E1-E3, seeds, plots, milestone text.

## 0. Your machine (Windows): use WSL2

stable-retro has no Windows wheel. Install WSL2 + Ubuntu, then inside Ubuntu:
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip git build-essential
git clone git@github.com:dzhusjr/sf-agent.git && cd sf-agent/sf_agent
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch && pip install -r requirements.txt
python -m pytest -q tests/
```
You only need this for tests, demos and plotting. Training runs on the VSC.

## 1. The ROM (never in git)

File: `Street Fighter II' - Special Champion Edition (USA).md` (Genesis), sha1 `a5aad1d108046d9388e33247610dafb4c6516e0b`.
Ask Roman for it (or find it yourself and check the sha1 with `shasum`). Put it in a folder `~/roms` and import:
```bash
python -m stable_retro.import ~/roms
```
Then sanity check: `python tools/probe_ram.py` (agent_x must go up when walking right, agent_y must change when jumping).

## 2. VSC setup (once)

Cluster facts: wice, account `lp_deeplearning_gt_2026b`, partition `batch` for CPU jobs, always `--clusters=wice`.
Use `$VSC_DATA`, not `$VSC_HOME`. Training is CPU-bound (emulator), so GPUs are not needed for E1-E4.

```bash
ssh <vsc-id>@login.hpc.kuleuven.be
cd $VSC_DATA
git clone git@github.com:dzhusjr/sf-agent.git      # or https; add an SSH key to GitHub if needed
cd sf-agent/sf_agent
mkdir -p $VSC_DATA/roms                            # then scp the ROM into it from your laptop
bash vsc/setup_vsc.sh $VSC_DATA/roms               # venv + deps + ROM import + tests
sbatch slurm/bench.slurm                           # steps/s for 1/8/16/32 envs
```
Read the bench output (`logs/`) to choose `n_envs` and check the time budget: jobs are limited to 12 h,
checkpoints every 250k steps. `seeds.slurm` uses `cpus-per-task - 4` envs (32 on 36 CPUs).
Check the credit balance of the account before large sweeps.

If `setup_vsc.sh` fails on the ROM import, the ROM folder is wrong. If it fails on pytest, send the traceback to Roman.

## 3. Baseline runs (E1-E3), 3 seeds each

```bash
sbatch slurm/seeds.slurm configs/ppo_mlp.yaml      # E1: no memory
sbatch slurm/seeds.slurm configs/lstm.yaml         # E2: LSTM
sbatch slurm/seeds.slurm configs/lstm_aux.yaml     # E3: LSTM + threat head
squeue --me --clusters=wice
sacct --clusters=wice -X
tail -f logs/*.out
```
Each seed lands in `runs/<run_name>/seed<N>/` (`model.zip`, `config.yaml`, `vecnormalize.pkl`, `progress.csv`).
Defaults: 5M steps, eval every 100k, 20 rounds per eval. 20 rounds gives a wide CI (about +-0.2), so for final
numbers re-evaluate with more rounds: `python eval.py runs/lstm/seed0 --rounds 100`.

Resume after a time-out: add `train.resume=runs/<run>/seed<N>/checkpoints/rl_model_<N>_steps.zip`.

Then the ablation (E3 sweep): `sbatch slurm/seeds.slurm configs/lstm_aux.yaml aux.coef=0.1 run_name=lstm_aux_c0.1`
for `aux.coef` in {0, 0.1, 0.5, 1.0}. `coef=0` is the control.

## 4. Plots and milestone

```bash
python tools/plot_runs.py runs/ppo_mlp runs/lstm runs/lstm_aux -o figures/win_rate.png
python tools/plot_runs.py runs/lstm runs/lstm_aux --metric aux/accuracy -o figures/aux_acc.png
```
Milestone (13/11) needs: baseline win-rate curves (E1, E2), a first look at the aux head (E3), and the experiment plan.
Copy `runs/` to a laptop with `scp -r` or OnDemand download to view TensorBoard (`tensorboard --logdir runs`).

## 5. Later experiments

- **E4 generalisation:** evaluate the best model on opponents it never saw (`blanka`, `dhalsim`, `balrog`) and on `hard_guile`.
  Use 100 rounds and compare with the same model on the training opponents:
  ```bash
  for o in blanka dhalsim balrog; do
    python eval.py runs/<best>/seed0 --rounds 100 --set "env.retro.state_paths=[states/ryu_vs_$o.state]"
  done
  python eval.py runs/<best>/seed0 --rounds 100 --set "env.retro.state_paths=[states/hard_guile.state]"
  ```
  Compare E1/E2/E3 on these: does memory or the threat head generalise better to unseen opponents?
  Make more states: `python tools/make_state.py states/<name>.state` (keys: wasd move, u/i/o punches, j/k/l kicks,
  e/Enter start, x KO the opponent, p save, q quit). If the argument does not end in `.state` (e.g. `states/ryu_vs_`) the opponent name is read from RAM and appended.
  Set LEVEL in OPTIONS first, press p at "ROUND 1". Commit the new `.state` (they are small).
- **E5 frame ablation:** `sbatch slurm/train_gpu.slurm configs/lstm_frame.yaml 0` (RAM + 84x84 frame; the last arg is the seed, run seeds 0-2).
- **E6 pixel-only baseline (your idea, Oleksandr):** `sbatch slurm/train_gpu.slurm configs/ppo_cnn.yaml 0`. PPO + CNN on 4 stacked 84x84 frames, no RAM features
  (`env.use_state=false`). Compare with `ppo_mlp` to show what reading RAM buys. It needs a GPU and more steps than the RAM agents.
- **Multi-opponent check** (before a long job): `python train.py -c configs/lstm.yaml --set train.n_envs=2 train.total_timesteps=4096`
  (loads all 9 train states; `info["state_name"]` tells which one an episode used).
- **Demo on a laptop:** copy a run folder, then `python demo.py runs/<name>/seed0` (q quits). Needs the ROM imported locally.

## Known issues and watch-outs

- A deterministic (argmax) eval once gave 0% wins due to a degenerate action, so `eval.deterministic: false` is the default. Keep it.
- Look at `eval/top_action_share` and `eval/action_entropy`: a high top share means the agent spams one move.
- The RAM state has no wall-position feature; the agent can't see how close it is to the corner. Optional improvement.
- Git identity: set `git config --global user.name/email` before committing.
- `ryu_vs_bison.state` starts with the player at 128/176 HP (all others 176/176). Minor, only affects the first round; re-record with `tools/make_state.py` if it bothers you.
- Don't commit `runs/`, `logs/` or any ROM. `states/*.state` are committed on purpose.
