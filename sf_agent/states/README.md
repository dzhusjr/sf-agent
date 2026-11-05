# Start states (SF2 Special Champion Edition, Genesis)

Saved emulator snapshots, recorded with `python tools/make_state.py states/<name>.state`
(small, contain no ROM data, so they ARE committed).

- `medium_guile.state`: Ryu vs Guile, mid CPU level. Fallback / single-opponent debugging (`--set env.retro.state_paths=null` uses it).
  A 200k-step LSTM test run gave round_win_rate ~0.3 and eval win_rate 0.35: not saturated, not hopeless.
- `hard_guile.state`: same, LEVEL at maximum. A 200k-step run had round_win_rate = 0 throughout.
  Use only for the generalisation experiment E4 (evaluate a model trained on medium against it).

The stock `Challenger` state shipped with stable-retro is too easy (a random agent wins).

- `ryu_vs_<opponent>.state` (12 files): Ryu vs every CPU opponent in ladder order, recorded by us with `tools/make_state.py`
  (key `x` KOs the opponent to climb the ladder). Saved at "ROUND 1", so the default `intro_frames=120` is right. Difficulty LEVEL as in `medium_guile`.
  Opponents: ryu (mirror), honda, chunli, zangief, blanka, dhalsim, guile, ken, balrog, vega, sagat, bison.
- `ken_vs_honda.state`: from Oleksandr's set (player = Ken). Saved after "FIGHT!", needs `--set env.retro.intro_frames=1`.
  His other states are other player characters and stay in `STATES/`.

## Train / held-out split for E4

Train on 9 opponents, test on 3 unseen ones (different play styles: rushdown, zoner, boxer):

- train: ryu, honda, chunli, zangief, guile, ken, vega, sagat, bison
- held-out: blanka, dhalsim, balrog

The train list is already the default in `configs/base.yaml` (`env.retro.state_paths`), so E1-E3 need no extra flags.

Evaluate one held-out opponent with `--set 'env.retro.state_paths=[states/ryu_vs_blanka.state]'` (same for dhalsim, balrog).
