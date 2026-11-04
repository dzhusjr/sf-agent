# Start states (SF2 Special Champion Edition, Genesis)

Saved emulator snapshots, recorded with `python tools/make_state.py states/<name>.state`
(small, contain no ROM data, so they ARE committed).

- `medium_guile.state`: Ryu vs Guile, mid CPU level. **Default for all experiments** (`configs/base.yaml`).
  A 200k-step LSTM test run gave round_win_rate ~0.3 and eval win_rate 0.35: not saturated, not hopeless.
- `hard_guile.state`: same, LEVEL at maximum. A 200k-step run had round_win_rate = 0 throughout.
  Use only for the generalisation experiment E4 (evaluate a model trained on medium against it).

The stock `Challenger` state shipped with stable-retro is too easy (a random agent wins).
