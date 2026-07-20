# perf — estimate training time *before* you train

The AI-Model-Factory calibrator: probe a machine's real ceilings, characterize a workload's
FLOPs/memory, and predict wall-clock training time — so a 2-day configuration gets redesigned
before it starts, not discovered on day 2.

## Why (the motivating numbers)

The identical 40-epoch CIFAR-100 resnet18 run takes **~2.1 min** on the dual RTX PRO 6000
workstation and **~10.6 min** on an RTX 2000 Ada laptop — 6.7× apart — while the spec-sheet FLOPS
ratio of those GPUs is an order of magnitude wider. The binding constraint changes with the
machine (per-step launch overhead on the big card, the 60 W power cap on the laptop), so accurate
estimation needs measurement, not spec sheets.

## Contents

| file | what it is |
|---|---|
| `training_time_estimator.ipynb` | run this on every machine (plugged in) — probes, estimates, calibrates, validates, saves a record |
| `collect.py` | the same collection run headless, no Jupyter needed — handy over SSH or on a machine without a browser |
| `perfkit.py` | the measurement/estimation library both of the above drive |
| `results/*.json` | one record per (machine, power-state, workload) run — commit these; estimates compound as the database grows |

## The three tiers (accuracy stated honestly)

- **Tier 0 — probe** (~3 min, once per machine + power state): burst and **sustained** GEMM
  TFLOPS, memory bandwidth, kernel-launch overhead, H2D bandwidth. Never marketing numbers.
- **Tier 1 — analytical roofline** (no target hardware needed):
  `t_step = max(compute/MFU, memory, launch)` over an MFU band, which answers *"2 minutes or 2 days?"*
  Order-of-magnitude by design; this is the **redesign gate**.
- **Tier 2 — calibration** (~2.5 min on the target): the real training step, warmed up and
  thermally soaked, extrapolated. **~5–10%** on desktops, **±10–15%** on power-capped laptops.
  (Published cross-GPU predictors — Habitat, ATC'21 — average ~11.8% error; nothing reliably
  beats ~5% without running on the exact target.)

## House rules learned from this repo's own data

- **Power state is machine identity**: battery vs AC differ enough to be separate records
  (the notebook warns and tags each record `ac`/`batt`).
- **Soak before you measure**: this laptop runs epochs 17% faster in its first ~90 s (boost
  clocks) than at steady state. A quick-sample estimator under-predicts every long run.
- **MFU is per-architecture**: on the same GPU, resnet18 ≈ 30% MFU, ViT ≈ 16%. Never reuse a
  machine-level efficiency factor across model families.
- **Never scale by raw TFLOPS ratio**: evaluate the full roofline `max()` on the target; if the
  binding term changes, calibrate there instead of transferring.
- **The calibration step must mirror the training recipe** (same dtype/flags/augmentation) — it
  currently copies `../cifar100_train.ipynb` section 4. Change one, change both.
