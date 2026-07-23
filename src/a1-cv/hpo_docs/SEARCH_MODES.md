# Search Modes

## Proxy

Every sampled candidate receives a low-fidelity budget: configurable epochs/steps, train fraction, validation fraction, and promotion count. Proxy changes are recorded in the trial resource record. The model family, label distribution, metric, optimizer semantics, and preprocessing are not silently changed.

## Successive halving

1. Run a broad proxy cohort.
2. Keep the strongest `promote_top_k` candidates by the declared primary objective.
3. Allocate increasing epoch, step, data-fraction, or seed budgets.
4. Persist checkpoints and continue survivors when compatible.
5. Record promotion and elimination events.
6. Optionally restart final survivors for repeated-seed full confirmation.
7. Construct the Pareto frontier from completed measurements.

## Full

All valid sampled candidates receive the complete configured budget. Performance pruning is disabled. Safety termination remains available for invalid trials, OOM, divergence, interruption, and impossible exhaustive sizes.

Fully discrete spaces can use exhaustive enumeration after an exact conditional combination count and a safety limit.

## Continuous

Continuous execution wraps proxy, successive halving, or full search in resumable bounded sessions. It stops on any configured trial, wall-time, GPU-hour, CPU-hour, cost, target-metric, no-improvement, Pareto-stagnation, or manual-interruption condition. Results and sampler state are persisted after each trial/session. Colab is not treated as indefinitely available.

## Recommended staged workflow

`proxy → successive halving → full confirmation → repeated seeds`

Each stage can be disabled, and individual modes can run independently.
