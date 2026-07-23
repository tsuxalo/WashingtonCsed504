# Estimation, Accounting, and Cost

## Calibration

Batch calibration runs representative forward/backward/optimizer steps, synchronizes CUDA/MPS timing, catches OOM, clears allocator state, and records throughput plus peak allocated/reserved memory. The largest fitting batch and highest-throughput batch are reported separately.

The search estimator consumes actual calibration records from the selected model, data strategy, augmentation, batch, optimizer, precision, validation, and checkpoint path. It uses medians and reports assumptions.

## Mode-specific projection

- Proxy: trial count, data fraction, epoch/step cap, validation overhead, promotion count.
- Successive halving: entrants and survivors per rung, incremental checkpoint continuation, final evaluations.
- Full: full budget per sampled or exhaustive candidate and seed.
- Continuous: expected work must be interpreted inside its configured stopping budget.

Each range is labeled calibrated projection, analytical estimate, or heuristic estimate. A projection is never labeled a measurement.

## Measured trial fields

Wall and CPU time, GPU/CPU hours, epochs, optimizer steps, training/validation/total examples, parameter count, repository perfkit FLOPs when available, approximate training FLOPs, throughput, peak allocated/reserved GPU memory, process RSS, checkpoint size, device, precision, memory format, and data strategy.

## Cost

Rates are user supplied. Missing rates produce compute quantities plus `cost unavailable`; zero is not silently interpreted as free. Subscription cost is identified as sunk rather than marginal. Storage is reported separately.

## Validation

`compare_estimate` reports absolute error, percentage error, median absolute percentage error, underestimation rate, overestimation rate, and calibration sample count.
