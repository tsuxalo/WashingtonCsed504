# HPO Architecture

## Layers

1. **Normalization:** `search_space.py` converts Python dictionaries, Python lists, CSV, JSON, and YAML into typed `ParameterSpec` objects.
2. **Validation:** `conditions.py` parses a restricted Boolean grammar; `constraints.py` rejects optimizer, architecture, precision, and model incompatibilities before model allocation.
3. **Repository adapters:** `adapters.py` imports the existing model, dataset, training, checkpoint, and performance modules using the repository's `src/a1-cv` execution convention.
4. **Trial execution:** `trial_runner.py` owns one model trial, applies runtime choices, resumes checkpoints, calls the repository loop, and records measured accounting.
5. **Search orchestration:** `study.py` uses Optuna ask/tell for sampling and durable trial history while explicitly implementing proxy, successive-halving, full, and bounded continuous sessions.
6. **Resource scheduling:** `hardware.py`, `scheduler.py`, and `parallel.py` create conservative CPU/GPU plans and persistent worker processes.
7. **Estimation and calibration:** `calibration.py` benchmarks feasible batches; `estimation.py` projects mode-specific work from measured records.
8. **Persistence/reporting:** SQLite, JSON, JSONL, atomic checkpoints, sampler pickle, CSV exports, Pareto selection, parameter importance, and study summaries.

## Why successive halving is explicit

Optuna supports multi-objective studies, but direct trial pruning is not generally available for multi-objective objectives. This framework therefore promotes staged candidates using one declared primary metric, records cost/time/memory as additional measurements, fully evaluates survivors, and computes a nondominated Pareto frontier afterward. This avoids claiming unsupported multi-objective pruning.

## Additive integration

The HPO package does not copy the repository's full model builders, resident datasets, training loop, checkpoints, GPU utility, or performance estimator. The adapter only reconstructs the small orchestration layer that was embedded in `train_run.py`.
