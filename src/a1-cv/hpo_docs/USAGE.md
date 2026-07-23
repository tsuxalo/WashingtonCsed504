# Installation, CLI, and Colab Usage

## Install additions

From the repository root:

```bash
python -m pip install -r src/a1-cv/hpo_requirements.txt
```

Do not reinstall PyTorch in Colab unless a verified incompatibility requires it.

## CLI

Run from the repository root or pass `--repo-root` before the subcommand:

```bash
PYTHONPATH=src/a1-cv python -m hpo.cli validate-space \
  --config src/a1-cv/hpo_configs/colab/resnet18_cifar10_successive_halving.yaml

PYTHONPATH=src/a1-cv python -m hpo.cli estimate \
  --config src/a1-cv/hpo_configs/colab/resnet18_cifar10_successive_halving.yaml

PYTHONPATH=src/a1-cv python -m hpo.cli search \
  --config src/a1-cv/hpo_configs/colab/resnet18_cifar10_successive_halving.yaml \
  --mode successive_halving

PYTHONPATH=src/a1-cv python -m hpo.cli search \
  --config src/a1-cv/hpo_configs/colab/resnet18_cifar10_continuous.yaml \
  --mode proxy --continuous

PYTHONPATH=src/a1-cv python -m hpo.cli report \
  --study-path hpo_outputs/resnet18-cifar10-colab-practical
```

Add `--smoke-test` to estimate/search commands for a tiny synthetic CPU path.

## Notebooks

- `hpo_smoke_test_colab.ipynb`: validates inputs, adapters, calibration, modes, pruning, persistence, resume, and exports.
- `hyperparameter_search_colab.ipynb`: user-facing persistent search and hyperparameter selection.

In Colab, select the accelerator first, then run setup. The notebook detects the assigned hardware at runtime and preserves Colab's installed CUDA PyTorch package.

## Output

Each study writes:

- SQLite study database;
- `trials.jsonl` and `events.jsonl`;
- `state.json` and sampler state;
- checkpoints;
- all-trials and Pareto CSV exports;
- selected-configuration JSON files;
- environment, resolved configuration, estimates, and summaries.
