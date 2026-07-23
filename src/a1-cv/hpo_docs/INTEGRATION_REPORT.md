# HPO Repository Integration Report

## Repository inspected

Authoritative inspection archive: `WashingtonCsed504-main(7).zip`.

The additions integrate with the existing repository rather than replacing it.

| Existing component | HPO use |
|---|---|
| `models.py` | `RepoModules` imports `models.build`; custom 32×32 ViT dimensions call the existing `make_vit` builder. |
| `cifar_data.py` | HPO creates the repository's GPU-resident CIFAR object, then adds a deterministic stratified train/validation view. |
| `imagenet_data.py` | HPO reuses `GpuImageNet32` when prepared ImageNet-32 data is available. |
| `train_loop.py` | Trials call the existing `train_one_epoch`, `evaluate`, `save_checkpoint`, and `load_checkpoint`. |
| `train_run.py` | Existing optimizer and augmentation recipes were reconstructed into adapter defaults; the CLI remains unchanged. |
| `train_fleet.py` | HPO follows its one-process-per-GPU design for independent trials. |
| `perf/perfkit.py` | HPO uses the repository FLOP counter and supports calibrated timing projections. |
| `src/common/gpu_check.py` | The typed HPO hardware profile can import this utility and supplements it with CPU/RAM/MPS/notebook fields. |
| Existing JSON/JSONL results | Registered as measured repository baselines, not universal optima. |

## Important findings

1. The original training path uses the CIFAR test split as the evaluation split. HPO must not tune against that split. `build_trial_dataset` creates a deterministic, class-stratified split from the original training data and reserves the test set for optional final confirmation.
2. CIFAR and ImageNet-32 are already resident-device datasets. HPO therefore defaults to zero DataLoader workers and does not create a competing loader.
3. The existing loop supports CUDA AMP, FP16 scaling, BF16, strong augmentation, channels-last CNN execution, checkpointing, and model-agnostic training.
4. The current `train_loop.py` does not implement gradient accumulation. Core HPO remains additive and rejects accumulation values above one unless the documented minimal compatibility patch is applied.
5. The repository's stored measured results include:
   - CIFAR-10 ResNet-18: 92.73% best recorded top-1, batch 512, learning rate 0.2, 30 epochs.
   - CIFAR-100 ResNet-18: 74.32%, batch 512, learning rate 0.2, 40 epochs.
   - CIFAR-10 ViT: 85.13%, learning rate 0.001, 200 epochs.
   - CIFAR-100 ViT: 62.59%, learning rate 0.001, 200 epochs.

These numbers were read from repository result files. They used the repository's historical evaluation convention and therefore are benchmark context, not directly comparable to the new HPO validation split without rerunning the reference track.

## Measured local inspection

- Existing lightweight test: `6 passed` for `src/common/test_gpu_check.py`.
- CPU thread oversubscription was measured on the inspection host: the default 56-thread pool caused a one-batch ResNet trial to exceed one minute; limiting the trial to two intra-op threads reduced it to about 1.1 seconds. The scheduler now caps CPU threads and accepts user overrides.
- A complete synthetic CPU successive-halving smoke study produced proxy, halving, and full records and completed successfully.

## Existing-file changes

No existing tracked file is silently modified by the additions archive. The core framework works without patches. A dedicated patch is supplied only for gradient accumulation above one. Optional README and `.gitignore` patches are also supplied separately.
