from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hpo.trial_runner import TrialResource, TrialRunner


@pytest.mark.cuda
def test_tiny_cuda_trial_when_available(tmp_path: Path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    repo_root = Path(__file__).resolve().parents[4]
    runner = TrialRunner(
        repo_root,
        dataset_config={"name": "synthetic", "train_examples": 16, "validation_examples": 8, "test_examples": 8, "num_classes": 10},
        model_config={"name": "vit", "hidden_dim": 48, "layers": 1, "heads": 3, "mlp_dim": 96, "patch_size": 8},
        runtime_config={"device": "cuda", "precision": "auto", "intraop_threads": 1, "interop_threads": 1, "seed": 42},
        output_dir=tmp_path,
    )
    result = runner.run(
        candidate_id="cuda-smoke",
        params={"optimizer": "adamw", "learning_rate": 0.001, "batch_size": 8, "weight_decay": 0.01, "channels_last": False, "strong_augmentation": False},
        resource=TrialResource(stage="full", target_epochs=1, max_steps=1, seed=42, continue_checkpoint=False),
        maximum_total_epochs=1,
    )
    assert result.status == "completed"
    assert result.metrics["device"] == "cuda"


@pytest.mark.cuda
def test_multi_gpu_scheduler_when_available():
    if torch.cuda.device_count() < 2:
        pytest.skip("fewer than two CUDA devices")
    from hpo.hardware import detect_hardware
    from hpo.scheduler import plan_resources
    plan = plan_resources(detect_hardware(), device="cuda")
    assert plan.concurrent_trials == torch.cuda.device_count()
