from __future__ import annotations

from pathlib import Path

import torch

from hpo.baselines import load_repository_baselines
from hpo.calibration import calibrate_batch_sizes


def test_cpu_batch_calibration():
    device = torch.device("cpu")

    def build_model():
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))

    def make_batch(batch_size):
        return torch.randn(batch_size, 3, 4, 4), torch.randint(0, 2, (batch_size,))

    report = calibrate_batch_sizes(
        build_model,
        make_batch,
        device=device,
        candidates=[2, 4],
        warmup_steps=0,
        measure_steps=1,
    )
    assert report.largest_fitting_batch == 4
    assert report.highest_throughput_batch in {2, 4}
    assert report.recommended_candidates


def test_repository_baselines_are_discovered(repo_root: Path):
    baselines = load_repository_baselines(repo_root)
    assert baselines
    assert any(item.dataset == "cifar10" and item.model == "resnet18" for item in baselines)
    assert all(item.measured for item in baselines)
