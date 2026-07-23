from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hpo.costing import estimate_cost
from hpo.hardware import HardwareProfile, detect_hardware, resolve_precision
from hpo.persistence import StudyFiles, append_jsonl, atomic_write_json, read_jsonl, stable_hash
from hpo.scheduler import plan_resources
from hpo.schemas import CostRates


def _cpu_profile(cores=16, threads=32, ram=64):
    return HardwareProfile(
        operating_system="test",
        notebook=False,
        colab=False,
        python_version="3",
        torch_version="test",
        cuda_available=False,
        cuda_version=None,
        cudnn_version=None,
        gpu_count=0,
        physical_cpu_cores=cores,
        logical_cpu_threads=threads,
        available_ram_gb=ram,
        total_ram_gb=ram,
    )


def test_hardware_detection_is_typed():
    profile = detect_hardware()
    assert profile.logical_cpu_threads >= 1
    assert profile.total_ram_gb > 0


def test_cpu_plan_caps_threads_and_honors_overrides():
    plan = plan_resources(_cpu_profile(), device="cpu", requested_concurrency=2)
    assert plan.concurrent_trials == 2
    assert 1 <= plan.intraop_threads <= 8
    overridden = plan_resources(
        _cpu_profile(),
        device="cpu",
        requested_concurrency=1,
        requested_intraop_threads=3,
        requested_interop_threads=2,
        requested_workers=1,
    )
    assert (overridden.intraop_threads, overridden.interop_threads, overridden.workers_per_trial) == (3, 2, 1)


def test_precision_cpu_and_mps_fallback(monkeypatch):
    profile = _cpu_profile()
    assert resolve_precision("auto", torch.device("cpu"), profile) == "fp32"
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_precision("fp16", torch.device("cpu"), profile)
    assert resolve_precision("auto", torch.device("mps"), profile) == "fp32"


def test_cost_missing_and_complete_rates():
    metrics = {"gpu_hours": 2, "cpu_hours": 1, "checkpoint_size_mb": 1024}
    unavailable = estimate_cost(metrics, CostRates())
    assert unavailable["estimated_cost_usd"] is None
    complete = estimate_cost(
        metrics,
        CostRates(gpu_usd_per_hour=1, cpu_usd_per_hour=0.5, storage_usd_per_gb_month=0.1),
    )
    assert complete["estimated_cost_usd"] == pytest.approx(2.6)


def test_atomic_json_jsonl_and_hash(tmp_path: Path):
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    path = atomic_write_json(tmp_path / "value.json", {"ok": True})
    assert path.read_text().strip().startswith("{")
    append_jsonl(tmp_path / "rows.jsonl", {"x": 1})
    append_jsonl(tmp_path / "rows.jsonl", {"x": 2})
    assert [row["x"] for row in read_jsonl(tmp_path / "rows.jsonl")] == [1, 2]


def test_study_files_roundtrip(tmp_path: Path):
    files = StudyFiles(tmp_path / "study")
    files.save_state({"value": 7})
    assert files.load_state()["value"] == 7
    files.record_event("started", value=1)
    files.record_result({"status": "completed"})
    assert read_jsonl(files.events)[0]["event"] == "started"
    assert read_jsonl(files.results)[0]["status"] == "completed"
