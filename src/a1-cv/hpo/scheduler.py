from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .hardware import HardwareProfile


@dataclass
class ResourcePlan:
    device: str
    concurrent_trials: int
    trial_processes: int
    intraop_threads: int
    interop_threads: int
    workers_per_trial: int
    cpu_core_reserve: int
    memory_reserve_gb: float
    rationale: list[str]

    def to_dict(self):
        return asdict(self)


def plan_resources(
    profile: HardwareProfile,
    *,
    device: str = "auto",
    requested_concurrency: int | None = None,
    requested_intraop_threads: int | None = None,
    requested_interop_threads: int | None = None,
    requested_workers: int | None = None,
    memory_reserve_gb: float = 1.0,
) -> ResourcePlan:
    rationale: list[str] = []
    if device == "auto":
        device = "cuda" if profile.cuda_available else "mps" if profile.mps_available else "cpu"
    logical = max(1, profile.logical_cpu_threads)
    physical = profile.physical_cpu_cores or max(1, logical // 2)
    reserve = max(1, min(2, physical // 4))

    if device == "cuda":
        if profile.gpu_count <= 1:
            concurrency = 1
            rationale.append("one active training trial per GPU; one-GPU concurrency requires calibration")
        else:
            concurrency = profile.gpu_count
            rationale.append("one independent process per visible GPU")
        if requested_concurrency is not None:
            concurrency = max(1, min(requested_concurrency, max(1, profile.gpu_count)))
        usable_cpu = max(1, physical - reserve)
        intra = max(1, usable_cpu // concurrency)
        workers = 0
        rationale.append("GPU-resident 32x32 data needs no DataLoader workers")
    elif device == "mps":
        concurrency = 1
        intra = max(1, physical - reserve)
        workers = 0
        rationale.append("MPS uses one active trial; repository ViT training may require CPU fallback")
    else:
        memory_limited = max(1, int(profile.available_ram_gb // 4))
        cpu_limited = max(1, (physical - reserve) // 2)
        concurrency = min(memory_limited, cpu_limited)
        if requested_concurrency is not None:
            concurrency = max(1, min(requested_concurrency, concurrency))
        # Large OpenMP thread pools are frequently slower for CIFAR-sized
        # models and tiny smoke studies. Use a conservative cap unless the
        # user explicitly overrides it.
        intra = min(8, max(1, (physical - reserve) // concurrency))
        workers = 0
        rationale.append("CPU trials divide physical cores and cap intra-op threads to avoid nested oversubscription")

    if requested_intraop_threads is not None:
        intra = max(1, int(requested_intraop_threads))
        rationale.append("user override applied for intra-op threads")
    interop = max(1, int(requested_interop_threads or 1))
    if requested_workers is not None:
        workers = max(0, int(requested_workers))
        rationale.append("user override applied for data workers")

    return ResourcePlan(
        device=device,
        concurrent_trials=concurrency,
        trial_processes=concurrency,
        intraop_threads=intra,
        interop_threads=interop,
        workers_per_trial=workers,
        cpu_core_reserve=reserve,
        memory_reserve_gb=memory_reserve_gb,
        rationale=rationale,
    )
