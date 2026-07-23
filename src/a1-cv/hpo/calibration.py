from __future__ import annotations

import gc
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F


@dataclass
class BatchMeasurement:
    batch_size: int
    status: str
    seconds_per_step: float | None
    examples_per_second: float | None
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None
    error: str | None = None


@dataclass
class BatchCalibration:
    measurements: list[BatchMeasurement]
    largest_fitting_batch: int | None
    highest_throughput_batch: int | None
    recommended_candidates: list[int]
    memory_headroom_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurements": [asdict(item) for item in self.measurements],
            "largest_fitting_batch": self.largest_fitting_batch,
            "highest_throughput_batch": self.highest_throughput_batch,
            "recommended_candidates": self.recommended_candidates,
            "memory_headroom_fraction": self.memory_headroom_fraction,
        }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def calibrate_batch_sizes(
    build_model: Callable[[], torch.nn.Module],
    make_batch: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    candidates: Iterable[int],
    precision: str = "fp32",
    warmup_steps: int = 2,
    measure_steps: int = 5,
    memory_headroom_fraction: float = 0.15,
    channels_last: bool = False,
) -> BatchCalibration:
    measurements: list[BatchMeasurement] = []
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = device.type == "cuda" and precision in {"fp16", "bf16"}

    for batch_size in sorted(set(int(item) for item in candidates if int(item) > 0)):
        model = optimizer = scaler = None
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = build_model().to(device).train()
            if channels_last:
                model = model.to(memory_format=torch.channels_last)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
            x, y = make_batch(batch_size)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            def step():
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                    loss = F.cross_entropy(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            for _ in range(warmup_steps):
                step()
            synchronize(device)
            windows = []
            for _ in range(3):
                start = time.perf_counter()
                for _ in range(measure_steps):
                    step()
                synchronize(device)
                windows.append((time.perf_counter() - start) / measure_steps)
            seconds = statistics.median(windows)
            allocated = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
            reserved = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None
            if device.type == "cuda":
                total = torch.cuda.get_device_properties(device).total_memory / 2**20
                if reserved and reserved > total * (1 - memory_headroom_fraction):
                    status = "insufficient_headroom"
                else:
                    status = "completed"
            else:
                status = "completed"
            measurements.append(BatchMeasurement(
                batch_size=batch_size,
                status=status,
                seconds_per_step=seconds,
                examples_per_second=batch_size / seconds,
                peak_allocated_mb=allocated,
                peak_reserved_mb=reserved,
            ))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
                error = f"{type(exc).__name__}: {exc}"
                status = "failed"
            else:
                error = str(exc)
                status = "oom"
            measurements.append(BatchMeasurement(batch_size, status, None, None, None, None, error))
        finally:
            del model, optimizer, scaler
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fitting = [item for item in measurements if item.status == "completed"]
    largest = max((item.batch_size for item in fitting), default=None)
    best = max(fitting, key=lambda item: item.examples_per_second or 0, default=None)
    recommended = sorted({
        item.batch_size
        for item in fitting
        if best and (item.examples_per_second or 0) >= 0.85 * (best.examples_per_second or 1)
    })
    return BatchCalibration(measurements, largest, None if best is None else best.batch_size, recommended, memory_headroom_fraction)
