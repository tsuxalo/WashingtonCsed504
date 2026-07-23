from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .persistence import read_jsonl


@dataclass
class StudyProgress:
    elapsed_seconds: float
    records: int
    status_counts: dict[str, int]
    stage_counts: dict[str, int]
    completed_candidates: int
    best_validation_top1: float | None
    cumulative_gpu_hours: float
    cumulative_cpu_hours: float
    cumulative_examples: int
    cumulative_cost_usd: float
    last_event: str | None
    projected_remaining_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def progress_snapshot(
    study_dir: str | Path,
    *,
    elapsed_seconds: float = 0.0,
    expected_records: int | None = None,
) -> StudyProgress:
    root = Path(study_dir)
    rows = read_jsonl(root / "trials.jsonl")
    events = read_jsonl(root / "events.jsonl")
    completed = [row for row in rows if row.get("status") == "completed" and row.get("metrics")]
    durations = [float(row["metrics"].get("wall_seconds", 0.0) or 0.0) for row in completed]
    remaining = None
    if expected_records is not None and durations:
        left = max(0, expected_records - len(rows))
        ordered = sorted(durations)
        median = ordered[len(ordered) // 2]
        remaining = left * median
    scores = [float(row["metrics"].get("validation_top1", 0.0)) for row in completed]
    return StudyProgress(
        elapsed_seconds=elapsed_seconds,
        records=len(rows),
        status_counts=dict(Counter(str(row.get("status")) for row in rows)),
        stage_counts=dict(Counter(str(row.get("stage")) for row in rows)),
        completed_candidates=len({str(row.get("candidate_id")) for row in completed}),
        best_validation_top1=max(scores) if scores else None,
        cumulative_gpu_hours=sum(float(row["metrics"].get("gpu_hours", 0.0) or 0.0) for row in completed),
        cumulative_cpu_hours=sum(float(row["metrics"].get("cpu_hours", 0.0) or 0.0) for row in completed),
        cumulative_examples=sum(int(row["metrics"].get("total_examples", 0) or 0) for row in completed),
        cumulative_cost_usd=sum(float(row["metrics"].get("known_component_total_usd", 0.0) or 0.0) for row in completed),
        last_event=None if not events else str(events[-1].get("event")),
        projected_remaining_seconds=remaining,
    )


def run_command_with_monitor(
    command: Sequence[str],
    *,
    cwd: str | Path,
    study_dir: str | Path,
    log_path: str | Path,
    environment: dict[str, str] | None = None,
    interval_seconds: float = 15.0,
    timeout_seconds: float | None = None,
    expected_records: int | None = None,
    on_update: Callable[[StudyProgress], None] | None = None,
) -> int:
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=environment or os.environ.copy(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if timeout_seconds is not None and elapsed >= timeout_seconds:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise TimeoutError(f"search exceeded {timeout_seconds} seconds")
                snapshot = progress_snapshot(
                    study_dir,
                    elapsed_seconds=elapsed,
                    expected_records=expected_records,
                )
                if on_update:
                    on_update(snapshot)
                else:
                    print(json.dumps(snapshot.to_dict(), indent=2), flush=True)
                time.sleep(interval_seconds)
            return int(process.wait())
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
