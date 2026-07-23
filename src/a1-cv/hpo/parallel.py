from __future__ import annotations

import multiprocessing as mp
import os
import queue
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .trial_runner import TrialResource, TrialResult, TrialRunner


def _worker_main(
    worker_id: int,
    device_type: str,
    device_index: int | None,
    repo_root: str,
    dataset_config: dict[str, Any],
    model_config: dict[str, Any],
    runtime_config: dict[str, Any],
    output_dir: str,
    task_queue: Any,
    result_queue: Any,
):
    runtime = dict(runtime_config)
    runtime["device"] = device_type
    runtime["gpu_id"] = device_index
    if device_type == "cuda" and device_index is not None:
        # Keep the repository's physical GPU numbering; TrialRunner calls torch.cuda.set_device.
        runtime["gpu_id"] = device_index
    try:
        import torch
        torch.set_num_threads(max(1, int(runtime.get("intraop_threads", 1))))
        try:
            torch.set_num_interop_threads(max(1, int(runtime.get("interop_threads", 1))))
        except RuntimeError:
            pass
        runner = TrialRunner(
            repo_root,
            dataset_config=dataset_config,
            model_config=model_config,
            runtime_config=runtime,
            output_dir=output_dir,
        )
        while True:
            task = task_queue.get()
            if task is None:
                break
            task_id = task["task_id"]
            try:
                resource = TrialResource(**task["resource"])
                result = runner.run(
                    candidate_id=task["candidate_id"],
                    trial_number=task.get("trial_number"),
                    params=task["params"],
                    resource=resource,
                    maximum_total_epochs=task.get("maximum_total_epochs"),
                )
                result_queue.put({"task_id": task_id, "worker_id": worker_id, "result": result.to_dict()})
            except BaseException as exc:  # worker must report and continue scheduling queued work
                result_queue.put({
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
    except BaseException as exc:
        result_queue.put({
            "task_id": None,
            "worker_id": worker_id,
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


class ParallelTrialExecutor:
    """Parent-owned queue with one persistent worker process per assigned device."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        dataset_config: dict[str, Any],
        model_config: dict[str, Any],
        runtime_config: dict[str, Any],
        output_dir: str | Path,
        device_type: str,
        device_indices: list[int | None],
    ):
        self.context = mp.get_context("spawn")
        self.task_queue = self.context.Queue()
        self.result_queue = self.context.Queue()
        self.processes = []
        for worker_id, device_index in enumerate(device_indices):
            process = self.context.Process(
                target=_worker_main,
                args=(
                    worker_id,
                    device_type,
                    device_index,
                    str(repo_root),
                    dataset_config,
                    model_config,
                    runtime_config,
                    str(output_dir),
                    self.task_queue,
                    self.result_queue,
                ),
                daemon=True,
            )
            process.start()
            self.processes.append(process)

    def run(self, tasks: Iterable[dict[str, Any]], *, timeout_seconds: float | None = None) -> list[dict[str, Any]]:
        submitted = list(tasks)
        for task in submitted:
            self.task_queue.put(task)
        results = []
        while len(results) < len(submitted):
            try:
                message = self.result_queue.get(timeout=timeout_seconds)
            except queue.Empty as exc:
                raise TimeoutError("parallel trial batch timed out") from exc
            if message.get("fatal_error"):
                raise RuntimeError(
                    f"worker {message['worker_id']} failed: {message['fatal_error']}\n{message.get('traceback', '')}"
                )
            results.append(message)
        return results

    def close(self):
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
