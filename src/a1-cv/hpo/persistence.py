from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def stable_hash(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, target)
    return target


def append_jsonl(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, default=str, separators=(",", ":")) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    result = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"corrupt JSONL at {source}:{line_number}: {exc}") from exc
    return result


class StudyFiles:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.results = self.root / "trials.jsonl"
        self.events = self.root / "events.jsonl"
        self.state = self.root / "state.json"
        self.environment = self.root / "environment.json"
        self.resolved_config = self.root / "resolved_config.json"
        self.sampler = self.root / "sampler.pkl"
        self.summary = self.root / "study_summary.json"

    def save_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.state, state)

    def load_state(self) -> dict[str, Any]:
        if not self.state.exists():
            return {"candidates": {}, "completed_param_hashes": [], "created_at": time.time()}
        return json.loads(self.state.read_text(encoding="utf-8"))

    def record_result(self, result: dict[str, Any]) -> None:
        append_jsonl(self.results, result)

    def record_event(self, event: str, **payload: Any) -> None:
        append_jsonl(self.events, {"timestamp": time.time(), "event": event, **payload})

    def save_sampler(self, sampler: Any) -> None:
        temporary = self.sampler.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(sampler, handle)
        os.replace(temporary, self.sampler)

    def load_sampler(self) -> Any | None:
        if not self.sampler.exists():
            return None
        with self.sampler.open("rb") as handle:
            return pickle.load(handle)
