from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .persistence import atomic_write_json, read_jsonl
from .schemas import ObjectiveSpec
from .selection import pareto_front, pareto_knee


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "candidate_id": row.get("candidate_id"),
        "trial_number": row.get("trial_number"),
        "stage": row.get("stage"),
        "status": row.get("status"),
    }
    for prefix in ("params", "metrics", "resource"):
        for key, value in (row.get(prefix) or {}).items():
            if key == "history":
                continue
            flat[f"{prefix}_{key}"] = value
    flat["failure_reason"] = row.get("failure_reason")
    flat["invalid_reason"] = row.get("invalid_reason")
    flat["prune_reason"] = row.get("prune_reason")
    return flat


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = [_flatten(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def export_reports(study_dir: str | Path, objectives: Sequence[ObjectiveSpec]) -> dict[str, Any]:
    root = Path(study_dir)
    results = read_jsonl(root / "trials.jsonl")
    completed = [row for row in results if row.get("status") == "completed" and row.get("metrics")]
    # Keep the highest-fidelity/latest result per candidate.
    latest: dict[str, dict[str, Any]] = {}
    order = {"proxy": 0, "halving": 1, "full": 2}
    for row in completed:
        current = latest.get(row["candidate_id"])
        if current is None or order.get(row.get("stage"), -1) >= order.get(current.get("stage"), -1):
            latest[row["candidate_id"]] = row
    final_rows = list(latest.values())
    front = pareto_front([row["metrics"] | {"_row": row} for row in final_rows], objectives)
    front_rows = [item["_row"] for item in front]
    knee_metric = pareto_knee(final_rows, objectives)
    write_csv(root / "all_trials.csv", results)
    write_csv(root / "completed_candidates.csv", final_rows)
    write_csv(root / "pareto_trials.csv", front_rows)
    if final_rows:
        best = max(final_rows, key=lambda row: float(row["metrics"].get("validation_top1", 0.0)))
        atomic_write_json(root / "best_validation_configuration.json", best)
    if knee_metric:
        atomic_write_json(root / "pareto_knee_configuration.json", knee_metric)
    plots = generate_plots(root, results)
    summary = {
        "records": len(results),
        "completed_candidates": len(final_rows),
        "pareto_candidates": len(front_rows),
        "plots": plots,
        "status_counts": {
            status: sum(row.get("status") == status for row in results)
            for status in sorted({str(row.get("status")) for row in results})
        },
    }
    atomic_write_json(root / "report_summary.json", summary)
    return summary


def generate_plots(study_dir: str | Path, rows: Sequence[dict[str, Any]] | None = None) -> dict[str, str]:
    """Generate optional static analysis plots, one figure per output file."""
    root = Path(study_dir)
    values = list(rows) if rows is not None else read_jsonl(root / "trials.jsonl")
    completed = [row for row in values if row.get("status") == "completed" and row.get("metrics")]
    if not completed:
        return {}
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    outputs: dict[str, str] = {}

    def scatter(name: str, x_key: str, x_label: str) -> None:
        points = [
            (row["metrics"].get(x_key), row["metrics"].get("validation_top1"))
            for row in completed
            if row["metrics"].get(x_key) is not None
            and row["metrics"].get("validation_top1") is not None
        ]
        if not points:
            return
        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.scatter([item[0] for item in points], [item[1] for item in points])
        axes.set_xlabel(x_label)
        axes.set_ylabel("Validation top-1")
        axes.set_title(name.replace("_", " ").title())
        figure.tight_layout()
        path = root / f"{name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        outputs[name] = str(path)

    scatter("accuracy_vs_time", "wall_seconds", "Trial wall time (seconds)")
    scatter("accuracy_vs_memory", "peak_gpu_memory_mb", "Peak GPU memory (MB)")
    scatter("accuracy_vs_cost", "estimated_cost_usd", "Estimated cost (USD)")

    ordered = sorted(
        completed,
        key=lambda row: (
            row.get("trial_number") is None,
            row.get("trial_number") if row.get("trial_number") is not None else 10**12,
        ),
    )
    if ordered:
        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.plot(
            list(range(len(ordered))),
            [float(row["metrics"].get("validation_top1", 0.0)) for row in ordered],
            marker="o",
        )
        axes.set_xlabel("Completed result order")
        axes.set_ylabel("Validation top-1")
        axes.set_title("Optimization history")
        figure.tight_layout()
        path = root / "optimization_history.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        outputs["optimization_history"] = str(path)

    proxy_by_candidate = {
        row["candidate_id"]: float(row["metrics"].get("validation_top1", 0.0))
        for row in values
        if row.get("stage") == "proxy" and row.get("status") == "completed"
    }
    full_pairs = [
        (
            proxy_by_candidate[row["candidate_id"]],
            float(row["metrics"].get("validation_top1", 0.0)),
        )
        for row in values
        if row.get("stage") == "full"
        and row.get("status") == "completed"
        and row.get("candidate_id") in proxy_by_candidate
    ]
    if full_pairs:
        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.scatter([item[0] for item in full_pairs], [item[1] for item in full_pairs])
        axes.set_xlabel("Proxy validation top-1")
        axes.set_ylabel("Full validation top-1")
        axes.set_title("Proxy versus full ranking")
        figure.tight_layout()
        path = root / "proxy_vs_full.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        outputs["proxy_vs_full"] = str(path)
    return outputs
