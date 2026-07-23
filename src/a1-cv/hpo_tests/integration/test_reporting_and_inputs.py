from __future__ import annotations

import json
from pathlib import Path

from hpo.notebook_api import normalize_notebook_space, normalize_uploaded_csv, preview_dataframe
from hpo.persistence import StudyFiles
from hpo.reporting import export_reports
from hpo.schemas import ObjectiveSpec


def test_notebook_inputs_and_preview():
    specs = normalize_notebook_space({
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128]},
    })
    assert list(preview_dataframe(specs)["name"]) == ["learning_rate", "batch_size"]
    uploaded = normalize_uploaded_csv(
        "name,type,low,high,choices,step,log,default,condition,enabled\n"
        "batch_size,categorical,,,64|128,,false,64,,true\n",
        filename="test_space.csv",
    )
    assert uploaded[0].choices == (64, 128)


def test_report_exports_pareto_and_best(tmp_path: Path):
    files = StudyFiles(tmp_path / "study")
    rows = [
        {
            "candidate_id": "a", "trial_number": 0, "stage": "full", "status": "completed",
            "params": {"lr": 0.1},
            "resource": {"target_epochs": 1},
            "metrics": {"validation_top1": 0.9, "wall_seconds": 20},
        },
        {
            "candidate_id": "b", "trial_number": 1, "stage": "full", "status": "completed",
            "params": {"lr": 0.01},
            "resource": {"target_epochs": 1},
            "metrics": {"validation_top1": 0.89, "wall_seconds": 10},
        },
    ]
    for row in rows:
        files.record_result(row)
    summary = export_reports(
        files.root,
        [ObjectiveSpec("validation_top1", "maximize", True), ObjectiveSpec("wall_seconds", "minimize")],
    )
    assert summary["pareto_candidates"] == 2
    assert (files.root / "all_trials.csv").exists()
    assert (files.root / "accuracy_vs_time.png").exists()
    assert (files.root / "optimization_history.png").exists()
    assert json.loads((files.root / "best_validation_configuration.json").read_text())["candidate_id"] == "a"
