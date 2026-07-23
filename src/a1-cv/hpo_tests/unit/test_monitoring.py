from pathlib import Path

from hpo.monitoring import progress_snapshot
from hpo.persistence import StudyFiles


def test_progress_snapshot(tmp_path: Path):
    files = StudyFiles(tmp_path / "study")
    files.record_result({
        "candidate_id": "a", "stage": "proxy", "status": "completed",
        "metrics": {
            "validation_top1": 0.8, "wall_seconds": 2,
            "gpu_hours": 0.1, "cpu_hours": 0.2,
            "total_examples": 100, "known_component_total_usd": 0.3,
        },
    })
    files.record_event("candidate_promoted")
    snapshot = progress_snapshot(files.root, elapsed_seconds=5, expected_records=3)
    assert snapshot.records == 1
    assert snapshot.best_validation_top1 == 0.8
    assert snapshot.projected_remaining_seconds == 4
    assert snapshot.last_event == "candidate_promoted"
