from hpo.baselines import Baseline
from hpo.benchmark import compare_with_reference, normalized_parameter_distance, proxy_reliability


def test_parameter_distance_and_reference_comparison():
    reference = Baseline("ref", "cifar10", "resnet18", {"learning_rate": 0.2, "optimizer": "sgd"}, 0.92, 10.0, "runs/x.json", True)
    row = {"params": {"learning_rate": 0.1, "optimizer": "sgd"}, "metrics": {"validation_top1": 0.915}}
    result = compare_with_reference(row, reference, accuracy_margin=0.01, ranges={"learning_rate": (0.01, 0.3)})
    assert result["within_configured_margin"] is True
    assert result["parameter_distance"]["compared_parameters"] == 2


def test_proxy_reliability():
    result = proxy_reliability({"a": 0.7, "b": 0.8, "c": 0.9}, {"a": 0.69, "b": 0.82, "c": 0.88}, top_k=2)
    assert result["sample_count"] == 3
    assert result["top_k_retention"] == 1.0


def test_discovery_summary():
    from hpo.benchmark import discovery_benchmark_summary
    reference = Baseline("ref", "cifar10", "resnet18", {"learning_rate": 0.2}, 0.9, 20.0, "runs/x", True)
    rows = [
        {"candidate_id":"a","status":"completed","started_at":1,"finished_at":11,"params":{"learning_rate":0.1},"metrics":{"validation_top1":0.89,"wall_seconds":10,"total_examples":100,"epochs_completed":1,"cpu_hours":0.01,"gpu_hours":0}},
        {"candidate_id":"b","status":"completed","started_at":1,"finished_at":16,"params":{"learning_rate":0.2},"metrics":{"validation_top1":0.91,"wall_seconds":15,"total_examples":100,"epochs_completed":1,"cpu_hours":0.01,"gpu_hours":0}},
    ]
    summary = discovery_benchmark_summary(rows, reference=reference, thresholds=[0.9])
    assert summary["best_candidate_id"] == "b"
    assert summary["time_to_accuracy_threshold_seconds"]["0.9"] == 15
    assert summary["reference_pareto_dominated_on_accuracy_time"] is True
