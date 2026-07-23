from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

from .baselines import load_repository_baselines
from .config import load_study_config
from .estimation import estimate_search
from .hardware import detect_hardware
from .persistence import atomic_write_json, read_jsonl
from .reporting import export_reports
from .schemas import ObjectiveSpec, ParameterSpec
from .search_space import combination_count, preview_rows
from .study import HpoStudy
from .trial_runner import TrialResource, TrialRunner


def _repo_root(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "a1-cv" / "models.py").exists():
            return candidate
    raise SystemExit("could not locate repository root; pass --repo-root")


def _json(value: Any):
    print(json.dumps(value, indent=2, default=str))


def _smoke_config(config):
    config.name = f"{config.name}-smoke"
    config.output_dir = config.output_dir / "smoke"
    config.storage_path = config.output_dir / f"{config.name}.db"
    config.dataset = {
        "name": "synthetic",
        "train_examples": 96,
        "validation_examples": 48,
        "test_examples": 48,
        "num_classes": 10,
    }
    # Use a deliberately tiny ViT for CLI smoke paths. ResNet-18 produces
    # ~100 MB optimizer checkpoints even for two synthetic steps, making a
    # software smoke test unnecessarily slow on constrained CI/Colab CPUs.
    config.model = {
        "name": "vit",
        "hidden_dim": 48,
        "layers": 1,
        "heads": 3,
        "mlp_dim": 96,
        "patch_size": 8,
    }
    config.search_space = [
        ParameterSpec(
            name="batch_size",
            type="fixed",
            default=16,
            source="cli-smoke",
        ),
        ParameterSpec(
            name="optimizer",
            type="fixed",
            default="adamw",
            source="cli-smoke",
        ),
        ParameterSpec(
            name="learning_rate",
            type="categorical",
            choices=(0.0005, 0.001),
            default=0.001,
            source="cli-smoke",
        ),
        ParameterSpec(
            name="weight_decay",
            type="fixed",
            default=0.01,
            source="cli-smoke",
        ),
        ParameterSpec(
            name="scheduler",
            type="fixed",
            default="cosine",
            source="cli-smoke",
        ),
        ParameterSpec(
            name="warmup_epochs",
            type="fixed",
            default=0,
            source="cli-smoke",
        ),
        ParameterSpec(
            name="label_smoothing",
            type="fixed",
            default=0.0,
            source="cli-smoke",
        ),
        ParameterSpec(
            name="gradient_clip",
            type="fixed",
            default=1.0,
            source="cli-smoke",
        ),
        ParameterSpec(
            name="channels_last",
            type="fixed",
            default=False,
            source="cli-smoke",
        ),
    ]
    config.runtime.device = "cpu"
    config.runtime.precision = "fp32"
    config.runtime.channels_last = False
    config.runtime.concurrent_trials = 1
    config.runtime.intraop_threads = 1
    config.runtime.interop_threads = 1
    config.proxy.trials = min(2, config.proxy.trials)
    config.proxy.budget.epochs = 1
    config.proxy.budget.max_steps = 2
    config.proxy.promote_top_k = 1
    config.successive_halving.rung_budgets = [2]
    config.successive_halving.minimum_trials_per_rung = 1
    config.full.trials = 1
    config.full.budget.epochs = 1
    config.full.budget.max_steps = 2
    config.full.budget.seeds = [42]
    config.continuous.maximum_trials = min(config.continuous.maximum_trials or 2, 2)
    return config


def validate_space_command(args):
    config = load_study_config(args.config)
    _json({
        "study": config.name,
        "mode": config.mode,
        "parameters": preview_rows(config.search_space),
        "finite_combination_count": combination_count(config.search_space),
        "objectives": [dataclasses.asdict(item) for item in config.objectives],
        "constraints": [dataclasses.asdict(item) for item in config.constraints],
    })


def hardware_command(_args):
    _json(detect_hardware().to_dict())


def estimate_command(args):
    config = load_study_config(args.config)
    if args.smoke_test:
        config = _smoke_config(config)
    repo = _repo_root(args.repo_root)
    study_dir = config.output_dir / config.name
    runtime = dataclasses.asdict(config.runtime)
    runtime["seed"] = config.seed
    runner = TrialRunner(
        repo,
        dataset_config=config.dataset,
        model_config=config.model,
        runtime_config=runtime,
        output_dir=study_dir / "calibration",
        hardware=detect_hardware(),
    )
    params = {
        spec.name: spec.default
        for spec in config.search_space
        if spec.default is not None
    }
    params.setdefault("batch_size", 64 if args.smoke_test else 128)
    params.setdefault("optimizer", "sgd" if str(config.model.get("name", "")).startswith("resnet") else "adamw")
    resource = TrialResource(
        stage="calibration",
        target_epochs=1,
        max_steps=args.calibration_steps,
        data_fraction=min(1.0, args.calibration_steps * params["batch_size"] / max(1, runner.dataset.train_examples)),
        validation_fraction=0.1 if not args.smoke_test else 1.0,
        seed=config.seed,
        continue_checkpoint=False,
    )
    result = runner.run(
        candidate_id="calibration",
        trial_number=None,
        params=params,
        resource=resource,
        maximum_total_epochs=1,
    )
    if result.status != "completed":
        raise SystemExit(f"calibration failed: {result.failure_reason or result.invalid_reason}")
    m = result.metrics
    record = {
        "batch_size": params["batch_size"],
        "seconds_per_example": m["wall_seconds"] / max(1, m["training_examples"]),
        "evaluation_seconds_per_epoch": 0.0,
        "checkpoint_seconds_per_epoch": 0.0,
        "checkpoint_size_mb": m["checkpoint_size_mb"],
        "peak_memory_mb": m["peak_gpu_memory_mb"],
        "train_flops_per_image": m["train_flops_per_image"],
        "cpu_seconds": m["cpu_seconds"],
        "elapsed_seconds": m["wall_seconds"],
    }
    estimate = estimate_search(
        config,
        train_examples=runner.dataset.train_examples,
        validation_examples=runner.dataset.validation_examples,
        calibration_records=[record],
        representative_batch_size=params["batch_size"],
    )
    path = study_dir / "pre_search_estimate.json"
    atomic_write_json(path, estimate.to_dict())
    _json({"calibration": result.to_dict(), "estimate": estimate.to_dict(), "saved": str(path)})


def search_command(args):
    config = load_study_config(args.config)
    if args.mode:
        config.mode = args.mode
    if args.continuous:
        config.continuous.enabled = True
        config.continuous.strategy = config.mode
    if args.smoke_test:
        config = _smoke_config(config)
    study = HpoStudy(config, repo_root=_repo_root(args.repo_root))
    if args.enqueue_repository_reference:
        candidates = [
            baseline for baseline in load_repository_baselines(study.repo_root)
            if baseline.dataset == config.dataset.get("name") and baseline.model == config.model.get("name")
        ]
        if candidates:
            study.enqueue_reference(max(candidates, key=lambda item: item.validation_top1 or 0))
    if args.session_trials is not None:
        if config.mode == "proxy":
            summary = study.run_proxy(trials=args.session_trials)
        elif config.mode == "full":
            summary = study.run_full(trials=args.session_trials)
        else:
            summary = study.run_successive_halving(trials=args.session_trials)
    else:
        summary = study.run()
    _json(summary)


def report_command(args):
    root = Path(args.study_path).resolve()
    objectives = [ObjectiveSpec("validation_top1", "maximize", True)]
    resolved = root / "resolved_config.json"
    if resolved.exists():
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        objectives = [ObjectiveSpec(**item) for item in raw.get("objectives", [])] or objectives
    _json(export_reports(root, objectives))


def baselines_command(args):
    _json([dataclasses.asdict(item) for item in load_repository_baselines(_repo_root(args.repo_root))])


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m hpo.cli")
    parser.add_argument("--repo-root", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-space")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=validate_space_command)

    hardware = sub.add_parser("hardware")
    hardware.set_defaults(func=hardware_command)

    estimate = sub.add_parser("estimate")
    estimate.add_argument("--config", required=True)
    estimate.add_argument("--calibration-steps", type=int, default=10)
    estimate.add_argument("--smoke-test", action="store_true")
    estimate.set_defaults(func=estimate_command)

    search = sub.add_parser("search")
    search.add_argument("--config", required=True)
    search.add_argument("--mode", choices=["proxy", "successive_halving", "full"])
    search.add_argument("--continuous", action="store_true")
    search.add_argument("--smoke-test", action="store_true")
    search.add_argument("--session-trials", type=int)
    search.add_argument("--enqueue-repository-reference", action="store_true")
    search.set_defaults(func=search_command)

    report = sub.add_parser("report")
    report.add_argument("--study-path", required=True)
    report.set_defaults(func=report_command)

    baselines = sub.add_parser("baselines")
    baselines.set_defaults(func=baselines_command)
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
