from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError
from .schemas import (
    ConstraintSpec,
    ContinuousConfig,
    CostRates,
    FullConfig,
    ObjectiveSpec,
    ProxyConfig,
    ResourceBudget,
    RuntimeConfig,
    StudyConfig,
    SuccessiveHalvingConfig,
)
from .search_space import load_space, normalize_space


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ConfigurationError("PyYAML is required for YAML configuration files") from exc
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ConfigurationError(f"study config must be YAML or JSON, got {suffix!r}")
    if not isinstance(value, dict):
        raise ConfigurationError(f"study config {path} must contain a mapping")
    return value


def _budget(value: dict[str, Any] | None, defaults: ResourceBudget | None = None) -> ResourceBudget:
    base = defaults or ResourceBudget()
    value = value or {}
    return ResourceBudget(
        epochs=value.get("epochs", base.epochs),
        max_steps=value.get("max_steps", value.get("maximum_steps", base.max_steps)),
        data_fraction=float(value.get("data_fraction", value.get("train_fraction", base.data_fraction))),
        validation_fraction=float(value.get("validation_fraction", base.validation_fraction)),
        validation_interval=int(value.get("validation_interval", base.validation_interval)),
        seeds=[int(item) for item in value.get("seeds", base.seeds)],
    )


def _space(raw: Any, base_dir: Path):
    if isinstance(raw, str):
        source = Path(raw)
        if not source.is_absolute():
            source = base_dir / source
        return load_space(source)
    if isinstance(raw, dict) and "path" in raw:
        source = Path(str(raw["path"]))
        if not source.is_absolute():
            source = base_dir / source
        return load_space(source)
    return normalize_space(raw, source_name="study.search_space")


def load_study_config(path_or_mapping: str | Path | dict[str, Any]) -> StudyConfig:
    if isinstance(path_or_mapping, (str, Path)):
        path = Path(path_or_mapping).resolve()
        raw = _load_mapping(path)
        base_dir = path.parent
    else:
        raw = dict(path_or_mapping)
        base_dir = Path.cwd()

    name = str(raw.get("study", {}).get("name", raw.get("name", "hpo-study"))).strip()
    mode = str(raw.get("search", {}).get("mode", raw.get("mode", "successive_halving")))
    if mode == "continuous":
        mode = str(raw.get("continuous", {}).get("strategy", "successive_halving"))
    if mode not in {"proxy", "successive_halving", "full"}:
        raise ConfigurationError(f"unsupported search mode {mode!r}")

    output_dir = Path(raw.get("study", {}).get("output_dir", raw.get("output_dir", "hpo_outputs")))
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    storage_path = Path(
        raw.get("study", {}).get("storage_path", raw.get("storage_path", output_dir / f"{name}.db"))
    )
    if not storage_path.is_absolute():
        storage_path = (base_dir / storage_path).resolve()

    space_raw = raw.get("search_space")
    if space_raw is None:
        raise ConfigurationError("study config is missing search_space")

    objectives = [
        ObjectiveSpec(
            name=str(item["name"]),
            direction=str(item.get("direction", "maximize")),  # type: ignore[arg-type]
            primary=bool(item.get("primary", False)),
            threshold=item.get("threshold"),
        )
        for item in raw.get("objectives", [{"name": "validation_top1", "direction": "maximize", "primary": True}])
    ]
    if not objectives:
        raise ConfigurationError("at least one objective is required")
    if sum(objective.primary for objective in objectives) > 1:
        raise ConfigurationError("only one objective may be marked primary")

    constraints = [
        ConstraintSpec(str(item["name"]), str(item["operator"]), item["value"])  # type: ignore[arg-type]
        for item in raw.get("constraints", [])
    ]
    runtime_raw = raw.get("runtime", {})
    runtime = RuntimeConfig(**{key: value for key, value in runtime_raw.items() if key in RuntimeConfig.__dataclass_fields__})

    proxy_raw = raw.get("proxy", {})
    proxy = ProxyConfig(
        enabled=bool(proxy_raw.get("enabled", True)),
        trials=int(proxy_raw.get("trials", 8)),
        budget=_budget(proxy_raw.get("budget", proxy_raw), ProxyConfig().budget),
        promote_top_k=int(proxy_raw.get("promote_top_k", proxy_raw.get("top_k", 4))),
        minimum_observations_before_pruning=int(proxy_raw.get("minimum_observations_before_pruning", 1)),
    )

    halving_raw = raw.get("successive_halving", raw.get("halving", {}))
    halving = SuccessiveHalvingConfig(
        enabled=bool(halving_raw.get("enabled", True)),
        resource_type=str(halving_raw.get("resource_type", "epochs")),  # type: ignore[arg-type]
        rung_budgets=[float(item) for item in halving_raw.get("rung_budgets", halving_raw.get("budgets", [2, 5, 12]))],
        reduction_factor=int(halving_raw.get("reduction_factor", 2)),
        minimum_trials_per_rung=int(halving_raw.get("minimum_trials_per_rung", 1)),
        continue_checkpoints=bool(halving_raw.get("continue_checkpoints", True)),
        promotion_metric=str(halving_raw.get("promotion_metric", "validation_top1")),
    )

    full_raw = raw.get("full", {})
    full = FullConfig(
        enabled=bool(full_raw.get("enabled", True)),
        trials=int(full_raw.get("trials", raw.get("search", {}).get("trials", 10))),
        budget=_budget(full_raw.get("budget", full_raw), FullConfig().budget),
        exhaustive=bool(full_raw.get("exhaustive", False)),
        maximum_combinations=int(full_raw.get("maximum_combinations", 500)),
        allow_performance_pruning=bool(full_raw.get("allow_performance_pruning", False)),
        allow_safety_termination=bool(full_raw.get("allow_safety_termination", True)),
        evaluate_test=bool(full_raw.get("evaluate_test", False)),
    )

    continuous_raw = raw.get("continuous", {})
    continuous = ContinuousConfig(**{
        key: value
        for key, value in continuous_raw.items()
        if key in ContinuousConfig.__dataclass_fields__
    })
    if continuous.enabled:
        continuous.strategy = mode  # type: ignore[assignment]

    rates_raw = raw.get("cost_rates", {})
    cost_rates = CostRates(**{
        key: value for key, value in rates_raw.items() if key in CostRates.__dataclass_fields__
    })

    config = StudyConfig(
        name=name,
        mode=mode,  # type: ignore[arg-type]
        output_dir=output_dir,
        storage_path=storage_path,
        dataset=dict(raw.get("dataset", {"name": "cifar10"})),
        model=dict(raw.get("model", {"name": "resnet18"})),
        search_space=_space(space_raw, base_dir),
        objectives=objectives,
        constraints=constraints,
        runtime=runtime,
        proxy=proxy,
        successive_halving=halving,
        full=full,
        continuous=continuous,
        cost_rates=cost_rates,
        sampler=str(raw.get("search", {}).get("sampler", raw.get("sampler", "tpe"))),
        seed=int(raw.get("study", {}).get("seed", raw.get("seed", 42))),
        resume=bool(raw.get("study", {}).get("resume", raw.get("resume", True))),
        metadata=dict(raw.get("metadata", {})),
    )
    validate_study_config(config)
    return config


def validate_study_config(config: StudyConfig) -> None:
    if config.proxy.trials <= 0:
        raise ConfigurationError("proxy.trials must be positive")
    if config.successive_halving.reduction_factor < 2:
        raise ConfigurationError("successive_halving.reduction_factor must be at least 2")
    if sorted(config.successive_halving.rung_budgets) != config.successive_halving.rung_budgets:
        raise ConfigurationError("successive_halving.rung_budgets must be increasing")
    for budget_name, budget in [("proxy", config.proxy.budget), ("full", config.full.budget)]:
        if not 0 < budget.data_fraction <= 1:
            raise ConfigurationError(f"{budget_name}.data_fraction must be in (0, 1]")
        if not 0 < budget.validation_fraction <= 1:
            raise ConfigurationError(f"{budget_name}.validation_fraction must be in (0, 1]")
    if len(config.objectives) > 1 and config.mode != "full":
        # Multi-objective promotion is allowed, but promotion must still use the primary metric.
        if not config.primary_objective:
            raise ConfigurationError("multi-objective staged search requires a primary objective")
