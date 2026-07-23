from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ParameterType = Literal["float", "int", "categorical", "bool", "fixed"]
Direction = Literal["maximize", "minimize"]
SearchMode = Literal["proxy", "successive_halving", "full"]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    type: ParameterType
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[Any, ...] = ()
    step: float | int | None = None
    log: bool = False
    default: Any = None
    condition: str | None = None
    enabled: bool = True
    source: str = "python"
    item: int | str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["choices"] = list(self.choices)
        return value


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    direction: Direction
    primary: bool = False
    threshold: float | None = None


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    operator: Literal["<=", ">=", "<", ">", "==", "!="]
    value: float | int | str | bool


@dataclass
class ResourceBudget:
    epochs: int | None = None
    max_steps: int | None = None
    data_fraction: float = 1.0
    validation_fraction: float = 1.0
    validation_interval: int = 1
    seeds: list[int] = field(default_factory=lambda: [42])


@dataclass
class ProxyConfig:
    enabled: bool = True
    trials: int = 8
    budget: ResourceBudget = field(
        default_factory=lambda: ResourceBudget(
            epochs=2,
            data_fraction=0.2,
            validation_fraction=0.5,
        )
    )
    promote_top_k: int = 4
    minimum_observations_before_pruning: int = 1


@dataclass
class SuccessiveHalvingConfig:
    enabled: bool = True
    resource_type: Literal["epochs", "steps", "data_fraction", "seeds"] = "epochs"
    rung_budgets: list[float] = field(default_factory=lambda: [2, 5, 12])
    reduction_factor: int = 2
    minimum_trials_per_rung: int = 1
    continue_checkpoints: bool = True
    promotion_metric: str = "validation_top1"


@dataclass
class FullConfig:
    enabled: bool = True
    trials: int = 10
    budget: ResourceBudget = field(default_factory=lambda: ResourceBudget(epochs=30))
    exhaustive: bool = False
    maximum_combinations: int = 500
    allow_performance_pruning: bool = False
    allow_safety_termination: bool = True
    evaluate_test: bool = False


@dataclass
class ContinuousConfig:
    enabled: bool = False
    strategy: SearchMode = "successive_halving"
    maximum_trials: int | None = None
    maximum_wall_time_hours: float | None = None
    maximum_gpu_hours: float | None = None
    maximum_cpu_hours: float | None = None
    maximum_cost_usd: float | None = None
    target_validation_metric: float | None = None
    stop_after_no_improvement_trials: int | None = 25
    minimum_improvement: float = 0.0005
    pareto_stagnation_trials: int | None = 25
    checkpoint_after_each_trial: bool = True


@dataclass
class RuntimeConfig:
    device: str = "auto"
    gpu_id: int | None = None
    precision: str = "auto"
    tf32: bool = True
    channels_last: bool | None = None
    compile: bool = False
    compile_mode: str | None = None
    concurrent_trials: int | None = None
    intraop_threads: int | None = None
    interop_threads: int | None = None
    workers: int | None = None
    memory_reserve_gb: float = 1.0
    data_strategy: str = "auto"
    asynchronous_checkpointing: bool = False


@dataclass
class CostRates:
    gpu_usd_per_hour: float | None = None
    cpu_usd_per_hour: float | None = None
    storage_usd_per_gb_month: float | None = None
    electricity_usd_per_kwh: float | None = None
    colab_subscription_usd: float | None = None
    colab_compute_unit_usd: float | None = None


@dataclass
class StudyConfig:
    name: str
    mode: SearchMode
    output_dir: Path
    storage_path: Path
    dataset: dict[str, Any]
    model: dict[str, Any]
    search_space: list[ParameterSpec]
    objectives: list[ObjectiveSpec]
    constraints: list[ConstraintSpec] = field(default_factory=list)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    successive_halving: SuccessiveHalvingConfig = field(default_factory=SuccessiveHalvingConfig)
    full: FullConfig = field(default_factory=FullConfig)
    continuous: ContinuousConfig = field(default_factory=ContinuousConfig)
    cost_rates: CostRates = field(default_factory=CostRates)
    sampler: str = "tpe"
    seed: int = 42
    resume: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_objective(self) -> ObjectiveSpec:
        primaries = [objective for objective in self.objectives if objective.primary]
        return primaries[0] if primaries else self.objectives[0]
