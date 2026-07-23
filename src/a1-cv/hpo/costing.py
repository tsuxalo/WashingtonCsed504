from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .schemas import CostRates


def estimate_cost(metrics: Mapping[str, Any], rates: CostRates) -> dict[str, Any]:
    components: dict[str, float | None] = {}
    gpu_hours = float(metrics.get("gpu_hours", 0.0) or 0.0)
    cpu_hours = float(metrics.get("cpu_hours", 0.0) or 0.0)
    storage_gb = float(metrics.get("checkpoint_size_mb", 0.0) or 0.0) / 1024
    components["gpu"] = None if rates.gpu_usd_per_hour is None else gpu_hours * rates.gpu_usd_per_hour
    components["cpu"] = None if rates.cpu_usd_per_hour is None else cpu_hours * rates.cpu_usd_per_hour
    components["storage_month"] = (
        None if rates.storage_usd_per_gb_month is None else storage_gb * rates.storage_usd_per_gb_month
    )
    known = [value for value in components.values() if value is not None]
    return {
        "estimated_cost_usd": sum(known) if len(known) == len(components) else None,
        "known_component_total_usd": sum(known),
        "components": components,
        "rates": asdict(rates),
        "missing_rates": [name for name, value in components.items() if value is None],
        "subscription_cost_is_sunk": rates.colab_subscription_usd is not None,
    }
