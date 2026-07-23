from __future__ import annotations

import operator
from typing import Any, Mapping, Sequence

from .exceptions import InvalidTrialError
from .schemas import ConstraintSpec

_OPS = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
}


def validate_candidate(params: Mapping[str, Any], model: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    optimizer = str(params.get("optimizer", "")).lower()
    if optimizer != "sgd":
        forbidden = [name for name in ("momentum", "nesterov", "dampening") if name in params]
        if forbidden:
            raise InvalidTrialError(f"{', '.join(forbidden)} apply only to SGD")
    if optimizer not in {"adam", "adamw"}:
        forbidden = [name for name in ("beta1", "beta2", "epsilon", "amsgrad") if name in params]
        if forbidden:
            raise InvalidTrialError(f"{', '.join(forbidden)} apply only to Adam/AdamW")
    if params.get("nesterov") and (float(params.get("momentum", 0)) <= 0 or float(params.get("dampening", 0)) != 0):
        raise InvalidTrialError("Nesterov requires momentum > 0 and dampening == 0")

    model_name = str(params.get("model", model.get("name", "")))
    if model_name.startswith("vit"):
        hidden = int(params.get("hidden_dim", model.get("hidden_dim", 384)))
        heads = int(params.get("heads", model.get("heads", 6)))
        patch = int(params.get("patch_size", model.get("patch_size", 4)))
        if hidden % heads:
            raise InvalidTrialError("ViT hidden dimension must be divisible by attention heads")
        if 32 % patch:
            raise InvalidTrialError("ViT patch size must divide the 32x32 input")
        if params.get("channels_last") or runtime.get("channels_last") is True:
            raise InvalidTrialError("channels_last is restricted to compatible CNN paths")
    if model_name.startswith("resnet") and any(name in params for name in ("hidden_dim", "heads", "patch_size")):
        raise InvalidTrialError("ViT architecture parameters cannot be used with a ResNet")

    precision = str(params.get("precision", runtime.get("precision", "auto")))
    device = str(runtime.get("device", "auto"))
    if precision in {"fp16", "bf16"} and device in {"cpu", "mps"}:
        raise InvalidTrialError(
            f"{precision} trial execution is currently supported only on compatible CUDA devices"
        )
    if int(params.get("batch_size", 1)) <= 0:
        raise InvalidTrialError("batch_size must be positive")
    if int(params.get("gradient_accumulation", 1)) <= 0:
        raise InvalidTrialError("gradient_accumulation must be positive")


def check_hard_constraints(metrics: Mapping[str, Any], constraints: Sequence[ConstraintSpec]) -> list[str]:
    violations = []
    for constraint in constraints:
        if constraint.name not in metrics:
            continue
        if not _OPS[constraint.operator](metrics[constraint.name], constraint.value):
            violations.append(
                f"{constraint.name}={metrics[constraint.name]!r} violates {constraint.operator} {constraint.value!r}"
            )
    return violations
