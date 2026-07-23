from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .conditions import SafeCondition, validate_condition_graph
from .exceptions import SearchSpaceError
from .schemas import ParameterSpec

_ALLOWED_TYPES = {"float", "int", "categorical", "bool", "fixed", "integer", "boolean"}


def _parse_bool(value: Any, *, field: str, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise SearchSpaceError(f"invalid Boolean value {value!r} for field {field!r}")


def _parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_choices(value: Any) -> tuple[Any, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SearchSpaceError(f"invalid JSON choices {value!r}") from exc
            if not isinstance(parsed, list):
                raise SearchSpaceError("choices JSON must be a list")
            return tuple(parsed)
        return tuple(_parse_scalar(item) for item in text.split("|") if item.strip())
    if isinstance(value, Sequence):
        return tuple(value)
    raise SearchSpaceError(f"choices must be a list or pipe-delimited string, got {type(value).__name__}")


def _normalize_type(value: Any) -> str:
    kind = str(value or "").strip().lower()
    aliases = {"integer": "int", "boolean": "bool"}
    kind = aliases.get(kind, kind)
    if kind not in {"float", "int", "categorical", "bool", "fixed"}:
        raise SearchSpaceError(f"unsupported parameter type {value!r}; expected {sorted(_ALLOWED_TYPES)}")
    return kind


def _source_error(source: str, item: Any, name: str | None, message: str) -> SearchSpaceError:
    prefix = f"{source}"
    if item is not None:
        prefix += f" item {item}"
    if name:
        prefix += f" parameter {name!r}"
    return SearchSpaceError(f"{prefix}: {message}")


def normalize_parameter(raw: Mapping[str, Any], *, source: str, item: int | str) -> ParameterSpec:
    name = str(raw.get("name", "")).strip()
    if not name:
        raise _source_error(source, item, None, "missing required field 'name'")
    try:
        kind = _normalize_type(raw.get("type"))
        low = _parse_scalar(raw.get("low"))
        high = _parse_scalar(raw.get("high"))
        step = _parse_scalar(raw.get("step"))
        choices = _parse_choices(raw.get("choices"))
        log = _parse_bool(raw.get("log"), field="log")
        enabled = _parse_bool(raw.get("enabled"), field="enabled", default=True)
        default = _parse_scalar(raw.get("default"))
        condition = str(raw.get("condition") or "").strip() or None
    except SearchSpaceError as exc:
        raise _source_error(source, item, name, str(exc)) from exc

    if kind in {"float", "int"}:
        if low is None or high is None:
            raise _source_error(source, item, name, "numeric parameters require low and high")
        if low > high:
            raise _source_error(source, item, name, f"low={low} exceeds high={high}")
        if log and (low <= 0 or high <= 0):
            raise _source_error(source, item, name, "logarithmic bounds must be positive")
        if step is not None and step <= 0:
            raise _source_error(source, item, name, "step must be positive")
        if log and step is not None:
            raise _source_error(source, item, name, "Optuna-style log ranges cannot also use step")
    elif kind == "categorical" and not choices:
        raise _source_error(source, item, name, "categorical choices cannot be empty")
    elif kind == "bool":
        choices = (False, True)
    elif kind == "fixed":
        if default is None and "value" in raw:
            default = _parse_scalar(raw.get("value"))
        if default is None:
            raise _source_error(source, item, name, "fixed parameters require default or value")

    spec = ParameterSpec(
        name=name,
        type=kind,  # type: ignore[arg-type]
        low=low,
        high=high,
        choices=choices,
        step=step,
        log=log,
        default=default,
        condition=condition,
        enabled=enabled,
        source=source,
        item=item,
        description=str(raw.get("description") or "").strip() or None,
    )
    validate_default(spec)
    if condition:
        SafeCondition(condition)
    return spec


def validate_default(spec: ParameterSpec) -> None:
    if spec.default is None:
        return
    value = spec.default
    if spec.type == "categorical" and value not in spec.choices:
        raise _source_error(spec.source, spec.item, spec.name, f"default {value!r} is not in choices")
    if spec.type == "bool" and not isinstance(value, bool):
        raise _source_error(spec.source, spec.item, spec.name, "Boolean default must be true or false")
    if spec.type in {"float", "int"}:
        if value < spec.low or value > spec.high:  # type: ignore[operator]
            raise _source_error(spec.source, spec.item, spec.name, "default is outside numeric bounds")
        if spec.type == "int" and not isinstance(value, int):
            raise _source_error(spec.source, spec.item, spec.name, "integer default must be an integer")


def normalize_space(source_value: Any, *, source_name: str = "python") -> list[ParameterSpec]:
    if isinstance(source_value, Mapping):
        rows = []
        for name, config in source_value.items():
            if isinstance(config, Mapping):
                rows.append({"name": name, **config})
            else:
                rows.append({"name": name, "type": "fixed", "default": config})
    elif isinstance(source_value, Sequence) and not isinstance(source_value, (str, bytes, bytearray)):
        rows = list(source_value)
    else:
        raise SearchSpaceError("search space must be a dictionary or list")

    specs: list[ParameterSpec] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise _source_error(source_name, index, None, "each list item must be a mapping")
        specs.append(normalize_parameter(row, source=source_name, item=index))
    return validate_space(specs)


def load_csv(path: str | Path) -> list[ParameterSpec]:
    source = Path(path)
    if not source.exists():
        raise SearchSpaceError(f"CSV search-space file not found: {source}")
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SearchSpaceError(f"CSV {source} has no header")
        specs = [
            normalize_parameter(row, source=str(source), item=reader.line_num)
            for row in reader
            if any(str(value or "").strip() for value in row.values())
        ]
    return validate_space(specs)


def load_space(path: str | Path) -> list[ParameterSpec]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return load_csv(source)
    if suffix == ".json":
        return normalize_space(json.loads(source.read_text(encoding="utf-8")), source_name=str(source))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise SearchSpaceError("PyYAML is required to load YAML search spaces") from exc
        return normalize_space(yaml.safe_load(source.read_text(encoding="utf-8")), source_name=str(source))
    raise SearchSpaceError(f"unsupported search-space file extension {suffix!r}")


def validate_space(specs: Iterable[ParameterSpec]) -> list[ParameterSpec]:
    enabled = [spec for spec in specs if spec.enabled]
    names = [spec.name for spec in enabled]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SearchSpaceError(f"duplicate parameter name(s): {', '.join(duplicates)}")
    conditions = {spec.name: SafeCondition(spec.condition) for spec in enabled if spec.condition}
    validate_condition_graph(names, conditions)
    return enabled


def active(spec: ParameterSpec, values: Mapping[str, Any]) -> bool:
    return not spec.condition or SafeCondition(spec.condition).evaluate(dict(values))


def suggest(trial: Any, specs: Sequence[ParameterSpec]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    pending = list(specs)
    while pending:
        progress = False
        for spec in list(pending):
            condition = SafeCondition(spec.condition) if spec.condition else None
            if condition and not condition.references.issubset(values):
                continue
            pending.remove(spec)
            progress = True
            if condition and not condition.evaluate(values):
                continue
            if spec.type == "float":
                values[spec.name] = trial.suggest_float(
                    spec.name,
                    float(spec.low),
                    float(spec.high),
                    step=float(spec.step) if spec.step is not None else None,
                    log=spec.log,
                )
            elif spec.type == "int":
                values[spec.name] = trial.suggest_int(
                    spec.name,
                    int(spec.low),
                    int(spec.high),
                    step=int(spec.step or 1),
                    log=spec.log,
                )
            elif spec.type in {"categorical", "bool"}:
                values[spec.name] = trial.suggest_categorical(spec.name, list(spec.choices))
            else:
                values[spec.name] = spec.default
        if not progress:
            raise SearchSpaceError("unable to order conditional search space; check condition dependencies")
    return values


def finite_values(spec: ParameterSpec) -> list[Any] | None:
    if spec.type in {"categorical", "bool"}:
        return list(spec.choices)
    if spec.type == "fixed":
        return [spec.default]
    if spec.type == "int" and not spec.log:
        step = int(spec.step or 1)
        return list(range(int(spec.low), int(spec.high) + 1, step))
    if spec.type == "float" and spec.step is not None and not spec.log:
        count = int(math.floor((float(spec.high) - float(spec.low)) / float(spec.step))) + 1
        return [float(spec.low) + index * float(spec.step) for index in range(count)]
    return None


def _ordered_finite_specs(specs: Sequence[ParameterSpec]) -> list[ParameterSpec]:
    """Topologically order a finite conditional space without evaluating code."""
    pending = list(specs)
    ordered: list[ParameterSpec] = []
    known: set[str] = set()
    while pending:
        progress = False
        for spec in list(pending):
            references = SafeCondition(spec.condition).references if spec.condition else frozenset()
            if not references.issubset(known):
                continue
            if finite_values(spec) is None:
                raise SearchSpaceError(
                    f"parameter {spec.name!r} is continuous; exhaustive enumeration is unavailable"
                )
            ordered.append(spec)
            known.add(spec.name)
            pending.remove(spec)
            progress = True
        if not progress:
            raise SearchSpaceError("unable to order conditional search space")
    return ordered


def _iter_combinations(
    specs: Sequence[ParameterSpec],
    *,
    stop_after: int | None = None,
):
    ordered = _ordered_finite_specs(specs)
    emitted = 0

    def walk(index: int, values: dict[str, Any]):
        nonlocal emitted
        if stop_after is not None and emitted >= stop_after:
            return
        if index >= len(ordered):
            emitted += 1
            yield dict(values)
            return
        spec = ordered[index]
        if spec.condition and not SafeCondition(spec.condition).evaluate(values):
            yield from walk(index + 1, values)
            return
        choices = finite_values(spec)
        assert choices is not None
        for value in choices:
            values[spec.name] = value
            yield from walk(index + 1, values)
            values.pop(spec.name, None)
            if stop_after is not None and emitted >= stop_after:
                break

    yield from walk(0, {})


def combination_count(specs: Sequence[ParameterSpec], *, maximum_enumeration: int = 100_000) -> int | None:
    try:
        count = 0
        for _candidate in _iter_combinations(specs, stop_after=maximum_enumeration + 1):
            count += 1
            if count > maximum_enumeration:
                return None
        return count
    except SearchSpaceError as exc:
        if "continuous" in str(exc):
            return None
        raise


def enumerate_combinations(specs: Sequence[ParameterSpec], *, limit: int) -> list[dict[str, Any]]:
    count = combination_count(specs, maximum_enumeration=limit)
    if count is None:
        # Distinguish a continuous space from a finite space above the safety limit.
        if any(finite_values(spec) is None for spec in specs):
            raise SearchSpaceError("exhaustive mode requires a fully discrete finite search space")
        raise SearchSpaceError(f"valid combination count exceeds safety limit {limit}")
    if count > limit:
        raise SearchSpaceError(f"valid combination count {count} exceeds safety limit {limit}")
    return list(_iter_combinations(specs))


def preview_rows(specs: Sequence[ParameterSpec]) -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in specs]
