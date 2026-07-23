from __future__ import annotations

import ast
from collections import defaultdict
from typing import Any, Iterable

from .exceptions import SearchSpaceError

_ALLOWED_COMPARE = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


class SafeCondition:
    """Restricted Boolean condition parser; never evaluates arbitrary Python."""

    def __init__(self, expression: str):
        self.expression = expression.strip()
        try:
            self.tree = ast.parse(self.expression, mode="eval")
        except SyntaxError as exc:
            raise SearchSpaceError(f"malformed condition {expression!r}: {exc.msg}") from exc
        self.references = frozenset(self._validate(self.tree.body))

    def _validate(self, node: ast.AST) -> set[str]:
        if isinstance(node, ast.Name):
            return {node.id}
        if isinstance(node, ast.Constant):
            return set()
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            refs: set[str] = set()
            for item in node.elts:
                refs.update(self._validate(item))
            return refs
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            refs: set[str] = set()
            for item in node.values:
                refs.update(self._validate(item))
            return refs
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._validate(node.operand)
        if isinstance(node, ast.Compare):
            refs = self._validate(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                if type(op) not in _ALLOWED_COMPARE:
                    raise SearchSpaceError(
                        f"condition {self.expression!r} uses unsupported operator {type(op).__name__}"
                    )
                refs.update(self._validate(comparator))
            return refs
        raise SearchSpaceError(
            f"condition {self.expression!r} contains unsupported syntax {type(node).__name__}"
        )

    def evaluate(self, values: dict[str, Any]) -> bool:
        return bool(self._evaluate(self.tree.body, values))

    def _evaluate(self, node: ast.AST, values: dict[str, Any]) -> Any:
        if isinstance(node, ast.Name):
            return values.get(node.id)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [self._evaluate(item, values) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._evaluate(item, values) for item in node.elts)
        if isinstance(node, ast.Set):
            return {self._evaluate(item, values) for item in node.elts}
        if isinstance(node, ast.BoolOp):
            items = [bool(self._evaluate(item, values)) for item in node.values]
            return all(items) if isinstance(node.op, ast.And) else any(items)
        if isinstance(node, ast.UnaryOp):
            return not bool(self._evaluate(node.operand, values))
        if isinstance(node, ast.Compare):
            current = self._evaluate(node.left, values)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                other = self._evaluate(comparator, values)
                if not _ALLOWED_COMPARE[type(op)](current, other):
                    return False
                current = other
            return True
        raise AssertionError(type(node))


def validate_condition_graph(names: Iterable[str], conditions: dict[str, SafeCondition]) -> None:
    known = set(names)
    graph: dict[str, set[str]] = defaultdict(set)
    for name, condition in conditions.items():
        unknown = sorted(condition.references - known)
        if unknown:
            raise SearchSpaceError(
                f"parameter {name!r} condition references unknown parameter(s): {', '.join(unknown)}"
            )
        graph[name].update(condition.references)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise SearchSpaceError(f"cyclic condition dependency detected at parameter {name!r}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in graph.get(name, ()):
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for item in known:
        visit(item)
