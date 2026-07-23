from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RegistryItem:
    name: str
    factory: Callable[..., Any]
    description: str = ""


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, RegistryItem] = {}

    def register(self, name: str, factory: Callable[..., Any], description: str = "") -> None:
        if name in self._items:
            raise ValueError(f"duplicate {self.kind} registration {name!r}")
        self._items[name] = RegistryItem(name, factory, description)

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} {name!r}; expected one of {sorted(self._items)}")
        return self._items[name].factory

    def names(self) -> list[str]:
        return sorted(self._items)
