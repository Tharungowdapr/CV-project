"""String -> class registry so configs can name components without imports.

Adding a calibrator must not require editing the experiment runner; it needs a
@register decorator and a YAML file, nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from reiduq.core.exceptions import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if key in self._items:
                raise ConfigError(f"{self.name}: duplicate registration for '{key}'")
            self._items[key] = cls
            return cls

        return deco

    def get(self, key: str) -> type[T]:
        if key not in self._items:
            raise ConfigError(
                f"{self.name}: unknown component '{key}'. Available: {sorted(self._items)}"
            )
        return self._items[key]

    def build(self, key: str, **kwargs: object) -> T:
        return self.get(key)(**kwargs)  # type: ignore[call-arg]

    def keys(self) -> list[str]:
        return sorted(self._items)


DETECTORS: Registry[object] = Registry("detectors")
SEGMENTERS: Registry[object] = Registry("segmenters")
TRACKERS: Registry[object] = Registry("trackers")
BACKBONES: Registry[object] = Registry("backbones")
CALIBRATORS: Registry[object] = Registry("calibrators")
AGGREGATORS: Registry[object] = Registry("aggregators")
DATASETS: Registry[object] = Registry("datasets")
