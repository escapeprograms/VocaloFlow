"""Generic config / dataclass helpers shared across modules."""

from __future__ import annotations

import dataclasses
from typing import Type, TypeVar

T = TypeVar("T")


def rebuild_dataclass_tolerant(saved, target_cls: Type[T]) -> T:
    """Construct a fresh ``target_cls`` instance, copying fields from ``saved``.

    Fields present on ``saved`` but not on ``target_cls`` are silently dropped;
    fields on ``target_cls`` but not on ``saved`` keep their class defaults.
    This lets checkpoints survive additive or subtractive schema changes.

    ``saved`` may be a dataclass instance (``asdict`` is used) or any object
    that supports ``vars()``.
    """
    fresh = target_cls()
    valid = {f.name for f in dataclasses.fields(fresh)}
    saved_dict = (
        dataclasses.asdict(saved) if dataclasses.is_dataclass(saved) else vars(saved)
    )
    for k, v in saved_dict.items():
        if k in valid:
            setattr(fresh, k, v)
    return fresh
