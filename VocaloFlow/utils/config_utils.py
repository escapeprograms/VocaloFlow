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

    ``saved`` may be a dataclass instance (``asdict`` is used), a plain dict,
    or any object that supports ``vars()``.  Accepting dict is important for
    checkpoints that pickle configs as dicts to stay portable across sys.path
    configurations (e.g. AdversarialFinetune's FinetuneConfig is stored this
    way so VocaloFlow's inference pipeline can unpickle the whole checkpoint).
    """
    fresh = target_cls()
    valid = {f.name for f in dataclasses.fields(fresh)}
    if dataclasses.is_dataclass(saved):
        saved_dict = dataclasses.asdict(saved)
    elif isinstance(saved, dict):
        saved_dict = saved
    else:
        saved_dict = vars(saved)
    for k, v in saved_dict.items():
        if k in valid:
            setattr(fresh, k, v)
    return fresh
