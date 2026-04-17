"""Path derivation helpers for AdversarialFinetune runs."""

from __future__ import annotations

import os


def derive_run_paths(config) -> None:
    """Set ``config.checkpoint_dir`` and ``config.log_dir`` from ``config.run_name``.

    No-op when ``run_name`` is empty.  Mutates ``config`` in place.
    """
    if not config.run_name:
        return
    config.checkpoint_dir = os.path.join("./checkpoints", config.run_name)
    config.log_dir = os.path.join("./logs", config.run_name)
