"""Cross-module import helpers and path constants for the Evaluation module.

Follows the pattern from AdversarialFinetune/ft_utils/__init__.py to avoid
namespace collisions between VocaloFlow/utils and DataSynthesizer/utils.
"""

from __future__ import annotations

import importlib.util as _ilu
import os
import sys
from datetime import datetime
from types import ModuleType

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))
VOCALOFLOW_DIR = os.path.join(REPO_ROOT, "VocaloFlow")
DATASYNTHESIZER_DIR = os.path.join(REPO_ROOT, "DataSynthesizer")
SOULX_DIR = os.path.join(REPO_ROOT, "SoulX-Singer")


def import_from_path(module_name: str, file_path: str) -> ModuleType:
    """Load *file_path* as a module registered under *module_name*."""
    spec = _ilu.spec_from_file_location(module_name, file_path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def setup_vocaloflow_sys_path() -> None:
    """Prepend VocaloFlow/ and SoulX-Singer/ to sys.path.

    Does NOT add DataSynthesizer — its ``utils`` package would shadow
    VocaloFlow's own ``utils``.  Use :func:`import_from_path` for
    DataSynthesizer imports instead.
    """
    for p in [VOCALOFLOW_DIR, SOULX_DIR]:
        if p not in sys.path:
            sys.path.insert(0, p)


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")
