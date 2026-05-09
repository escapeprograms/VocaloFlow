"""Cross-module import helpers and path constants for the RVCBaseline module.

Follows the pattern from Evaluation/eval_utils/imports.py.
"""

from __future__ import annotations

import importlib.util as _ilu
import os
import sys
from datetime import datetime
from types import ModuleType

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RVC_BASELINE_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(RVC_BASELINE_DIR, ".."))
VOCALOFLOW_DIR = os.path.join(REPO_ROOT, "VocaloFlow")
EVAL_DIR = os.path.join(REPO_ROOT, "Evaluation")
APPLIO_DIR = os.path.join(REPO_ROOT, "Applio")
DATA_DIR = os.path.join(REPO_ROOT, "Data", "Rachie")


def import_from_path(module_name: str, file_path: str) -> ModuleType:
    """Load *file_path* as a module registered under *module_name*."""
    spec = _ilu.spec_from_file_location(module_name, file_path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")
