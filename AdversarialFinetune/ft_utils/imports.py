"""Cross-module import helpers for AdversarialFinetune.

AdversarialFinetune pulls code from two sibling packages (VocaloFlow and
AdversarialPostnet) that both define top-level ``configs``/``training`` /
``model`` / ``utils`` packages.  Only one such package name can be on
``sys.path`` at once, so we use two tricks:

* VocaloFlow is placed on ``sys.path`` (``setup_vocaloflow_sys_path``) and its
  subpackages are imported normally.
* AdversarialPostnet files are loaded by file path via ``import_from_path``
  under unique module names (``ap_discriminator`` etc.).
"""

from __future__ import annotations

import importlib.util as _ilu
import os
import sys
from types import ModuleType


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
VOCALOFLOW_DIR = os.path.join(REPO_ROOT, "VocaloFlow")
ADV_POSTNET_DIR = os.path.join(REPO_ROOT, "AdversarialPostnet")


def import_from_path(module_name: str, file_path: str) -> ModuleType:
    """Load ``file_path`` as a module registered under ``module_name``.

    Register in ``sys.modules`` BEFORE ``exec_module`` so that decorators which
    resolve types via ``cls.__module__`` lookups (e.g. ``@dataclass``) don't
    hit ``None`` and crash.
    """
    spec = _ilu.spec_from_file_location(module_name, file_path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def setup_vocaloflow_sys_path() -> None:
    """Prepend ``VocaloFlow/`` to ``sys.path`` if not already there.

    Must be called before ``from configs.default import VocaloFlowConfig`` and
    sibling VocaloFlow imports.  Idempotent.
    """
    if VOCALOFLOW_DIR not in sys.path:
        sys.path.insert(0, VOCALOFLOW_DIR)
