"""One-shot migration for AdversarialFinetune checkpoints.

Two historical defects are repaired:

1. ``finetune_config`` was stored as a ``FinetuneConfig`` dataclass instance,
   preventing VocaloFlow's inference pipeline from unpickling our files
   (it has no ``finetune_config`` on its sys.path).  Fix: rewrite as
   ``dataclasses.asdict(config)``.

2. ``config`` was momentarily populated with the wrong value — a stale call
   site passed ``wandb.run.id`` (a short string) where ``vf_config``
   belonged, so ``ckpt["config"]`` ended up as a string instead of a
   ``VocaloFlowConfig``.  Fix: recover the correct config by reading it
   from the pretrained checkpoint that ``finetune_config['pretrained_run']``
   points at.

Both repairs are idempotent (re-running on an already-fixed file is a
no-op for that concern).  A ``.bak`` sidecar is written on the first
fixup so originals are recoverable.

Usage (from AdversarialFinetune/):
    python migrate_checkpoint.py checkpoints/4-17-adv/checkpoint_65000.pt
    python migrate_checkpoint.py checkpoints/4-17-adv/          # all in dir
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import shutil
import sys

# Bootstrap VocaloFlow on sys.path BEFORE torch is used to unpickle, because
# the ``config`` key holds a ``VocaloFlowConfig`` whose module path is
# ``configs.default`` — only resolvable when VocaloFlow/ is on sys.path.
# ``finetune_config`` (the legacy key we're migrating) is already on sys.path
# since this script lives alongside it in AdversarialFinetune/.
from ft_utils import setup_vocaloflow_sys_path

setup_vocaloflow_sys_path()

import torch  # noqa: E402


def _fix_finetune_config(ckpt: dict) -> str | None:
    """Convert ``finetune_config`` to a plain dict if it's still a dataclass.

    Returns a short action string if anything changed, else ``None``.
    """
    ft_cfg = ckpt.get("finetune_config")
    if ft_cfg is None:
        return None
    if isinstance(ft_cfg, dict):
        return None
    if not dataclasses.is_dataclass(ft_cfg):
        return None
    ckpt["finetune_config"] = dataclasses.asdict(ft_cfg)
    return f"finetune_config: {type(ft_cfg).__name__} -> dict"


def _fix_config(ckpt: dict) -> str | None:
    """Recover ``ckpt["config"]`` if it is not a ``VocaloFlowConfig``.

    Reads ``finetune_config['pretrained_run']`` / ``pretrained_vocaloflow_dir``
    (handling both dict and dataclass forms) to locate the pretrained
    checkpoint, then copies that file's ``config`` into ours.  Returns a
    short action string if anything changed, else ``None``.
    """
    # Import inside the function so this script stays importable even on
    # machines where VocaloFlow isn't yet on sys.path for some reason.
    from configs.default import VocaloFlowConfig

    cfg = ckpt.get("config")
    if isinstance(cfg, VocaloFlowConfig):
        return None                                     # already correct

    ft = ckpt.get("finetune_config")
    if ft is None:
        raise ValueError("Cannot recover 'config': no 'finetune_config' key to find the pretrained run")

    if isinstance(ft, dict):
        pretrained_run = ft.get("pretrained_run")
        vf_dir = ft.get("pretrained_vocaloflow_dir", "../VocaloFlow")
    else:                                               # FinetuneConfig instance
        pretrained_run = getattr(ft, "pretrained_run", None)
        vf_dir = getattr(ft, "pretrained_vocaloflow_dir", "../VocaloFlow")

    if not pretrained_run:
        raise ValueError("Cannot recover 'config': finetune_config has no pretrained_run")

    # Resolve the pretrained checkpoint directory relative to the repo, not
    # to the current working directory, so this works regardless of where the
    # script is invoked from.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vf_dir_abs = os.path.abspath(os.path.join(repo_root, "AdversarialFinetune", vf_dir)) \
        if not os.path.isabs(vf_dir) else vf_dir
    pretrained_ckpt_dir = os.path.join(vf_dir_abs, "checkpoints", pretrained_run)
    if not os.path.isdir(pretrained_ckpt_dir):
        raise FileNotFoundError(
            f"Cannot recover 'config': pretrained dir not found: {pretrained_ckpt_dir}"
        )

    pretrained_paths = sorted(glob.glob(os.path.join(pretrained_ckpt_dir, "checkpoint_*.pt")))
    if not pretrained_paths:
        raise FileNotFoundError(
            f"Cannot recover 'config': no checkpoints in {pretrained_ckpt_dir}"
        )

    # Any pretrained checkpoint from the same run has the same architecture;
    # pick the first one to minimise I/O (no need to sort-by-step).
    source = pretrained_paths[0]
    src_ckpt = torch.load(source, map_location="cpu", weights_only=False)
    src_config = src_ckpt.get("config")
    if not isinstance(src_config, VocaloFlowConfig):
        raise ValueError(
            f"Cannot recover 'config': pretrained checkpoint {source} has non-VocaloFlowConfig config"
        )

    ckpt["config"] = src_config
    return f"config: {type(cfg).__name__} -> VocaloFlowConfig (from {os.path.basename(source)})"


def migrate_one(path: str, *, backup: bool = True, dry_run: bool = False) -> str:
    """Rewrite a single checkpoint in place.  Returns a one-line status string.

    Runs both fixups (finetune_config dict conversion + config recovery).
    Either or both can be a no-op.  The file is rewritten only if at least
    one fixup made a change.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    actions: list[str] = []
    for fn in (_fix_finetune_config, _fix_config):
        msg = fn(ckpt)
        if msg:
            actions.append(msg)

    if not actions:
        return f"OK    {path} — already correct"

    if dry_run:
        return f"DRY   {path} — would apply: {'; '.join(actions)}"

    if backup and not os.path.exists(path + ".bak"):
        shutil.copy2(path, path + ".bak")
    torch.save(ckpt, path)
    sidecar_note = "  (.bak saved)" if backup and os.path.exists(path + ".bak") else ""
    return f"WROTE {path} — {'; '.join(actions)}{sidecar_note}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "path",
        help="Path to a single checkpoint_*.pt OR a directory containing them.",
    )
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip writing .bak sidecar (default: keep one)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        paths = sorted(glob.glob(os.path.join(args.path, "checkpoint_*.pt")))
        if not paths:
            print(f"No checkpoint_*.pt under {args.path}", file=sys.stderr)
            sys.exit(1)
    else:
        paths = [args.path]

    for p in paths:
        print(migrate_one(p, backup=not args.no_backup, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
