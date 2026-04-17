"""Differentiable ODE unrolling over a VocaloFlow velocity field.

Mirrors VocaloFlow/inference/inference.py sample_ode(), but:
  * no @torch.no_grad — gradients flow through every step;
  * no CFG (caller passes full conditioning);
  * no diagnostics;
  * optional gradient checkpointing per step to cap activation memory.

Used by the adversarial fine-tune training loop to produce x_1_hat from the
prior mel in ``ode_num_steps`` Euler or midpoint steps.  The resulting tensor
feeds the reconstruction / adversarial / feature-matching losses; its gradient
propagates back through all ``ode_num_steps`` forward passes into the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def _call_model(
    model: nn.Module,
    x_t: Tensor,
    t: Tensor,
    x_0: Tensor,
    f0: Tensor,
    voicing: Tensor,
    phoneme_ids: Tensor,
    padding_mask: Tensor | None,
    grad_checkpoint: bool,
) -> Tensor:
    """Single model forward, optionally wrapped in torch.utils.checkpoint."""
    if not grad_checkpoint:
        return model(x_t, t, x_0, f0, voicing, phoneme_ids, padding_mask)

    # checkpoint requires a function taking only tensors; capture padding_mask
    # in closure since it may be None.
    def _run(x_t_, t_, x_0_, f0_, voicing_, phoneme_ids_):
        return model(x_t_, t_, x_0_, f0_, voicing_, phoneme_ids_, padding_mask)

    return checkpoint(
        _run, x_t, t, x_0, f0, voicing, phoneme_ids, use_reentrant=False,
    )


def unroll_ode(
    model: nn.Module,
    x_0: Tensor,
    f0: Tensor,
    voicing: Tensor,
    phoneme_ids: Tensor,
    padding_mask: Tensor | None,
    num_steps: int,
    method: str = "euler",
    grad_checkpoint: bool = False,
) -> Tensor:
    """Differentiable ODE integration from t=0 to t=1.

    Args:
        model: VocaloFlow velocity field.  Caller controls .train() / .eval().
        x_0: (B, T, 128) prior mel-spectrogram (serves as the initial state).
        f0, voicing, phoneme_ids: conditioning, (B, T) each.
        padding_mask: (B, T) bool, True = valid frame, or None.
        num_steps: Number of integration steps.
        method: "euler" (1 model call per step) or "midpoint" (2 per step).
        grad_checkpoint: Wrap each model call in torch.utils.checkpoint to
            trade compute for activation memory.  Useful when num_steps >= 4
            with larger batch sizes.

    Returns:
        (B, T, 128) integrated mel x_1_hat.
    """
    B = x_0.shape[0]
    device = x_0.device
    dt = 1.0 / num_steps
    x_t = x_0

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((B,), t_val, device=device, dtype=x_0.dtype)

        if method == "euler":
            v = _call_model(
                model, x_t, t, x_0, f0, voicing, phoneme_ids,
                padding_mask, grad_checkpoint,
            )
            x_t = x_t + dt * v

        elif method == "midpoint":
            v1 = _call_model(
                model, x_t, t, x_0, f0, voicing, phoneme_ids,
                padding_mask, grad_checkpoint,
            )
            x_mid = x_t + 0.5 * dt * v1
            t_mid = torch.full((B,), t_val + 0.5 * dt, device=device, dtype=x_0.dtype)
            v2 = _call_model(
                model, x_mid, t_mid, x_0, f0, voicing, phoneme_ids,
                padding_mask, grad_checkpoint,
            )
            x_t = x_t + dt * v2

        else:
            raise ValueError(f"Unknown ODE method: {method!r}")

    return x_t


# ────────────────────────────────────────────────────────────────────────────
# Smoke tests (run directly: python ode_unroll.py  — from AdversarialFinetune/)
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys

    # Bridge to VocaloFlow.  Put VocaloFlow on sys.path so its own
    # ``configs``/``model``/``inference`` packages resolve; this is safe in
    # this __main__ block because we don't have any AdversarialFinetune
    # sub-packages with colliding names (flat layout by design).
    _THIS = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.abspath(os.path.join(_THIS, ".."))
    sys.path.insert(0, os.path.join(_REPO, "VocaloFlow"))

    from configs.default import VocaloFlowConfig      # noqa: E402
    from model.vocaloflow import VocaloFlow            # noqa: E402
    from inference.inference import sample_ode         # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = VocaloFlowConfig()
    cfg.num_wavenet_blocks = 8
    cfg.num_convnext_blocks = 0
    model = VocaloFlow(cfg).to(device).eval()

    B, T = 2, 64
    x_0 = torch.randn(B, T, 128, device=device)
    f0 = torch.rand(B, T, device=device) * 400
    voicing = (torch.rand(B, T, device=device) > 0.5).float()
    phoneme_ids = torch.randint(0, cfg.phoneme_vocab_size, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)

    # --- Test 1: numerical equivalence with sample_ode --------------------
    print("Test 1: unroll_ode vs sample_ode (no-grad path)")
    for method, n in [("euler", 4), ("midpoint", 8)]:
        with torch.no_grad():
            out_ref = sample_ode(
                model, x_0, f0, voicing, phoneme_ids,
                num_steps=n, method=method, padding_mask=mask,
                diagnostics=False, cfg_scale=1.0,
            )
            out_new = unroll_ode(
                model, x_0, f0, voicing, phoneme_ids, mask,
                num_steps=n, method=method, grad_checkpoint=False,
            )
        diff = (out_ref - out_new).abs().max().item()
        print(f"  method={method:8s} steps={n}  max|diff|={diff:.3e}")
        assert diff < 1e-4, f"unroll_ode drifted from sample_ode: {diff}"

    # --- Test 2: gradient flows through every step ------------------------
    print("\nTest 2: gradient flow through 4 Euler steps")
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    out = unroll_ode(
        model, x_0, f0, voicing, phoneme_ids, mask,
        num_steps=4, method="euler", grad_checkpoint=False,
    )
    out.sum().backward()

    # Grad should reach most trainable parameters via at least one of the 4
    # ODE steps.  Note: a fresh VocaloFlow uses AdaLN-Zero, which gates
    # DiT attention/FFN residuals by a modulation initialised to 0 — so
    # dit_blocks.*.qkv/ffn weights receive zero gradient until the AdaLN
    # linear is trained away from zero.  This does NOT indicate a broken
    # graph; it's the same behaviour as VocaloFlow's own training loop.
    # Once we load pretrained weights (in the real fine-tune), AdaLN is
    # non-zero and every parameter gets gradient.
    n_total = 0
    n_with_grad = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_total += 1
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            n_with_grad += 1
    print(f"  params with non-zero grad: {n_with_grad}/{n_total} "
          f"(AdaLN-Zero blocks attention/FFN weights in fresh models)")
    assert n_with_grad > 0.5 * n_total, (
        "fewer than half of parameters received gradient — ODE graph likely broken"
    )

    # Spot-check: input_proj is upstream of the AdaLN-Zero gate, so it must
    # always receive gradient when the ODE graph is intact.
    for name, p in model.named_parameters():
        if name == "input_proj.weight":
            s = p.grad.abs().sum().item()
            assert s > 0, f"input_proj.weight got no grad — ODE graph broken"
            print(f"  input_proj.weight: |grad|.sum = {s:.3e}")
            break

    # --- Test 3: gradient checkpointing works + same output ---------------
    print("\nTest 3: grad_checkpoint=True matches grad_checkpoint=False")
    model.eval()
    with torch.no_grad():
        a = unroll_ode(
            model, x_0, f0, voicing, phoneme_ids, mask,
            num_steps=4, method="euler", grad_checkpoint=False,
        )
        b = unroll_ode(
            model, x_0, f0, voicing, phoneme_ids, mask,
            num_steps=4, method="euler", grad_checkpoint=True,
        )
    diff = (a - b).abs().max().item()
    print(f"  max|diff|={diff:.3e}")
    assert diff < 1e-5, "grad_checkpoint changed numerics"

    print("\nAll ODE unroll smoke tests passed.")
