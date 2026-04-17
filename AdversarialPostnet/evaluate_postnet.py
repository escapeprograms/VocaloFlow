"""Evaluate the adversarial post-network on VocaloFlow output mels.

Loads a VocaloFlow output mel (.npy), runs it through the trained postnet,
vocodes both to audio, and saves side-by-side spectrogram visualizations.

Usage (from AdversarialPostnet/):
    # From an existing demo mel:
    python evaluate_postnet.py \
        --input-mel ../demo/let_it_go/4-16-wavenet/output_mel.npy \
        --checkpoint checkpoints/my-run/checkpoint_20000.pt \
        --output-dir eval_output/let_it_go

    # From a raw predicted_mel in the dataset:
    python evaluate_postnet.py \
        --input-mel ../Data/Rachie/006b5d1db6a447039c30443310b60c6f/line_0/predicted_mel.npy \
        --target-mel ../Data/Rachie/006b5d1db6a447039c30443310b60c6f/line_0/target_mel.npy \
        --checkpoint checkpoints/my-run/checkpoint_20000.pt \
        --output-dir eval_output/sample_chunk
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_VOCALOFLOW_DIR = os.path.join(_REPO_ROOT, "VocaloFlow")
_DATASYNTHESIZER_DIR = os.path.join(_REPO_ROOT, "DataSynthesizer")

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import importlib.util as _ilu


def _import_from_path(module_name: str, file_path: str):
    spec = _ilu.spec_from_file_location(module_name, file_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vocoders_mod = _import_from_path(
    "ds_vocoders", os.path.join(_DATASYNTHESIZER_DIR, "utils", "vocoders.py")
)
invert_mel_to_audio_soulxsinger = _vocoders_mod.invert_mel_to_audio_soulxsinger

from model.postnet import PostNet
from configs.postnet_config import PostnetConfig

SR = 24000


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_postnet(checkpoint_path: str, device: torch.device) -> PostNet:
    """Load a trained PostNet from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", PostnetConfig())

    postnet = PostNet(
        mel_channels=128,
        num_blocks=config.postnet_num_blocks,
        kernel_size=config.postnet_kernel_size,
    ).to(device)

    postnet.load_state_dict(ckpt["postnet_state_dict"])
    postnet.eval()

    step = ckpt.get("step", "?")
    params = sum(p.numel() for p in postnet.parameters())
    print(f"[eval] Loaded PostNet from step {step} ({params:,} params)")
    return postnet


def mel_to_wav(mel: np.ndarray) -> np.ndarray:
    """Vocode a (T, 128) mel to audio via SoulX-Singer Vocos."""
    mel_transposed = mel.T.astype(np.float32)  # (128, T)
    audio = invert_mel_to_audio_soulxsinger(mel_transposed, config={})
    return audio


def plot_mel_comparison(
    mels: dict[str, np.ndarray],
    output_path: str,
    title: str = "",
) -> None:
    """Plot mel spectrograms side-by-side and save to file.

    Args:
        mels: Dict mapping label -> (T, 128) mel array.
        output_path: Path to save the figure.
        title: Optional overall title.
    """
    n = len(mels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)

    for ax, (label, mel) in zip(axes[0], mels.items()):
        im = ax.imshow(
            mel.T, aspect="auto", origin="lower",
            interpolation="none",
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("Mel bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Saved spectrogram: {output_path}")


def plot_mel_difference(
    mel_a: np.ndarray,
    mel_b: np.ndarray,
    label_a: str,
    label_b: str,
    output_path: str,
) -> None:
    """Plot the absolute difference between two mels."""
    diff = np.abs(mel_a - mel_b)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    im = ax.imshow(diff.T, aspect="auto", origin="lower", interpolation="none")
    ax.set_title(f"|{label_a} - {label_b}|  (mean={diff.mean():.4f})", fontsize=11)
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Mel bin")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Saved difference plot: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate adversarial postnet on VocaloFlow output mels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-mel", required=True,
                        help="Path to VocaloFlow output mel .npy (T, 128)")
    parser.add_argument("--target-mel", default=None,
                        help="Optional ground truth target mel .npy for comparison")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to postnet checkpoint .pt")
    parser.add_argument("--output-dir", default="eval_output",
                        help="Directory to save outputs")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # -- Load mel ----------------------------------------------------------
    predicted_mel = np.load(args.input_mel).astype(np.float32)  # (T, 128)
    print(f"[eval] Input mel: {predicted_mel.shape}  "
          f"({predicted_mel.shape[0] * 480 / SR:.2f}s)")

    # -- Load postnet and run ----------------------------------------------
    postnet = load_postnet(args.checkpoint, device)

    with torch.no_grad():
        mel_tensor = torch.from_numpy(predicted_mel).unsqueeze(0).to(device)
        sharpened_tensor = postnet(mel_tensor)
        sharpened_mel = sharpened_tensor[0].cpu().numpy()

    # -- Stats -------------------------------------------------------------
    diff = np.abs(sharpened_mel - predicted_mel)
    print(f"[eval] |sharpened - predicted|  mean={diff.mean():.4f}  "
          f"max={diff.max():.4f}")

    # -- Vocode both -------------------------------------------------------
    print("[eval] Vocoding predicted mel (no postnet)...")
    audio_predicted = mel_to_wav(predicted_mel)
    sf.write(os.path.join(args.output_dir, "predicted.wav"), audio_predicted, SR)

    print("[eval] Vocoding sharpened mel (with postnet)...")
    audio_sharpened = mel_to_wav(sharpened_mel)
    sf.write(os.path.join(args.output_dir, "sharpened.wav"), audio_sharpened, SR)

    # -- Spectrogram comparison --------------------------------------------
    mels_to_plot = {
        "Predicted (VocaloFlow)": predicted_mel,
        "Sharpened (PostNet)": sharpened_mel,
    }

    if args.target_mel:
        target_mel = np.load(args.target_mel).astype(np.float32)
        # Align lengths
        T = min(predicted_mel.shape[0], target_mel.shape[0])
        target_mel = target_mel[:T]
        mels_to_plot["Target (ground truth)"] = target_mel

        print("[eval] Vocoding target mel...")
        audio_target = mel_to_wav(target_mel)
        sf.write(os.path.join(args.output_dir, "target.wav"), audio_target, SR)

        # Difference plots
        plot_mel_difference(
            predicted_mel[:T], target_mel,
            "Predicted", "Target",
            os.path.join(args.output_dir, "diff_predicted_vs_target.png"),
        )
        plot_mel_difference(
            sharpened_mel[:T], target_mel,
            "Sharpened", "Target",
            os.path.join(args.output_dir, "diff_sharpened_vs_target.png"),
        )

    plot_mel_comparison(
        mels_to_plot,
        os.path.join(args.output_dir, "mel_comparison.png"),
    )

    # Predicted vs sharpened difference
    plot_mel_difference(
        predicted_mel, sharpened_mel,
        "Predicted", "Sharpened",
        os.path.join(args.output_dir, "diff_predicted_vs_sharpened.png"),
    )

    # -- Save mels ---------------------------------------------------------
    np.save(os.path.join(args.output_dir, "predicted_mel.npy"), predicted_mel)
    np.save(os.path.join(args.output_dir, "sharpened_mel.npy"), sharpened_mel)

    print(f"\n[eval] All outputs saved to {args.output_dir}/")
    print(f"  predicted.wav  — VocaloFlow output (no postnet)")
    print(f"  sharpened.wav  — PostNet output")
    if args.target_mel:
        print(f"  target.wav     — Ground truth")
    print(f"  mel_comparison.png — Side-by-side spectrograms")


if __name__ == "__main__":
    main()
