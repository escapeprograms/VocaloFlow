"""Extract a single averaged ECAPA-TDNN speaker embedding from target voice clips.

Produces a single (192,) embedding vector representing the target speaker,
averaged across multiple reference clips for robustness.  The output is saved
as a .pt file and can be used as a global speaker identity vector.

Usage:
    cd "Honors Thesis"
    python SpeakerEmbedding/extract_embedding.py \
        --audio-files path/to/clip1.wav path/to/clip2.wav \
        --output speaker_embedding.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torchaudio
from torch import Tensor


def load_ecapa_model(device: str = "cpu"):
    """Load the SpeechBrain ECAPA-TDNN speaker verification model."""
    import shutil
    import speechbrain.utils.fetching as sb_fetch
    from speechbrain.inference.speaker import EncoderClassifier

    # Windows lacks symlink privileges; patch to copy instead
    _orig_link = sb_fetch.link_with_strategy
    def _copy_instead(src, dst, *args, **kwargs):
        dst = Path(dst)
        src = Path(src)
        if dst.exists():
            dst.unlink()
        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
    sb_fetch.link_with_strategy = _copy_instead

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path(__file__).parent / "checkpoints" / "ecapa-tdnn"),
        run_opts={"device": device},
    )

    sb_fetch.link_with_strategy = _orig_link
    return model


def extract_single(model, audio_path: str, target_sr: int = 16000) -> Tensor:
    """Extract a single L2-normalized (192,) embedding from one audio file."""
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    embedding = model.encode_batch(waveform)  # (1, 1, 192)
    embedding = embedding.squeeze()  # (192,)
    embedding = torch.nn.functional.normalize(embedding, dim=0)
    return embedding


def extract_averaged(
    model, audio_paths: list[str], target_sr: int = 16000
) -> Tensor:
    """Extract and average L2-normalized embeddings across multiple clips."""
    embeddings = []
    for path in audio_paths:
        emb = extract_single(model, path, target_sr)
        embeddings.append(emb)
        print(f"  {Path(path).name}: extracted (192,)")

    if len(embeddings) > 1:
        stacked = torch.stack(embeddings)
        sims = torch.mm(stacked, stacked.T)
        print(f"\n  Pairwise cosine similarities:")
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                print(f"    {Path(audio_paths[i]).name} <-> "
                      f"{Path(audio_paths[j]).name}: {sims[i, j]:.4f}")

    averaged = torch.stack(embeddings).mean(dim=0)
    averaged = torch.nn.functional.normalize(averaged, dim=0)
    return averaged


def main():
    parser = argparse.ArgumentParser(
        description="Extract averaged ECAPA-TDNN speaker embedding"
    )
    parser.add_argument(
        "--audio-files", nargs="+", required=True,
        help="One or more audio files from the target speaker",
    )
    parser.add_argument(
        "--output", default=str(Path(__file__).parent / "embeddings" / "speaker_embedding.pt"),
        help="Output path for the .pt file (default: SpeakerEmbedding/embeddings/speaker_embedding.pt)",
    )
    parser.add_argument(
        "--name", default=None,
        help="Speaker name — creates SpeakerEmbedding/embeddings/<name>/speaker_embedding.pt",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    output = args.output
    if args.name:
        output = str(Path(__file__).parent / "embeddings" / args.name / "speaker_embedding.pt")

    os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"Loading ECAPA-TDNN on {args.device}...")
    model = load_ecapa_model(args.device)

    print(f"Extracting embeddings from {len(args.audio_files)} clip(s)...")
    embedding = extract_averaged(model, args.audio_files)

    torch.save(embedding.cpu(), output)
    print(f"\nSaved speaker embedding to {output}")
    print(f"  Shape: {embedding.shape}, dtype: {embedding.dtype}")
    print(f"  L2 norm: {embedding.norm().item():.6f}")


if __name__ == "__main__":
    main()
