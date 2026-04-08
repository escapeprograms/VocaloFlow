import os
import numpy as np
import librosa
from typing import Dict

# Fix for Windows permissions (WinError 1314) when SpeechBrain/HuggingFace attempts to create symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Global cache for the HiFiGAN model so we don't reload it every slice
_HIFIGAN_MODEL = None

def mel_to_log_mel(mel: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """
    Applies log compression to a linear-scale power mel spectrogram.
    
    HiFiGAN (and most neural vocoders) are trained on log-mel, not raw power values.
    SpeechBrain internally computes: log(mel + floor).
    
    Args:
        mel: Linear-scale mel spectrogram, shape (n_mels, time).
        floor: Small constant to avoid log(0).
    
    Returns:
        log_mel: Log-compressed mel spectrogram, same shape.
    """
    return np.log(np.maximum(mel, floor))

def invert_mel_to_audio_griffin_lim(mel_spectrogram: np.ndarray, config: Dict[str, int]) -> np.ndarray:
    """
    Inverts a mel-spectrogram back into an audio waveform using Griffin-Lim phase reconstruction.
    
    Args:
        mel_spectrogram: The warped mel-spectrogram (shape: n_mels x time).
        config: STFT parameters used for extraction (sample_rate, n_fft, hop_length).
        
    Returns:
        audio: The reconstructed waveform.
    """
    sr = config.get("sample_rate", 22050)
    n_fft = config.get("n_fft", 1024)
    hop_length = config.get("hop_length", 256)
    
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spectrogram, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    return audio

def invert_mel_to_audio_hifigan(mel_spectrogram: np.ndarray, config: Dict[str, int]) -> np.ndarray:
    """
    Inverts a mel-spectrogram back into an audio waveform using a pre-trained SpeechBrain HiFiGAN.
    
    Args:
        mel_spectrogram: The warped mel-spectrogram (shape: n_mels x time).
        config: Configuration dictionary (unused by pretrained model, but kept for signature consistency).
        
    Returns:
        audio: The reconstructed waveform as a 1D numpy array.
    """
    global _HIFIGAN_MODEL
    
    try:
        import torch
        from speechbrain.inference.vocoders import HIFIGAN
    except ImportError:
        raise ImportError("To use the HiFiGAN vocoder, you must install torch and speechbrain. "
                          "Please run: pip install torch speechbrain")

    # Load model lazily
    if _HIFIGAN_MODEL is None:
        print("Loading SpeechBrain HiFiGAN pretrained model...")
        storage_dir = "tmp_hifigan_libritts"
        
        # Manually snapshot download into local folder to bypass Windows symlink issues
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="speechbrain/tts-hifigan-libritts-22050Hz", 
                local_dir=storage_dir, 
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"Warning: Manual snapshot download failed ({e}). Proceeding to SpeechBrain default as fallback.")

        # Load from local dir directly (source=storage_dir prevents re-downloading)
        _HIFIGAN_MODEL = HIFIGAN.from_hparams(source=storage_dir, savedir=storage_dir)

    # CRITICAL: HiFiGAN was trained on log-compressed mel, not linear-scale power values.
    # Apply log compression to match training distribution: log(mel + 1e-6)
    log_mel = mel_to_log_mel(mel_spectrogram)
    
    # librosa output is (n_mels, time)
    # SpeechBrain HiFiGAN expects input of shape (batch, n_mels, time)
    mel_tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
    
    # Run vocoder inference
    with torch.no_grad():
        audio_tensor = _HIFIGAN_MODEL.decode_batch(mel_tensor)
        
    # Squeeze batch dimension and retrieve numpy array
    audio = audio_tensor.squeeze().cpu().numpy()
    
    return audio

def invert_mel_to_audio(mel_spectrogram: np.ndarray, config: Dict[str, int], vocoder: str = "griffin_lim") -> np.ndarray:
    """
    Factory wrapper to invert a mel-spectrogram using the selected vocoder strategy.
    
    Args:
        mel_spectrogram: The mel-spectrogram to invert.
        config: Global STFT configuration.
        vocoder: "griffin_lim", "hifigan", or "soulxsinger".
        
    Returns:
        audio: The 1D reconstructed waveform.
    """
    if vocoder.lower() == "hifigan":
        return invert_mel_to_audio_hifigan(mel_spectrogram, config)
    elif vocoder.lower() == "soulxsinger":
        return invert_mel_to_audio_soulxsinger(mel_spectrogram, config)
    else:
        return invert_mel_to_audio_griffin_lim(mel_spectrogram, config)


# SoulX-Singer's exact mel parameters (from mel_transform.py defaults)
SOULX_MEL_CONFIG = {
    "sample_rate": 24000,
    "n_fft": 1920,
    "hop_length": 480,
    "win_length": 1920,
    "n_mels": 128,
    "fmin": 0,
    "fmax": 12000,
    "mel_mean": -4.92,
    "mel_var": 8.14,
    "clip_val": 1e-5,
}

# Global cache for the SoulX-Singer Vocos model
_SOULX_VOCODER = None

def mel_to_soulx_mel(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Re-extracts a mel spectrogram from a waveform using SoulX-Singer's exact STFT pipeline.
    
    SoulX-Singer uses 24kHz / 128 mels / hop=480 / n_fft=1920 with log-compression
    and z-score normalization. This function resamples the waveform if needed,
    then applies the full pipeline to produce a mel that Vocos expects as input.
    
    Args:
        y: Raw audio waveform (1D numpy array).
        sr: Source sample rate of `y` (will be resampled to 24000 if different).
        
    Returns:
        soulx_mel: Normalized log-mel of shape (128, time) ready for Vocos.
    """
    import math
    cfg = SOULX_MEL_CONFIG
    target_sr = cfg["sample_rate"]
    
    # Resample to 24kHz if needed
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Compute mel spectrogram using librosa with SoulX-Singer parameters
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=target_sr,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        win_length=cfg["win_length"],
        n_mels=cfg["n_mels"],
        fmin=cfg["fmin"],
        fmax=cfg["fmax"],
        center=False,
    )
    
    # Apply log compression: log(clip(mel, 1e-5))
    log_mel = np.log(np.clip(mel, a_min=cfg["clip_val"], a_max=None))
    
    # Apply z-score normalization: (mel - mean) / sqrt(var)
    normalized_mel = (log_mel - cfg["mel_mean"]) / math.sqrt(cfg["mel_var"])
    
    return normalized_mel  # shape: (128, time)


def invert_mel_to_audio_soulxsinger(mel_spectrogram: np.ndarray, config: Dict[str, int]) -> np.ndarray:
    """
    Inverts a mel-spectrogram back into audio using SoulX-Singer's own Vocos vocoder.
    
    IMPORTANT: `mel_spectrogram` must already be in SoulX-Singer's mel space 
    (128 mels, log-compressed, z-score normalized). Use `mel_to_soulx_mel` to produce it
    from a waveform, or pass directly if it was produced by SoulX-Singer's inference pipeline.
    
    Args:
        mel_spectrogram: Normalized log-mel of shape (128, time).
        config: Unused — kept for signature consistency with other vocoder functions.
        
    Returns:
        audio: Reconstructed waveform at 24kHz as a 1D numpy array.
    """
    global _SOULX_VOCODER
    
    try:
        import torch
        import sys
        SOULX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SoulX-Singer"))
        if SOULX_DIR not in sys.path:
            sys.path.insert(0, SOULX_DIR)
        from soulxsinger.models.modules.vocoder import load_vocos_model
    except ImportError as e:
        raise ImportError(f"Could not import SoulX-Singer vocoder: {e}. "
                          "Ensure the SoulX-Singer directory is at ../SoulX-Singer.")
    
    if _SOULX_VOCODER is None:
        ckpt_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "SoulX-Singer", "pretrained_models", "SoulX-Singer", "model.pt"
        ))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"SoulX-Singer checkpoint not found at: {ckpt_path}")
        
        print(f"Loading SoulX-Singer Vocos vocoder from {ckpt_path}...")
        vocos_model = load_vocos_model(ckpt_path=None, config=None)  # Build model architecture only
        
        # The checkpoint is the full SoulXSinger model: state_dict contains vocoder.model.* keys.
        # Extract and remap just the vocoder sub-weights.
        full_ckpt = torch.load(ckpt_path, map_location="cpu")
        full_sd = full_ckpt.get("state_dict", full_ckpt)
        
        prefix = "vocoder.model."
        vocoder_sd = {
            k[len(prefix):]: v
            for k, v in full_sd.items()
            if k.startswith(prefix)
        }
        
        if not vocoder_sd:
            raise RuntimeError(
                f"No vocoder weights found under prefix '{prefix}' in checkpoint. "
                f"First 5 keys: {list(full_sd.keys())[:5]}"
            )
        
        missing, unexpected = vocos_model.load_state_dict(vocoder_sd, strict=True)
        if missing:
            print(f"  Warning: Missing vocoder keys: {missing[:5]}")
        
        vocos_model.eval()
        _SOULX_VOCODER = vocos_model

    # Input shape: (n_mels, time) → Vocos expects (batch, n_mels, time)
    mel_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        audio_tensor = _SOULX_VOCODER(mel_tensor)
    
    # Output shape: (1, 1, T) → squeeze to (T,)
    audio = audio_tensor.squeeze().cpu().numpy()
    return audio

