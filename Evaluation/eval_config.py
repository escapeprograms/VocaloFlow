"""Configuration dataclass for VocaloFlow evaluation pipeline."""

from dataclasses import asdict, dataclass


@dataclass
class EvalConfig:
    """Configuration for modular VocaloFlow evaluation.

    All metric toggles default to True except ContentVec (SVC-motivation only).
    """

    # -- Checkpoint & model ------------------------------------------------
    checkpoint_path: str = ""
    device: str = "auto"

    # -- Data --------------------------------------------------------------
    data_dir: str = "../Data/Rachie"
    manifest_path: str = "../Data/Rachie/manifest.csv"
    dali_annot_dir: str = "../DALI/DALI_v2.0/annot_tismir"
    max_dtw_cost: float = 100.0
    val_fraction: float = 0.05
    seed: int = 42
    max_eval_chunks: int = 0

    # -- RVC baseline (pre-generated audio) --------------------------------
    rvc_audio_dir: str = ""

    # -- Inference ---------------------------------------------------------
    num_ode_steps: int = 32
    ode_method: str = "midpoint"
    cfg_scale: float = 2.0
    chunk_size: int = 256
    overlap: int = 16

    # -- Metric toggles ----------------------------------------------------
    enable_f0_rmse: bool = True
    enable_f0_pearson: bool = True
    enable_ffe_gpe_vde: bool = True
    enable_mcd: bool = True
    enable_wer: bool = True
    enable_mos: bool = True
    enable_duration_mae: bool = True
    enable_contentvec: bool = False

    # -- Eval context ------------------------------------------------------
    eval_context: str = "quality"

    # -- DALI USTX (controllability) ---------------------------------------
    dali_ustx_dir: str = "./dali_ustx"

    # -- External model paths ----------------------------------------------
    rmvpe_model_path: str = "../SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"
    phoneset_path: str = "../SoulX-Singer/soulxsinger/utils/phoneme/phone_set.json"
    speaker_embedding_path: str = "../SpeakerEmbedding/embeddings/Rachie/speaker_embedding.pt"

    # -- WER ---------------------------------------------------------------
    whisper_model: str = "large-v3"

    # -- MOS ---------------------------------------------------------------
    mos_model: str = "utmos"

    # -- Audio params (must match VocaloFlow) ------------------------------
    sr: int = 24000
    hop: int = 480
    n_mels: int = 128

    # -- MCD ---------------------------------------------------------------
    mcd_n_mfcc: int = 13

    # -- Output ------------------------------------------------------------
    output_dir: str = "./eval_output"
    save_plots: bool = True
    num_plot_samples: int = 8

    # -- YAML serialization ------------------------------------------------

    def to_yaml(self, path: str) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def needs_audio(self) -> bool:
        """True if any enabled metric requires vocoded audio."""
        return (
            self.enable_f0_rmse
            or self.enable_f0_pearson
            or self.enable_ffe_gpe_vde
            or self.enable_wer
            or self.enable_mos
            or self.enable_contentvec
        )

    def needs_f0_extraction(self) -> bool:
        """True if any enabled metric requires RMVPE F0 from predicted audio."""
        return (
            self.enable_f0_rmse
            or self.enable_f0_pearson
            or self.enable_ffe_gpe_vde
        )
