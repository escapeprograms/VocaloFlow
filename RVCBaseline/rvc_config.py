"""Configuration dataclass for the RVC baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RVCBaselineConfig:
    """Configuration for RVC baseline training and inference.

    Data-selection fields (data_dir, manifest_path, max_dtw_cost,
    val_fraction, seed) MUST match EvalConfig defaults so that
    run_inference.py selects the same validation set as run_eval.py.
    """

    # -- Data (must match EvalConfig for val-set consistency) ---------------
    data_dir: str = "../Data/Rachie"
    manifest_path: str = "../Data/Rachie/manifest.csv"
    max_dtw_cost: float = 100.0
    val_fraction: float = 0.05
    seed: int = 42

    # -- Training data prep ------------------------------------------------
    max_training_chunks: int = 1000
    training_data_dir: str = "./training_data"

    # -- RVC training ------------------------------------------------------
    rvc_model_name: str = "Rachie-RVC"
    rvc_epochs: int = 300
    rvc_sample_rate: int = 40000
    f0_method: str = "rmvpe"

    # -- Inference ---------------------------------------------------------
    max_eval_chunks: int = 0
    output_dir: str = "./output"
    index_ratio: float = 0.0
    f0_shift: int = 0

    # -- DALI prior inference (Context 2) ---------------------------------
    mode: str = "quality"
    dali_ustx_dir: str = "../Evaluation/dali_ustx"
    dali_output_dir: str = "./output_dali"

    # -- Model paths (populated after training) ----------------------------
    rvc_model_path: str = ""
    rvc_index_path: str = ""

    # -- YAML serialization ------------------------------------------------

    def to_yaml(self, path: str) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "RVCBaselineConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
