"""
Type-safe configuration system for CleanUNet2 with WavLM embeddings.
Uses Pydantic dataclasses for runtime validation and IDE support.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ============================================================================
# WAVLM-SPECIFIC CONFIGURATION
# ============================================================================

@dataclass
class WavLMConfig:
    """WavLM embedding extractor configuration."""
    model_name: str = "microsoft/wavlm-large"
    embedding_dim: int = 1024
    hidden_states_layer: int = 24
    normalize_embeddings: bool = True


# ============================================================================
# CORE MODEL CONFIGURATION
# ============================================================================

@dataclass
class CleanUNetParams:
    """CleanUNet model parameters."""
    channels: int = 1
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    segment_size: Optional[int] = None
    filter_channels: int = 128
    kernel_size: int = 5
    dilation_cycle: int = 4
    n_layers: int = 4
    use_attention: bool = True
    dropout: float = 0.1


@dataclass
class CleanSpecNetParams:
    """CleanSpecNet model parameters."""
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    n_layers: int = 4
    n_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    wavlm: WavLMConfig = field(default_factory=WavLMConfig)
    cleanunet: CleanUNetParams = field(default_factory=CleanUNetParams)
    cleanspecnet: CleanSpecNetParams = field(default_factory=CleanSpecNetParams)


# ============================================================================
# LOSS CONFIGURATION
# ============================================================================

@dataclass
class STFTConfig:
    """Multi-resolution STFT loss configuration."""
    fft_sizes: List[int] = field(default_factory=lambda: [1024, 2048, 512])
    hop_sizes: List[int] = field(default_factory=lambda: [160, 320, 80])
    win_lengths: List[int] = field(default_factory=lambda: [1024, 2048, 512])


@dataclass
class LossConfig:
    """Loss function configuration."""
    l1_weight: float = 0.5
    stft_weight: float = 0.3
    phase_weight: float = 0.1
    spec_weight: float = 0.1
    stft_config: STFTConfig = field(default_factory=STFTConfig)


# ============================================================================
# OPTIMIZER & SCHEDULER CONFIGURATION
# ============================================================================

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "AdamW"
    lr: float = 5.0e-5
    betas: tuple = (0.9, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 1.0e-4
    amsgrad: bool = False


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    enabled: bool = False
    warmup_steps: int = 1000
    total_steps: Optional[int] = None


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    enabled: bool = False
    techniques: List[str] = field(default_factory=lambda: ["pitch_shift", "time_stretch"])
    probability: float = 0.5


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_dir: str = "."
    train_list_path: str = "filelists/train.csv"
    val_list_path: Optional[str] = "filelists/test.csv"
    val_split: Optional[float] = None
    batch_size: int = 32
    num_workers: int = 4
    persistent_workers: bool = False
    segment_size: Optional[int] = None
    sampling_rate: int = 16000
    use_preextracted_embeddings: bool = False
    augmentations: Optional[AugmentationConfig] = None


# ============================================================================
# TRAINER CONFIGURATION
# ============================================================================

@dataclass
class CheckpointConfig:
    """Checkpoint saving configuration."""
    save_dir: str = "experiments/stage1_baseline/checkpoints"
    monitor_metric: str = "val_loss"
    save_top_k: int = 3
    save_last: bool = True
    mode: str = "min"  # "min" for loss, "max" for metrics


@dataclass
class TrainerConfig:
    """PyTorch Lightning Trainer configuration."""
    max_epochs: int = 300
    val_check_interval: float = 1.0
    log_every_n_steps: int = 10
    precision: str = "16-mixed"
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    enable_progress_bar: bool = True
    num_sanity_val_steps: int = 2
    accelerator: str = "auto"
    devices: int = 1


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

@dataclass
class LoggingConfig:
    """Logging configuration."""
    tensorboard_dir: str = "experiments/stage1_baseline/tensorboard"
    wandb_enabled: bool = False
    wandb_project: str = "cleanunet2_wavlm"
    wandb_entity: Optional[str] = None
    log_metrics_interval: int = 10


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Training pipeline configuration."""
    stage: int = 1  # 1 or 2
    stage1_checkpoint: Optional[str] = None


# ============================================================================
# ROOT TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainConfig:
    """Complete training configuration."""
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.pipeline.stage not in [1, 2]:
            raise ValueError(f"pipeline.stage must be 1 or 2, got {self.pipeline.stage}")

        if self.pipeline.stage == 2 and not self.pipeline.stage1_checkpoint:
            raise ValueError("Stage 2 requires stage1_checkpoint to be specified")


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

@dataclass
class AudioProcessingConfig:
    """Audio processing for inference."""
    sampling_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64


@dataclass
class InferenceInputOutput:
    """Input/output paths for inference."""
    input_dir: str = "test_samples/"
    output_dir: str = "denoised_output/"
    checkpoint_path: str = ""


@dataclass
class InferenceConfig:
    """Complete inference configuration."""
    checkpoint_path: str = ""
    input_output: InferenceInputOutput = field(default_factory=InferenceInputOutput)
    audio_processing: AudioProcessingConfig = field(default_factory=AudioProcessingConfig)
    device: str = "auto"  # "auto", "cuda", or "cpu"
    batch_size: int = 4
    compute_metrics: bool = False
    wavlm_config: WavLMConfig = field(default_factory=WavLMConfig)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration dictionary."""
    required_keys = ["pipeline", "data", "model", "trainer"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required configuration section: {key}")


def create_train_config_from_dict(config_dict: Dict[str, Any]) -> TrainConfig:
    """Create TrainConfig from dictionary with proper nesting."""
    try:
        # Convert nested dicts to dataclass instances
        if "model" in config_dict and isinstance(config_dict["model"], dict):
            model_dict = config_dict["model"]
            if "wavlm" in model_dict and isinstance(model_dict["wavlm"], dict):
                model_dict["wavlm"] = WavLMConfig(**model_dict["wavlm"])
            if "cleanunet" in model_dict and isinstance(model_dict["cleanunet"], dict):
                model_dict["cleanunet"] = CleanUNetParams(**model_dict["cleanunet"])
            if "cleanspecnet" in model_dict and isinstance(model_dict["cleanspecnet"], dict):
                model_dict["cleanspecnet"] = CleanSpecNetParams(**model_dict["cleanspecnet"])
            config_dict["model"] = ModelConfig(**model_dict)

        # Convert other nested configs
        for section in ["loss", "optimizer", "scheduler", "data", "trainer", "checkpoint", "logging", "pipeline"]:
            if section in config_dict and isinstance(config_dict[section], dict):
                section_class = globals()[section.capitalize() + "Config"] if section != "pipeline" else PipelineConfig
                if section == "data" and "augmentations" in config_dict[section]:
                    if isinstance(config_dict[section]["augmentations"], dict):
                        config_dict[section]["augmentations"] = AugmentationConfig(**config_dict[section]["augmentations"])
                config_dict[section] = section_class(**config_dict[section])

        return TrainConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Error creating TrainConfig: {str(e)}")
