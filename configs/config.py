"""
Configuration schema for CleanUNet2 training (Wav2Vec2 ONLY).

This module defines the configuration structure for both Stage 1 and Stage 2 training
using Pydantic v2 for runtime validation and type safety.

Example usage:
    from configs.config import TrainConfig
    import yaml

    with open('configs/train.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)

    config = TrainConfig(**cfg_dict)
    print(config.pipeline.stage)  # Access nested fields
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ============================================================================
# Loss Configuration
# ============================================================================

@dataclass
class STFTConfig:
    """Multi-resolution STFT loss configuration."""
    fft_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    hop_sizes: List[int] = field(default_factory=lambda: [50, 120, 240])
    win_lengths: List[int] = field(default_factory=lambda: [240, 600, 1200])


@dataclass
class LossConfig:
    """Loss function weights and parameters."""
    weight_waveform: float = 10.0
    weight_spec: float = 5.0
    weight_phase: float = 5.0

    # Waveform loss parameters
    ell_p: int = 1  # 1 for L1, 2 for L2
    ell_p_lambda: float = 1.0

    # STFT loss parameters
    stft_lambda: float = 0.5
    sc_lambda: float = 0.8  # Spectral convergence
    mag_lambda: float = 0.2  # Magnitude

    # Multi-resolution STFT
    stft_config: STFTConfig = field(default_factory=STFTConfig)

    # Stage 2: Latent replication weight (gamma parameter)
    gamma_latent: float = 0.05


# ============================================================================
# Model Architecture Configuration
# ============================================================================

@dataclass
class CleanUNetParams:
    """CleanUNet (waveform domain) architecture parameters."""
    channels_input: int = 1
    channels_output: int = 1
    channels_H: int = 64
    max_H: int = 768
    encoder_n_layers: int = 8
    kernel_size: int = 4
    stride: int = 2
    tsfm_n_layers: int = 5
    tsfm_n_head: int = 8
    tsfm_d_model: int = 512
    tsfm_d_inner: int = 2048


@dataclass
class CleanSpecNetParams:
    """CleanSpecNet (spectrogram domain) architecture parameters."""
    input_channels: int = 513  # 1 + FFT_SIZE/2
    num_conv_layers: int = 5
    kernel_size: int = 4
    stride: int = 1
    conv_hidden_dim: int = 64
    hidden_dim: int = 512
    num_attention_layers: int = 5
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class Wav2Vec2Config:
    """Wav2Vec2 embeddings configuration."""
    model_name: str = "facebook/wav2vec2-xls-r-2b"
    layer: int = 24  # Extract from middle layer
    input_dim: int = 1024  # Output dimension (fixed for XLS-R 2B)
    cache_dir: str = "cached_embeddings/wav2vec2"
    use_preextracted: bool = True  # Load from cache vs. extract on-the-fly


@dataclass
class ModelConfig:
    """Model architecture and Wav2Vec2 configuration."""
    name: str = "cleanunet2_wav2vec2"
    conditioning_type: str = "film"  # "addition", "concatenation", "film"

    # Training flags
    train_cleanunet: bool = True
    train_cleanspecnet: bool = True

    # Wav2Vec2 embeddings (ALWAYS used - no xvector support)
    embeddings: Wav2Vec2Config = field(default_factory=Wav2Vec2Config)

    # Architecture parameters
    cleanunet: CleanUNetParams = field(default_factory=CleanUNetParams)
    cleanspecnet: CleanSpecNetParams = field(default_factory=CleanSpecNetParams)

    # Optional: Load vanilla checkpoint for warm start
    vanilla_checkpoint: Optional[str] = None


# ============================================================================
# Optimizer Configuration
# ============================================================================

@dataclass
class OptimizerConfig:
    """Adam optimizer configuration."""
    name: str = "adam"
    lr: float = 5.0e-05
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 0.0


# ============================================================================
# Data Configuration
# ============================================================================

@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    enabled: bool = False
    techniques: List[str] = field(default_factory=list)
    probability: float = 0.5


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""
    data_dir: str
    train_list_path: str = "filelists/train.csv"
    val_list_path: str = "filelists/test.csv"

    batch_size: int = 32
    num_workers: int = 4
    persistent_workers: bool = True

    sample_rate: int = 16000
    segment_size: int = 32000  # ~1 second at 16kHz

    # Wav2Vec2 embeddings
    use_preextracted_embeddings: bool = True

    # Data augmentation (Stage 1 specific)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Quick test mode
    quick_test: bool = False
    quick_test_samples: int = 30


# ============================================================================
# Trainer Configuration
# ============================================================================

@dataclass
class TrainerConfig:
    """PyTorch Lightning Trainer configuration."""
    accelerator: str = "auto"
    devices: int = 1
    max_epochs: int = 300
    precision: str = "16-mixed"
    log_every_n_steps: int = 50
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 5.0
    deterministic: bool = False
    benchmark: bool = True


# ============================================================================
# Checkpoint Configuration
# ============================================================================

@dataclass
class CheckpointConfig:
    """Checkpoint saving configuration."""
    save_dir: str
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    mode: str = "min"
    every_n_epochs: int = 10
    resume_from_checkpoint: Optional[str] = None


# ============================================================================
# Logger Configuration
# ============================================================================

@dataclass
class TensorBoardLoggerConfig:
    """TensorBoard logger configuration."""
    save_dir: str
    name: str = "tb_logs"
    default_hp_metric: bool = False


@dataclass
class WandbLoggerConfig:
    """Weights & Biases logger configuration."""
    project: str = "CleanUNet2_Wav2Vec2"
    name: str = "training_run"
    offline: bool = False
    log_model: bool = False


@dataclass
class LoggingConfig:
    """Unified logging configuration."""
    loggers: List[str] = field(default_factory=lambda: ["tensorboard"])
    tensorboard: TensorBoardLoggerConfig = field(default_factory=TensorBoardLoggerConfig)
    wandb: WandbLoggerConfig = field(default_factory=WandbLoggerConfig)


# ============================================================================
# Callbacks Configuration
# ============================================================================

@dataclass
class EarlyStoppingConfig:
    """Early stopping callback configuration."""
    enabled: bool = True
    monitor: str = "val_loss"
    patience: int = 20
    min_delta: float = 0.001
    mode: str = "min"


@dataclass
class CallbacksConfig:
    """Callbacks configuration."""
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


# ============================================================================
# Debug Configuration
# ============================================================================

@dataclass
class DebugConfig:
    """Debug and testing configuration."""
    quick_test: bool = False
    quick_test_samples: int = 30
    quick_test_epochs: int = 1
    seed: int = 1234


# ============================================================================
# Pipeline Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    stage: int = 1  # 1 or 2
    name: str = "wav2vec2_stage1"
    description: str = "Stage 1 training without augmentation"


# ============================================================================
# Main Training Configuration (Root)
# ============================================================================

@dataclass
class TrainConfig:
    """Complete training configuration for CleanUNet2 (Wav2Vec2 ONLY).

    This is the root configuration class that brings together all sub-configs.
    Load from YAML and instantiate with:

        import yaml
        from configs.config import TrainConfig

        with open('configs/train.yaml', 'r') as f:
            cfg_dict = yaml.safe_load(f)

        config = TrainConfig(**cfg_dict)
    """

    pipeline: PipelineConfig
    trainer: TrainerConfig
    model: ModelConfig
    losses: LossConfig
    optimizer: OptimizerConfig
    data: DataConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # Optional: Stage 1 checkpoint for Stage 2 training
    stage1_checkpoint: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Stage must be 1 or 2
        if self.pipeline.stage not in [1, 2]:
            raise ValueError(f"pipeline.stage must be 1 or 2, got {self.pipeline.stage}")

        # Stage 2 requires stage1_checkpoint
        if self.pipeline.stage == 2 and self.stage1_checkpoint is None:
            raise ValueError("Stage 2 training requires 'stage1_checkpoint' to be set")


# ============================================================================
# Inference Configuration
# ============================================================================

@dataclass
class InferenceAudioConfig:
    """Audio processing for inference."""
    sample_rate: int = 16000
    normalize: bool = True


@dataclass
class InferenceRuntimeConfig:
    """Runtime settings for inference."""
    device: str = "auto"  # "auto", "cuda", or "cpu"
    force_cpu: bool = False
    use_amp: bool = True
    verbose: bool = True
    save_metrics_report: bool = False
    metrics_report_path: str = "inference_metrics.json"


@dataclass
class InferenceInputOutputConfig:
    """Input/output file configuration."""
    input_dir: str
    input_pattern: str = "*.wav"
    output_dir: str = "denoised_output/"
    format: str = "wav"
    overwrite: bool = False


@dataclass
class InferenceConfig:
    """Complete inference configuration.

    Usage:
        import yaml
        from configs.config import InferenceConfig

        with open('configs/inference.yaml', 'r') as f:
            cfg_dict = yaml.safe_load(f)

        config = InferenceConfig(**cfg_dict)
    """

    checkpoint_path: str
    audio: InferenceAudioConfig = field(default_factory=InferenceAudioConfig)
    input_output: InferenceInputOutputConfig = field(default_factory=InferenceInputOutputConfig)
    runtime: InferenceRuntimeConfig = field(default_factory=InferenceRuntimeConfig)
    embeddings: Wav2Vec2Config = field(default_factory=Wav2Vec2Config)

    def __post_init__(self):
        """Validate inference configuration."""
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required for inference")
