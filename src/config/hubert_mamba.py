from dataclasses import dataclass, field

from .base import DataloaderConfig, GeneralConfig, WandbConfig


@dataclass
class HubertMambaDataConfig:
    manifest_dir: str = "/path/to/manifest"
    label_dir: str = "/path/to/labels"
    labels: list[str] = field(default_factory=lambda: ["km"])
    train_split: str = "train"
    valid_split: str = "valid"
    sample_rate: int = 16000
    label_rate: float = 100.0
    num_classes: int = 100
    max_sample_size: int = 250000
    min_sample_size: int = 32000
    pad_audio: bool = False
    random_crop: bool = True
    normalize: bool = False


@dataclass
class HubertMambaModelConfig:
    name: str = "HubertMamba"
    tag: str = "mamba_base_iter1"
    description: str = "Mamba-based HuBERT masked unit pretraining"
    variant: str = "mamba"  # mamba, extbimamba, innbimamba, mamba_mlp
    size: str = "base"

    conv_feature_layers: list[list[int]] = field(
        default_factory=lambda: [[512, 10, 5]] + [[512, 3, 2] for _ in range(4)] + [[512, 2, 2] for _ in range(2)]
    )
    conv_bias: bool = False
    extractor_mode: str = "default"

    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = -1
    final_dim: int = 256
    logit_temp: float = 0.1
    layer_norm_eps: float = 1e-5

    conv_pos: int = 128
    conv_pos_groups: int = 16
    mask_prob: float = 0.80
    mask_length: int = 10
    mask_min_masks: int = 2

    dropout_input: float = 0.1
    dropout_features: float = 0.1
    dropout: float = 0.1
    activation_dropout: float = 0.0
    encoder_layerdrop: float = 0.05
    feature_grad_mult: float = 0.1
    untie_final_proj: bool = True

    mamba_ssm_state_expand: int = 16
    mamba_conv_kernel_size: int = 4
    mamba_block_expand: int = 2
    use_fast_path: bool = True


@dataclass
class HubertCriterionConfig:
    pred_masked_weight: float = 1.0
    pred_nomask_weight: float = 0.0
    feature_penalty_weight: float = 10.0


@dataclass
class HubertSolverConfig:
    optimizer: str = "Adam"
    adam_betas: tuple[float, float] = (0.9, 0.98)
    adam_eps: float = 1e-6
    weight_decay: float = 0.01
    scheduler: str = "polynomial_decay"
    lr: list[float] = field(default_factory=lambda: [5e-4])
    max_updates: int = 250000
    warmup_updates: int = 20000
    update_freq: int = 2
    max_grad_norm: float = 10.0
    amp_dtype: str = "fp16"
    log_interval_updates: int = 200
    validate_interval_updates: int = 10000
    save_interval_updates: int = 25000
    keep_last_checkpoints: int = 1
    criterion: HubertCriterionConfig = field(default_factory=HubertCriterionConfig)


@dataclass
class HubertMambaConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    model: HubertMambaModelConfig = field(default_factory=HubertMambaModelConfig)
    data: HubertMambaDataConfig = field(default_factory=HubertMambaDataConfig)
    solver: HubertSolverConfig = field(default_factory=HubertSolverConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
