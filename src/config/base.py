from dataclasses import dataclass, field


@dataclass
class GeneralConfig:
    device: str = "cuda"
    device_id: str = "0"
    work_dir: str = "default"
    seed: int = 39
    deterministic: bool = False
    freeze: list = field(default_factory=list)
    unfreeze: list = field(default_factory=list)
    ckpt: dict = field(
        default_factory=lambda: {
            "path": "",
            "modules": {
                "from": ["all"],
                "to": ["all"],
            },
        }
    )
    testing_ckpt: str = "default"
    train: bool = True
    eval: bool = False


@dataclass
class DataloaderConfig:
    name: str = "HubertPretrainDataset"
    num_workers: int = 20
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    non_blocking_transfer: bool = True
    token_batch_size: int = 0
    batch_size: dict = field(
        default_factory=lambda: {
            "train": 128,
            "dev": 128,
            "eval": 128,
        }
    )


@dataclass
class WandbConfig:
    enable: bool = True
    entity: str = "icebird"
    project: str = "hubert-mamba"
