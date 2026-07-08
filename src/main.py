import functools
import logging
import os
import time
from dataclasses import asdict
from logging import Logger

import click
import torch

try:
    import wandb
except ImportError:
    wandb = None

from config import HubertMambaConfig, config_to_yaml, load_config
from controller import HubertMambaController
from data import get_hubert_dataloader
from model.hubert_mamba.loader import load_model
from utils import set_seed, setup_freeze, setup_logger, setup_tf32


def handle_exceptions(logger: Logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"An error occurred in {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


@click.command()
@click.option("--config-name", help="Configuration name to load", required=True)
def main(config_name) -> None:
    cfg = load_config(config_name)
    if not isinstance(cfg, HubertMambaConfig):
        raise ValueError(f"Unsupported config type: {type(cfg)}. This entrypoint supports HuBERT-Mamba only.")

    if cfg.general.device == "cuda" and cfg.general.device_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.general.device_id)

    os.makedirs(cfg.general.work_dir, exist_ok=True)
    logger = setup_logger(
        name="main",
        project_root=os.getcwd(),
        log_file=f'{cfg.general.work_dir}/{time.strftime("%Y%m%d_%H%M%S")}_main.log',
        level=logging.INFO,
    )
    logger.info("Using config: \n%s", config_to_yaml(cfg))
    pipeline(logger, cfg)


def pipeline(logger: logging.Logger, cfg: HubertMambaConfig):
    set_seed(cfg.general.seed, cfg.general.deterministic)

    if cfg.general.eval and not cfg.general.train and not cfg.general.ckpt.get("path"):
        cfg.general.ckpt["path"] = cfg.general.testing_ckpt

    model = load_model(logger, cfg)
    _log_model_size(logger, model)
    setup_freeze(cfg, logger, model)
    setup_tf32(cfg, logger)

    wandb_run = _setup_wandb(logger, cfg)
    dataloaders = _build_dataloaders(logger, cfg)
    controller = HubertMambaController(logger, cfg, wandb_run, model, dataloaders)
    controller.run()

    if wandb_run is not None:
        wandb_run.finish()


def _build_dataloaders(logger: logging.Logger, cfg: HubertMambaConfig):
    if cfg.general.train:
        logger.info("Loading HuBERT train and valid datasets")
        return [
            get_hubert_dataloader(cfg, cfg.data.train_split, True),
            get_hubert_dataloader(cfg, cfg.data.valid_split, False),
        ]
    if cfg.general.eval:
        logger.info("Loading HuBERT validation dataset")
        return [get_hubert_dataloader(cfg, cfg.data.valid_split, False)]
    raise ValueError("At least one of general.train or general.eval must be true")


def _setup_wandb(logger: logging.Logger, cfg: HubertMambaConfig):
    if not cfg.wandb.enable:
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed. Disable wandb in the config or install wandb.")
    wandb_run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.model.name,
        tags=[cfg.model.tag],
        notes=cfg.model.description,
        config=asdict(cfg),
    )
    logger.info("WandB initialized: Project - %s, Run Name - %s", cfg.wandb.project, cfg.model.name)
    return wandb_run


def _log_model_size(logger: logging.Logger, model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model Total Parameters: %s", f"{total_params:,}")
    logger.info("Model Trainable Parameters: %s", f"{trainable_params:,}")
    return total_params, trainable_params


if __name__ == "__main__":
    main()
