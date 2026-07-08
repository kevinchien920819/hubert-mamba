import logging

import torch
from config.hubert_mamba import HubertMambaConfig

from .model import HubertMambaModel


def _load_checkpoint(logger: logging.Logger, model: torch.nn.Module, cfg: HubertMambaConfig) -> None:
    ckpt_path = cfg.general.ckpt.get("path", "")
    if not ckpt_path:
        return
    if ckpt_path == "self":
        ckpt_path = f"{cfg.general.work_dir}/checkpoint_last.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded checkpoint from %s", ckpt_path)
    if missing:
        logger.warning("Missing checkpoint keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected checkpoint keys: %s", unexpected)


def load_model(logger: logging.Logger, cfg: HubertMambaConfig) -> HubertMambaModel:
    model = HubertMambaModel(cfg.model, num_classes=cfg.data.num_classes)
    _load_checkpoint(logger, model, cfg)
    model.to(cfg.general.device)
    logger.info("Load HubertMamba model:\n%s", model)
    return model

