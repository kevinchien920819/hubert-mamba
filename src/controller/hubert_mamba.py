from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from logging import Logger
from typing import Any

import torch
import torch.nn.functional as F
from config.hubert_mamba import HubertMambaConfig
from data import Batch
from tqdm.auto import tqdm


class HubertMambaController:
    def __init__(
        self,
        logger: Logger,
        cfg: HubertMambaConfig,
        wandb_run: Any,
        model: torch.nn.Module,
        dataloaders: list[torch.utils.data.DataLoader],
    ):
        self.logger = logger
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.model = model
        self.mode = self._resolve_mode()
        if self.mode == "train":
            self.train_dataloader = dataloaders[0]
            self.valid_dataloader = dataloaders[1] if len(dataloaders) > 1 else None
        else:
            self.train_dataloader = None
            self.valid_dataloader = dataloaders[-1] if dataloaders else None
        self.device = cfg.general.device
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.best_valid_loss = float("inf")
        self.update = 0
        self.scaler = None
        self.amp_dtype = cfg.solver.amp_dtype
        if self.amp_dtype not in {"fp16", "bf16", "none"}:
            raise ValueError(f"Unsupported amp_dtype: {self.amp_dtype}")
        self.amp_enabled = self.amp_dtype != "none" and self.device_type == "cuda"
        if self.mode == "train" and self.train_dataloader is not None and hasattr(self.train_dataloader, "__len__") and len(self.train_dataloader) == 0:
            raise ValueError("HubertMambaController received an empty train dataloader")
        if self.mode == "eval" and self.valid_dataloader is None:
            raise ValueError("HubertMambaController eval mode requires a validation dataloader")

    def _resolve_mode(self) -> str:
        if self.cfg.general.train:
            return "train"
        if self.cfg.general.eval:
            return "eval"
        raise ValueError("At least one of general.train or general.eval must be true")

    def _setup_optimizer(self):
        if self.cfg.solver.optimizer != "Adam":
            raise ValueError(f"HubertMambaController supports Adam, got {self.cfg.solver.optimizer}")
        return torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.cfg.solver.lr[0]),
            betas=tuple(self.cfg.solver.adam_betas),
            eps=float(self.cfg.solver.adam_eps),
            weight_decay=float(self.cfg.solver.weight_decay),
        )

    def _setup_scheduler(self):
        if self.cfg.solver.scheduler != "polynomial_decay":
            raise ValueError(f"Unsupported HuBERT scheduler: {self.cfg.solver.scheduler}")

        warmup = int(self.cfg.solver.warmup_updates)
        total = int(self.cfg.solver.max_updates)

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            progress = (step - warmup) / float(max(1, total - warmup))
            return max(0.0, 1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _loss(self, output):
        criterion = self.cfg.solver.criterion
        device = output.features.device
        zero = output.features.sum() * 0.0
        masked_loss = zero
        unmasked_loss = zero
        masked_acc = zero
        masked_count = torch.tensor(0.0, device=device)

        if output.targets_masked.numel() > 0:
            masked_loss = F.cross_entropy(output.logits_masked, output.targets_masked)
            masked_pred = output.logits_masked.argmax(dim=-1)
            masked_acc = (masked_pred == output.targets_masked).float().mean()
            masked_count = torch.tensor(float(output.targets_masked.numel()), device=device)

        if criterion.pred_nomask_weight > 0 and output.targets_unmasked.numel() > 0:
            unmasked_loss = F.cross_entropy(output.logits_unmasked, output.targets_unmasked)

        total_loss = (
            criterion.pred_masked_weight * masked_loss
            + criterion.pred_nomask_weight * unmasked_loss
            + criterion.feature_penalty_weight * output.feature_penalty
        )
        return total_loss, {
            "loss": total_loss.detach(),
            "loss_masked": masked_loss.detach(),
            "loss_unmasked": unmasked_loss.detach(),
            "feature_penalty": output.feature_penalty.detach(),
            "masked_acc": masked_acc.detach(),
            "masked_count": masked_count.detach(),
        }

    def _batch_iter(self):
        while True:
            for batch in self.train_dataloader:
                yield batch

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        dtype = torch.float16 if self.amp_dtype == "fp16" else torch.bfloat16
        return torch.amp.autocast(self.device_type, dtype=dtype)

    def _run_batch(self, batch: Batch, backward: bool, scaler=None):
        batch.to(self.device, non_blocking=self.cfg.dataloader.non_blocking_transfer)
        with torch.enable_grad() if backward else torch.no_grad():
            with self._autocast_context():
                output = self.model(batch)
                loss, metrics = self._loss(output)
        return loss, metrics

    @staticmethod
    def _merge_metrics(metrics: list[dict]) -> dict:
        if not metrics:
            return {}
        merged = {}
        counts = torch.stack([m["masked_count"].float() for m in metrics]) if "masked_count" in metrics[0] else None
        total_count = counts.sum() if counts is not None else torch.tensor(0.0)
        for key in metrics[0].keys():
            values = [m[key].float() for m in metrics]
            stacked = torch.stack(values)
            if key == "masked_count":
                merged[key] = stacked.sum().item()
            elif key in {"loss", "loss_masked", "masked_acc"} and counts is not None and total_count.item() > 0:
                merged[key] = ((stacked * counts).sum() / total_count).item()
            else:
                merged[key] = stacked.mean().item()
        return merged

    def validate(self) -> dict:
        if self.valid_dataloader is None:
            return {}
        self.model.eval()
        metrics = []
        for batch in tqdm(self.valid_dataloader, total=len(self.valid_dataloader), unit="batch", desc="HuBERT valid"):
            loss, batch_metrics = self._run_batch(batch, backward=False)
            if torch.isfinite(loss):
                metrics.append(batch_metrics)
        merged = self._merge_metrics(metrics)
        if merged:
            self.logger.info(
                "[HubertMambaController] valid update=%d loss=%.6f masked_acc=%.6f",
                self.update,
                merged["loss"],
                merged["masked_acc"],
            )
            if self.cfg.wandb.enable and self.wandb_run is not None:
                self.wandb_run.log({f"valid/{k}": v for k, v in merged.items()}, step=self.update)
        return merged

    def _save_checkpoint(self, name: str, valid_loss: float | None = None):
        os.makedirs(self.cfg.general.work_dir, exist_ok=True)
        path = f"{self.cfg.general.work_dir}/{name}"
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "update": self.update,
            "best_valid_loss": self.best_valid_loss,
            "config": self.cfg,
        }
        if valid_loss is not None:
            payload["valid_loss"] = valid_loss
        if self.scaler is not None and self.scaler.is_enabled():
            payload["scaler"] = self.scaler.state_dict()
        torch.save(payload, path)
        self.logger.info("[HubertMambaController] saved checkpoint: %s", path)

    def _resume_training_state(self):
        ckpt_path = self.cfg.general.ckpt.get("path", "")
        if not ckpt_path:
            return
        if ckpt_path == "self":
            ckpt_path = f"{self.cfg.general.work_dir}/checkpoint_last.pt"
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and self.scaler is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.update = int(checkpoint.get("update", 0))
        self.best_valid_loss = float(checkpoint.get("best_valid_loss", self.best_valid_loss))
        self.logger.info("[HubertMambaController] resumed training state from %s at update %d", ckpt_path, self.update)

    def train(self):
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled and self.amp_dtype == "fp16")
        scaler = self.scaler
        self._resume_training_state()
        train_iter = self._batch_iter()
        start = time.time()

        bar = tqdm(range(self.update + 1, self.cfg.solver.max_updates + 1), unit="update", desc="HuBERT train")
        for update in bar:
            self.update = update
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            accum_metrics = []

            for _ in range(max(1, int(self.cfg.solver.update_freq))):
                batch = next(train_iter)
                loss, metrics = self._run_batch(batch, backward=True, scaler=scaler)
                if not torch.isfinite(loss):
                    self.logger.warning("[HubertMambaController] skip NaN/Inf loss at update %d", update)
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_metrics = []
                    break
                scaled = loss / max(1, int(self.cfg.solver.update_freq))
                if scaler.is_enabled():
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                accum_metrics.append(metrics)

            if not accum_metrics:
                continue

            if scaler.is_enabled():
                scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.solver.max_grad_norm)
            if scaler.is_enabled():
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()

            merged = self._merge_metrics(accum_metrics)
            lr = self.scheduler.get_last_lr()[0]
            bar.set_postfix({"loss": f"{merged['loss']:.4f}", "lr": f"{lr:.2e}"})

            if update % self.cfg.solver.log_interval_updates == 0 or update == 1:
                elapsed_min = (time.time() - start) / 60.0
                self.logger.info(
                    "[HubertMambaController] update=%d/%d loss=%.6f masked_acc=%.6f lr=%.8f elapsed=%.2fm",
                    update,
                    self.cfg.solver.max_updates,
                    merged["loss"],
                    merged["masked_acc"],
                    lr,
                    elapsed_min,
                )
                if self.cfg.wandb.enable and self.wandb_run is not None:
                    log = {f"train/{k}": v for k, v in merged.items()}
                    log["train/lr"] = lr
                    self.wandb_run.log(log, step=update)

            run_validation = (
                self.valid_dataloader is not None
                and self.cfg.solver.validate_interval_updates > 0
                and update % self.cfg.solver.validate_interval_updates == 0
            )
            if run_validation:
                valid = self.validate()
                valid_loss = valid.get("loss", math.inf)
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self._save_checkpoint("checkpoint_best.pt", valid_loss=valid_loss)

            if update % self.cfg.solver.save_interval_updates == 0:
                self._save_checkpoint("checkpoint_last.pt")

        if self.valid_dataloader is not None:
            valid = self.validate()
            valid_loss = valid.get("loss", math.inf)
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self._save_checkpoint("checkpoint_best.pt", valid_loss=valid_loss)
        self._save_checkpoint("checkpoint_last.pt")

    def run(self):
        if self.mode == "train":
            return self.train()
        return self.validate()
