from __future__ import annotations

import random
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from config.hubert_mamba import HubertMambaModelConfig
from data.dataclass import Batch
from data.utils import align_targets_to_length, get_feat_extract_output_lengths, lengths_to_padding_mask
from torch import Tensor, nn


@dataclass
class HubertMambaOutput:
    logits_masked: Tensor
    targets_masked: Tensor
    logits_unmasked: Tensor
    targets_unmasked: Tensor
    feature_penalty: Tensor
    features: Tensor
    padding_mask: Tensor
    mask_indices: Tensor


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, scale: float):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def _load_mamba_class():
    try:
        from mamba_ssm import Mamba
        return Mamba
    except ImportError:
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
            return Mamba
        except ImportError as exc:
            warnings.warn(
                "mamba_ssm is not installed; using the local TorchMambaFallback mixer. "
                "Install mamba-ssm in a compatible CUDA environment for paper-faithful Mamba kernels.",
                RuntimeWarning,
                stacklevel=2,
            )
            return TorchMambaFallback


class TorchMambaFallback(nn.Module):
    """Small dependency-free mixer used when mamba_ssm is unavailable."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_fast_path: bool = True,
        **_: object,
    ):
        super().__init__()
        inner_dim = int(d_model * expand)
        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=inner_dim,
        )
        self.state_proj = nn.Sequential(
            nn.Linear(inner_dim, max(1, int(d_state))),
            nn.SiLU(),
            nn.Linear(max(1, int(d_state)), inner_dim),
        )
        self.out_proj = nn.Linear(inner_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        mixed = self.depthwise_conv(value.transpose(1, 2)).transpose(1, 2)
        if mixed.shape[1] > x.shape[1]:
            mixed = mixed[:, : x.shape[1]]
        elif mixed.shape[1] < x.shape[1]:
            mixed = F.pad(mixed, (0, 0, 0, x.shape[1] - mixed.shape[1]))
        mixed = F.silu(mixed) + self.state_proj(value)
        return self.out_proj(mixed * torch.sigmoid(gate))


class ConvFeatureEncoder(nn.Module):
    def __init__(self, cfg: HubertMambaModelConfig):
        super().__init__()
        if cfg.extractor_mode not in {"default", "layer_norm"}:
            raise ValueError(f"Unsupported extractor_mode: {cfg.extractor_mode}")
        layers = []
        in_channels = 1
        for idx, (out_channels, kernel, stride) in enumerate(cfg.conv_feature_layers):
            conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                bias=cfg.conv_bias,
            )
            modules = [conv]
            if idx == 0 and cfg.extractor_mode == "default":
                modules.append(nn.GroupNorm(out_channels, out_channels))
            elif cfg.extractor_mode == "layer_norm":
                modules.append(TransposeLastLayerNorm(out_channels))
            modules.append(nn.GELU())
            layers.append(nn.Sequential(*modules))
            in_channels = out_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, wavform: Tensor) -> Tensor:
        x = wavform.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)


class TransposeLastLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ConvPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int, groups: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        pos = self.conv(x.transpose(1, 2))
        if pos.shape[-1] > x.shape[1]:
            pos = pos[..., : x.shape[1]]
        return self.activation(pos.transpose(1, 2))


class TimeMasker(nn.Module):
    def __init__(self, embed_dim: int, mask_prob: float, mask_length: int, min_masks: int):
        super().__init__()
        self.mask_embedding = nn.Parameter(torch.empty(embed_dim).uniform_())
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.min_masks = min_masks

    def _compute_mask(self, lengths: Tensor, max_len: int, device: torch.device) -> Tensor:
        mask = torch.zeros((lengths.shape[0], max_len), device=device, dtype=torch.bool)
        if self.mask_prob <= 0 or self.mask_length <= 0:
            return mask
        for b, length_t in enumerate(lengths):
            length = int(length_t.item())
            if length <= 0:
                continue
            if length <= self.mask_length:
                mask[b, :length] = True
                continue
            expected = int(round(self.mask_prob * length / self.mask_length))
            num_spans = max(self.min_masks, expected)
            max_starts = max(1, length - self.mask_length + 1)
            num_spans = min(num_spans, max_starts)
            starts = torch.randperm(max_starts, device=device)[:num_spans]
            offsets = torch.arange(self.mask_length, device=device)
            span_idx = starts[:, None] + offsets[None, :]
            mask[b, span_idx.reshape(-1)] = True
        return mask

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        mask = self._compute_mask(lengths, x.shape[1], x.device)
        x = x.clone()
        x[mask] = self.mask_embedding.to(dtype=x.dtype)
        return x, mask


class MambaMixer(nn.Module):
    def __init__(self, cfg: HubertMambaModelConfig):
        super().__init__()
        Mamba = _load_mamba_class()
        kwargs = dict(
            d_model=cfg.encoder_embed_dim,
            d_state=cfg.mamba_ssm_state_expand,
            d_conv=cfg.mamba_conv_kernel_size,
            expand=cfg.mamba_block_expand,
            use_fast_path=cfg.use_fast_path,
        )
        self.variant = cfg.variant
        self.forward_mamba = Mamba(**kwargs)
        if cfg.variant == "extbimamba":
            self.backward_mamba = Mamba(**kwargs)
        else:
            self.backward_mamba = None

    def _bidirectional(self, x: Tensor) -> Tensor:
        if self.variant == "extbimamba":
            y_fwd = self.forward_mamba(x)
            y_bwd = torch.flip(self.backward_mamba(torch.flip(x, dims=[1])), dims=[1])
            return y_fwd + y_bwd
        if self.variant == "innbimamba":
            y_fwd = self.forward_mamba(x)
            y_bwd = torch.flip(self.forward_mamba(torch.flip(x, dims=[1])), dims=[1])
            return y_fwd + y_bwd
        raise ValueError(f"Unsupported bidirectional Mamba variant: {self.variant}")

    def _bidirectional_unpadded(self, x: Tensor, padding_mask: Tensor | None) -> Tensor:
        if padding_mask is None or not padding_mask.any():
            return self._bidirectional(x)
        out = torch.zeros_like(x)
        valid_lengths = (~padding_mask).sum(dim=1)
        for b, length_t in enumerate(valid_lengths):
            length = int(length_t.item())
            if length <= 0:
                continue
            out[b : b + 1, :length] = self._bidirectional(x[b : b + 1, :length])
        return out

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if self.variant == "mamba" or self.variant == "mamba_mlp":
            return self.forward_mamba(x)
        if self.variant == "extbimamba":
            return self._bidirectional_unpadded(x, padding_mask)
        if self.variant == "innbimamba":
            return self._bidirectional_unpadded(x, padding_mask)
        raise ValueError(f"Unsupported Mamba variant: {self.variant}")


class HubertMambaEncoderLayer(nn.Module):
    def __init__(self, cfg: HubertMambaModelConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.encoder_embed_dim, eps=cfg.layer_norm_eps)
        self.mixer = MambaMixer(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ffn = None
        ffn_dim = cfg.encoder_ffn_embed_dim
        if cfg.variant == "mamba_mlp" and ffn_dim <= 0:
            ffn_dim = 4 * cfg.encoder_embed_dim
        if ffn_dim and ffn_dim > 0:
            self.ffn = nn.Sequential(
                nn.LayerNorm(cfg.encoder_embed_dim, eps=cfg.layer_norm_eps),
                nn.Linear(cfg.encoder_embed_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(cfg.activation_dropout),
                nn.Linear(ffn_dim, cfg.encoder_embed_dim),
                nn.Dropout(cfg.dropout),
            )

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        residual = x
        x = residual + self.dropout(self.mixer(self.norm(x), padding_mask))
        if self.ffn is not None:
            x = x + self.ffn(x)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class HubertMambaModel(nn.Module):
    def __init__(self, cfg: HubertMambaModelConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        conv_dim = cfg.conv_feature_layers[-1][0]
        self.feature_extractor = ConvFeatureEncoder(cfg)
        self.post_extract_proj = nn.Linear(conv_dim, cfg.encoder_embed_dim)
        self.layer_norm = nn.LayerNorm(conv_dim, eps=cfg.layer_norm_eps)
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.pos_conv = ConvPositionalEncoding(cfg.encoder_embed_dim, cfg.conv_pos, cfg.conv_pos_groups)
        self.masker = TimeMasker(cfg.encoder_embed_dim, cfg.mask_prob, cfg.mask_length, cfg.mask_min_masks)
        self.layers = nn.ModuleList([HubertMambaEncoderLayer(cfg) for _ in range(cfg.encoder_layers)])
        self.final_proj = nn.Linear(cfg.encoder_embed_dim, cfg.final_dim)
        self.label_embs_concat = nn.Parameter(torch.empty(num_classes, cfg.final_dim))
        nn.init.uniform_(self.label_embs_concat)

    def _extract_features(self, wavform: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.feature_extractor(wavform)
        feature_penalty = features.float().pow(2).mean()
        if self.cfg.feature_grad_mult != 1.0:
            features = GradMultiply.apply(features, self.cfg.feature_grad_mult)
        feature_lengths = get_feat_extract_output_lengths(lengths, self.cfg.conv_feature_layers).to(wavform.device)
        return features, feature_lengths, feature_penalty

    def _encode(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        padding_mask = lengths_to_padding_mask(feature_lengths, features.shape[1])
        x = self.dropout_features(features)
        x = self.layer_norm(x)
        x = self.post_extract_proj(x)
        x, mask_indices = self.masker(x, feature_lengths)
        x = self.dropout_input(x + self.pos_conv(x))
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for layer in self.layers:
            if self.training and self.cfg.encoder_layerdrop > 0 and random.random() < self.cfg.encoder_layerdrop:
                continue
            x = layer(x, padding_mask)
        return x, padding_mask, mask_indices

    def _compute_logits(self, features: Tensor) -> Tensor:
        projected = self.final_proj(features)
        projected = F.normalize(projected, dim=-1)
        label_embs = F.normalize(self.label_embs_concat, dim=-1)
        return torch.matmul(projected, label_embs.transpose(0, 1)) / self.cfg.logit_temp

    def forward(self, batch: Batch) -> HubertMambaOutput:
        if batch.target is None or batch.target_length is None:
            raise ValueError("HubertMambaModel requires batch.target and batch.target_length")
        features, feature_lengths, feature_penalty = self._extract_features(batch.wavform, batch.length)
        encoded, padding_mask, mask_indices = self._encode(features, feature_lengths)
        logits = self._compute_logits(encoded)

        targets = align_targets_to_length(
            batch.target,
            batch.target_length,
            feature_lengths,
            max_output_len=logits.shape[1],
        )
        valid = (~padding_mask) & (targets != -100)
        masked = mask_indices & valid
        unmasked = (~mask_indices) & valid

        return HubertMambaOutput(
            logits_masked=logits[masked],
            targets_masked=targets[masked],
            logits_unmasked=logits[unmasked],
            targets_unmasked=targets[unmasked],
            feature_penalty=feature_penalty,
            features=encoded,
            padding_mask=padding_mask,
            mask_indices=mask_indices,
        )
