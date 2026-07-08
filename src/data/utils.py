from typing import Optional

import torch
from torch import Tensor


def conv_output_length(lengths: Tensor, kernel_size: int, stride: int, padding: int = 0, dilation: int = 1) -> Tensor:
    lengths = lengths.to(dtype=torch.long)
    return torch.div(
        lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride,
        rounding_mode="floor",
    ).add(1).clamp_min(0)


def get_feat_extract_output_lengths(lengths: Tensor, conv_layers: list[list[int]]) -> Tensor:
    out = lengths.to(dtype=torch.long)
    for layer in conv_layers:
        if len(layer) == 3:
            _, kernel, stride = layer
        elif len(layer) == 2:
            kernel, stride = layer
        else:
            raise ValueError(f"conv layer must have 2 or 3 values, got {layer}")
        out = conv_output_length(out, kernel, stride)
    return out


def lengths_to_padding_mask(lengths: Tensor, max_len: Optional[int] = None) -> Tensor:
    lengths = lengths.to(dtype=torch.long)
    if max_len is None:
        max_len = int(lengths.max().item()) if lengths.numel() else 0
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


def align_targets_to_length(targets: Tensor, target_lengths: Tensor, output_lengths: Tensor, max_output_len: int) -> Tensor:
    aligned = torch.full(
        (targets.shape[0], max_output_len),
        -100,
        device=targets.device,
        dtype=targets.dtype,
    )
    for b in range(targets.shape[0]):
        src_len = int(target_lengths[b].item())
        dst_len = min(int(output_lengths[b].item()), max_output_len)
        if src_len <= 0 or dst_len <= 0:
            continue
        src = targets[b, :src_len]
        if src_len == dst_len:
            aligned[b, :dst_len] = src
            continue
        index = torch.linspace(0, src_len - 1, dst_len, device=targets.device)
        aligned[b, :dst_len] = src[index.round().long()]
    return aligned
