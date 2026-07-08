from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset

from .dataclass import Batch, Sample


def _load_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "HuBERT waveform loading requires torchaudio. Install torchaudio matching the local PyTorch build."
        ) from exc
    return torchaudio


class HubertPretrainDataset(TorchDataset):
    def __init__(
        self,
        manifest_dir: str,
        label_dir: str,
        split: str,
        label_name: str = "km",
        sample_rate: int = 16000,
        label_rate: float = 100.0,
        max_sample_size: int = 250000,
        min_sample_size: int = 32000,
        pad_audio: bool = False,
        random_crop: bool = True,
        normalize: bool = False,
    ):
        self.manifest_dir = Path(manifest_dir)
        self.label_dir = Path(label_dir)
        self.split = split
        self.label_name = label_name
        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size
        self.pad_audio = pad_audio
        self.random_crop = random_crop
        self.normalize = normalize

        self.data = self._load_manifest()
        labels = self._load_labels()
        if len(labels) != len(self.data):
            raise ValueError(
                f"Label count mismatch for {split}: manifest has {len(self.data)} rows, "
                f"{self.label_dir / f'{split}.{label_name}'} has {len(labels)} rows."
            )

        paired = self._filter_samples(list(zip(self.data, labels)))
        if not paired:
            raise ValueError(
                f"No usable samples for {split}; check min_sample_size={self.min_sample_size}, "
                f"pad_audio={self.pad_audio}, and manifest lengths."
            )

        self.data = []
        for sample, target in paired:
            sample.target = target
            self.data.append(sample)

    def _load_manifest(self) -> list[Sample]:
        manifest_path = self.manifest_dir / f"{self.split}.tsv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        rows = manifest_path.read_text(encoding="utf-8").splitlines()
        if not rows:
            raise ValueError(f"Empty manifest: {manifest_path}")

        root = Path(rows[0])
        data = []
        for line in rows[1:]:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rel_path, frames = parts[0], int(parts[1])
            audio_path = Path(rel_path)
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            data.append(
                Sample(
                    filename=audio_path.stem,
                    path=str(audio_path),
                    length=frames,
                )
            )

        if not data:
            raise ValueError(f"Manifest has no samples: {manifest_path}")
        return data

    def _load_labels(self) -> list[torch.Tensor]:
        label_path = self.label_dir / f"{self.split}.{self.label_name}"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")

        labels = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                labels.append(torch.empty(0, dtype=torch.long))
            else:
                labels.append(torch.tensor([int(x) for x in line.split()], dtype=torch.long))
        return labels

    def _filter_samples(self, samples: list[tuple[Sample, torch.Tensor]]) -> list[tuple[Sample, torch.Tensor]]:
        if self.pad_audio or self.min_sample_size <= 0:
            return samples
        return [(sample, target) for sample, target in samples if int(sample.length) >= self.min_sample_size]

    def get_lengths(self) -> list[int]:
        return [min(int(item.length), int(self.max_sample_size)) for item in self.data]

    def __len__(self):
        return len(self.data)

    def _load_audio(self, path: str) -> torch.Tensor:
        torchaudio = _load_torchaudio()
        wavform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            wavform = torchaudio.transforms.Resample(sr, self.sample_rate)(wavform)
        if wavform.ndim == 2:
            wavform = wavform.mean(dim=0)
        wavform = wavform.float()
        if self.normalize:
            wavform = (wavform - wavform.mean()) / wavform.std().clamp_min(1e-5)
        return wavform

    def __getitem__(self, idx) -> Sample:
        sample = self.data[idx]
        wavform = self._load_audio(sample.path)
        target = sample.target
        valid_length = int(wavform.numel())

        if wavform.numel() > self.max_sample_size:
            start = random.randint(0, wavform.numel() - self.max_sample_size) if self.random_crop else 0
            end = start + self.max_sample_size
            wavform = wavform[start:end]
            valid_length = int(wavform.numel())
            label_start = int(round(start / self.sample_rate * self.label_rate))
            label_end = int(round(end / self.sample_rate * self.label_rate))
            target = target[label_start:label_end]
        elif self.pad_audio and wavform.numel() < self.min_sample_size:
            valid_length = int(wavform.numel())
            wavform = F.pad(wavform, (0, self.min_sample_size - wavform.numel()))

        return Sample(
            filename=sample.filename,
            path=sample.path,
            length=valid_length,
            wavform=wavform,
            target=target,
        )

    def collate_fn(self, batch: list[Sample]) -> Batch:
        wavforms = [item.wavform for item in batch]
        targets = [item.target for item in batch]
        wavform = pad_sequence(wavforms, batch_first=True, padding_value=0.0)
        target = pad_sequence(targets, batch_first=True, padding_value=-100)
        return Batch(
            path=[item.path for item in batch],
            wavform=wavform,
            length=torch.tensor([int(item.length) for item in batch], dtype=torch.long),
            target=target.long(),
            target_length=torch.tensor([int(t.numel()) for t in targets], dtype=torch.long),
        )
