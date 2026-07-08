from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Sample:
    filename: str = ""
    path: str = ""
    length: int = 0
    wavform: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None


@dataclass
class Batch:
    path: list[str]
    wavform: torch.Tensor
    length: torch.Tensor
    target: Optional[torch.Tensor] = None
    target_length: Optional[torch.Tensor] = None

    def to(self, device, non_blocking: bool = False):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device, non_blocking=non_blocking))
        return self

    def __len__(self):
        return len(self.path)
