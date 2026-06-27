from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class Fidelity(Enum):
    LOSSLESS = "fp16"
    COMPRESSED = "int8"
    SKETCH = "int8_channel"


class Placement(Enum):
    GPU = "gpu"
    CPU = "cpu"
    DISK = "disk"


@dataclass
class DiffusionState:
    request_id: str
    step_idx: int
    total_steps: int
    latent: torch.Tensor | None
    fidelity: Fidelity
    placement: Placement
    value_score: float
    size_bytes: int
    p_resume: float = 0.0
    scale: torch.Tensor | float | None = None
    disk_path: str | None = None
    metadata: dict[str, Any] | None = None
