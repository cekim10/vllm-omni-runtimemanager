from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.state.diffusion_state import Fidelity


class FidelityPolicy:
    """Map state value to a checkpoint fidelity tier."""

    def __init__(self, theta_h: float = 0.7, theta_w: float = 0.3) -> None:
        self.theta_h = theta_h
        self.theta_w = theta_w

    def assign(self, value_score: float) -> Fidelity:
        if value_score >= self.theta_h:
            return Fidelity.LOSSLESS
        if value_score >= self.theta_w:
            return Fidelity.COMPRESSED
        return Fidelity.SKETCH

    def compress(
        self,
        latent: torch.Tensor,
        fidelity: Fidelity,
    ) -> tuple[torch.Tensor, torch.Tensor | float | None]:
        if fidelity == Fidelity.LOSSLESS:
            return latent.detach().to(device="cpu").contiguous(), None

        tensor = latent.detach().to(dtype=torch.float32, device="cpu").contiguous()

        if fidelity == Fidelity.COMPRESSED:
            scale = tensor.abs().max().clamp_min(1e-8) / 127.0
            quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
            return quantized, scale

        reduce_dims = tuple(range(2, tensor.ndim))
        if not reduce_dims:
            scale = tensor.abs().max().clamp_min(1e-8) / 127.0
        else:
            scale = tensor.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / 127.0
        quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
        return quantized, scale

    def decompress(
        self,
        latent: torch.Tensor,
        scale: torch.Tensor | float | None,
        fidelity: Fidelity,
    ) -> torch.Tensor:
        if fidelity == Fidelity.LOSSLESS:
            return latent
        if scale is None:
            raise ValueError(f"Missing scale for fidelity={fidelity.value}")
        return latent.to(torch.float32) * scale

    @staticmethod
    def estimate_size_bytes(latent: torch.Tensor, scale: Any = None) -> int:
        size = int(latent.nelement() * latent.element_size())
        if isinstance(scale, torch.Tensor):
            size += int(scale.nelement() * scale.element_size())
        return size
