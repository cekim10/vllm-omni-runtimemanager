#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


DEFAULT_PROMPT_SET: list[dict[str, str]] = [
    {
        "prompt_id": "preflight_000",
        "motion_category": "static_low_motion",
        "prompt": "A ceramic teacup on a windowsill at sunrise, dust motes drifting in the light.",
    },
    {
        "prompt_id": "preflight_001",
        "motion_category": "static_low_motion",
        "prompt": "A mountain lake at dawn with faint mist slowly moving above the water.",
    },
    {
        "prompt_id": "preflight_002",
        "motion_category": "camera_motion",
        "prompt": "A slow cinematic pan across a bookstore aisle with warm lighting and stacked books.",
    },
    {
        "prompt_id": "preflight_003",
        "motion_category": "camera_motion",
        "prompt": "A gentle zoom toward a lighthouse on a rocky coast as waves roll in.",
    },
    {
        "prompt_id": "preflight_004",
        "motion_category": "single_object_motion",
        "prompt": "A red fox trotting across a snowy field, leaving a clean trail behind.",
    },
    {
        "prompt_id": "preflight_005",
        "motion_category": "single_object_motion",
        "prompt": "A ballerina spinning on a rehearsal stage under a single spotlight.",
    },
    {
        "prompt_id": "preflight_006",
        "motion_category": "multi_object_motion",
        "prompt": "Several cyclists crossing a city intersection while pedestrians move along the sidewalks.",
    },
    {
        "prompt_id": "preflight_007",
        "motion_category": "fast_complex_motion",
        "prompt": "A surfer carving through a breaking wave with spray flying in every direction.",
    },
    {
        "prompt_id": "preflight_008",
        "motion_category": "fast_complex_motion",
        "prompt": "A street dance performance with multiple dancers moving quickly under neon lights.",
    },
    {
        "prompt_id": "preflight_009",
        "motion_category": "scene_change_occlusion",
        "prompt": "A yellow taxi passes behind a row of street trees and briefly disappears from view.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight temporal-dimension kill test on the native Wan2.2 T2V path."
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Primary video model. Default is the native Wan2.2 T2V path.",
    )
    parser.add_argument(
        "--prompt-set",
        default="experiments/temporal_dimension_prompt_set.json",
        help="JSON file with prompt_id, prompt, and motion_category.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/temporal_dimension_preflight",
        help="Directory for CSVs, report, and saved videos.",
    )
    parser.add_argument("--num-prompts", type=int, default=10, help="Limit prompts for the initial preflight.")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--boundary-ratio", type=float, default=0.875)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument(
        "--disable-semantic-metric",
        action="store_true",
        help="Skip prompt-conditioned CLIP scoring for intermediate videos.",
    )
    parser.add_argument(
        "--semantic-model",
        default="openai/clip-vit-base-patch32",
        help="Prompt-conditioned image-text model used for semantic trajectory scoring.",
    )
    parser.add_argument(
        "--semantic-frame-count",
        type=int,
        default=4,
        help="How many frames per video to sample for semantic scoring.",
    )
    parser.add_argument(
        "--semantic-device",
        default="cpu",
        help="Device for semantic scoring, usually cpu to avoid perturbing diffusion memory.",
    )
    parser.add_argument(
        "--capture-progress",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0],
        help="Measured trajectory progress points in [0,1].",
    )
    return parser.parse_args()


def _read_prompt_set(path: Path, limit: int) -> list[dict[str, Any]]:
    candidate_paths = [path]
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        candidate_paths.append(repo_root / path)
        candidate_paths.append(repo_root / "experiments" / path.name)

    resolved_path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if resolved_path is None:
        print(
            f"[preflight] prompt set {path} not found; "
            "falling back to the built-in 10-prompt representative set."
        )
        prompts = DEFAULT_PROMPT_SET
    else:
        prompts = json.loads(resolved_path.read_text())
    if not isinstance(prompts, list):
        raise ValueError(f"Prompt set must be a list: {path}")
    return prompts[:limit]


def _build_omni(args: argparse.Namespace) -> Omni:
    return Omni(
        model=args.model,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=DiffusionParallelConfig(),
    )


def _unwrap_output(result: Any) -> OmniRequestOutput:
    return OmniRequestOutput.unwrap_result(result)


def _normalize_frames(output: OmniRequestOutput) -> np.ndarray:
    frames: Any = output.images
    if isinstance(frames, list):
        if not frames:
            raise ValueError("No frames found in output.images.")
        if len(frames) == 1:
            frames = frames[0]

    if isinstance(frames, tuple) and len(frames) == 2:
        frames = frames[0]
    if isinstance(frames, dict):
        frames = frames.get("frames") or frames.get("video")

    if isinstance(frames, torch.Tensor):
        tensor = frames.detach().cpu()
        if tensor.ndim == 5 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim == 4 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        if tensor.is_floating_point():
            if tensor.min() < 0.0 or tensor.max() > 1.0:
                tensor = (tensor.clamp(-1.0, 1.0) + 1.0) * 0.5
            tensor = (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        else:
            tensor = tensor.to(torch.uint8)
        return tensor.numpy()

    if isinstance(frames, list):
        stacked: list[np.ndarray] = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            frame_np = np.asarray(frame)
            if frame_np.ndim == 3 and frame_np.shape[0] in (3, 4):
                frame_np = np.transpose(frame_np, (1, 2, 0))
            if frame_np.dtype != np.uint8:
                if np.issubdtype(frame_np.dtype, np.floating):
                    if frame_np.min() < 0.0 or frame_np.max() > 1.0:
                        frame_np = (np.clip(frame_np, -1.0, 1.0) + 1.0) * 0.5
                    frame_np = np.clip(frame_np, 0.0, 1.0)
                    frame_np = np.round(frame_np * 255.0).astype(np.uint8)
                else:
                    frame_np = frame_np.astype(np.uint8)
            stacked.append(frame_np)
        return np.stack(stacked, axis=0)

    arr = np.asarray(frames)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"Unsupported video output shape: {arr.shape}")
    if arr.shape[-1] not in (3, 4) and arr.shape[1] in (3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if arr.min() < 0.0 or arr.max() > 1.0:
                arr = (np.clip(arr, -1.0, 1.0) + 1.0) * 0.5
            arr = np.clip(arr, 0.0, 1.0)
            arr = np.round(arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _save_video(path: Path, frames_u8: np.ndarray, fps: float) -> None:
    path.write_bytes(mux_video_audio_bytes(frames_u8, None, fps=fps))


def _video_tensor(video: np.ndarray, size: int = 64) -> torch.Tensor:
    tensor = torch.from_numpy(video).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2).contiguous()
    gray = 0.2989 * tensor[:, 0] + 0.5870 * tensor[:, 1] + 0.1140 * tensor[:, 2]
    gray = gray.unsqueeze(1)
    return F.interpolate(gray, size=(size, size), mode="bilinear", align_corners=False).squeeze(1)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) < 1e-12:
        return 1.0 if float(torch.linalg.vector_norm(a - b)) < 1e-12 else 0.0
    value = float(torch.dot(a, b) / denom)
    return max(0.0, min(1.0, 0.5 * (value + 1.0)))


def _ratio_similarity(a: float, b: float, eps: float = 1e-6) -> float:
    a = max(float(a), 0.0)
    b = max(float(b), 0.0)
    hi = max(a, b, eps)
    lo = max(min(a, b), 0.0)
    return max(0.0, min(1.0, lo / hi))


def _safe_relative(value: float, reference: float, eps: float = 1e-6) -> float:
    return max(0.0, min(1.5, float(value) / max(float(reference), eps)))


def _motion_energy_profile(video_small: torch.Tensor) -> torch.Tensor:
    if video_small.shape[0] < 2:
        return torch.ones(1)
    diffs = (video_small[1:] - video_small[:-1]).abs()
    return diffs.mean(dim=(1, 2))


def _temporal_transition_profile(video_small: torch.Tensor) -> torch.Tensor:
    if video_small.shape[0] < 2:
        return torch.ones(1)
    flat = video_small.reshape(video_small.shape[0], -1)
    prof = []
    for idx in range(video_small.shape[0] - 1):
        prof.append(_cosine_similarity(flat[idx], flat[idx + 1]))
    return torch.tensor(prof, dtype=torch.float32)


def _flow_magnitude_profile(video_small: torch.Tensor) -> torch.Tensor:
    try:
        import cv2
    except ImportError:
        return torch.zeros(max(video_small.shape[0] - 1, 1), dtype=torch.float32)

    if video_small.shape[0] < 2:
        return torch.zeros(1, dtype=torch.float32)
    mags = []
    frames = (video_small.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    for idx in range(frames.shape[0] - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[idx],
            frames[idx + 1],
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        mags.append(float(np.linalg.norm(flow, axis=2).mean()))
    return torch.tensor(mags, dtype=torch.float32)


def _flow_direction_histogram(video_small: torch.Tensor, bins: int = 8) -> torch.Tensor:
    try:
        import cv2
    except ImportError:
        return torch.zeros(max(video_small.shape[0] - 1, 1) * bins, dtype=torch.float32)

    if video_small.shape[0] < 2:
        return torch.zeros(bins, dtype=torch.float32)
    histograms = []
    frames = (video_small.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    edges = np.linspace(0.0, 2.0 * np.pi, bins + 1, dtype=np.float32)
    for idx in range(frames.shape[0] - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[idx],
            frames[idx + 1],
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        mag = np.linalg.norm(flow, axis=2)
        ang = np.mod(np.arctan2(flow[..., 1], flow[..., 0]), 2.0 * np.pi)
        hist, _ = np.histogram(ang, bins=edges, weights=mag)
        denom = float(hist.sum())
        if denom > 1e-6:
            hist = hist / denom
        histograms.append(hist.astype(np.float32))
    return torch.from_numpy(np.concatenate(histograms, axis=0))


def _delta_map_similarity(video_small: torch.Tensor, final_small: torch.Tensor) -> float:
    if video_small.shape[0] < 2 or final_small.shape[0] < 2:
        return 1.0
    steps = min(video_small.shape[0], final_small.shape[0]) - 1
    delta = (video_small[1 : steps + 1] - video_small[:steps]).abs()
    final_delta = (final_small[1 : steps + 1] - final_small[:steps]).abs()
    diff = (delta - final_delta).abs().mean()
    scale = final_delta.mean() + 1e-6
    return max(0.0, min(1.0, 1.0 - float(diff / scale)))


def _temporal_metrics(video: np.ndarray, final_video: np.ndarray) -> dict[str, float]:
    small = _video_tensor(video)
    final_small = _video_tensor(final_video)

    energy_profile = _motion_energy_profile(small)
    final_energy_profile = _motion_energy_profile(final_small)
    transition_profile = _temporal_transition_profile(small)
    final_transition_profile = _temporal_transition_profile(final_small)
    flow_profile = _flow_magnitude_profile(small)
    final_flow_profile = _flow_magnitude_profile(final_small)
    flow_direction = _flow_direction_histogram(small)
    final_flow_direction = _flow_direction_histogram(final_small)

    energy_cosine = _cosine_similarity(energy_profile, final_energy_profile)
    transition_cosine = _cosine_similarity(transition_profile, final_transition_profile)
    motion_profile_cosine = _cosine_similarity(
        energy_profile / (energy_profile.mean() + 1e-6),
        final_energy_profile / (final_energy_profile.mean() + 1e-6),
    )
    motion_energy_mean = float(energy_profile.mean())
    final_motion_energy_mean = float(final_energy_profile.mean())
    flow_magnitude_mean = float(flow_profile.mean())
    final_flow_magnitude_mean = float(final_flow_profile.mean())
    motion_energy_ratio = _ratio_similarity(motion_energy_mean, final_motion_energy_mean)
    flow_magnitude_cosine = _cosine_similarity(flow_profile, final_flow_profile)
    flow_magnitude_ratio = _ratio_similarity(flow_magnitude_mean, final_flow_magnitude_mean)
    flow_direction_cosine = _cosine_similarity(flow_direction, final_flow_direction)
    flicker_similarity = _delta_map_similarity(small, final_small)

    shape_composite = float(
        (energy_cosine + transition_cosine + motion_profile_cosine) / 3.0
    )
    dynamic_composite = float(
        (
            motion_energy_ratio
            + flow_magnitude_cosine
            + flow_magnitude_ratio
            + flow_direction_cosine
            + flicker_similarity
        )
        / 5.0
    )

    return {
        "temporal_metric_1": energy_cosine,
        "temporal_metric_2": transition_cosine,
        "motion_metric": motion_profile_cosine,
        "motion_energy_mean_raw": motion_energy_mean,
        "final_motion_energy_mean_raw": final_motion_energy_mean,
        "motion_energy_ratio": motion_energy_ratio,
        "flow_magnitude_mean_raw": flow_magnitude_mean,
        "final_flow_magnitude_mean_raw": final_flow_magnitude_mean,
        "flow_magnitude_cosine": flow_magnitude_cosine,
        "flow_magnitude_ratio": flow_magnitude_ratio,
        "flow_direction_cosine": flow_direction_cosine,
        "flicker_similarity": flicker_similarity,
        "temporal_shape_composite": shape_composite,
        "temporal_dynamic_composite": dynamic_composite,
    }


def _spatial_metric(video: np.ndarray, final_video: np.ndarray) -> float:
    small = _video_tensor(video)
    final_small = _video_tensor(final_video)
    steps = min(small.shape[0], final_small.shape[0])
    sims = [_cosine_similarity(small[idx], final_small[idx]) for idx in range(steps)]
    return float(sum(sims) / max(len(sims), 1))


def _temporal_composite(row: dict[str, Any]) -> float:
    return float(
        (
            float(row["temporal_metric_1"])
            + float(row["temporal_metric_2"])
            + float(row["motion_metric"])
        )
        / 3.0
    )


def _convergence_step(rows: list[dict[str, Any]], key: str, threshold: float) -> int | None:
    for row in sorted(rows, key=lambda item: int(item["step"])):
        if float(row[key]) >= threshold:
            return int(row["step"])
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _sample_video_frames(video: np.ndarray, frame_count: int) -> list[np.ndarray]:
    if video.ndim != 4:
        raise ValueError(f"Expected [T,H,W,C] video array, got {video.shape}")
    total = video.shape[0]
    if total == 0:
        return []
    frame_count = max(1, min(frame_count, total))
    indices = np.linspace(0, total - 1, num=frame_count, dtype=int)
    return [video[idx] for idx in indices]


class SemanticMetricEvaluator:

    def __init__(self, model_name: str, device: str, frame_count: int) -> None:
        self.model_name = model_name
        self.device = device
        self.frame_count = frame_count
        self._model = None
        self._processor = None
        self._disabled_reason: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._disabled_reason is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:
            self._disabled_reason = f"transformers unavailable: {exc}"
            return
        try:
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
        except Exception as exc:
            self._disabled_reason = f"failed to load semantic model {self.model_name}: {exc}"
            self._model = None
            self._processor = None

    @property
    def disabled_reason(self) -> str | None:
        self._ensure_loaded()
        return self._disabled_reason

    @torch.inference_mode()
    def score_video(self, prompt: str, video: np.ndarray) -> float:
        self._ensure_loaded()
        if self._model is None or self._processor is None:
            return float("nan")

        from PIL import Image

        sampled_frames = _sample_video_frames(video, self.frame_count)
        if not sampled_frames:
            return float("nan")
        pil_frames = [Image.fromarray(frame.astype(np.uint8), mode="RGB") for frame in sampled_frames]

        text_inputs = self._processor(text=[prompt], return_tensors="pt", padding=True)
        image_inputs = self._processor(images=pil_frames, return_tensors="pt")
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
        image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}

        text_features = self._model.get_text_features(**text_inputs)
        image_features = self._model.get_image_features(**image_inputs)
        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        sims = image_features @ text_features.T
        return float(sims.mean().item())


def _judgment(convergence_rows: list[dict[str, Any]]) -> str:
    valid_rows = [
        row
        for row in convergence_rows
        if row["temporal_dynamic_95_step"] is not None and row["spatial_95_step"] is not None
    ]
    if not valid_rows:
        return "NO-GO"
    earlier = sum(int(row["temporal_dynamic_95_step"]) < int(row["spatial_95_step"]) for row in valid_rows)
    majority = earlier / len(valid_rows)
    median_gap = _median([float(row["dynamic_convergence_gap_fraction"]) for row in valid_rows])
    if majority >= 0.6 and median_gap >= 0.2:
        return "CONDITIONAL GO"
    if majority < 0.5 or median_gap < 0.05:
        return "NO-GO"
    return "CONDITIONAL GO"


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    trajectory_rows: list[dict[str, Any]],
    convergence_rows: list[dict[str, Any]],
) -> None:
    judgment = _judgment(convergence_rows)
    valid_rows = [
        row
        for row in convergence_rows
        if row["temporal_dynamic_95_step"] is not None and row["spatial_95_step"] is not None
    ]
    majority = (
        sum(int(row["temporal_dynamic_95_step"]) < int(row["spatial_95_step"]) for row in valid_rows) / len(valid_rows)
        if valid_rows
        else float("nan")
    )
    median_gap = _median([float(row["dynamic_convergence_gap_fraction"]) for row in valid_rows])
    shape_saturated = sum(int(row.get("temporal_shape_95_step", -1) == 0) for row in convergence_rows)
    dynamic_saturated = sum(int(row.get("temporal_dynamic_95_step", -1) == 0) for row in convergence_rows)
    semantic_available = any(not math.isnan(float(row.get("semantic_95_step", float("nan")))) for row in convergence_rows)

    lines = [
        "# Temporal Dimension Kill Test (Preflight)",
        "",
        "## Baseline",
        "",
        f"- Model: `{args.model}`",
        f"- Resolution: `{args.width}x{args.height}`",
        f"- Frames: `{args.num_frames}`",
        f"- Denoising steps: `{args.num_inference_steps}`",
        f"- Guidance scale: `{args.guidance_scale}`",
        f"- Flow shift: `{args.flow_shift}`",
        f"- Boundary ratio: `{args.boundary_ratio}`",
        f"- FPS: `{args.fps}`",
        "",
        "## CONFIRMED",
        "",
        "- The native Wan2.2 text-to-video path can emit measurement-only trajectory checkpoints without replacing the denoising loop.",
        "- Intermediate latents can be decoded through the normal VAE path and saved as deterministic artifacts.",
        "- Preflight convergence rows were generated from identical baseline trajectories, not separate reruns with changed prompts.",
        "",
        "## NEGATIVE RESULTS",
        "",
        "- This preflight does not yet test oracle temporal reduction; it only checks whether temporal-vs-spatial convergence separation is visible.",
        "- A temporal metric that saturates at step 0 should be treated as a calibration failure, not as evidence of early temporal convergence.",
        "",
        "## UNKNOWN",
        "",
        "- Whether the observed temporal metrics remain stable under a richer 24-30 prompt sweep.",
        "- Whether an oracle temporal-compute reduction yields meaningful DiT-time savings at near-baseline quality.",
        "- Whether a fixed policy captures most of the eventual oracle gain.",
        "",
        "## Preflight Summary",
        "",
        f"- Prompts analyzed: `{len(convergence_rows)}`",
        f"- Majority with temporal_dynamic_95 earlier than spatial_95: `{majority:.3f}`" if valid_rows else "- Majority with temporal_dynamic_95 earlier than spatial_95: `n/a`",
        f"- Median dynamic convergence gap fraction: `{median_gap:.3f}`" if valid_rows else "- Median dynamic convergence gap fraction: `n/a`",
        f"- Temporal-shape metric saturated at step 0 for `{shape_saturated}/{len(convergence_rows)}` prompts",
        f"- Temporal-dynamic metric saturated at step 0 for `{dynamic_saturated}/{len(convergence_rows)}` prompts",
        f"- Semantic convergence available: `{semantic_available}`",
        "",
        "## Judgment",
        "",
        judgment,
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    artifact_root = output_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    prompt_set = _read_prompt_set(Path(args.prompt_set), args.num_prompts)
    config_path = output_dir / "preflight_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2))

    omni = _build_omni(args)
    semantic_evaluator = None if args.disable_semantic_metric else SemanticMetricEvaluator(
        model_name=args.semantic_model,
        device=args.semantic_device,
        frame_count=args.semantic_frame_count,
    )
    trajectory_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []

    for prompt_idx, prompt_entry in enumerate(prompt_set):
        prompt_id = str(prompt_entry["prompt_id"])
        category = str(prompt_entry["motion_category"])
        prompt = str(prompt_entry["prompt"])
        request_seed = args.seed + prompt_idx
        request_label = f"{prompt_id}_seed{request_seed}"
        prompt_artifact_dir = artifact_root / request_label
        prompt_artifact_dir.mkdir(parents=True, exist_ok=True)

        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(request_seed)
        sampling = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            fps=args.fps,
            seed=request_seed,
        )
        sampling.extra_args = {
            "trajectory_probe": {
                "artifact_dir": str(prompt_artifact_dir),
                "request_label": request_label,
                "capture_progress": args.capture_progress,
                "fps": args.fps,
                "save_decoded": True,
                "save_latents": False,
                "save_mp4": True,
            },
            "flow_shift": args.flow_shift,
        }

        request = {"prompt": prompt}
        t0 = time.perf_counter()
        result = omni.generate(request, sampling)
        e2e_ms = (time.perf_counter() - t0) * 1000.0

        output = _unwrap_output(result)
        final_video = _normalize_frames(output)
        final_frames_path = prompt_artifact_dir / f"{request_label}_final_frames.pt"
        final_mp4_path = prompt_artifact_dir / f"{request_label}_final.mp4"
        torch.save(torch.from_numpy(final_video), final_frames_path)
        _save_video(final_mp4_path, final_video, fps=args.fps)
        final_semantic_metric_raw = (
            semantic_evaluator.score_video(prompt, final_video)
            if semantic_evaluator is not None
            else float("nan")
        )
        if semantic_evaluator is not None and semantic_evaluator.disabled_reason is not None:
            print(f"[preflight] semantic metric disabled: {semantic_evaluator.disabled_reason}")

        probe_meta_path = output.custom_output.get("trajectory_probe_metadata_path")
        if not probe_meta_path:
            raise ValueError(f"Missing trajectory probe metadata for {request_label}.")
        probe_meta = json.loads(Path(probe_meta_path).read_text())

        prompt_rows: list[dict[str, Any]] = []
        for record in probe_meta["records"]:
            frames_path = record.get("frames_path")
            if not frames_path:
                continue
            checkpoint_video = torch.load(frames_path, map_location="cpu").numpy()
            temporal_metrics = _temporal_metrics(checkpoint_video, final_video)
            spatial_metric = _spatial_metric(checkpoint_video, final_video)
            semantic_metric_raw = (
                semantic_evaluator.score_video(prompt, checkpoint_video)
                if semantic_evaluator is not None
                else float("nan")
            )
            row = {
                "prompt_id": prompt_id,
                "category": category,
                "step": int(record["step_index"]),
                "progress": float(record["progress"]),
                "step_latency_ms": float(record["step_latency_ms"]),
                "cumulative_dit_ms": float(record["cumulative_dit_ms"]),
                "latent_bytes": int(np.prod(record["latent_shape"]) * 4),
                **temporal_metrics,
                "spatial_metric": spatial_metric,
                "semantic_metric_raw": semantic_metric_raw,
                "final_semantic_metric_raw": final_semantic_metric_raw,
                "semantic_metric": _safe_relative(semantic_metric_raw, final_semantic_metric_raw)
                if not math.isnan(semantic_metric_raw) and not math.isnan(final_semantic_metric_raw)
                else float("nan"),
                "timestep": record["timestep"],
                "free_gpu_bytes": record["free_gpu_bytes"],
                "peak_reserved_bytes": record["peak_reserved_bytes"],
                "peak_allocated_bytes": record["peak_allocated_bytes"],
                "baseline_e2e_ms": e2e_ms,
                "peak_memory_mb": float(output.peak_memory_mb or 0.0),
                "frames_path": frames_path,
            }
            row["temporal_composite"] = _temporal_composite(row)
            trajectory_rows.append(row)
            prompt_rows.append(row)

        convergence_row = {
            "prompt_id": prompt_id,
            "category": category,
            "first_capture_step": int(prompt_rows[0]["step"]) if prompt_rows else None,
            "first_capture_timestep": float(prompt_rows[0]["timestep"]) if prompt_rows else None,
            "temporal_90_step": _convergence_step(prompt_rows, "temporal_composite", 0.90),
            "temporal_95_step": _convergence_step(prompt_rows, "temporal_composite", 0.95),
            "temporal_99_step": _convergence_step(prompt_rows, "temporal_composite", 0.99),
            "temporal_shape_90_step": _convergence_step(prompt_rows, "temporal_shape_composite", 0.90),
            "temporal_shape_95_step": _convergence_step(prompt_rows, "temporal_shape_composite", 0.95),
            "temporal_shape_99_step": _convergence_step(prompt_rows, "temporal_shape_composite", 0.99),
            "temporal_dynamic_90_step": _convergence_step(prompt_rows, "temporal_dynamic_composite", 0.90),
            "temporal_dynamic_95_step": _convergence_step(prompt_rows, "temporal_dynamic_composite", 0.95),
            "temporal_dynamic_99_step": _convergence_step(prompt_rows, "temporal_dynamic_composite", 0.99),
            "spatial_90_step": _convergence_step(prompt_rows, "spatial_metric", 0.90),
            "spatial_95_step": _convergence_step(prompt_rows, "spatial_metric", 0.95),
            "spatial_99_step": _convergence_step(prompt_rows, "spatial_metric", 0.99),
            "semantic_90_step": _convergence_step(prompt_rows, "semantic_metric", 0.90)
            if any(not math.isnan(float(row["semantic_metric"])) for row in prompt_rows)
            else float("nan"),
            "semantic_95_step": _convergence_step(prompt_rows, "semantic_metric", 0.95)
            if any(not math.isnan(float(row["semantic_metric"])) for row in prompt_rows)
            else float("nan"),
            "semantic_99_step": _convergence_step(prompt_rows, "semantic_metric", 0.99)
            if any(not math.isnan(float(row["semantic_metric"])) for row in prompt_rows)
            else float("nan"),
        }
        if convergence_row["temporal_95_step"] is not None and convergence_row["spatial_95_step"] is not None:
            gap_steps = int(convergence_row["spatial_95_step"]) - int(convergence_row["temporal_95_step"])
            convergence_row["convergence_gap_steps"] = gap_steps
            convergence_row["convergence_gap_fraction"] = gap_steps / max(args.num_inference_steps, 1)
        else:
            convergence_row["convergence_gap_steps"] = None
            convergence_row["convergence_gap_fraction"] = float("nan")
        if convergence_row["temporal_dynamic_95_step"] is not None and convergence_row["spatial_95_step"] is not None:
            gap_steps = int(convergence_row["spatial_95_step"]) - int(convergence_row["temporal_dynamic_95_step"])
            convergence_row["dynamic_convergence_gap_steps"] = gap_steps
            convergence_row["dynamic_convergence_gap_fraction"] = gap_steps / max(args.num_inference_steps, 1)
        else:
            convergence_row["dynamic_convergence_gap_steps"] = None
            convergence_row["dynamic_convergence_gap_fraction"] = float("nan")
        convergence_rows.append(convergence_row)

        print(
            f"[preflight] prompt_id={prompt_id} category={category} "
            f"e2e_ms={e2e_ms:.1f} checkpoints={len(prompt_rows)}"
        )

    trajectory_fields = [
        "prompt_id",
        "category",
        "step",
        "progress",
        "step_latency_ms",
        "cumulative_dit_ms",
        "latent_bytes",
        "temporal_metric_1",
        "temporal_metric_2",
        "motion_metric",
        "motion_energy_mean_raw",
        "final_motion_energy_mean_raw",
        "motion_energy_ratio",
        "flow_magnitude_mean_raw",
        "final_flow_magnitude_mean_raw",
        "flow_magnitude_cosine",
        "flow_magnitude_ratio",
        "flow_direction_cosine",
        "flicker_similarity",
        "temporal_shape_composite",
        "temporal_dynamic_composite",
        "spatial_metric",
        "semantic_metric_raw",
        "final_semantic_metric_raw",
        "semantic_metric",
        "temporal_composite",
        "timestep",
        "free_gpu_bytes",
        "peak_reserved_bytes",
        "peak_allocated_bytes",
        "baseline_e2e_ms",
        "peak_memory_mb",
        "frames_path",
    ]
    convergence_fields = [
        "prompt_id",
        "category",
        "first_capture_step",
        "first_capture_timestep",
        "temporal_90_step",
        "temporal_95_step",
        "temporal_99_step",
        "temporal_shape_90_step",
        "temporal_shape_95_step",
        "temporal_shape_99_step",
        "temporal_dynamic_90_step",
        "temporal_dynamic_95_step",
        "temporal_dynamic_99_step",
        "spatial_90_step",
        "spatial_95_step",
        "spatial_99_step",
        "semantic_90_step",
        "semantic_95_step",
        "semantic_99_step",
        "convergence_gap_steps",
        "convergence_gap_fraction",
        "dynamic_convergence_gap_steps",
        "dynamic_convergence_gap_fraction",
    ]

    _write_csv(output_dir / "trajectory_metrics.csv", trajectory_rows, trajectory_fields)
    _write_csv(output_dir / "convergence_summary.csv", convergence_rows, convergence_fields)
    _write_report(
        output_dir / "temporal_dimension_killtest.md",
        args=args,
        trajectory_rows=trajectory_rows,
        convergence_rows=convergence_rows,
    )


if __name__ == "__main__":
    main()
