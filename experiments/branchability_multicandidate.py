#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp2_memory_pressure import (
    _artifact_dir,
    _attach_engine_owner,
    _extract_first_image,
    _load_prompts,
    _run_full_baseline,
    _save_image,
)
from tools.diffusion_state_recovery_smoke import (
    _build_prompt,
    _build_sampling_params,
    _get_inline_diffusion_engine,
    _initialize_omni_with_retry,
    _make_request,
    _preflight_real_model_args,
    _wait_for_checkpoint,
)
from vllm_omni.diffusion.state import Fidelity

DEFAULT_PROMPTS = [
    "A brass astrolabe on a wooden desk",
    "A neon-lit alley in the rain at midnight",
    "An observatory on a snowy mountain at sunrise",
]

CSV_FIELDS = [
    "model",
    "prompt_id",
    "prompt",
    "run_idx",
    "candidate_count",
    "branch_step",
    "total_steps",
    "mode",
    "branch_noise_scale",
    "independent_total_wall_sec",
    "condition_total_wall_sec",
    "actual_wall_saved_sec",
    "actual_wall_saved_frac",
    "estimated_step_saved",
    "estimated_step_saved_frac",
    "mean_pairwise_ssim_distance",
    "mean_pairwise_mse",
    "mean_pairwise_lpips",
    "mean_pairwise_clip_distance",
    "mean_clip_prompt_score",
    "min_clip_prompt_score",
    "diversity_retention_vs_independent",
    "clip_diversity_retention_vs_independent",
    "lpips_retention_vs_independent",
]


@dataclass
class CandidateResult:
    seed: int
    image: np.ndarray
    wall_sec: float
    image_path: Path


class OptionalMetricSuite:
    def __init__(
        self,
        *,
        enable_clip: bool,
        enable_lpips: bool,
        clip_model_name: str,
        metric_device: str,
    ) -> None:
        self.enable_clip = enable_clip
        self.enable_lpips = enable_lpips
        self.clip_model_name = clip_model_name
        self.metric_device = metric_device
        self._clip = None
        self._lpips = None
        self.warnings: list[str] = []

    def _resolve_device(self) -> str:
        if self.metric_device != "auto":
            return self.metric_device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_clip(self) -> bool:
        if not self.enable_clip:
            return False
        if self._clip is not None:
            return True
        try:
            from transformers import CLIPModel, CLIPProcessor

            device = self._resolve_device()
            processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            model = CLIPModel.from_pretrained(self.clip_model_name).to(device)
            model.eval()
            self._clip = (processor, model, device)
            return True
        except Exception as exc:
            self.warnings.append(f"CLIP metrics disabled: {exc}")
            self.enable_clip = False
            return False

    def _ensure_lpips(self) -> bool:
        if not self.enable_lpips:
            return False
        if self._lpips is not None:
            return True
        try:
            import lpips

            device = self._resolve_device()
            model = lpips.LPIPS(net="alex").to(device)
            model.eval()
            self._lpips = (model, device)
            return True
        except Exception as exc:
            self.warnings.append(f"LPIPS metrics disabled: {exc}")
            self.enable_lpips = False
            return False

    def compute_clip_metrics(self, prompt: str, images: list[np.ndarray]) -> dict[str, float | None]:
        if not self._ensure_clip():
            return {
                "mean_pairwise_clip_distance": None,
                "mean_clip_prompt_score": None,
                "min_clip_prompt_score": None,
            }

        from PIL import Image

        processor, model, device = self._clip
        pil_images = [Image.fromarray(image.astype(np.uint8), mode="RGB") for image in images]
        with torch.inference_mode():
            image_inputs = processor(images=pil_images, return_tensors="pt").to(device)
            image_features = model.get_image_features(**image_inputs)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)

            text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
            text_features = model.get_text_features(**text_inputs)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

        clip_distances: list[float] = []
        for idx in range(len(images)):
            for jdx in range(idx + 1, len(images)):
                sim = torch.sum(image_features[idx] * image_features[jdx]).item()
                clip_distances.append(1.0 - sim)

        prompt_scores = torch.matmul(image_features, text_features[0]).detach().cpu().tolist()
        return {
            "mean_pairwise_clip_distance": mean(clip_distances) if clip_distances else 0.0,
            "mean_clip_prompt_score": mean(prompt_scores) if prompt_scores else None,
            "min_clip_prompt_score": min(prompt_scores) if prompt_scores else None,
        }

    def compute_lpips(self, images: list[np.ndarray]) -> float | None:
        if not self._ensure_lpips():
            return None

        model, device = self._lpips
        tensors = []
        for image in images:
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
            tensor = tensor * 2.0 - 1.0
            tensors.append(tensor.to(device))

        distances: list[float] = []
        with torch.inference_mode():
            for idx in range(len(tensors)):
                for jdx in range(idx + 1, len(tensors)):
                    distance = model(tensors[idx], tensors[jdx]).mean().item()
                    distances.append(float(distance))
        return mean(distances) if distances else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Branch-and-share multicandidate exploration for diffusion serving."
    )
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo", help="Diffusion model name or local path.")
    parser.add_argument("--output", type=Path, required=True, help="CSV output path.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional directory for baseline/branch output artifacts.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Optional newline-delimited prompt file. Defaults to a short built-in set.",
    )
    parser.add_argument(
        "--candidate-counts",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Number of candidates to compare per request.",
    )
    parser.add_argument(
        "--branch-steps",
        type=int,
        nargs="+",
        default=[0, 5, 10, 20, 30, 40],
        help="Shared-prefix branch points. 0 means fully independent generation.",
    )
    parser.add_argument("--runs-per-condition", type=int, default=1, help="Number of runs per prompt/condition.")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Total denoising steps.")
    parser.add_argument(
        "--branch-noise-scale",
        type=float,
        default=0.03,
        help="Gaussian noise scale applied at the branch point, relative to latent std.",
    )
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG guidance scale.")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="Qwen true CFG scale.")
    parser.add_argument("--stage-configs-path", type=str, default=None, help="Optional Omni stage config YAML.")
    parser.add_argument("--gpu-budget-bytes", type=int, default=0, help="State manager GPU tier budget.")
    parser.add_argument(
        "--cpu-budget-bytes",
        type=int,
        default=1 << 30,
        help="State manager CPU tier budget. Defaults high to avoid placement confounds.",
    )
    parser.add_argument("--theta-h", type=float, default=0.7, help="LOSSLESS threshold.")
    parser.add_argument("--theta-w", type=float, default=0.3, help="COMPRESSED threshold.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--init-timeout", type=int, default=600, help="Omni init timeout in seconds.")
    parser.add_argument("--stage-init-timeout", type=int, default=600, help="Per-stage init timeout in seconds.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile for easier debugging.")
    parser.add_argument("--enable-cpu-offload", action="store_true", help="Enable CPU offload from the first attempt.")
    parser.add_argument(
        "--enable-layerwise-offload",
        action="store_true",
        help="Enable layerwise offload from the first attempt.",
    )
    parser.add_argument(
        "--retry-with-offload",
        action="store_true",
        default=True,
        help="Retry model initialization with CPU/layerwise offload after CUDA OOM.",
    )
    parser.add_argument(
        "--no-retry-with-offload",
        dest="retry_with_offload",
        action="store_false",
        help="Disable the automatic OOM retry path.",
    )
    parser.add_argument(
        "--disable-auto-offload",
        action="store_true",
        help="Disable the conservative single-GPU auto-offload heuristic.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help="Polling interval in seconds while waiting for the branch checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds while waiting for the branch checkpoint.",
    )
    parser.add_argument(
        "--disable-clip-metrics",
        action="store_true",
        help="Disable CLIP-based diversity and prompt-alignment metrics.",
    )
    parser.add_argument(
        "--disable-lpips-metrics",
        action="store_true",
        help="Disable LPIPS-based diversity metrics.",
    )
    parser.add_argument(
        "--clip-model-name",
        default="openai/clip-vit-base-patch32",
        help="Transformers CLIP model used for optional diversity metrics.",
    )
    parser.add_argument(
        "--metric-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for optional CLIP/LPIPS metrics.",
    )
    parser.set_defaults(backend="real-model", disk_path=None, strict_equality=False, step_delay=0.02)
    return parser.parse_args()


def _write_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(CSV_FIELDS) + "\n", encoding="utf-8")


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    import csv

    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def _ensure_branch_steps(args: argparse.Namespace) -> list[int]:
    branch_steps: list[int] = []
    for step in args.branch_steps:
        if step < 0 or step >= args.num_inference_steps:
            raise ValueError(
                f"Branch step must satisfy 0 <= step < num_inference_steps; got {step} with total={args.num_inference_steps}."
            )
        branch_steps.append(int(step))
    return sorted(dict.fromkeys(branch_steps))


def _estimate_saved_steps(candidate_count: int, branch_step: int, total_steps: int) -> tuple[int, float]:
    independent_total = candidate_count * total_steps
    shared_total = candidate_count * total_steps if branch_step == 0 else branch_step + candidate_count * (total_steps - branch_step)
    saved_steps = independent_total - shared_total
    return saved_steps, saved_steps / max(independent_total, 1)


def _candidate_seed(args: argparse.Namespace, prompt_id: int, run_idx: int, candidate_idx: int) -> int:
    return args.seed + prompt_id * 10_000 + run_idx * 1_000 + candidate_idx


def _prompt_and_sampling(engine: Any, args: argparse.Namespace, *, prompt: str, seed: int) -> tuple[Any, Any]:
    prompt_args = argparse.Namespace(**vars(args))
    prompt_args.prompt = prompt
    prompt_args.seed = seed
    return _build_prompt(engine.omni, prompt_args), _build_sampling_params(prompt_args)


def _pairwise_image_metrics(images: list[np.ndarray]) -> dict[str, float]:
    if len(images) < 2:
        return {
            "mean_pairwise_ssim_distance": 0.0,
            "mean_pairwise_mse": 0.0,
        }

    try:
        from skimage.metrics import structural_similarity as structural_similarity
    except Exception:
        structural_similarity = None

    ssim_distances: list[float] = []
    mses: list[float] = []
    for idx in range(len(images)):
        for jdx in range(idx + 1, len(images)):
            lhs = images[idx].astype(np.float32)
            rhs = images[jdx].astype(np.float32)
            diff = lhs - rhs
            mses.append(float(np.mean(diff * diff)))
            if structural_similarity is not None:
                score = structural_similarity(
                    images[idx],
                    images[jdx],
                    channel_axis=-1,
                    data_range=255,
                )
                ssim_distances.append(float(1.0 - score))

    return {
        "mean_pairwise_ssim_distance": mean(ssim_distances) if ssim_distances else 0.0,
        "mean_pairwise_mse": mean(mses) if mses else 0.0,
    }


async def _run_one_request(engine: Any, request: Any) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    outputs = await engine.step(request)
    elapsed = time.perf_counter() - start
    return _extract_first_image(outputs), elapsed


async def _run_independent_candidates(
    engine: Any,
    args: argparse.Namespace,
    *,
    prompt_id: int,
    prompt: str,
    run_idx: int,
    candidate_count: int,
    artifact_dir: Path,
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for candidate_idx in range(candidate_count):
        seed = _candidate_seed(args, prompt_id, run_idx, candidate_idx)
        prompt_obj, sampling_params = _prompt_and_sampling(engine, args, prompt=prompt, seed=seed)
        request_id = f"independent-p{prompt_id}-r{run_idx}-c{candidate_idx}"
        image, wall_sec = await _run_one_request(engine, _make_request(request_id, prompt_obj, sampling_params))
        image_path = (
            artifact_dir
            / "independent"
            / f"prompt_{prompt_id:02d}"
            / f"run_{run_idx:02d}"
            / f"candidates_{candidate_count:02d}"
            / f"candidate_{candidate_idx:02d}.png"
        )
        _save_image(image, image_path)
        results.append(CandidateResult(seed=seed, image=image, wall_sec=wall_sec, image_path=image_path))
    return results


def _branch_request_from_state(
    engine: Any,
    *,
    root_request_id: str,
    branch_request_id: str,
    prompt_obj: Any,
    sampling_params: Any,
    branch_seed: int,
    branch_noise_scale: float,
) -> Any:
    restored_request = engine.restore_request_from_state(
        _make_request(root_request_id, prompt_obj, sampling_params),
        request_id=root_request_id,
    )
    restored_request.request_id = branch_request_id
    restored_request.sampling_params.seed = branch_seed

    latents = restored_request.sampling_params.latents
    if latents is None:
        raise RuntimeError(f"Missing restored latents for request {root_request_id}.")

    generator = torch.Generator(device=latents.device).manual_seed(branch_seed)
    noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    latent_std = latents.detach().to(torch.float32).std().item()
    if not math.isfinite(latent_std) or latent_std == 0:
        latent_std = 1.0
    restored_request.sampling_params.latents = latents + noise * (latent_std * branch_noise_scale)
    return restored_request


async def _run_shared_prefix_candidates(
    engine: Any,
    args: argparse.Namespace,
    *,
    prompt_id: int,
    prompt: str,
    run_idx: int,
    candidate_count: int,
    branch_step: int,
    artifact_dir: Path,
) -> tuple[list[CandidateResult], float]:
    if branch_step == 0:
        raise ValueError("branch_step=0 should be handled by the independent baseline path.")

    root_seed = _candidate_seed(args, prompt_id, run_idx, 0)
    prompt_obj, sampling_params = _prompt_and_sampling(engine, args, prompt=prompt, seed=root_seed)
    root_request_id = f"shared-root-p{prompt_id}-r{run_idx}-b{branch_step}"
    start_total = time.perf_counter()
    initial_task = asyncio.create_task(
        engine.async_add_req_and_wait_for_response(_make_request(root_request_id, prompt_obj, sampling_params))
    )
    checkpoint_state = await _wait_for_checkpoint(
        engine,
        request_id=root_request_id,
        target_step=branch_step,
        poll_interval=args.poll_interval,
        timeout_s=args.checkpoint_timeout,
    )
    engine.abort(root_request_id)
    initial_output = await initial_task
    if not initial_output.aborted:
        raise RuntimeError(f"Expected shared-prefix root request {root_request_id} to abort at branch step.")

    results: list[CandidateResult] = []
    for candidate_idx in range(candidate_count):
        branch_seed = _candidate_seed(args, prompt_id, run_idx, candidate_idx)
        branch_request_id = f"shared-branch-p{prompt_id}-r{run_idx}-b{branch_step}-c{candidate_idx}"
        branched_request = _branch_request_from_state(
            engine,
            root_request_id=root_request_id,
            branch_request_id=branch_request_id,
            prompt_obj=prompt_obj,
            sampling_params=sampling_params,
            branch_seed=branch_seed,
            branch_noise_scale=args.branch_noise_scale,
        )
        image, wall_sec = await _run_one_request(engine, branched_request)
        image_path = (
            artifact_dir
            / "shared_prefix"
            / f"prompt_{prompt_id:02d}"
            / f"run_{run_idx:02d}"
            / f"candidates_{candidate_count:02d}"
            / f"branch_{branch_step:02d}"
            / f"candidate_{candidate_idx:02d}.png"
        )
        _save_image(image, image_path)
        results.append(CandidateResult(seed=branch_seed, image=image, wall_sec=wall_sec, image_path=image_path))
    end_total = time.perf_counter()

    state_manager = getattr(engine, "state_manager", None)
    if state_manager is not None:
        state_manager.release_request(root_request_id)

    return results, end_total - start_total


def _condition_metrics(
    prompt: str,
    candidates: list[CandidateResult],
    metric_suite: OptionalMetricSuite,
) -> dict[str, float | None]:
    images = [candidate.image for candidate in candidates]
    metrics = _pairwise_image_metrics(images)
    metrics["mean_pairwise_lpips"] = metric_suite.compute_lpips(images)
    metrics.update(metric_suite.compute_clip_metrics(prompt, images))
    return metrics


async def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    prompts = _load_prompts(args) if args.prompts_file is not None else list(DEFAULT_PROMPTS)
    _preflight_real_model_args(args)
    branch_steps = _ensure_branch_steps(args)

    output_path = args.output
    artifact_dir = _artifact_dir(args)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.disk_path is None:
        args.disk_path = artifact_dir / "checkpoints"
    Path(args.disk_path).mkdir(parents=True, exist_ok=True)
    _write_csv_header(output_path)

    metric_suite = OptionalMetricSuite(
        enable_clip=not args.disable_clip_metrics,
        enable_lpips=not args.disable_lpips_metrics,
        clip_model_name=args.clip_model_name,
        metric_device=args.metric_device,
    )

    omni, init_config = _initialize_omni_with_retry(args)
    rows: list[dict[str, Any]] = []
    try:
        engine = _get_inline_diffusion_engine(omni)
        _attach_engine_owner(omni, engine)
        state_manager = getattr(engine, "state_manager", None)
        if state_manager is None:
            raise RuntimeError("Diffusion state manager is disabled for this engine.")
        original_assign = state_manager.fid_policy.assign
        state_manager.fid_policy.assign = lambda _value_score: Fidelity.LOSSLESS
        try:
            for prompt_id, prompt in enumerate(prompts):
                for run_idx in range(args.runs_per_condition):
                    for candidate_count in args.candidate_counts:
                        independent_candidates = await _run_independent_candidates(
                            engine,
                            args,
                            prompt_id=prompt_id,
                            prompt=prompt,
                            run_idx=run_idx,
                            candidate_count=candidate_count,
                            artifact_dir=artifact_dir,
                        )
                        independent_total_wall = sum(candidate.wall_sec for candidate in independent_candidates)
                        independent_metrics = _condition_metrics(prompt, independent_candidates, metric_suite)

                        independent_row = {
                            "model": args.model,
                            "prompt_id": prompt_id,
                            "prompt": prompt,
                            "run_idx": run_idx,
                            "candidate_count": candidate_count,
                            "branch_step": 0,
                            "total_steps": args.num_inference_steps,
                            "mode": "independent",
                            "branch_noise_scale": 0.0,
                            "independent_total_wall_sec": independent_total_wall,
                            "condition_total_wall_sec": independent_total_wall,
                            "actual_wall_saved_sec": 0.0,
                            "actual_wall_saved_frac": 0.0,
                            "estimated_step_saved": 0,
                            "estimated_step_saved_frac": 0.0,
                            **independent_metrics,
                            "diversity_retention_vs_independent": 1.0,
                            "clip_diversity_retention_vs_independent": 1.0
                            if independent_metrics["mean_pairwise_clip_distance"] is not None
                            else None,
                            "lpips_retention_vs_independent": 1.0
                            if independent_metrics["mean_pairwise_lpips"] is not None
                            else None,
                        }
                        rows.append(independent_row)
                        _append_csv_row(output_path, independent_row)
                        print(
                            f"[independent] prompt_id={prompt_id} run_idx={run_idx} "
                            f"candidates={candidate_count} wall_sec={independent_total_wall:.3f} "
                            f"pairwise_ssim_dist={independent_metrics['mean_pairwise_ssim_distance']:.6f}",
                            flush=True,
                        )

                        for branch_step in branch_steps:
                            if branch_step == 0:
                                continue
                            shared_candidates, shared_total_wall = await _run_shared_prefix_candidates(
                                engine,
                                args,
                                prompt_id=prompt_id,
                                prompt=prompt,
                                run_idx=run_idx,
                                candidate_count=candidate_count,
                                branch_step=branch_step,
                                artifact_dir=artifact_dir,
                            )
                            shared_metrics = _condition_metrics(prompt, shared_candidates, metric_suite)
                            estimated_step_saved, estimated_step_saved_frac = _estimate_saved_steps(
                                candidate_count,
                                branch_step,
                                args.num_inference_steps,
                            )

                            baseline_ssim_dist = independent_metrics["mean_pairwise_ssim_distance"] or 0.0
                            baseline_clip_dist = independent_metrics["mean_pairwise_clip_distance"]
                            baseline_lpips = independent_metrics["mean_pairwise_lpips"]
                            shared_ssim_dist = shared_metrics["mean_pairwise_ssim_distance"] or 0.0
                            shared_clip_dist = shared_metrics["mean_pairwise_clip_distance"]
                            shared_lpips = shared_metrics["mean_pairwise_lpips"]

                            row = {
                                "model": args.model,
                                "prompt_id": prompt_id,
                                "prompt": prompt,
                                "run_idx": run_idx,
                                "candidate_count": candidate_count,
                                "branch_step": branch_step,
                                "total_steps": args.num_inference_steps,
                                "mode": "shared_prefix",
                                "branch_noise_scale": args.branch_noise_scale,
                                "independent_total_wall_sec": independent_total_wall,
                                "condition_total_wall_sec": shared_total_wall,
                                "actual_wall_saved_sec": independent_total_wall - shared_total_wall,
                                "actual_wall_saved_frac": (independent_total_wall - shared_total_wall)
                                / max(independent_total_wall, 1e-8),
                                "estimated_step_saved": estimated_step_saved,
                                "estimated_step_saved_frac": estimated_step_saved_frac,
                                **shared_metrics,
                                "diversity_retention_vs_independent": (
                                    shared_ssim_dist / baseline_ssim_dist if baseline_ssim_dist > 0 else None
                                ),
                                "clip_diversity_retention_vs_independent": (
                                    shared_clip_dist / baseline_clip_dist
                                    if baseline_clip_dist not in (None, 0.0) and shared_clip_dist is not None
                                    else None
                                ),
                                "lpips_retention_vs_independent": (
                                    shared_lpips / baseline_lpips
                                    if baseline_lpips not in (None, 0.0) and shared_lpips is not None
                                    else None
                                ),
                            }
                            rows.append(row)
                            _append_csv_row(output_path, row)
                            print(
                                f"[shared_prefix] prompt_id={prompt_id} run_idx={run_idx} "
                                f"candidates={candidate_count} branch_step={branch_step} "
                                f"wall_saved_frac={row['actual_wall_saved_frac']:.4f} "
                                f"ssim_retention={row['diversity_retention_vs_independent']}",
                                flush=True,
                            )
        finally:
            state_manager.fid_policy.assign = original_assign
    finally:
        omni.close()

    aggregates: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            f"mode={row['mode']}/candidates={row['candidate_count']}/branch_step={row['branch_step']}"
        )
        aggregates.setdefault(key, []).append(row)

    aggregate_rows = []
    for key, group in sorted(aggregates.items()):
        def maybe_mean(name: str) -> float | None:
            values = [row[name] for row in group if row[name] is not None]
            return mean(values) if values else None

        aggregate_rows.append(
            {
                "group": key,
                "num_rows": len(group),
                "mean_condition_total_wall_sec": maybe_mean("condition_total_wall_sec"),
                "mean_actual_wall_saved_frac": maybe_mean("actual_wall_saved_frac"),
                "mean_estimated_step_saved_frac": maybe_mean("estimated_step_saved_frac"),
                "mean_pairwise_ssim_distance": maybe_mean("mean_pairwise_ssim_distance"),
                "mean_pairwise_lpips": maybe_mean("mean_pairwise_lpips"),
                "mean_pairwise_clip_distance": maybe_mean("mean_pairwise_clip_distance"),
                "mean_diversity_retention_vs_independent": maybe_mean("diversity_retention_vs_independent"),
                "mean_lpips_retention_vs_independent": maybe_mean("lpips_retention_vs_independent"),
                "mean_clip_diversity_retention_vs_independent": maybe_mean("clip_diversity_retention_vs_independent"),
            }
        )

    summary = {
        "model": args.model,
        "num_inference_steps": args.num_inference_steps,
        "candidate_counts": args.candidate_counts,
        "branch_steps": branch_steps,
        "branch_noise_scale": args.branch_noise_scale,
        "runs_per_condition": args.runs_per_condition,
        "init_config": init_config,
        "csv_path": str(output_path),
        "artifact_dir": str(artifact_dir),
        "metric_warnings": metric_suite.warnings,
        "aggregates": aggregate_rows,
        "num_rows": len(rows),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run_experiment(args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
