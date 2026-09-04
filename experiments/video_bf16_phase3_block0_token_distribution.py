"""Phase-3 descriptive follow-up: token/frame distribution of CLEAN-vs-PLUS1 differences.

Read-only, CPU-only decomposition of already-persisted Phase-3 block-boundary artifacts.
It answers one descriptive question about the existing run: after block 0, are the
differing hidden-state elements confined to the single token that differs before block 0
(token 2781, the patch containing latent coordinate 516515), or spread across tokens/frames?

This is NOT a primary result and defines no new hypothesis, gate, or decision. It binds
every tensor it reads to the Phase-3 manifest record (file sha256 + runtime identity) and
records the Phase-3 provenance hash so the output cannot be detached from the run it describes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("results/video_bf16_first_divergence_localization_phase3")
OUT = ROOT / "phase3" / "descriptive_followup" / "block0_token_distribution.json"
BOUNDARIES = ("pre_block_hidden_state", "after_block_000", "after_block_001", "after_block_002")
BRANCHES = ("positive", "negative")
PAIR = ("CLEAN", "PLUS1")
TOKENS_PER_FRAME = 30 * 52  # patch grid (T=9, H=60/2, W=104/2) -> 9 x 30 x 52 = 14040 tokens
FRAMES = 9
REFERENCE_TOKEN = 2781  # patch token containing latent coordinate 516515 (channel 9, f=1, h=46, w=51)


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_identity(bits: np.ndarray, shape: list[int]) -> str:
    header = canonical_json({"version": "phase3-runtime-tensor-identity-v1", "runtime_dtype": "torch.bfloat16", "shape": shape})
    return hashlib.sha256(header + b"\0" + np.ascontiguousarray(bits, dtype=np.uint16).tobytes()).hexdigest()


def load_bound(record: dict) -> tuple[np.ndarray, dict]:
    artifact = record["artifact"]
    path = ROOT / artifact["relative_path"]
    bits = np.load(path, allow_pickle=False)
    shape = [int(x) for x in artifact["shape"]]
    if bits.dtype != np.uint16 or bits.size != int(np.prod(shape)):
        raise SystemExit(f"STOP: artifact encoding/shape mismatch for {path}")
    if sha256_file(path) != artifact["file_sha256"] or runtime_identity(bits, shape) != artifact["runtime_canonical_identity"]:
        raise SystemExit(f"STOP: artifact does not match its Phase-3 manifest record: {path}")
    return bits.reshape(shape), {"relative_path": artifact["relative_path"], "file_sha256": artifact["file_sha256"], "runtime_canonical_identity": artifact["runtime_canonical_identity"]}


def token_position(token: int) -> dict:
    return {"token": int(token), "frame": int(token // TOKENS_PER_FRAME), "h": int((token % TOKENS_PER_FRAME) // 52), "w": int(token % 52)}


def main() -> None:
    manifest = json.loads((ROOT / "phase3" / "phase3_manifest.json").read_text())
    provenance = json.loads((ROOT / "provenance.json").read_text())
    if manifest["provenance_hash"] != provenance["provenance_hash"]:
        raise SystemExit("STOP: Phase-3 manifest is not bound to the result-root provenance")
    records = {
        (name, branch, row["boundary"]): row
        for name, trajectory in manifest["trajectories"].items()
        for branch, data in trajectory["branches"].items()
        for row in data["records"]
    }
    reference = token_position(REFERENCE_TOKEN)
    results = {}
    sources = {}
    for boundary in BOUNDARIES:
        for branch in BRANCHES:
            lhs, lhs_src = load_bound(records[(PAIR[0], branch, boundary)])
            rhs, rhs_src = load_bound(records[(PAIR[1], branch, boundary)])
            sources[f"{boundary}/{branch}"] = {PAIR[0]: lhs_src, PAIR[1]: rhs_src}
            if lhs.shape != rhs.shape or lhs.shape[0] != 1 or lhs.shape[1] != FRAMES * TOKENS_PER_FRAME:
                raise SystemExit(f"STOP: unexpected hidden-state shape at {boundary}/{branch}: {lhs.shape}")
            differing = lhs[0] != rhs[0]  # [tokens, features]
            per_token = differing.sum(axis=1)
            per_feature = differing.sum(axis=0)
            affected = np.nonzero(per_token)[0]
            total = int(differing.sum())
            top = sorted(affected.tolist(), key=lambda t: (-int(per_token[t]), t))[:10]
            results[f"{boundary}/{branch}"] = {
                "pair": f"{PAIR[0]}_VS_{PAIR[1]}",
                "total_differing_elements": total,
                "hidden_elements_total": int(differing.size),
                "tokens_with_any_difference": int(affected.size),
                "tokens_total": int(per_token.size),
                "frames_with_any_difference": int((np.bincount(affected // TOKENS_PER_FRAME, minlength=FRAMES) > 0).sum()),
                "affected_tokens_per_frame": np.bincount(affected // TOKENS_PER_FRAME, minlength=FRAMES).tolist(),
                "features_with_any_difference": int((per_feature > 0).sum()),
                "features_total": int(per_feature.size),
                "reference_token": reference,
                "reference_token_differing_features": int(per_token[REFERENCE_TOKEN]),
                "reference_token_share_of_differences": (int(per_token[REFERENCE_TOKEN]) / total) if total else None,
                "affected_tokens_sharing_reference_h_w": int(sum(1 for t in affected.tolist() if token_position(t)["h"] == reference["h"] and token_position(t)["w"] == reference["w"])),
                "per_affected_token_difference_count_min_median_max": (
                    [int(per_token[affected].min()), int(np.median(per_token[affected])), int(per_token[affected].max())] if affected.size else None
                ),
                "top_tokens_by_difference_count": [dict(token_position(t), differing_features=int(per_token[t])) for t in top],
            }
    # branch overlap of affected tokens after block 0
    overlap = {}
    affected_sets = {}
    for branch in BRANCHES:
        lhs, _ = load_bound(records[(PAIR[0], branch, "after_block_000")])
        rhs, _ = load_bound(records[(PAIR[1], branch, "after_block_000")])
        affected_sets[branch] = set(np.nonzero((lhs[0] != rhs[0]).sum(axis=1))[0].tolist())
    overlap = {
        "boundary": "after_block_000",
        "positive_and_negative": len(affected_sets["positive"] & affected_sets["negative"]),
        "positive_only": len(affected_sets["positive"] - affected_sets["negative"]),
        "negative_only": len(affected_sets["negative"] - affected_sets["positive"]),
    }
    document = {
        "kind": "phase3_descriptive_followup",
        "descriptive_only": True,
        "primary_result": False,
        "question": "After block 0, are CLEAN-vs-PLUS1 hidden-state differences confined to the single pre-block differing token, or spread across tokens and frames?",
        "note": "PLUS1 and HISTORICAL_PLUS14 are bit-exact at every boundary listed here (Phase-3 P3-G22 and pairwise rows), so CLEAN_VS_HISTORICAL_PLUS14 would reproduce these numbers exactly.",
        "phase3_provenance_hash": provenance["provenance_hash"],
        "phase3_git_commit": provenance["git_commit"],
        "phase3_anchor_manifest_sha256": manifest["manifest_sha256"],
        "token_geometry": {"frames": FRAMES, "tokens_per_frame": TOKENS_PER_FRAME, "grid_h": 30, "grid_w": 52, "patch_size": [1, 2, 2]},
        "sources": sources,
        "results": results,
        "after_block_000_affected_token_overlap_between_branches": overlap,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=1, sort_keys=True))
    for key, row in results.items():
        print(f"{key:36s} total={row['total_differing_elements']:>9d} tokens={row['tokens_with_any_difference']:>6d}/{row['tokens_total']} frames={row['frames_with_any_difference']}/9 ref_token_share={row['reference_token_share_of_differences']}")
    print("overlap:", overlap)
    print("written:", OUT)


if __name__ == "__main__":
    main()
