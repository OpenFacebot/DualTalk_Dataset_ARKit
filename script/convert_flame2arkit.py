#!/usr/bin/env python3
"""Convert DualTalk FLAME samples to ARKit motion, driven by metadata jsonl.

Responsibilities of this script (IO / dimension alignment only; all math
lives in flame2arkit.py):

1. Load the ARKit51-to-FLAME103 forward matrix (SHA256 verified).
2. Read a metadata jsonl produced by gen_metadata.py; each record carries the
   relative paths of the source FLAME npz files (speaker1_flame/speaker2_flame).
3. Load each FLAME npz; optionally smooth the temporal sequences (controlled
   by --smooth / --no-smooth), then align into the FLAME106 layout:
       flame106[t] = concat(exp[t, 0:50], zeros(50), pose[t, 0:6])
   DualTalk only observes expression dims 0-49, so dims 50-99 are zero padding.
4. Convert with Flame2ARKit_Linear.convert -> 61D motion.  Head rotation
   calibration is controlled by --head-center/--head-calibration-frames/
   --head-signs/--head-gains/--head-offsets/--head-limits; the sequence-level
   neutral pose is computed here and folded into the per-frame offsets.
   Afterwards the head channels are optionally stabilized (--head-stabilize):
   median spike removal + Savitzky-Golay smoothing + per-frame angular speed
   cap, to suppress sudden jitter and wild head swings.
5. Save the motion as .npy into the output folder (layout mirrors the source
   relative path) and write the output path back into the metadata
   (speaker*_arkit field), rewriting the jsonl atomically.

Usage:
    python convert_flame2arkit.py --metadata metadata/ood.jsonl \
        --output-dir /path/to/arkit_output [--overwrite]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

# The per-frame BVLS solves are tiny (43 unknowns); multithreaded BLAS only
# adds thread-management overhead there, and each worker process should own
# its core.  Must be set before numpy loads its BLAS backend.
for _blas_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from flame2arkit import (
    FLAME_EXPR_DIM,
    FLAME_EXPR_OBSERVED,
    FLAME106_DIM,
    MOTION61_ORDER,
    Flame2ARKit_Linear,
    validate_head_calibration,
)

DEFAULT_MATRIX = Path("/xuxuanyang/DATA/_data/DualTalk_Dataset/script/mat_final.npy")
EXPECTED_MATRIX_SHA256 = (
    "f055de09c64182696499a26c2d6109349c627195bcd40c6adc3dd27f3922b34b"
)
SPEAKERS = ("speaker1", "speaker2")

# Temporal smoothing (sequence-level; flame2arkit.py stays frame-independent).
SMOOTH_WINDOW = 7
SMOOTH_POLYORDER = 3

# Head channels inside the 61D motion vector (HeadYaw/HeadPitch/HeadRoll).
HEAD_CHANNEL_SLICE = slice(52, 55)


def smooth_sequence(values: np.ndarray) -> np.ndarray:
    """Savitzky-Golay temporal smoothing; passthrough if too short."""
    if len(values) < SMOOTH_WINDOW:
        return values.astype(np.float64, copy=True)
    return savgol_filter(
        values, SMOOTH_WINDOW, SMOOTH_POLYORDER, axis=0, mode="interp"
    ).astype(np.float64, copy=False)


def median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Per-column rolling median; removes isolated 1-2 frame spikes."""
    window = int(window)
    if window <= 1 or len(values) < 2:
        return values.astype(np.float64, copy=True)
    if window % 2 == 0:
        window += 1
    return median_filter(values, size=(window, 1), mode="nearest").astype(
        np.float64, copy=False
    )


def clamp_frame_delta(values: np.ndarray, max_delta: float) -> np.ndarray:
    """Cap the per-frame change; sequential integration keeps history coherent.

    Each frame moves toward the raw value by at most max_delta, so the result
    always stays between the previous output and the raw value (no overshoot).
    """
    if max_delta <= 0 or len(values) < 2:
        return values.astype(np.float64, copy=True)
    out = values.astype(np.float64, copy=True)
    for frame in range(1, len(out)):
        step = np.clip(out[frame] - out[frame - 1], -max_delta, max_delta)
        out[frame] = out[frame - 1] + step
    return out


def stabilize_head_pose(
    head: np.ndarray,
    *,
    median_window: int,
    smooth_window: int,
    max_delta: float,
) -> np.ndarray:
    """Post-conversion headpose constraint: spike removal + smoothing + speed cap.

    Applied on the calibrated HeadYaw/Pitch/Roll euler channels; all three
    stages are individually disabled via their parameters.
    """
    result = median_smooth(head, median_window)
    smooth_window = int(smooth_window)
    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window >= SMOOTH_POLYORDER + 2 and len(result) >= smooth_window:
        result = savgol_filter(
            result, smooth_window, SMOOTH_POLYORDER, axis=0, mode="interp"
        ).astype(np.float64, copy=False)
    result = clamp_frame_delta(result, max_delta)
    # Savitzky-Golay can overshoot near boundaries; keep the output inside the
    # range already enforced by the per-frame head limits.
    result = np.clip(result, head.min(axis=0), head.max(axis=0))
    if not np.isfinite(result).all():
        raise ValueError("Stabilized head rotation contains NaN or Inf")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--metadata", type=Path, required=True,
        help="Metadata jsonl file; source FLAME paths are read from it.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for converted .npy files (mirrors metadata-relative paths).",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Base directory that the metadata-relative paths resolve against. "
             "Defaults to the parent of the metadata file's parent "
             "(i.e. metadata/.. = DualTalk_Dataset root).",
    )
    parser.add_argument(
        "--smooth", action=argparse.BooleanOptionalAction, default=True,
        help="Smooth expression and neck sequences (Savitzky-Golay) before "
             "conversion, matching the base script. Use --no-smooth to disable.",
    )
    parser.add_argument(
        "--head-center", choices=("none", "first", "median"), default="none",
        help="Neutral-pose correction for HeadYaw/Pitch/Roll. Default keeps "
             "the base behavior (no centering).",
    )
    parser.add_argument(
        "--head-calibration-frames", type=int, default=25,
        help="Frames used by median neutral-pose correction.",
    )
    parser.add_argument(
        "--head-signs", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        metavar=("YAW", "PITCH", "ROLL"),
        help="Per-axis direction signs; each value must be -1 or 1.",
    )
    parser.add_argument(
        "--head-gains", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        metavar=("YAW", "PITCH", "ROLL"),
        help="Per-axis non-negative rotation gains.",
    )
    parser.add_argument(
        "--head-offsets", type=float, nargs=3, default=(0.0, 0.0, 0.0),
        metavar=("YAW", "PITCH", "ROLL"),
        help="Per-axis output offsets in radians.",
    )
    parser.add_argument(
        "--head-stabilize", action=argparse.BooleanOptionalAction, default=True,
        help="Post-conversion constraint on HeadYaw/Pitch/Roll: median spike "
             "removal + Savitzky-Golay smoothing + per-frame angular speed "
             "cap. Use --no-head-stabilize to disable.",
    )
    parser.add_argument(
        "--head-median-window", type=int, default=5,
        help="Median filter window for head spike removal; <=1 disables.",
    )
    parser.add_argument(
        "--head-smooth-window", type=int, default=7,
        help="Savitzky-Golay window for head smoothing; <5 disables.",
    )
    parser.add_argument(
        "--head-max-delta", type=float, default=0.3,
        help="Max head rotation change per frame in radians; 0 disables.",
    )
    parser.add_argument(
        "--head-limits", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        metavar=("YAW", "PITCH", "ROLL"),
        help="Positive per-axis absolute limits in radians.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max records; 0 means all.")
    parser.add_argument(
        "--num-workers", type=int, default=16,
        help="Parallel worker processes converting samples; 1 means sequential.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matrix(path: Path) -> np.ndarray:
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_MATRIX_SHA256:
        raise ValueError(f"Matrix SHA256 mismatch: {actual_sha}")
    matrix = np.load(path, allow_pickle=False)
    return matrix


def atomic_write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.save appends ".npy" unless the name already ends with it.
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.npy", dir=path.parent)
    try:
        os.close(fd)
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_flame106(npz_path: Path, smooth: bool) -> tuple[np.ndarray, str]:
    """Load a DualTalk FLAME npz and pad it into the FLAME106 layout.

    With smooth=True the expression and neck sequences are smoothed before
    alignment (jaw is left raw), reproducing the base script behavior.
    """
    payload = npz_path.read_bytes()
    source_sha = hashlib.sha256(payload).hexdigest()
    with np.load(npz_path, allow_pickle=False) as sample:
        if "exp" not in sample.files or "pose" not in sample.files:
            raise ValueError(f"{npz_path}: missing exp or pose")
        expression = np.asarray(sample["exp"], dtype=np.float64)
        pose = np.asarray(sample["pose"], dtype=np.float64)
    if expression.ndim != 2 or expression.shape[1] < FLAME_EXPR_OBSERVED:
        raise ValueError(f"{npz_path}: expected exp [T,>={FLAME_EXPR_OBSERVED}], got {expression.shape}")
    if pose.shape != (len(expression), 6):
        raise ValueError(f"{npz_path}: expected pose [T,6], got {pose.shape}")
    if not np.isfinite(expression).all() or not np.isfinite(pose).all():
        raise ValueError(f"{npz_path}: source contains NaN or Inf")

    if smooth:
        expression = expression.copy()
        pose = pose.copy()
        expression[:, :FLAME_EXPR_OBSERVED] = smooth_sequence(
            expression[:, :FLAME_EXPR_OBSERVED]
        )
        pose[:, :3] = smooth_sequence(pose[:, :3])

    frames = len(expression)
    flame106 = np.zeros((frames, FLAME106_DIM), dtype=np.float64)
    flame106[:, :FLAME_EXPR_OBSERVED] = expression[:, :FLAME_EXPR_OBSERVED]
    # Dims 50:100 stay zero (unobserved FLAME expression dims).
    flame106[:, FLAME_EXPR_DIM:FLAME_EXPR_DIM + 6] = pose
    return flame106, source_sha


def output_npy_path(output_dir: Path, relative: str) -> Path:
    """Map a metadata-relative flame path to its .npy output path."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe metadata path: {relative}")
    return output_dir / rel.with_suffix(".npy")


def compute_head_neutral(
    flame106: np.ndarray, center_mode: str, calibration_frames: int,
) -> tuple[np.ndarray, int]:
    """Sequence-level neutral head pose (YXZ Euler) for centering.

    Returns (neutral[3], frames_used).  flame2arkit.py stays frame-independent,
    so this multi-frame statistic lives here.
    """
    if center_mode == "none":
        return np.zeros(3, dtype=np.float64), 0
    raw_head = Flame2ARKit_Linear.rotvec_to_head_euler(
        flame106[:, FLAME_EXPR_DIM:FLAME_EXPR_DIM + 3]
    )
    if center_mode == "first":
        return raw_head[0].copy(), 1
    used = min(calibration_frames, len(raw_head))
    return np.median(raw_head[:used], axis=0), used


def convert_sample(
    converter: Flame2ARKit_Linear,
    args: argparse.Namespace,
    dataset_root: Path,
    signs: np.ndarray,
    gains: np.ndarray,
    offsets: np.ndarray,
    limits: np.ndarray,
    speaker: str,
    flame_rel: str,
) -> tuple[dict[str, object], Path, int]:
    """Convert one speaker's FLAME npz into ARKit motion and save it.

    Returns (metadata fields to merge back, npy path, frame count); runs
    inside worker processes, so it must not touch shared state.
    """
    npz_path = dataset_root / flame_rel
    npy_path = output_npy_path(args.output_dir, flame_rel)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    # 3. Dimension alignment (+ optional smoothing) + conversion.
    flame106, source_sha = load_flame106(npz_path, smooth=args.smooth)
    # Sequence-level neutral pose folded into the per-frame offsets:
    #   (raw - neutral)*signs*gains + offsets
    #     == raw*signs*gains + (offsets - neutral*signs*gains)
    neutral, neutral_frames = compute_head_neutral(
        flame106, args.head_center, args.head_calibration_frames
    )
    offsets_eff = offsets - neutral * signs * gains
    motion61 = converter.convert(
        flame106,
        head_signs=signs,
        head_gains=gains,
        head_offsets=offsets_eff,
        head_limits=limits,
    )
    # 3.5 Post-conversion constraint on the head channels.
    if args.head_stabilize:
        motion61[:, HEAD_CHANNEL_SLICE] = stabilize_head_pose(
            motion61[:, HEAD_CHANNEL_SLICE],
            median_window=args.head_median_window,
            smooth_window=args.head_smooth_window,
            max_delta=args.head_max_delta,
        )
    if not (np.isfinite(motion61).all()):
        raise ValueError("Converted motion contains NaN or Inf")
    # 4. Save and record the output path in metadata.
    save_npy(npy_path, motion61.astype(np.float32))
    return {
        f"{speaker}_arkit": npy_path.relative_to(args.output_dir).as_posix(),
        f"{speaker}_arkit_sha256": sha256_file(npy_path),
        f"{speaker}_flame_sha256": source_sha,
        f"{speaker}_frames": int(len(motion61)),
        f"{speaker}_smoothed": bool(args.smooth),
        f"{speaker}_head_correction": {
            "center_mode": args.head_center,
            "calibration_frames_requested": args.head_calibration_frames,
            "calibration_frames_used": int(neutral_frames),
            "neutral_yaw_pitch_roll": neutral.tolist(),
            "signs": signs.tolist(),
            "gains": gains.tolist(),
            "offsets": offsets.tolist(),
            "limits": limits.tolist(),
            "stabilize": {
                "enabled": args.head_stabilize,
                "median_window": args.head_median_window,
                "smooth_window": args.head_smooth_window,
                "max_delta_rad_per_frame": args.head_max_delta,
            },
        },
    }, npy_path, len(motion61)


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.head_calibration_frames <= 0:
        raise ValueError("--head-calibration-frames must be positive")
    if args.head_median_window < 1:
        raise ValueError("--head-median-window must be >= 1")
    if args.head_smooth_window < 0:
        raise ValueError("--head-smooth-window must be >= 0")
    if args.head_max_delta < 0:
        raise ValueError("--head-max-delta must be >= 0")
    # Validate head calibration before creating or overwriting any outputs.
    signs, gains, offsets, limits = validate_head_calibration(
        args.head_signs, args.head_gains, args.head_offsets, args.head_limits
    )
    metadata_path = args.metadata.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    dataset_root = (
        args.dataset_root or metadata_path.parent.parent
    ).resolve()

    # 1. Load the forward matrix and initialize the converter.
    converter = Flame2ARKit_Linear()
    converter.set_matrix(load_matrix(args.matrix))

    # 2. Read metadata records.
    with metadata_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        records = records[: args.limit]
    print(
        f"smooth={'on' if args.smooth else 'off'} head_center={args.head_center} "
        f"head_stabilize={'on' if args.head_stabilize else 'off'} "
        f"workers={args.num_workers}"
    )

    # 3-4. Convert per (record, speaker) in a worker pool.  The per-frame
    # BVLS solve is pure Python and holds the GIL, so real parallelism needs
    # processes (on Linux the fork context inherits the loaded state cheaply);
    # record updates and counters stay on the main thread.
    converted_samples = skipped = failed = 0
    pending: dict[object, tuple[int, str, str]] = {}

    def harvest(future) -> None:
        nonlocal converted_samples, skipped, failed
        index, speaker, name = pending.pop(future)
        try:
            fields, npy_path, frames = future.result()
            records[index].update(fields)
            converted_samples += 1
            print(
                f"written: name={name} speaker={speaker} "
                f"frames={frames} npy={npy_path}"
            )
        except Exception as exc:
            failed += 1
            print(
                f"failed: name={name} speaker={speaker} "
                f"error={type(exc).__name__}: {exc}"
            )

    try:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for index, record in enumerate(records):
                name = record.get("name", "<unknown>")
                for speaker in SPEAKERS:
                    flame_rel = record.get(f"{speaker}_flame")
                    if not flame_rel:
                        print(
                            f"skipped: name={name} speaker={speaker} "
                            f"reason=no flame path"
                        )
                        skipped += 1
                        continue
                    npy_path = output_npy_path(args.output_dir, flame_rel)
                    if npy_path.is_file() and not args.overwrite:
                        # Already converted: keep/repair the metadata pointer only.
                        arkit_rel = npy_path.relative_to(args.output_dir).as_posix()
                        if record.get(f"{speaker}_arkit") != arkit_rel:
                            record[f"{speaker}_arkit"] = arkit_rel
                        skipped += 1
                        continue
                    future = pool.submit(
                        convert_sample,
                        converter, args, dataset_root,
                        signs, gains, offsets, limits,
                        speaker, flame_rel,
                    )
                    pending[future] = (index, speaker, name)
                # Reap finished tasks as we go so the queue stays bounded.
                done, _ = wait(pending, timeout=0)
                for future in done:
                    harvest(future)
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    harvest(future)
    except KeyboardInterrupt:
        print("interrupted: cancelling remaining conversions")
        for future in pending:
            future.cancel()
        raise

    # 5. Write the updated metadata (output paths) back atomically.
    atomic_write_jsonl(metadata_path, records)

    print(
        f"summary: converted={converted_samples} skipped={skipped} failed={failed} "
        f"output_dir={args.output_dir} metadata={metadata_path}"
    )
    print(f"motion_order ({len(MOTION61_ORDER)}): {', '.join(MOTION61_ORDER)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
