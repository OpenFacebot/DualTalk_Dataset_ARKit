#!/usr/bin/env python3
"""Convert one ARKit .npy motion file to a render-ready CSV file.

Input : one [T, 61] float32 npy file produced by convert_flame2arkit.py
        (columns ordered as flame2arkit.MOTION61_ORDER).
Output: one CSV matching the renderer reference format, e.g.
        ARKit/YI30GzlffAA_sub_video_6_001_speaker1.csv

CSV format (verified against the reference file):
    Header : Timecode,BlendshapeCount,<61 channel names>
    Rows   : <timecode>,61,<61 values with --decimals decimal places (default 4)>
    Timecode HH:MM:SS:FF.000 advances one SECOND per row and the frame
    field stays 00 (the reference convention of the renderer; DualTalk
    data is 25 fps but the renderer only consumes the row order).

Usage (one-to-one):
    python convert_arkit_for_render.py --npy path/to/x.npy --output path/to/x.csv
    python convert_arkit_for_render.py --npy x.npy --output out_dir/   # -> out_dir/x.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

from flame2arkit import MOTION61_DIM, MOTION61_ORDER

CSV_HEADER = ("Timecode", "BlendshapeCount") + tuple(MOTION61_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", type=Path, required=True, help="Input .npy motion file.")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output .csv path, or a directory (then <dir>/<stem>.csv is used).",
    )
    parser.add_argument(
        "--decimals", type=int, default=4,
        help="Decimal places of the CSV values; default 4 matches the "
             "renderer reference file.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def format_timecode(frame_index: int) -> str:
    """Renderer timecode: one second per row, frame field fixed at 00."""
    hours, remainder = divmod(frame_index, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:00.000"


def load_motion_npy(path: Path) -> np.ndarray:
    motion = np.load(path, allow_pickle=False)
    if motion.ndim != 2 or motion.shape[1] != MOTION61_DIM:
        raise ValueError(f"{path}: expected [T,{MOTION61_DIM}] motion, got {motion.shape}")
    if not np.isfinite(motion).all():
        raise ValueError(f"{path}: motion contains NaN or Inf")
    return motion


def write_render_csv(path: Path, motion: np.ndarray, decimals: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(CSV_HEADER)
            for frame, values in enumerate(motion):
                writer.writerow(
                    (format_timecode(frame), str(MOTION61_DIM),
                     *(f"{v:.{decimals}f}" for v in values))
                )
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    if args.decimals < 0:
        raise ValueError("--decimals must be >= 0")
    npy_path = args.npy.resolve()
    if not npy_path.is_file():
        raise FileNotFoundError(npy_path)
    csv_path = args.output / f"{npy_path.stem}.csv" if args.output.is_dir() \
        else args.output.resolve()
    if csv_path.is_file() and not args.overwrite:
        raise FileExistsError(f"{csv_path} exists; use --overwrite")

    motion = load_motion_npy(npy_path)
    write_render_csv(csv_path, motion, decimals=args.decimals)
    print(f"written: {npy_path} -> {csv_path} frames={len(motion)} decimals={args.decimals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
