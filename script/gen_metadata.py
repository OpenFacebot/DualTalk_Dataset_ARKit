#!/usr/bin/env python3
"""Generate DualTalk metadata jsonl files for train/test/ood splits.

Each jsonl line records one dialogue pair:
    {name, speaker1_text, speaker1_audio, speaker1_flame, speaker1_arkit,
     speaker2_text, speaker2_audio, speaker2_flame, speaker2_arkit}
Missing fields are filled with null. All paths are relative to the dataset
root, e.g. "train/xxx_speaker1.wav" or "ARKit/train/xxx_speaker1/xxx_speaker1.csv".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SPLITS = ("train", "test", "ood")
SPEAKERS = ("speaker1", "speaker2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path("/xuxuanyang/DATA/_data/DualTalk_Dataset"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for *.jsonl outputs. Defaults to <dataset-dir>/metadata.",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
    )
    return parser.parse_args()


def read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def relative_path(path: Path, dataset_dir: Path) -> str | None:
    """Return POSIX relative path if the file exists, otherwise None."""
    return path.relative_to(dataset_dir).as_posix() if path.is_file() else None


def collect_pairs(split_dir: Path) -> dict[str, dict[str, dict[str, Path]]]:
    """Group files into {pair_name: {speaker: {ext: path}}}."""
    pairs: dict[str, dict[str, dict[str, Path]]] = {}
    for path in sorted(split_dir.iterdir()):
        if not path.is_file():
            continue
        for speaker in SPEAKERS:
            suffix = f"_{speaker}"
            if path.stem.endswith(suffix):
                name = path.stem[: -len(suffix)]
                pairs.setdefault(name, {}).setdefault(speaker, {})[path.suffix] = path
                break
        else:
            print(f"[warn] unrecognized file skipped: {path}")
    return pairs


def build_record(
    name: str,
    pair: dict[str, dict[str, Path]],
    split: str,
    dataset_dir: Path,
) -> dict[str, object]:
    record: dict[str, object] = {"name": name}
    for speaker in SPEAKERS:
        files = pair.get(speaker, {})
        # ARKit CSV follows the converter layout: ARKit/{split}/{stem}/{stem}.csv
        stem = f"{name}_{speaker}"
        arkit_csv = dataset_dir / "ARKit" / split / stem / f"{stem}.csv"
        record[f"{speaker}_text"] = read_text(files.get(".txt"))
        record[f"{speaker}_audio"] = relative_path(files.get(".wav"), dataset_dir)
        record[f"{speaker}_flame"] = relative_path(files.get(".npz"), dataset_dir)
        record[f"{speaker}_arkit"] = relative_path(arkit_csv, dataset_dir)
    return record


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = (args.output_dir or dataset_dir / "metadata").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            print(f"[warn] split dir not found, skipped: {split_dir}")
            continue
        pairs = collect_pairs(split_dir)
        out_path = output_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for name in sorted(pairs):
                record = build_record(name, pairs[name], split, dataset_dir)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{split}: {len(pairs)} pairs -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
