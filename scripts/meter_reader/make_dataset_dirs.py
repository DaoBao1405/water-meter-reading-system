from __future__ import annotations

import argparse
from pathlib import Path

import yaml


SPLITS = ("train", "val", "test")


def class_names(scheme: str) -> list[str]:
    if scheme == "simple":
        return ["meter", "digit_panel"] + [f"digit_{digit}" for digit in range(10)]
    if scheme == "color":
        return (
            ["meter", "digit_panel"]
            + [f"black_{digit}" for digit in range(10)]
            + [f"red_{digit}" for digit in range(10)]
        )
    raise ValueError(f"Unsupported scheme: {scheme}")


def create_dataset_tree(dataset_dir: Path, scheme: str) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    data_yaml = {
        "path": dataset_dir.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names(scheme))},
    }
    (dataset_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YOLO dataset folders for water meter reading.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/meter_digits"),
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--scheme",
        choices=("simple", "color"),
        default="simple",
        help="Class scheme. Use 'color' when you label black and red digits separately.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_dataset_tree(args.dataset_dir, args.scheme)
    print(f"Created YOLO dataset at: {args.dataset_dir.resolve()}")
    print(f"Class scheme: {args.scheme}")
    print(f"Config file: {(args.dataset_dir / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
