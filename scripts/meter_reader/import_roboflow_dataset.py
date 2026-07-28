from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import yaml


def normalize_yaml(dataset_dir: Path) -> Path:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found after extraction: {data_yaml}")

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if "names" not in data:
        raise ValueError(f"Missing names in {data_yaml}")

    normalized = {
        "path": dataset_dir.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images" if (dataset_dir / "valid" / "images").exists() else "val/images",
        "names": data["names"],
    }
    if (dataset_dir / "test" / "images").exists():
        normalized["test"] = "test/images"
    if "nc" in data:
        normalized["nc"] = data["nc"]

    data_yaml.write_text(
        yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return data_yaml


def extract_zip(zip_path: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty. Use --overwrite to replace existing files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a Roboflow YOLOv8 zip and normalize data.yaml.")
    parser.add_argument("--zip", type=Path, required=True, help="Path to Roboflow YOLOv8 zip file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/meter_digits_roboflow"),
        help="Destination dataset directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.zip.exists():
        raise FileNotFoundError(args.zip)

    extract_zip(args.zip, args.output_dir, args.overwrite)
    data_yaml = normalize_yaml(args.output_dir)
    print(f"Extracted dataset to: {args.output_dir.resolve()}")
    print(f"Normalized config: {data_yaml.resolve()}")


if __name__ == "__main__":
    main()
