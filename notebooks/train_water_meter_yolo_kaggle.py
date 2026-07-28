# Kaggle notebook sample: train a YOLO model for water-meter digit detection.
# Upload the cleaned Roboflow zip as a Kaggle Dataset, add it as Notebook Input,
# enable a GPU, then paste this file into a notebook cell and run it.

from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pyyaml"])

import yaml
from ultralytics import YOLO


# Training settings. Lower BATCH to 8 if Kaggle reports an out-of-memory error.
EPOCHS = 50
IMAGE_SIZE = 640
BATCH = 16
MODEL_NAME = "yolo11n.pt"
RUN_NAME = "water_meter_yolo11n"

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
EXTRACT_DIR = WORK_ROOT / "water_meter_dataset"
RUNS_DIR = WORK_ROOT / "runs" / "water_meter"


def find_input() -> tuple[str, Path]:
    zip_files = sorted(INPUT_ROOT.rglob("*.zip"))
    if zip_files:
        return "zip", zip_files[0]

    yaml_files = sorted(INPUT_ROOT.rglob("data.yaml"))
    if yaml_files:
        return "folder", yaml_files[0]

    available = [str(path) for path in INPUT_ROOT.iterdir()]
    raise FileNotFoundError(
        "No .zip or data.yaml found. Add the uploaded dataset under Add Input. "
        f"Available inputs: {available}"
    )


def find_data_yaml(dataset_dir: Path) -> Path:
    candidates = sorted(dataset_dir.rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError("The archive does not contain data.yaml.")
    return candidates[0]


input_type, source_path = find_input()
print("Input type:", input_type)
print("Using:", source_path)
if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)

if input_type == "zip":
    EXTRACT_DIR.mkdir(parents=True)
    with zipfile.ZipFile(source_path) as archive:
        archive.extractall(EXTRACT_DIR)
else:
    shutil.copytree(source_path.parent, EXTRACT_DIR)

data_yaml = find_data_yaml(EXTRACT_DIR)
dataset_dir = data_yaml.parent
config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))

required_keys = {"train", "val", "names"}
missing_keys = required_keys - set(config)
if missing_keys:
    raise ValueError(f"data.yaml is missing required keys: {sorted(missing_keys)}")

# Roboflow exports paths relative to the original archive. Make them local to Kaggle.
normalized_config = {
    "path": str(dataset_dir),
    "train": "train/images",
    "val": "valid/images",
    "names": config["names"],
    "nc": config.get("nc", len(config["names"])),
}
data_yaml.write_text(
    yaml.safe_dump(normalized_config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

print("Dataset:", dataset_dir)
print("Config:", data_yaml)
for split in ("train", "valid"):
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    print(f"{split}/images: {len(list(image_dir.glob('*')))}")
    print(f"{split}/labels: {len(list(label_dir.glob('*')))}")

print("Classes:", normalized_config["names"])

model = YOLO(MODEL_NAME)
model.train(
    data=str(data_yaml),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH,
    device=0,
    workers=2,
    project=str(RUNS_DIR),
    name=RUN_NAME,
    patience=10,
    seed=42,
)

best_weights = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
if not best_weights.exists():
    raise FileNotFoundError(f"Training completed but best.pt was not found at {best_weights}")

# Evaluate the best checkpoint and write prediction examples to the Kaggle output area.
best_model = YOLO(str(best_weights))
metrics = best_model.val(data=str(data_yaml), imgsz=IMAGE_SIZE, device=0)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

sample_images = sorted((dataset_dir / "valid" / "images").glob("*"))[:10]
best_model.predict(
    source=[str(image) for image in sample_images],
    imgsz=IMAGE_SIZE,
    conf=0.25,
    save=True,
    project=str(WORK_ROOT / "predictions"),
    name="validation_samples",
)

archive_base = WORK_ROOT / "water_meter_yolo_output"
archive_path = shutil.make_archive(
    str(archive_base),
    "zip",
    root_dir=RUNS_DIR / RUN_NAME,
)

print("Best weights:", best_weights)
print("Download this output archive:", archive_path)
