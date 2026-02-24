"""
Kịch bản chuẩn bị dữ liệu.
============================
Tải dataset, phân chia train/val/test, và tạo data.yaml
cho Ultralytics YOLO training pipeline.

Tương thích: DVC pipeline stage `prepare_data`.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

from src.utils.helpers import load_config, seed_everything
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_data(config_path: str = "configs/config.yaml",
                 output_dir: str = "data/processed") -> None:
    """
    Chuẩn bị dữ liệu cho pipeline huấn luyện.

    - Tải dataset (COCO128 hoặc custom) thông qua Ultralytics API
    - Tổ chức thư mục train/val/test
    - Tạo data.yaml cho YOLO
    - Tạo class_distribution.json cho DVC plots

    Args:
        config_path: Đường dẫn tệp cấu hình.
        output_dir: Thư mục xuất dữ liệu đã xử lý.
    """
    config = load_config(config_path)
    data_config = config["data"]
    seed_everything(config["training"]["seed"])

    output_path = Path(output_dir)
    image_size = data_config.get("image_size", 640)
    dataset_name = data_config.get("dataset_name", "coco128")

    logger.info("=" * 60)
    logger.info("CHUẨN BỊ DỮ LIỆU")
    logger.info("=" * 60)
    logger.info("Dataset: %s", dataset_name)
    logger.info("Image size: %d", image_size)
    logger.info("Output: %s", output_dir)

    # ── Bước 1: Tải dataset qua Ultralytics ──
    from ultralytics.data.utils import check_det_dataset

    data_yaml_name = data_config.get("data_yaml", f"{dataset_name}.yaml")
    dataset_info = check_det_dataset(data_yaml_name)

    train_images_src = Path(dataset_info.get("train", ""))
    val_images_src = Path(dataset_info.get("val", ""))
    class_names = dataset_info.get("names", {})

    logger.info("Số lớp: %d", len(class_names))
    logger.info("Thư mục train: %s", train_images_src)
    logger.info("Thư mục val:   %s", val_images_src)

    # ── Bước 2: Sao chép ảnh và nhãn vào output_dir ──
    for split_name, src_dir in [("train", train_images_src),
                                 ("val", val_images_src)]:
        img_out = output_path / split_name / "images"
        lbl_out = output_path / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            logger.warning("Thư mục nguồn không tồn tại: %s", src_dir)
            continue

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        for img_file in sorted(src_dir.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue

            # Copy ảnh
            shutil.copy2(img_file, img_out / img_file.name)

            # Copy nhãn tương ứng
            label_dir = src_dir.parent / "labels"
            label_file = label_dir / img_file.with_suffix(".txt").name
            if label_file.exists():
                shutil.copy2(label_file, lbl_out / label_file.name)

        n_images = len(list(img_out.glob("*")))
        n_labels = len(list(lbl_out.glob("*")))
        logger.info("  %s: %d ảnh, %d nhãn", split_name, n_images, n_labels)

    # ── Bước 3: Tạo data.yaml ──
    data_yaml_content = {
        "path": str(output_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": class_names,
        "nc": len(class_names),
    }

    data_yaml_path = output_path / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False,
                  allow_unicode=True)
    logger.info("Đã tạo: %s", data_yaml_path)

    # ── Bước 4: Tạo class_distribution.json (DVC plots) ──
    class_counter: Counter[int] = Counter()
    labels_dir = output_path / "train" / "labels"
    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text().strip().split("\n"):
            if line.strip():
                cls_id = int(line.strip().split()[0])
                class_counter[cls_id] += 1

    distribution = [
        {
            "class_id": cls_id,
            "class_name": class_names.get(cls_id, str(cls_id)),
            "count": count,
        }
        for cls_id, count in sorted(class_counter.items())
    ]

    dist_path = output_path / "class_distribution.json"
    dist_path.write_text(json.dumps(distribution, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    logger.info("Đã tạo: %s", dist_path)

    logger.info("=" * 60)
    logger.info("HOÀN TẤT CHUẨN BỊ DỮ LIỆU")
    logger.info("=" * 60)


def main() -> None:
    """Entry point cho CLI."""
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu cho YOLO training")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Đường dẫn tệp cấu hình")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Thư mục xuất dữ liệu")
    args = parser.parse_args()
    prepare_data(config_path=args.config, output_dir=args.output)


if __name__ == "__main__":
    main()
