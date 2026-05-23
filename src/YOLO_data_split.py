from pathlib import Path
import random
import shutil


def split_yolo_data(input_dir, output_dir, train_ratio=0.8):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    (output_dir / "train/images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train/labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "val/images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val/labels").mkdir(parents=True, exist_ok=True)

    class_names = []

    for cls in input_dir.iterdir():
        if not cls.is_dir():
            continue

        class_names.append(cls.name)

        images = sorted((cls / "images").glob("*"))

        pairs = []

        for img in images:
            label = cls / "labels" / f"{img.stem}.txt"

            if label.exists():
                pairs.append((img, label))
            else:
                print(f"Missing label: {img.name}")

        random.shuffle(pairs)

        split_idx = int(len(pairs) * train_ratio)

        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        for img, lbl in train_pairs:
            new_img = f"{cls.name}_{img.name}"
            new_lbl = f"{cls.name}_{lbl.name}"

            shutil.copy(img, output_dir / "train/images" / new_img)

            shutil.copy(lbl, output_dir / "train/labels" / new_lbl)

        for img, lbl in val_pairs:
            new_img = f"{cls.name}_{img.name}"
            new_lbl = f"{cls.name}_{lbl.name}"

            shutil.copy(img, output_dir / "val/images" / new_img)
            shutil.copy(lbl, output_dir / "val/labels" / new_lbl)

        print(f"{cls.name}: {len(train_pairs)} train | {len(val_pairs)} val")

    with open(output_dir / "classes.txt", "w") as f:
        f.write("\n".join(class_names))


split_yolo_data("Nepali_Food_Image", "Dataset")
