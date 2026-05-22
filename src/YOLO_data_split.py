from pathlib import Path
import random
import shutil


def split_yolo_data(input_dir, output_dir, train_ratio=0.8):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directories
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "val").mkdir(parents=True, exist_ok=True)
    classes_names = []
    classes = [folder for folder in input_dir.iterdir() if folder.is_dir()]
    for cls in classes:
        classes_names.append(cls.name)
        images = list(cls.glob("images/*"))
        labels = list(cls.glob("labels/*"))
        # Shuffle data
        combined = list(zip(images, labels))
        random.shuffle(combined)
        images[:], labels[:] = zip(*combined)
        # Split data
        split_idx = int(train_ratio * len(images))
        train_images, val_images = images[:split_idx], images[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        # Create class subdirectories in output
        (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
        # Copy training data
        for img, lbl in zip(train_images, train_labels):
            shutil.copy(img, output_dir / "train" / "images" / img.name)
            shutil.copy(lbl, output_dir / "train" / "labels" / lbl.name)
        # Copy validation data
        for img, lbl in zip(val_images, val_labels):
            shutil.copy(img, output_dir / "val" / "images" / img.name)
            shutil.copy(lbl, output_dir / "val" / "labels" / lbl.name)
    # Save class names to a file
    with open(output_dir / "classes.txt", "w") as f:
        for class_name in classes_names:
            f.write(f"{class_name}\n")


split_yolo_data(input_dir="Nepali_Food_Image", output_dir="Dataset")
