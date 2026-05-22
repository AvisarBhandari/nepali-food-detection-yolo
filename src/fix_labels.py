from pathlib import Path

mapping = {1: 0, 2: 1, 5: 2, 7: 3, 9: 4, 10: 5, 12: 6}

keep_classes = [
    "chatpate",
    "dal_bhat",
    "gundruk",
    "momo",
    "samosa",
    "sel_roti",
    "yomari",
]

dataset = Path("Nepali_Food_Image")

# remove unwanted folders
for folder in dataset.iterdir():
    if folder.is_dir() and folder.name not in keep_classes:
        import shutil

        shutil.rmtree(folder)

        print("Removed:", folder.name)


# remap remaining labels
for cls in keep_classes:
    label_dir = dataset / cls / "labels"

    for txt_file in label_dir.glob("*.txt"):
        new_lines = []

        with open(txt_file) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            old = int(parts[0])

            parts[0] = str(mapping[old])

            new_lines.append(" ".join(parts))

        with open(txt_file, "w") as f:
            f.write("\n".join(new_lines))

print("Done.")
