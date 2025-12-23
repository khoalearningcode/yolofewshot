# count_samples.py
from collections import Counter
from pathlib import Path

def count_yolo_labels(label_dir):
    label_dir = Path(label_dir)
    class_counts = Counter()
    total_boxes = 0
    for lbl_file in label_dir.rglob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    total_boxes += 1
    return class_counts, total_boxes

# Chạy
train_counts, train_total = count_yolo_labels("/home/fit02/nguyen_workspace/nguyen/yolofewshot/dataset/converted/labels/train")
val_counts, val_total = count_yolo_labels("/home/fit02/nguyen_workspace/nguyen/yolofewshot/dataset/converted/labels/val")

print("TRAIN CLASS DISTRIBUTION:")
for cid, count in sorted(train_counts.items()):
    print(f"  {cid}: {count} boxes")

print(f"\nTotal train boxes: {train_total}")
print(f"Total val boxes: {val_total}")