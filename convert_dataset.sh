#!/bin/bash

# Convert dataset với stratified split theo từng video_id

cd "$(dirname "$0")" || exit

echo "════════════════════════════════════════════════════════════"
echo "🚀 Converting dataset với split theo từng video_id"
echo "════════════════════════════════════════════════════════════"
echo ""

# Remove old converted data
echo "Removing old converted data..."
rm -rf dataset/converted

# Show original annotation counts
echo ""
echo "📊 Original annotation counts:"
python3 << 'EOF'
import json
from collections import defaultdict
from pathlib import Path

ann_path = Path("/home/caokhoa/Documents/AEROYEYES/raw_data/train/annotations/annotations.json")
with open(ann_path) as f:
    items = json.load(f)

video_objects = defaultdict(int)
total_objects = 0

for it in items:
    video_id = str(it.get('video_id'))
    ann_list = it.get('annotations', [])
    for ann in ann_list:
        bboxes = ann.get('bboxes', [])
        video_objects[video_id] += len(bboxes)
        total_objects += len(bboxes)

print(f"Total videos: {len(video_objects)}")
print(f"Total objects: {total_objects}")
print("\nObjects per video:")
for vid in sorted(video_objects.keys()):
    print(f"  {vid:20} {video_objects[vid]:6} objects")
EOF

# Run conversion
echo ""
echo "Running conversion (input: raw_data/train)..."
python tools/convert_to_yolo.py \
  --input_root /home/caokhoa/Documents/AEROYEYES/raw_data/train \
  --output_root dataset/converted \
  --frame_stride 2 \
  --dataset_yaml ultralytics/cfg/datasets/fewshot.yaml \
  --val_ratio 0.2 \
  --seed 42

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📊 Verifying split..."
echo "════════════════════════════════════════════════════════════"
echo ""

python3 << 'EOF'
from pathlib import Path
from collections import defaultdict

class_names = ["Backpack", "Jacket", "Laptop", "Lifering", "MobilePhone", "Person", "WaterBottle"]

# Count per split
split_totals = {}
split_per_class = {}

for split in ['train', 'val']:
    labels_dir = Path(f"dataset/converted/labels/{split}")
    class_count = defaultdict(int)
    video_count = defaultdict(int)
    
    for txt_file in labels_dir.glob("*.txt"):
        video_id = "_".join(txt_file.stem.split('_')[:-1])
        video_count[video_id] += 1
        
        with open(txt_file) as f:
            for line in f:
                class_id = int(line.split()[0])
                class_count[class_id] += 1
    
    split_totals[split] = sum(class_count.values())
    split_per_class[split] = class_count
    
    print(f"{'='*70}")
    print(f"{split.upper()} SET")
    print(f"{'='*70}")
    print(f"Videos: {len(video_count)}, Images: {len(list(labels_dir.glob('*.txt')))}, Objects: {split_totals[split]}")
    print()
    print("Objects by class:")
    print(f"  ID | {'Class':<15} | {'Count':>6}")
    print(f"  ---+{'-'*17}+{'-'*8}")
    for cid in range(7):
        count = split_per_class[split][cid]
        status = "✓" if count > 0 else "✗"
        print(f"  {status}{cid} | {class_names[cid]:<15} | {count:>6}")
    print(f"     |{' '*17}|{'-'*8}")
    print(f"     | {'TOTAL':<15} | {split_totals[split]:>6}")
    print()

# Verify
print(f"{'='*70}")
print("📋 VERIFICATION")
print(f"{'='*70}")
total = split_totals['train'] + split_totals['val']
print(f"Train + Val total: {total}")
print(f"Train: {split_totals['train']} ({split_totals['train']/total*100:.1f}%)")
print(f"Val:   {split_totals['val']} ({split_totals['val']/total*100:.1f}%)")
print()

# Check if all classes present
all_train = set(split_per_class['train'].keys())
all_val = set(split_per_class['val'].keys())
all_classes = set(range(7))

missing_train = all_classes - all_train
missing_val = all_classes - all_val

if missing_train:
    print(f"⚠️  Train missing: {[class_names[i] for i in sorted(missing_train)]}")
else:
    print(f"✅ Train: All 7 classes present")

if missing_val:
    print(f"⚠️  Val missing: {[class_names[i] for i in sorted(missing_val)]}")
else:
    print(f"✅ Val: All 7 classes present")

EOF

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Done!"
echo "════════════════════════════════════════════════════════════"
