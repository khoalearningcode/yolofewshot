import os
import re
import json
import cv2
import math
import argparse
from pathlib import Path
from collections import defaultdict

# Conversion script: annotations.json + videos -> YOLO detection dataset
# - Discovers class names from video_id prefix (strip trailing digits)
# - Extracts only annotated frames (optionally with stride)
# - Writes images and labels in YOLO format
# - Generates a dataset YAML at ultralytics/cfg/datasets/fewshot.yaml


def slug_to_class(name: str) -> str:
    # Map 'Person1' -> 'Person', keep original case of prefix
    m = re.match(r"^([A-Za-z_]+?)[0-9]*$", name)
    return m.group(1) if m else name


def xyxy_to_yolo(x1, y1, x2, y2, iw, ih):
    # Ensure bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw - 1, x2), min(ih - 1, y2)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / iw, cy / ih, w / iw, h / ih


def parse_args():
    p = argparse.ArgumentParser(description="Convert custom video+box annotations to YOLO format")
    p.add_argument('--input_root', type=str, default=str(Path('dataset') / 'train'), help='Root containing annotations/ and samples/')
    p.add_argument('--annotations', type=str, default='annotations/annotations.json', help='Relative path to annotations.json from input_root')
    p.add_argument('--samples_dir', type=str, default='samples', help='Relative path to samples directory from input_root')
    p.add_argument('--output_root', type=str, default=str(Path('dataset/converted')), help='Output YOLO dataset root')
    p.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio by video id')
    p.add_argument('--frame_stride', type=int, default=2, help='Sample every Nth annotated frame')
    p.add_argument('--frame_offset', type=int, default=0, help='Add this offset to decoded frame index before matching annotations (use 1 if annotations are 1-based)')
    p.add_argument('--jpeg_quality', type=int, default=90, help='JPEG encode quality')
    p.add_argument('--dataset_yaml', type=str, default=str(Path('ultralytics') / 'cfg' / 'datasets' / 'fewshot.yaml'), help='Path to write dataset yaml')
    return p.parse_args()


def main():
    args = parse_args()
    in_root = Path(args.input_root)
    ann_path = in_root / args.annotations
    samples_root = in_root / args.samples_dir
    out_root = Path(args.output_root)

    # Read annotations.json
    with open(ann_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    # Build per-video frame->boxes mapping and class discovery
    video_to_frames = dict()
    class_names = []
    class_index = dict()

    for it in items:
        video_id = it.get('video_id')
        base = str(video_id)
        cls_slug = slug_to_class(base.split('_')[0])
        if cls_slug not in class_index:
            class_index[cls_slug] = len(class_names)
            class_names.append(cls_slug)
        cid = class_index[cls_slug]

        # Collect all bboxes for frames
        frame_map = defaultdict(list)
        ann_list = it.get('annotations', [])
        for ann in ann_list:
            bboxes = ann.get('bboxes', [])
            for b in bboxes:
                fr = int(b.get('frame'))
                x1 = float(b.get('x1'))
                y1 = float(b.get('y1'))
                x2 = float(b.get('x2'))
                y2 = float(b.get('y2'))
                frame_map[fr].append((cid, x1, y1, x2, y2))
        video_to_frames[base] = frame_map

    # Train/val split by deterministic hash on video id
    def is_val(video_id: str) -> bool:
        return (abs(hash(video_id)) % 1000) / 1000.0 < args.val_ratio

    # Prepare output dirs
    for split in ['train', 'val']:
        (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Iterate videos
    jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
    total_images, total_labels = 0, 0

    for video_id, frame_map in video_to_frames.items():
        split = 'val' if is_val(video_id) else 'train'
        video_path = samples_root / video_id / 'drone_video.mp4'
        if not video_path.exists():
            print(f"[WARN] Video not found: {video_path}")
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Cannot open video: {video_path}")
            continue
        # Build quick lookup
        wanted_frames = sorted(frame_map.keys())
        if args.frame_stride > 1:
            wanted_frames = [f for i, f in enumerate(wanted_frames) if i % args.frame_stride == 0]
        wanted_set = set(wanted_frames)
        if not wanted_frames:
            cap.release()
            continue

        # Read frames sequentially and dump when in wanted_set
        fidx = 0
        next_target_idx = 0
        max_target = wanted_frames[-1]
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # current frame index depends on video encoding; we assume first read is 0
            # Apply optional frame offset to align with annotations
            f = fidx + args.frame_offset
            if f in wanted_set:
                ih, iw = frame.shape[:2]
                boxes = frame_map.get(f, [])
                # Write image
                img_name = f"{video_id}_{f:06d}.jpg"
                img_out = out_root / 'images' / split / img_name
                cv2.imwrite(str(img_out), frame, jpg_params)
                total_images += 1
                # Write label
                lbl_out = out_root / 'labels' / split / (img_out.stem + '.txt')
                with open(lbl_out, 'w', encoding='utf-8') as lf:
                    for (cid, x1, y1, x2, y2) in boxes:
                        x, y, w, h = xyxy_to_yolo(x1, y1, x2, y2, iw, ih)
                        lf.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                        total_labels += 1
            fidx += 1
            if fidx > max_target + 1 and next_target_idx >= len(wanted_frames):
                # Heuristic early stop if we've passed all targets
                pass
        cap.release()

    # Write dataset YAML
    yaml_path = Path(args.dataset_yaml)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    # Use absolute path for robustness
    abs_root = str(out_root.resolve()).replace('\\', '/')
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yf.write(f"path: {abs_root}\n")
        yf.write("train: images/train\n")
        yf.write("val: images/val\n")
        yf.write("names:\n")
        for i, n in enumerate(class_names):
            yf.write(f"  {i}: {n}\n")

    print(f"[DONE] Images written: {total_images}, Labels written: {total_labels}")
    print(f"[YAML] {yaml_path}")
    print(f"[DATA ROOT] {abs_root}")


if __name__ == '__main__':
    main()
