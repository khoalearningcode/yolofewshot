# Few-shot Training on Custom Video Dataset

## 1) Quick setup
# Requirements: Python 3.10, CUDA drivers installed
pip install -r requirements.txt
pip install opencv-python pyyaml huggingface-hub==0.26.3

# Pretrained checkpoint (smallest model, segmentation) for linear probing
huggingface-cli download jameslahm/yoloe yoloe-v8s-seg.pt --local-dir pretrain
Copy-Item pretrain\yoloe-v8s-seg.pt .

# Text encoder MobileCLIP BLT for text PE
Invoke-WebRequest -Uri https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt -OutFile mobileclip_blt.pt

## 2) Prepare data
# Required structure:
# - Annotations: dataset/train/annotations/annotations.json
# - Video: dataset/train/samples/<video_id>/video.mp4
# Conventions:
# - Class name = prefix of video_id before the last "_" (e.g., Backpack_0 -> Backpack)
# - Split: *_0 -> train, *_1 -> val, others -> 80/20 by hash
# Tip: create empty val dirs if needed
mkdir dataset\converted\images\val -Force
mkdir dataset\converted\labels\val -Force

# Convert (use absolute paths for out_root/yaml_out to avoid path issues)
python tools\convert_to_yolo.py --dataset_root dataset\train --annotations dataset\train\annotations\annotations.json --out_root "yoloe-fewshot\dataset\converted" --yaml_out "yoloe-fewshot\ultralytics\cfg\datasets\fewshot.yaml" --seg
# Result:
# - Images: dataset/converted/images/{train,val}/<video_id>/frame_*.jpg
# - Labels: dataset/converted/labels/{train,val}/<video_id>/frame_*.txt
# - YAML: ultralytics/cfg/datasets/my_vid.yaml (absolute path)

## 3) Train (Linear Probing, smallest model)
python train_fewshot_pe_vid.py
# Adjust batch/epochs/device in train_pe_vid.py if needed



## Notes
- If you see “images not found”: ensure both images/train and images/val exist (val can be empty).
- If you see “mobileclip_blt.pt” missing: download it to the project root (see setup above).
- Prefer absolute path in YAML to avoid Ultralytics datasets_dir auto-prefixing.