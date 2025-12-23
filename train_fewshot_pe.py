import os
import argparse
from numpy._typing import _32Bit
import torch
from pathlib import Path
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_pe import YOLOEPETrainer
from ultralytics.utils import yaml_load, LOGGER
import random
import numpy as np

def make_freeze_list(model):
    head_index = len(model.model.model) - 1
    # Freeze ONLY backbone layers [0..head_index-1]; keep the entire head trainable
    freeze = [str(f) for f in range(0, head_index)]
    return freeze


def parse_args():
    p = argparse.ArgumentParser(description='Few-shot PE fine-tuning (detection)')
    p.add_argument('--data', type=str, default=str(Path('ultralytics') / 'cfg' / 'datasets' / 'fewshot.yaml'))
    p.add_argument('--model', type=str, default='yoloe-v8m.yaml', help='Detection model YAML, e.g., yoloe-v8s.yaml')
    p.add_argument('--weights', type=str, default='yoloe-v8m.pt', help='Pretrained detection .pt path (for backbone init)')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr0', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=0.0005) #0.01
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--exp_name', type=str, default='fewshot-pe-8m')
    p.add_argument('--save_pe', type=str, default='fewshot-pe-8m.pt')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--patience', type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load detection model from YAML to avoid auto 'segment' task
    model = YOLOE(args.model)

    # Prepare PE from dataset names
    model.eval()  # required for get_text_pe
    names_dict = yaml_load(args.data).get('names')
    # names can be list or dict
    if isinstance(names_dict, dict):
        names = [names_dict[k] for k in sorted(names_dict.keys(), key=lambda x: int(x))]
    else:
        names = list(names_dict)

    tpe = model.get_text_pe(names)
    torch.save({'names': names, 'pe': tpe}, args.save_pe)
    LOGGER.info(f"Saved initial PE to {args.save_pe}")

    # Freeze majority
    freeze = make_freeze_list(model)

    # Train
    model.train(
        data=args.data,
        task='detect',
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer='AdamW',
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        warmup_bias_lr=0.0,
        close_mosaic=15,
        cos_lr=True,
        mosaic=0.8,
        mixup=0.1,
        max_det=1000,
        cache='disk',
        workers=args.workers,
        device=args.device,
        project='runs/detect',
        name=args.exp_name,
        trainer=YOLOEPETrainer,
        freeze=freeze,
        train_pe_path=args.save_pe,
        pretrained=args.weights,   #load backbone from detection checkpoint
        patience=args.patience,
        box=7.5,
        nbs=64,
        scale=0.5,
    )


if __name__ == '__main__':
    main()
