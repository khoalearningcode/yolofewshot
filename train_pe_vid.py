from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_pe import YOLOEPESegTrainer
import os
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import yaml_load, LOGGER
import torch

import argparse


def main():
    os.environ["PYTHONHASHSEED"] = "0"

    parser = argparse.ArgumentParser(description="Train (linear probing) on custom video-derived dataset")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/my_vid.yaml", help="Path to dataset YAML")
    parser.add_argument("--weights", default="yoloe-v8s-seg.pt", help="Pretrained segmentation weights path (.pt)")
    parser.add_argument("--model_cfg", default="yoloe-v8s-seg.yaml", help="Segmentation model config YAML to infer scale")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--device", default="0", help="CUDA device id(s) or 'cpu'")
    args = parser.parse_args()

    scale = guess_model_scale(args.model_cfg)
    cfg_dir = "ultralytics/cfg"
    default_cfg_path = f"{cfg_dir}/default.yaml"
    extend_cfg_path = f"{cfg_dir}/coco_{scale}_train.yaml"
    defaults = yaml_load(default_cfg_path)
    extends = yaml_load(extend_cfg_path)
    assert all(k in defaults for k in extends)
    LOGGER.info(f"Extends: {extends}")

    model = YOLOE(args.weights)

    # Ensure pe is set for classes from custom dataset
    names = list(yaml_load(args.data)["names"].values())
    tpe = model.get_text_pe(names)
    pe_path = "custom-pe.pt"
    torch.save({"names": names, "pe": tpe}, pe_path)

    # Linear probing-style freeze similar to the original script
    head_index = len(model.model.model) - 1
    freeze = [str(f) for f in range(0, head_index)]
    for name, child in model.model.model[-1].named_children():
        if 'cv3' not in name:
            freeze.append(f"{head_index}.{name}")
    freeze.extend([
        f"{head_index}.cv3.0.0", f"{head_index}.cv3.0.1",
        f"{head_index}.cv3.1.0", f"{head_index}.cv3.1.1",
        f"{head_index}.cv3.2.0", f"{head_index}.cv3.2.1",
    ])

    model.train(
        data=args.data,
        epochs=args.epochs,
        close_mosaic=5,
        batch=args.batch,
        optimizer="AdamW",
        lr0=1e-3,
        warmup_bias_lr=0.0,
        weight_decay=0.025,
        momentum=0.9,
        workers=8,
        device=args.device,
        **extends,
        trainer=YOLOEPESegTrainer,
        train_pe_path=pe_path,
        freeze=freeze,  
    )


if __name__ == "__main__":
    main()
