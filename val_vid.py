import argparse
from ultralytics import YOLOE


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOE model on custom video-derived dataset")
    parser.add_argument("--weights", default=r"C:\Users\VALTEC-07\Desktop\yoloe\runs\segment\train6\weights\best.pt", help="Path to model weights (.pt)")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/my_vid.yaml", help="Path to dataset YAML")
    parser.add_argument("--device", default="0", help="CUDA device, e.g., '0' or 'cpu'")
    args = parser.parse_args()

    model = YOLOE(args.weights)
    model.val(data=args.data, device=args.device)


if __name__ == "__main__":
    main()
