import argparse
from pathlib import Path
import json
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from ultralytics.utils import yaml_load

# python scripts/eval_fewshot_vp.py --mode vp-seg --weight best.pt --seg_weights pretrain/yoloe-v8l-seg.pt --source "c:\Users\VALTEC-07\Desktop\yoloe\prompt.jpg" --target "c:\Users\VALTEC-07\Desktop\yoloe\target.mp4" --bbox 486,7,551,65 --class_id 0 --ann "c:\Users\VALTEC-07\Desktop\yoloe\dataset\train\annotations\annotations.json" --video_id Backpack_0 --ann_index 0 --vid_stride 1 --conf 0.001 --save --eval_out "runs/eval/stiou_target.json"
def parse_args():
    p = argparse.ArgumentParser(description='YOLOE Visual Prompt Evaluation (ST-IoU)')
    p.add_argument('--weight', type=str, dest='weights', help='Alias of --weights')
    p.add_argument('--data', type=str, default=str(Path('ultralytics') / 'cfg' / 'datasets' / 'fewshot.yaml'))
    p.add_argument('--source', type=str, required=True, help='Source image path for prompts')
    p.add_argument('--target', type=str, required=True, help='Target image or video path to evaluate on')
    p.add_argument('--bbox', type=str, default=None, help='Single bbox as x1,y1,x2,y2 if no prompts_json provided')
    p.add_argument('--class_id', type=int, default=0, help='Class id for the bbox when using --bbox')
    p.add_argument('--prompts_json', type=str, default=None, help='JSON file with prompts dict: {"bboxes": [..], "cls": [..]}')
    p.add_argument('--save', action='store_true', help='Save output visuals')
    p.add_argument('--seg_weights', type=str, default=str(Path('pretrain') / 'yoloe-v8l-seg.pt'), help='Segmentation checkpoint for VP if detection weights are provided')
    p.add_argument('--mode', type=str, default='vp-seg', choices=['vp-seg', 'detect'])
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--vid_stride', type=int, default=1)
    p.add_argument('--ann', type=str, required=True, help='Path to annotations.json')
    p.add_argument('--video_id', type=str, default=None, help='Video ID to select from annotations (defaults to target stem)')
    p.add_argument('--ann_index', type=int, default=0, help='Index of annotation track within the video annotations list')
    p.add_argument('--eval_out', type=str, default=None, help='Optional path to save evaluation metrics as JSON')
    p.add_argument('--frame0', type=int, default=0, help='Absolute frame index corresponding to predicted frame i=0')
    return p.parse_args()


def iou_xyxy(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def load_gt_track(ann_path: str, video_id: str | None, ann_index: int, target_stem: str | None = None):
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('annotations.json must be a list of video entries')
    if video_id is None:
        raise ValueError('video_id must not be None')
    entry = None
    for item in data:
        if item.get('video_id') == video_id:
            entry = item
            break
    if entry is None and target_stem is not None:
        def norm(s: str) -> str:
            s = s.lower()
            return ''.join(ch for ch in s if ch.isalnum())
        q = norm(target_stem)
        candidates = []
        for item in data:
            vid = str(item.get('video_id'))
            v = norm(vid)
            if q in v or v in q:
                candidates.append(item)
        if len(candidates) == 1:
            entry = candidates[0]
            print(f"Resolved video_id to '{entry.get('video_id')}' by fuzzy match")
        elif len(candidates) > 1:
            entry = candidates[0]
            print("Multiple candidates found, selecting the first:", [c.get('video_id') for c in candidates[:5]])
        else:
            all_ids = [str(item.get('video_id')) for item in data]
            preview = all_ids[:10]
            raise ValueError(f"video_id {video_id} not found in annotations. Available examples: {preview}")
    anns = entry.get('annotations', [])
    if not anns:
        raise ValueError(f'No annotations found for video_id {video_id}')
    if ann_index < 0 or ann_index >= len(anns):
        raise IndexError(f'ann_index {ann_index} out of range [0, {len(anns)-1}]')
    bboxes = anns[ann_index].get('bboxes', [])
    frames = [bb['frame'] for bb in bboxes]
    if not frames:
        return {}, entry.get('video_id'), 0
    min_f = min(frames)
    gt = {}
    for bb in bboxes:
        f = int(bb['frame'] - min_f)
        gt[f] = np.array([bb['x1'], bb['y1'], bb['x2'], bb['y2']], dtype=float)
    return gt, entry.get('video_id'), int(min_f)


def build_prompts(args):
    if args.prompts_json:
        with open(args.prompts_json, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        prompts['bboxes'] = [np.array(b) for b in prompts['bboxes']]
        prompts['cls'] = [np.array(c) for c in prompts['cls']]
    else:
        if args.bbox is None:
            raise ValueError('Provide --bbox or --prompts_json')
        x1, y1, x2, y2 = map(float, args.bbox.split(','))
        x1, x2 = (x1, x2) if x2 > x1 else (min(x1, x2), max(x1, x2))
        y1, y2 = (y1, y2) if y2 > y1 else (min(y1, y2), max(y1, y2))
        prompts = dict(
            bboxes=[np.array([[x1, y1, x2, y2]])],
            cls=[np.array([args.class_id])]
        )
    return prompts


def main():
    args = parse_args()

    if not args.weights:
        raise SystemExit('Please provide --weight <weights.pt>')

    model = YOLOE(args.weights)

    prompts = None
    if args.mode == 'vp-seg':
        prompts = build_prompts(args)

    names_dict = yaml_load(args.data).get('names')
    if isinstance(names_dict, dict):
        names = [names_dict[k] for k in sorted(names_dict.keys(), key=lambda x: int(x))]
    else:
        names = list(names_dict)

    if args.mode == 'detect':
        results = model.predict(args.target, save=args.save, conf=args.conf, vid_stride=args.vid_stride)
        print(f'Evaluate for detect mode is not implemented with ST-IoU. Results frames: {len(results)}')
        return

    try:
        _ = model.model.model[-1].savpe  # type: ignore[attr-defined]
    except AttributeError:
        model = YOLOE(args.seg_weights)

    model.predict(
        args.source,
        save=args.save,
        prompts=prompts,
        predictor=YOLOEVPSegPredictor,
        return_vpe=True,
        conf=args.conf,
    )

    num_vpe = model.predictor.vpe.shape[1]
    if args.prompts_json:
        try:
            cls_idx = list(np.array(prompts['cls'][0]).astype(int).tolist())
        except Exception:
            cls_idx = list(range(num_vpe))
    else:
        cls_idx = [int(args.class_id)] if num_vpe == 1 else list(range(num_vpe))
    names_vp = [names[i] if isinstance(names, list) and 0 <= i < len(names) else f"object{i}" for i in cls_idx]
    model.set_classes(names_vp, model.predictor.vpe)
    model.predictor = None

    results = model.predict(args.target, save=args.save, conf=args.conf, vid_stride=args.vid_stride)

    if args.prompts_json:
        desired_orig_cls = int(np.array(prompts['cls'][0]).flatten()[0])
        try:
            target_pos = cls_idx.index(desired_orig_cls)
        except ValueError:
            target_pos = 0
    else:
        target_pos = 0

    pred_map = {}
    for i, r in enumerate(results):
        if r.boxes is None or r.boxes.xyxy is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=int)
        conf = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.zeros((xyxy.shape[0],), dtype=float)
        m = np.where(cls == target_pos)[0]
        if m.size == 0:
            continue
        k = int(m[np.argmax(conf[m])])
        frame_idx = (args.frame0 - min_f) + i * max(1, int(args.vid_stride))
        pred_map[frame_idx] = xyxy[k]

    video_id = args.video_id or Path(args.target).stem
    gt_map, resolved_vid, min_f = load_gt_track(args.ann, video_id, args.ann_index, Path(args.target).stem)

    frames_union = set(pred_map.keys()) | set(gt_map.keys())
    frames_inter = set(pred_map.keys()) & set(gt_map.keys())

    sum_iou = 0.0
    for f in frames_inter:
        sum_iou += iou_xyxy(pred_map[f], gt_map[f])

    stiou = (sum_iou / float(len(frames_union))) if len(frames_union) > 0 else 0.0
    miou_intersection = (sum_iou / float(len(frames_inter))) if len(frames_inter) > 0 else 0.0

    metrics = {
        'video_id': resolved_vid,
        'ann_index': args.ann_index,
        'frames_union': len(frames_union),
        'frames_intersection': len(frames_inter),
        'ST_IoU': stiou,
        'mean_IoU_on_intersection': miou_intersection,
        'gt_min_frame': min_f,
        'pred_frame0': args.frame0,
    }

    print(json.dumps(metrics, indent=2))

    if args.eval_out:
        out_path = Path(args.eval_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
