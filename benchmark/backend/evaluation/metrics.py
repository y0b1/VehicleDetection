import math
import numpy as np
from typing import List, Dict, Tuple, Optional

# COCO vehicle class IDs: bicycle, car, motorcycle, bus, train, truck
VEHICLE_CLASS_IDS = frozenset({1, 2, 3, 5, 6, 7})


def filter_to_vehicles(detections: Dict) -> Dict:
    """Keep only COCO vehicle-class detections (IDs 1,2,3,5,6,7)."""
    out: Dict = {'boxes': [], 'scores': [], 'class_ids': []}
    for box, score, cls in zip(
        detections.get('boxes', []),
        detections.get('scores', []),
        detections.get('class_ids', []),
    ):
        if cls in VEHICLE_CLASS_IDS:
            out['boxes'].append(box)
            out['scores'].append(score)
            out['class_ids'].append(cls)
    return out


class MetricsCalculator:
    """Calculate detection metrics (precision, recall, F1, proper mAP, temporal consistency)."""

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 < x1 or y2 < y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def calculate_precision_recall(
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> Tuple[float, float, float]:
        """
        Class-aware precision, recall, F1.
        A prediction is a TP only if IoU >= threshold AND class_id matches GT.
        """
        tp = fp = fn = 0

        for frame_pred, frame_gt in zip(predictions, ground_truth):
            pred_boxes   = frame_pred.get('boxes', [])
            pred_classes = frame_pred.get('class_ids', [])
            gt_boxes     = frame_gt.get('boxes', [])
            gt_classes   = frame_gt.get('class_ids', [])

            matched_gt: set = set()

            for pred_box, pred_cls in zip(pred_boxes, pred_classes):
                best_iou = 0.0
                best_j   = -1
                for j, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                    if j in matched_gt or gt_cls != pred_cls:
                        continue
                    iou = MetricsCalculator.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_j   = j

                if best_iou >= iou_threshold and best_j >= 0:
                    tp += 1
                    matched_gt.add(best_j)
                else:
                    fp += 1

            fn += len(gt_boxes) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    @staticmethod
    def calculate_ap(
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> float:
        """
        Proper Average Precision: sort all detections by confidence descending,
        walk the ranked list computing cumulative TP/FP, integrate AUC of the
        precision-recall curve (PASCAL VOC style).
        Class-aware: requires class_id match for TP.
        """
        # Flatten all detections with their frame index
        all_dets: List[Tuple] = []
        for frame_idx, pred in enumerate(predictions):
            for box, score, cls in zip(
                pred.get('boxes', []),
                pred.get('scores', []),
                pred.get('class_ids', []),
            ):
                all_dets.append((score, frame_idx, box, cls))

        total_gt = sum(len(gt.get('boxes', [])) for gt in ground_truth)
        if total_gt == 0 or not all_dets:
            return 0.0

        # Sort by confidence descending
        all_dets.sort(key=lambda x: x[0], reverse=True)

        # Per-frame set of matched GT indices
        matched_gt = [set() for _ in ground_truth]

        tps = np.zeros(len(all_dets))
        fps = np.zeros(len(all_dets))

        for i, (score, frame_idx, box, cls) in enumerate(all_dets):
            gt      = ground_truth[frame_idx]
            gt_boxes   = gt.get('boxes', [])
            gt_classes = gt.get('class_ids', [])

            best_iou = 0.0
            best_j   = -1
            for j, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                if j in matched_gt[frame_idx] or gt_cls != cls:
                    continue
                iou = MetricsCalculator.calculate_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j   = j

            if best_iou >= iou_threshold and best_j >= 0:
                tps[i] = 1
                matched_gt[frame_idx].add(best_j)
            else:
                fps[i] = 1

        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)
        recalls    = cum_tp / total_gt
        precisions = cum_tp / (cum_tp + cum_fp)

        # Prepend (recall=0, precision=1) start point for proper AUC
        recalls    = np.concatenate([[0.0], recalls])
        precisions = np.concatenate([[1.0], precisions])

        ap = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))
        return min(ap, 1.0)

    @staticmethod
    def calculate_map5095(
        predictions: List[Dict],
        ground_truth: List[Dict],
    ) -> float:
        """COCO-style mAP@[0.5:0.95] — average proper AP over 10 IoU thresholds."""
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        aps = [
            MetricsCalculator.calculate_ap(predictions, ground_truth, t)
            for t in thresholds
        ]
        return float(np.mean(aps))

    @staticmethod
    def calculate_temporal_consistency(predictions: List[Dict]) -> float:
        """
        Frame-to-frame detection stability for video sequences.

        For each consecutive frame pair, greedily match vehicle detections by
        IoU >= 0.5 (class-aware). Returns mean IoU of all matched pairs.

        Interpretation:
          1.0 = boxes are perfectly stable across frames
          0.0 = matched boxes exist but have minimal overlap
          nan = single image or no vehicle detections found
        """
        if len(predictions) < 2:
            return float('nan')

        iou_scores: List[float] = []

        for prev, curr in zip(predictions[:-1], predictions[1:]):
            prev_boxes   = prev.get('boxes', [])
            prev_classes = prev.get('class_ids', [])
            curr_boxes   = curr.get('boxes', [])
            curr_classes = curr.get('class_ids', [])

            if not prev_boxes or not curr_boxes:
                continue

            matched_curr: set = set()
            for pb, pc in zip(prev_boxes, prev_classes):
                best_iou = 0.0
                best_j   = -1
                for j, (cb, cc) in enumerate(zip(curr_boxes, curr_classes)):
                    if j in matched_curr or cc != pc:
                        continue
                    iou = MetricsCalculator.calculate_iou(pb, cb)
                    if iou > best_iou:
                        best_iou = iou
                        best_j   = j
                if best_j >= 0 and best_iou >= 0.5:
                    iou_scores.append(best_iou)
                    matched_curr.add(best_j)

        return float(np.mean(iou_scores)) if iou_scores else float('nan')


def calculate_metrics(
    predictions: List[Dict],
    ground_truth: Optional[List[Dict]] = None,
) -> Dict:
    """
    Calculate all evaluation metrics.

    Filters both predictions and GT to COCO vehicle classes before computing.
    Temporal consistency is included only for multi-frame (video) inputs.
    """
    preds_v = [filter_to_vehicles(p) for p in predictions]

    if ground_truth is None:
        ground_truth = [{
            'boxes': [[10, 10, 100, 100], [150, 150, 300, 300]],
            'scores': [1.0, 1.0],
            'class_ids': [2, 7],  # car, truck
        } for _ in predictions]

    gt_v = [filter_to_vehicles(g) for g in ground_truth]

    calc = MetricsCalculator()

    prec, rec, f1 = calc.calculate_precision_recall(preds_v, gt_v, iou_threshold=0.5)
    map50          = calc.calculate_ap(preds_v, gt_v, iou_threshold=0.5)
    map5095        = calc.calculate_map5095(preds_v, gt_v)
    temporal       = calc.calculate_temporal_consistency(preds_v)

    result = {
        'precision': round(prec,    4),
        'recall':    round(rec,     4),
        'f1':        round(f1,      4),
        'mAP50':     round(map50,   4),
        'mAP5095':   round(map5095, 4),
    }

    if not math.isnan(temporal):
        result['temporal_consistency'] = round(temporal, 4)

    return result
