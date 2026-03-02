import numpy as np
from typing import List, Dict, Tuple


class MetricsCalculator:
    """Calculate detection metrics (precision, recall, F1, mAP)."""

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    @staticmethod
    def calculate_precision_recall(
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score.

        Args:
            predictions: List of detection dicts with 'boxes', 'scores', 'class_ids'
            ground_truth: List of gt dicts with same format (one dict per frame)
            iou_threshold: IoU threshold for match

        Returns:
            Tuple of (precision, recall, f1_score)
        """
        tp = 0
        fp = 0
        fn = 0

        for frame_pred, frame_gt in zip(predictions, ground_truth):
            pred_boxes = frame_pred.get('boxes', [])
            gt_boxes = frame_gt.get('boxes', [])

            matched_gt = set()

            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1

                for j, gt_box in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    iou = MetricsCalculator.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn += len(gt_boxes) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    @staticmethod
    def calculate_ap(
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5
    ) -> float:
        """
        Calculate Average Precision (simplified).

        Args:
            predictions: List of detection dicts
            ground_truth: List of gt dicts (one per frame)
            iou_threshold: IoU threshold

        Returns:
            AP score between 0-1
        """
        total_gt = sum(len(gt.get('boxes', [])) for gt in ground_truth)

        if total_gt == 0:
            return 0.0

        tp_count = 0
        total_pred = sum(len(pred.get('boxes', [])) for pred in predictions)

        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = pred.get('boxes', [])
            gt_boxes = gt.get('boxes', [])

            matched_gt = set()
            for pred_box in pred_boxes:
                for k, gt_box in enumerate(gt_boxes):
                    if k in matched_gt:
                        continue
                    if MetricsCalculator.calculate_iou(pred_box, gt_box) >= iou_threshold:
                        tp_count += 1
                        matched_gt.add(k)
                        break

        ap = tp_count / max(total_pred, total_gt) if max(total_pred, total_gt) > 0 else 0.0
        return min(ap, 1.0)

    @staticmethod
    def calculate_map5095(
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> float:
        """
        Calculate COCO-style mAP@[0.5:0.95] — average AP over 10 IoU thresholds.
        """
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        aps = [
            MetricsCalculator.calculate_ap(predictions, ground_truth, iou_threshold=t)
            for t in thresholds
        ]
        return float(np.mean(aps))


def calculate_metrics(predictions: List[Dict], ground_truth: List[Dict] = None) -> Dict:
    """
    Calculate all evaluation metrics.

    Args:
        predictions: List of detection dicts from model
        ground_truth: List of gt dicts (optional, uses mock if None)

    Returns:
        Dict with all metrics
    """
    if ground_truth is None:
        ground_truth = [{
            'boxes': [[10, 10, 100, 100], [150, 150, 300, 300]],
            'scores': [1.0, 1.0],
            'class_ids': [2, 6]
        } for _ in predictions]

    calc = MetricsCalculator()

    prec, rec, f1 = calc.calculate_precision_recall(predictions, ground_truth, iou_threshold=0.5)
    map50 = calc.calculate_ap(predictions, ground_truth, iou_threshold=0.5)
    map5095 = calc.calculate_map5095(predictions, ground_truth)

    return {
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'mAP50': round(map50, 4),
        'mAP5095': round(map5095, 4),
    }
