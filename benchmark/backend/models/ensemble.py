import numpy as np
from ensemble_boxes import weighted_boxes_fusion, nms

IMG_SIZE = 640  # fixed frame dimension used by all runners


def _normalize(boxes, img_size=IMG_SIZE):
    """Convert absolute pixel boxes [x1,y1,x2,y2] to [0,1] range."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.array(boxes, dtype=np.float32) / img_size
    return np.clip(arr, 0.0, 1.0)


def _denormalize(boxes, img_size=IMG_SIZE):
    """Convert [0,1] boxes back to absolute pixel coordinates."""
    if len(boxes) == 0:
        return []
    return (np.array(boxes, dtype=np.float32) * img_size).tolist()


class EnsembleRunner:
    """Ensemble methods for combining detections from multiple models."""

    @staticmethod
    def nms_ensemble(detections_list, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression across detections from multiple models.

        Args:
            detections_list: List of detection dicts, one per model
            iou_threshold: IoU threshold for NMS

        Returns:
            Combined detection dict
        """
        boxes_list = []
        scores_list = []
        labels_list = []

        for det in detections_list:
            boxes = det.get('boxes', [])
            scores = det.get('scores', [])
            class_ids = det.get('class_ids', [])

            if len(boxes) > 0:
                boxes_list.append(_normalize(boxes))
                scores_list.append(np.array(scores, dtype=np.float32))
                labels_list.append(np.array(class_ids, dtype=np.float32))
            else:
                boxes_list.append(np.zeros((0, 4), dtype=np.float32))
                scores_list.append(np.array([], dtype=np.float32))
                labels_list.append(np.array([], dtype=np.float32))

        try:
            out_boxes, out_scores, out_labels = nms(
                boxes_list, scores_list, labels_list,
                weights=None, iou_thr=iou_threshold
            )
            return {
                'boxes': _denormalize(out_boxes),
                'scores': out_scores.tolist() if len(out_scores) > 0 else [],
                'class_ids': out_labels.astype(int).tolist() if len(out_labels) > 0 else [],
            }
        except Exception as e:
            print(f"NMS ensemble error: {e}")
            return {'boxes': [], 'scores': [], 'class_ids': []}

    @staticmethod
    def wbf_ensemble(detections_list, weights=None):
        """
        Apply Weighted Boxes Fusion across detections from multiple models.

        Args:
            detections_list: List of detection dicts, one per model
            weights: Per-model weights (default: equal)

        Returns:
            Combined detection dict
        """
        if weights is None:
            weights = [1.0] * len(detections_list)

        boxes_list = []
        scores_list = []
        labels_list = []

        for det in detections_list:
            boxes = det.get('boxes', [])
            scores = det.get('scores', [])
            class_ids = det.get('class_ids', [])

            if len(boxes) > 0:
                boxes_list.append(_normalize(boxes))
                scores_list.append(np.array(scores, dtype=np.float32))
                labels_list.append(np.array(class_ids, dtype=np.float32))
            else:
                boxes_list.append(np.zeros((0, 4), dtype=np.float32))
                scores_list.append(np.array([], dtype=np.float32))
                labels_list.append(np.array([], dtype=np.float32))

        try:
            out_boxes, out_scores, out_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=weights, iou_thr=0.55, skip_box_thr=0.0
            )
            return {
                'boxes': _denormalize(out_boxes),
                'scores': out_scores.tolist() if len(out_scores) > 0 else [],
                'class_ids': out_labels.astype(int).tolist() if len(out_labels) > 0 else [],
            }
        except Exception as e:
            print(f"WBF ensemble error: {e}")
            return {'boxes': [], 'scores': [], 'class_ids': []}
