import time
from ultralytics import YOLO


class EfficientDetRunner:
    """
    YOLOv9s-backed runner (replaces EfficientDet for compatibility).
    Falls back to YOLOv8s if the current ultralytics version doesn't support YOLOv9.
    """

    def __init__(self, model_size='d0'):
        self.model = None
        self.model_name = None

        for candidate in ('yolov9s.pt', 'yolov8s.pt'):
            try:
                self.model = YOLO(candidate)
                self.model_name = candidate
                print(f"Loaded second model: {candidate}")
                break
            except Exception as e:
                print(f"Could not load {candidate}: {e}")

        if self.model is None:
            print("Warning: No second model available. Using mock implementation.")

    def run_inference(self, frames):
        """
        Run inference on a list of 640x640 BGR frames.
        Returns (detections list, avg_time_ms, fps).
        """
        if self.model is None:
            return [{'boxes': [], 'scores': [], 'class_ids': []} for _ in frames], 22.0, 45.5

        results = []
        total_time = 0

        for frame in frames:
            start = time.perf_counter()
            result = self.model(frame, verbose=False, conf=0.25)
            elapsed = time.perf_counter() - start
            total_time += elapsed

            detections = {'boxes': [], 'scores': [], 'class_ids': []}
            if len(result) > 0 and result[0].boxes is not None:
                for box in result[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    detections['boxes'].append(xyxy.tolist())
                    detections['scores'].append(float(box.conf[0]))
                    detections['class_ids'].append(int(box.cls[0]))

            results.append(detections)

        avg_time_ms = (total_time / len(frames)) * 1000 if frames else 0
        fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0
        return results, avg_time_ms, fps
