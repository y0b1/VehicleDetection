import time
import numpy as np
from ultralytics import YOLO
import cv2


class YOLOv8Runner:
    def __init__(self, model_size='n'):
        """Initialize YOLOv8 model with auto device detection."""
        try:
            self.model = YOLO(f'yolov8{model_size}.pt')
            self.model_size = model_size
        except Exception as e:
            print(f"Failed to load YOLOv8: {e}")
            self.model = None

    def run_inference(self, frames):
        """
        Run inference on a batch of frames.
        
        Args:
            frames: List of numpy arrays (BGR images, 640x640)
            
        Returns:
            List of detections per frame: [{'boxes': [], 'scores': [], 'class_ids': []}]
        """
        if self.model is None:
            return [{'boxes': [], 'scores': [], 'class_ids': []} for _ in frames]
        
        results = []
        total_time = 0
        
        for frame in frames:
            start = time.perf_counter()
            result = self.model(frame, verbose=False, conf=0.25)
            elapsed = time.perf_counter() - start
            total_time += elapsed
            
            # Extract detections
            detections = {
                'boxes': [],
                'scores': [],
                'class_ids': []
            }
            
            if len(result) > 0 and result[0].boxes is not None:
                boxes = result[0].boxes
                for box in boxes:
                    # box.xyxy returns [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    detections['boxes'].append(xyxy.tolist())
                    detections['scores'].append(conf)
                    detections['class_ids'].append(cls_id)
            
            results.append(detections)
        
        avg_time_ms = (total_time / len(frames)) * 1000 if frames else 0
        fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0
        
        return results, avg_time_ms, fps
