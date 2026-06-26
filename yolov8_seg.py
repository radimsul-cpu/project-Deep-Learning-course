import cv2
import numpy as np
from ultralytics import YOLO
from .base import BaseDetector

class YOLOv8SegDetector(BaseDetector):
    """
    Детектор на основе YOLOv8 с поддержкой сегментации.
    Возвращает bounding boxes (как и обычный детектор), но использует модель сегментации.
    """
    def __init__(self, model_path='yolov8n-seg.pt', conf_thresh=0.5):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, image):
        # YOLO ожидает RGB, OpenCV даёт BGR
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(rgb_img, conf=self.conf_thresh, verbose=False)
        
        bboxes = []
        scores = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy.tolist()[0]
                    bboxes.append([int(x1), int(y1), int(x2), int(y2)])
                    scores.append(float(box.conf))
        return bboxes, scores