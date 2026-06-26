import cv2
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

class DeepSortWrapper:
    def __init__(self, detector, reid_model, max_cosine_distance=0.3, nn_budget=200):
        self.detector = detector
        self.reid = reid_model
        
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(metric)

    def update(self, bgr_img):
        bboxes, scores = self.detector.detect(bgr_img)
        # Отладочный вывод убран
        
        crops = []
        for (x1, y1, x2, y2) in bboxes:
            h, w, _ = bgr_img.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 > 0 and y2 - y1 > 0:
                crops.append(bgr_img[y1:y2, x1:x2])
        
        features = self.reid.extract_features(crops) if crops else []
        
        detections = []
        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            if i < len(features):
                detections.append(
                    Detection(bbox, score, features[i])
                )
        
        self.tracker.predict()
        self.tracker.update(detections)
        
        results = []
        for track in self.tracker.tracks:
            # Фильтры временно сняты – можно вернуть после отладки
            bbox = track.to_tlbr()
            results.append({
                'track_id': track.track_id,
                'bbox': bbox,
                'score': 1.0
            })
        return results