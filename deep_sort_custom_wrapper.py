import cv2
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

class DeepSortWrapper:
    def __init__(self, detector, reid_model, max_cosine_distance=0.2, nn_budget=100):
        self.detector = detector
        self.reid = reid_model
        
        # Инициализация метрики (cosine distance)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(metric)

    def update(self, bgr_img):
        # 1. Детекция
        bboxes, scores = self.detector.detect(bgr_img)
        
        # 2. Вырезаем кропы
        crops = []
        for (x1, y1, x2, y2) in bboxes:
            h, w, _ = bgr_img.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 > 0 and y2 - y1 > 0:
                crops.append(bgr_img[y1:y2, x1:x2])
        
        # 3. Извлечение дескрипторов
        features = self.reid.extract_features(crops) if crops else []
        
        # 4. Формируем объекты Detection для DeepSORT
        detections = []
        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            if i < len(features):
                detections.append(
                    Detection(bbox, score, features[i])
                )
        
        # 5. Запуск трекинга
        self.tracker.predict()
        self.tracker.update(detections)
        
        # 6. Возвращаем результат: track_id, bbox, score
        results = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()  # [x1, y1, x2, y2]
            # track_id всегда положительное число (в DeepSORT это гарантировано)
            results.append({
                'track_id': track.track_id,
                'bbox': bbox,
                'score': 1.0
            })
        return results