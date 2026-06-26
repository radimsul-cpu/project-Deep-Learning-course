import numpy as np
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> tuple[list, list]:
        """
        Принимает BGR изображение (np.array).
        Возвращает: (list_of_bboxes, list_of_scores)
        bbox формат: [x1, y1, x2, y2] (абсолютные координаты)
        """
        pass