import numpy as np
from abc import ABC, abstractmethod

class BaseReID(ABC):
    @abstractmethod
    def extract_features(self, crops: list[np.ndarray]) -> np.ndarray:
        """
        Принимает список вырезанных изображений (crops) в формате BGR (np.array).
        Возвращает: np.array размера (N, feature_dim) с дескрипторами.
        """
        pass