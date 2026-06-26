import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

class IdentityManager:
    def __init__(self, max_history=100, similarity_threshold=0.6, window_seconds=1.0, fps=30):
        # База данных: { identity_id: list_of_descriptors }
        self.database = defaultdict(list)
        # Центроиды кластеров (для быстрого поиска)
        self.centroids = {}
        self.next_id = 1
        
        # Параметры
        self.sim_threshold = similarity_threshold
        self.max_history = max_history
        self.window_frames = int(window_seconds * fps)  # теперь fps передаётся извне
        self.fps = fps

    def search(self, query_desc, adaptive=True):
        """
        Поиск наиболее похожей личности в базе.
        Если adaptive=True, порог вычисляется адаптивно.
        Возвращает: (identity_id, confidence) или (None, 0)
        """
        if not self.centroids:
            return None, 0.0
            
        centroids = list(self.centroids.values())
        ids = list(self.centroids.keys())
        
        # Берём до 5 ближайших соседей для адаптивного порога
        n_neighbors = min(5, len(centroids))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        knn.fit(centroids)
        distances, indices = knn.kneighbors(query_desc.reshape(1, -1))
        
        if adaptive and n_neighbors > 1:
            # Адаптивный порог: среднее + 1.5 * стандартное отклонение
            threshold = np.mean(distances[0]) + 1.5 * np.std(distances[0])
        else:
            # Фиксированный порог (преобразуем similarity_threshold в расстояние)
            threshold = 1 - self.sim_threshold
        
        best_dist = distances[0][0]
        if best_dist <= threshold:
            best_id = ids[indices[0][0]]
            return best_id, 1 - best_dist
        return None, 0.0

    def add_to_database(self, desc, identity_id=None):
        """Добавляет дескриптор в базу. Если identity_id None - создаёт новую личность."""
        if identity_id is None:
            identity_id = self.next_id
            self.next_id += 1
        
        self.database[identity_id].append(desc)
        if len(self.database[identity_id]) > self.max_history:
            self.database[identity_id] = self.database[identity_id][-self.max_history:]
        
        # Обновляем центроид (среднее всех дескрипторов)
        self.centroids[identity_id] = np.mean(self.database[identity_id], axis=0)
        self.centroids[identity_id] /= np.linalg.norm(self.centroids[identity_id])
        
        return identity_id