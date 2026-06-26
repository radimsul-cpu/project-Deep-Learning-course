import torchreid
import cv2
import torch
import numpy as np
from .base import BaseReID

class OSNetReID(BaseReID):
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # Загружаем предобученную модель OSNet (x1_0 - базовая версия)
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,  # Это значение не важно, так как мы берем фичи
            pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Трансформации для OSNet
        self.transform = torchreid.transforms.Compose([
            torchreid.transforms.Resize((256, 128)),
            torchreid.transforms.ToTensor(),
            torchreid.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, crops):
        if not crops:
            return np.array([])
        batch = []
        for img in crops:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img_rgb).unsqueeze(0)
            batch.append(tensor)
        batch = torch.cat(batch, dim=0).to(self.device)
        with torch.no_grad():
            features = self.model(batch).cpu().numpy()
        # L2-нормализация (важно для cosine distance)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        return features