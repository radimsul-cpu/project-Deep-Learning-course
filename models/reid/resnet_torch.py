import torch
import torchvision.transforms as T
from torchvision import models
import cv2
import numpy as np
from .base import BaseReID

class ResNetReID(BaseReID):
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        # Используем ResNet50, убираем последний слой (выход 2048)
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Identity()  # Возвращает вектор 2048
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),  # Стандартный размер для REID
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, crops):
        if not crops:
            return np.array([])
        
        batch = []
        for img in crops:
            # OpenCV -> BGR -> RGB (для ToPILImage)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img_rgb).unsqueeze(0)
            batch.append(tensor)
        
        batch = torch.cat(batch, dim=0).to(self.device)
        with torch.no_grad():
            features = self.model(batch).cpu().numpy()
        # Нормализуем признаки (L2)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        return features