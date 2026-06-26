import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2
from .base import BaseDetector

class FasterRCNNDetector(BaseDetector):
    def __init__(self, device='cuda', conf_thresh=0.5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_thresh = conf_thresh

        # Загружаем предобученный Faster R-CNN (ResNet50 FPN) из torchvision
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image):
        # OpenCV выдаёт BGR, а PyTorch требует RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = F.to_tensor(rgb_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)[0]

        bboxes = []
        scores = []
        h, w, _ = image.shape

        for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
            if score >= self.conf_thresh:
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                # Обрезаем координаты до границ кадра
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                bboxes.append([x1, y1, x2, y2])
                scores.append(float(score))

        return bboxes, scores