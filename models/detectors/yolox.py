import cv2
import torch
import numpy as np
from yolox.exp import get_exp
from yolox.data.data_augment import preproc as yolox_preproc
from yolox.utils import postprocess
from .base import BaseDetector

class YOLOXDetector(BaseDetector):
    def __init__(self, model_path='yolox_s.pth', device='cpu', conf_thresh=0.5, input_size=(640, 640), exp_name='yolox-s'):
        self.device = torch.device(device)
        self.conf_thresh = conf_thresh
        self.input_size = input_size
        self.num_classes = 80

        exp = get_exp(None, exp_name)
        model = exp.get_model()
        ckpt = torch.load(model_path, map_location=self.device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        self.model = model.to(self.device)
        self.model.eval()

    def detect(self, image):
        img_h, img_w, _ = image.shape

        img, ratio = yolox_preproc(image, self.input_size)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = postprocess(outputs, self.num_classes, self.conf_thresh, nms_thre=0.45)

        bboxes = []
        scores = []
        if outputs[0] is not None:
            bboxes_raw = outputs[0][:, 0:4] / ratio
            scores_raw = outputs[0][:, 4]

            for bbox, score in zip(bboxes_raw, scores_raw):
                x1, y1, x2, y2 = bbox.tolist()
                # Обрезка по границам кадра
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                bboxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(float(score))

        return bboxes, scores