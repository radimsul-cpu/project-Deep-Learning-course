import os
import cv2
import configparser
import argparse
import numpy as np
import torch
from collections import defaultdict

from models.detectors.yolov8 import YOLOv8Detector
from models.detectors.yolox import YOLOXDetector
from models.detectors.faster_rcnn import FasterRCNNDetector
from models.detectors.yolov8_seg import YOLOv8SegDetector

from models.reid.resnet_torch import ResNetReID
from models.reid.resnet101 import ResNet101ReID
from models.reid.mobilenet import MobileNetReID

from deep_sort_custom_wrapper import DeepSortWrapper
from reid_manager.identity_manager import IdentityManager


def read_seqinfo(seq_root):
    info_path = os.path.join(seq_root, 'seqinfo.ini')
    config = configparser.ConfigParser()
    config.read(info_path)

    seq_name = config['Sequence']['name']
    im_dir = config['Sequence']['imDir']
    frame_rate = int(config['Sequence']['frameRate'])
    seq_len = int(config['Sequence']['seqLength'])
    im_width = int(config['Sequence']['imWidth'])
    im_height = int(config['Sequence']['imHeight'])
    im_ext = config['Sequence']['imExt']

    return seq_name, im_dir, frame_rate, seq_len, im_width, im_height, im_ext


def main(args):
    # ---------- Выбор детектора с передачей conf_thresh ----------
    if args.detector == 'yolov8':
        detector = YOLOv8Detector(args.detector_weights, conf_thresh=args.conf_thresh)
    elif args.detector == 'yolox':
        # Для YOLOX тоже можно добавить conf_thresh, но пока оставляем как есть
        detector = YOLOXDetector(args.detector_weights, conf_thresh=args.conf_thresh)
    elif args.detector == 'faster_rcnn':
        detector = FasterRCNNDetector(conf_thresh=args.conf_thresh)
    elif args.detector == 'yolov8-seg':
        # Для сегментационной модели тоже можно передать порог
        detector = YOLOv8SegDetector(args.detector_weights, conf_thresh=args.conf_thresh)
    else:
        raise ValueError(f"Неизвестный детектор: {args.detector}")

    # ---------- Выбор ReID модели ----------
    if args.reid == 'resnet':
        reid_model = ResNetReID(device='cpu')
    elif args.reid == 'resnet101':
        reid_model = ResNet101ReID(device='cpu')
    elif args.reid == 'mobilenet':
        reid_model = MobileNetReID(device='cpu')
    else:
        raise ValueError(f"Неизвестная REID модель: {args.reid}")

    # ---------- Трекер (с увеличенным nn_budget) ----------
    tracker = DeepSortWrapper(
        detector,
        reid_model,
        max_cosine_distance=args.max_cosine_dist,
        nn_budget=200   # увеличено с 100
    )

    # ---------- IdentityManager с уменьшенным окном (1 секунда) ----------
    seq_name, im_dir, fps, seq_len, width, height, im_ext = read_seqinfo(args.sequence_dir)
    identity_manager = IdentityManager(
        similarity_threshold=args.reid_threshold,
        window_seconds=1.0,   # уменьшено с 2.0
        fps=fps
    )

    img_dir = os.path.join(args.sequence_dir, im_dir)
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith(im_ext)])
    frame_paths = [os.path.join(img_dir, f) for f in frame_files]

    output_dir = os.path.join('output', seq_name)
    os.makedirs(output_dir, exist_ok=True)
    out_det_file = os.path.join(output_dir, 'det.txt')

    # Храним (frame_idx, identity, score) для взвешенного голосования
    track_identities = defaultdict(list)
    total_tracks_written = 0

    with open(out_det_file, 'w') as f_out:
        for frame_idx, frame_path in enumerate(frame_paths, start=1):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            # ---------- Трекинг ----------
            tracks = tracker.update(frame)

            # ---------- Обновление идентичностей ----------
            current_identities = {}   # track_id -> (мажоритарная личность или -1)

            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Извлекаем дескриптор и ищем личность (адаптивный поиск)
                desc = reid_model.extract_features([crop])[0]
                identity, _ = identity_manager.search(desc, adaptive=True)
                if identity is None:
                    identity = identity_manager.add_to_database(desc)

                # Сохраняем (кадр, личность, уверенность детекции)
                track_identities[track_id].append((frame_idx, identity, track['score']))

                # Удаляем записи старше окна
                while track_identities[track_id] and \
                      track_identities[track_id][0][0] < frame_idx - identity_manager.window_frames:
                    track_identities[track_id].pop(0)

                # Взвешенное голосование по сумме уверенностей
                identity_scores = {}
                for _, ident, score in track_identities[track_id]:
                    if ident != -1:
                        identity_scores[ident] = identity_scores.get(ident, 0.0) + score
                if identity_scores:
                    most_common = max(identity_scores, key=identity_scores.get)
                    current_identities[track_id] = most_common
                else:
                    current_identities[track_id] = -1

            # ---------- Улучшенное разрешение конфликтов ----------
            # Построить обратное отображение: личность -> список (track_id, суммарная уверенность)
            identity_to_tracks = {}
            for tid, ident in current_identities.items():
                if ident != -1:
                    total_score = sum(score for _, _, score in track_identities[tid])
                    identity_to_tracks.setdefault(ident, []).append((tid, total_score))

            # Для каждой личности, принадлежащей более чем одному треку
            for ident, track_list in identity_to_tracks.items():
                if len(track_list) > 1:
                    # Находим трек с максимальной суммарной уверенностью
                    best_tid = max(track_list, key=lambda x: x[1])[0]
                    # Сбрасываем у всех остальных
                    for tid, _ in track_list:
                        if tid != best_tid:
                            current_identities[tid] = -1

            # ---------- Визуализация ----------
            if not args.skip_viz:
                for track in tracks:
                    track_id = track['track_id']
                    bbox = track['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    ident = current_identities.get(track_id, -1)
                    if ident != -1:
                        cv2.putText(frame, f"ID:{track_id} P:{ident}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ---------- Запись в MOT формат ----------
            for track in tracks:
                track_id = track['track_id']
                if track_id <= 0:
                    continue
                x1, y1, x2, y2 = map(int, track['bbox'])
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                w, h = x2 - x1, y2 - y1
                f_out.write(f"{frame_idx},{track_id},{x1},{y1},{w},{h},{track['score']:.4f},-1,-1,-1\n")
                total_tracks_written += 1

    print(f"Результаты сохранены в: {out_det_file}")
    print(f"Всего записано {total_tracks_written} строк в det.txt")

    if total_tracks_written > 0:
        with open(out_det_file, 'r') as f:
            lines = f.readlines()
            print("Первые 5 строк det.txt:")
            for i in range(min(5, len(lines))):
                print(lines[i].strip())
    else:
        print("ВНИМАНИЕ: det.txt пуст! Трекер не создал ни одного подтверждённого трека.")

    cv2.destroyAllWindows()

    if args.gt_dir:
        from utils.evaluator import evaluate_metrics
        gt_path = os.path.join(args.gt_dir, 'gt', 'gt.txt')
        evaluate_metrics(gt_path, out_det_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_dir', type=str, required=True,
                        help='Путь к папке с последовательностью')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Путь к папке с GT (для оценки)')
    parser.add_argument('--detector', type=str, default='yolov8',
                        choices=['yolov8', 'yolox', 'faster_rcnn', 'yolov8-seg'])
    parser.add_argument('--detector_weights', type=str, default='yolov8n.pt')
    parser.add_argument('--conf_thresh', type=float, default=0.6,
                        help='Порог уверенности детектора')
    parser.add_argument('--reid', type=str, default='resnet',
                        choices=['resnet', 'resnet101', 'mobilenet'])
    parser.add_argument('--max_cosine_dist', type=float, default=0.2)
    parser.add_argument('--reid_threshold', type=float, default=0.6)
    parser.add_argument('--skip_viz', action='store_true', help='Пропустить показ окна')
    parser.add_argument('--original', action='store_true',
                        help='Запустить оригинальный DeepSORT (без улучшений)')
    args = parser.parse_args()
    main(args)