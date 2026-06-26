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
from models.reid.resnet_torch import ResNetReID
from models.reid.osnet import OSNetReID
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
    if args.detector == 'yolov8':
        detector = YOLOv8Detector(args.detector_weights)
    elif args.detector == 'yolox':
        detector = YOLOXDetector(args.detector_weights)
    elif args.detector == 'faster_rcnn':
        detector = FasterRCNNDetector()
    else:
        raise ValueError(f"Неизвестный детектор: {args.detector}")

    if args.reid == 'resnet':
        reid_model = ResNetReID(device='cpu')
    elif args.reid == 'osnet':
        reid_model = OSNetReID(device='cpu')
    else:
        raise ValueError(f"Неизвестная REID модель: {args.reid}")

    tracker = DeepSortWrapper(
        detector,
        reid_model,
        max_cosine_distance=args.max_cosine_dist,
        nn_budget=100
    )

    identity_manager = IdentityManager(similarity_threshold=args.reid_threshold)

    seq_name, im_dir, fps, seq_len, width, height, im_ext = read_seqinfo(args.sequence_dir)
    img_dir = os.path.join(args.sequence_dir, im_dir)

    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith(im_ext)])
    frame_paths = [os.path.join(img_dir, f) for f in frame_files]

    output_dir = os.path.join('output', seq_name)
    os.makedirs(output_dir, exist_ok=True)
    out_det_file = os.path.join(output_dir, 'det.txt')

    track_identities = defaultdict(list)
    total_tracks_written = 0

    with open(out_det_file, 'w') as f_out:
        for frame_idx, frame_path in enumerate(frame_paths, start=1):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            tracks = tracker.update(frame)

            # Доп. задание (Standalone REID)
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                desc = reid_model.extract_features([crop])[0]
                identity, _ = identity_manager.search(desc)
                if identity is None:
                    identity = identity_manager.add_to_database(desc)
                track_identities[track_id].append((frame_idx, identity))
                while track_identities[track_id] and \
                      track_identities[track_id][0][0] < frame_idx - identity_manager.window_frames:
                    track_identities[track_id].pop(0)
                identities_in_window = [i for _, i in track_identities[track_id]]
                if identities_in_window:
                    most_common = max(set(identities_in_window), key=identities_in_window.count)
                    cv2.putText(frame, f"ID:{track_id} P:{most_common}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Запись в MOT формат с обрезкой координат
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

            # Визуализация и выход по 'q'
            if not args.skip_viz:
                cv2.imshow('Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

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
                        help='Путь к папке с последовательностью (например, MOT15/train/TUD-Campus)')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Путь к папке с GT (обычно совпадает с sequence_dir). Если указано, посчитает метрики.')
    parser.add_argument('--detector', type=str, default='yolov8', choices=['yolov8', 'yolox', 'faster_rcnn'])
    parser.add_argument('--detector_weights', type=str, default='yolov8n.pt')
    parser.add_argument('--reid', type=str, default='resnet', choices=['resnet', 'osnet'])
    parser.add_argument('--max_cosine_dist', type=float, default=0.2)
    parser.add_argument('--reid_threshold', type=float, default=0.6)
    parser.add_argument('--visualize', action='store_true', help='Показывать видео в реальном времени')
    parser.add_argument('--skip_viz', action='store_true', help='Пропустить показ окна с видео')
    args = parser.parse_args()
    main(args)