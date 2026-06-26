import os
import numpy as np

if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

from trackeval import Evaluator, metrics
from trackeval.datasets._base_dataset import _BaseDataset

class CustomDirectDataset(_BaseDataset):
    def __init__(self, seq_name, gt_data, tracker_data, seq_length):
        super().__init__()
        self.seq_list = [seq_name]
        self.class_list = ['pedestrian']
        self.tracker_list = ['my_tracker']
        self.gt_data = gt_data
        self.tracker_data = tracker_data
        self.seq_lengths = {seq_name: seq_length}
        self.output_fol = ''

    def get_name(self):
        return 'CustomDataset'

    def get_display_name(self, tracker):
        return tracker

    def get_output_fol(self, tracker):
        return self.output_fol

    def _load_raw_file(self, tracker, seq, is_gt):
        data = self.gt_data[seq] if is_gt else self.tracker_data[tracker][seq]

        ids_list = []
        bboxes_list = []
        for f in range(1, self.seq_lengths[seq] + 1):
            if f in data and len(data[f]) > 0:
                if not is_gt:
                    mask = data[f][:, 0] != -1
                    if np.any(mask):
                        ids_list.append(data[f][mask, 0].astype(int))
                        bboxes_list.append(data[f][mask, 1:5])
                    else:
                        ids_list.append(np.empty(0, dtype=int))
                        bboxes_list.append(np.empty((0, 4)))
                else:
                    ids_list.append(data[f][:, 0].astype(int))
                    bboxes_list.append(data[f][:, 1:5])
            else:
                ids_list.append(np.empty(0, dtype=int))
                bboxes_list.append(np.empty((0, 4)))

        classes_list = [np.ones_like(ids) for ids in ids_list]
        dets_list = [np.ones_like(ids) for ids in ids_list]
        zero_marked_list = [np.zeros_like(ids) for ids in ids_list]

        return {
            'ids': ids_list,
            'bboxes': bboxes_list,
            'classes': classes_list,
            'dets': dets_list,
            'zero_marked': zero_marked_list
        }

    def get_raw_seq_data(self, tracker, seq):
        gt_raw = self._load_raw_file(tracker, seq, is_gt=True)
        tracker_raw = self._load_raw_file(tracker, seq, is_gt=False)

        # Перемаппинг ID для GT
        gt_ids_mapped = []
        gt_id_map = {}
        next_gt_id = 0
        for ids_frame in gt_raw['ids']:
            mapped_ids = np.zeros_like(ids_frame)
            for i, old_id in enumerate(ids_frame):
                if old_id not in gt_id_map:
                    gt_id_map[old_id] = next_gt_id
                    next_gt_id += 1
                mapped_ids[i] = gt_id_map[old_id]
            gt_ids_mapped.append(mapped_ids)

        # Перемаппинг ID для трекера
        tracker_ids_mapped = []
        tracker_id_map = {}
        next_tracker_id = 0
        for ids_frame in tracker_raw['ids']:
            mapped_ids = np.zeros_like(ids_frame)
            for i, old_id in enumerate(ids_frame):
                if old_id not in tracker_id_map:
                    tracker_id_map[old_id] = next_tracker_id
                    next_tracker_id += 1
                mapped_ids[i] = tracker_id_map[old_id]
            tracker_ids_mapped.append(mapped_ids)

        raw_data = {}
        raw_data['gt_ids'] = gt_ids_mapped
        raw_data['gt_bboxes'] = gt_raw['bboxes']
        raw_data['gt_classes'] = gt_raw['classes']
        raw_data['gt_dets'] = gt_raw['dets']
        raw_data['gt_zero_marked'] = gt_raw['zero_marked']
        raw_data['tracker_ids'] = tracker_ids_mapped
        raw_data['tracker_bboxes'] = tracker_raw['bboxes']
        raw_data['tracker_classes'] = tracker_raw['classes']
        raw_data['tracker_dets'] = tracker_raw['dets']
        raw_data['tracker_zero_marked'] = tracker_raw['zero_marked']

        raw_data['num_gt_dets'] = sum(len(d) for d in raw_data['gt_dets'])
        raw_data['num_tracker_dets'] = sum(len(d) for d in raw_data['tracker_dets'])
        raw_data['num_gt_ids'] = len(gt_id_map)
        raw_data['num_tracker_ids'] = len(tracker_id_map)
        raw_data['num_timesteps'] = self.seq_lengths[seq]

        similarity_scores = []
        for t in range(self.seq_lengths[seq]):
            gt_bboxes = raw_data['gt_bboxes'][t]
            tracker_bboxes = raw_data['tracker_bboxes'][t]
            sim = self._calculate_similarities(gt_bboxes, tracker_bboxes)
            similarity_scores.append(sim)
        raw_data['similarity_scores'] = similarity_scores

        return raw_data

    def get_preprocessed_seq_data(self, raw_seq_data, cls):
        return raw_seq_data

    def _calculate_similarities(self, gt_bboxes, tracker_bboxes, similarity_type='iou'):
        if len(gt_bboxes) == 0 or len(tracker_bboxes) == 0:
            return np.empty((len(gt_bboxes), len(tracker_bboxes)))

        box_1 = np.expand_dims(gt_bboxes, 1)
        box_2 = np.expand_dims(tracker_bboxes, 0)

        int_w = np.maximum(0, np.minimum(box_1[:, :, 0] + box_1[:, :, 2], box_2[:, :, 0] + box_2[:, :, 2]) - np.maximum(box_1[:, :, 0], box_2[:, :, 0]))
        int_h = np.maximum(0, np.minimum(box_1[:, :, 1] + box_1[:, :, 3], box_2[:, :, 1] + box_2[:, :, 3]) - np.maximum(box_1[:, :, 1], box_2[:, :, 1]))
        inters = int_w * int_h
        uni = box_1[:, :, 2] * box_1[:, :, 3] + box_2[:, :, 2] * box_2[:, :, 3] - inters

        return inters / (uni + 1e-10)

    def get_default_dataset_config(self):
        return {}

    def get_output_fields(self, tracker):
        return []


def parse_mot_txt(file_path):
    data = {}
    max_frame = 1
    if not os.path.exists(file_path):
        return data, max_frame

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = [float(x) for x in line.replace(',', ' ').split()]
            if len(parts) < 6:
                continue
            frame, obj_id = int(parts[0]), int(parts[1])
            bbox = parts[2:6]
            if frame > max_frame:
                max_frame = frame
            if frame not in data:
                data[frame] = []
            data[frame].append([obj_id] + bbox)

    for frame in data:
        data[frame] = np.array(data[frame])
    return data, max_frame


def evaluate_metrics(gt_path, det_path):
    if os.path.isdir(gt_path):
        possible_gt_file = os.path.join(gt_path, 'gt', 'gt.txt')
        if os.path.exists(possible_gt_file):
            gt_path = possible_gt_file
        else:
            possible_gt_file = os.path.join(gt_path, 'gt.txt')
            if os.path.exists(possible_gt_file):
                gt_path = possible_gt_file

    seq_name = os.path.basename(os.path.dirname(os.path.dirname(gt_path)))
    if not seq_name or seq_name in ['gt', 'train']:
        seq_name = os.path.basename(os.path.dirname(gt_path))
    if not seq_name or seq_name in ['gt', 'train']:
        seq_name = 'TUD-Campus'

    gt_parsed, gt_frames = parse_mot_txt(gt_path)
    det_parsed, det_frames = parse_mot_txt(det_path)

    if not gt_parsed:
        print(f"Ошибка: Не удалось найти или прочитать файл разметки по пути: {gt_path}")
        return 0.0

    seq_length = max(gt_frames, det_frames)

    gt_data = {seq_name: gt_parsed}
    tracker_data = {'my_tracker': {seq_name: det_parsed}}

    dataset = CustomDirectDataset(seq_name, gt_data, tracker_data, seq_length)

    # Убираем аргумент threshold — он не поддерживается
    metrics_list = [metrics.HOTA(), metrics.CLEAR(), metrics.Identity()]

    evaluator = Evaluator({
        'PRINT_RESULTS': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'DISPLAY_LESS_PROGRESS': True,
        'OUTPUT_SUMMARY': False,
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': False
    })

    try:
        results = evaluator.evaluate([dataset], metrics_list)
    except Exception as e:
        print(f"Ошибка при расчете метрик TrackEval: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

    if isinstance(results, tuple):
        results = results[0]

    if not results:
        print("Ошибка: результаты оценки пусты.")
        return 0.0

    try:
        res = results['CustomDataset']['my_tracker'][seq_name]['pedestrian']
        hota = res['HOTA']['HOTA(0)']
        mota = res['CLEAR']['MOTA']
        idf1 = res['Identity']['IDF1']
    except KeyError as e:
        print(f"Ошибка: отсутствует ключ {e} в результатах.")
        return 0.0

    print("\n======= ОЦЕНКА КАЧЕСТВА (HOTA, MOTA, IDF1) =======")
    print(f"Sequence: {seq_name}")
    print(f"HOTA: {hota:.4f}")
    print(f"MOTA: {mota:.4f}")
    print(f"IDF1: {idf1:.4f}")
    print("====================================================\n")

    return hota