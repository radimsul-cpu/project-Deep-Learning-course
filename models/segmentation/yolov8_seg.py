#!/usr/bin/env python3
import os
import sys

import cv2

WINDOW_NAME = "video-annotator"
KEY_ESC = 27
KEY_BACK = {ord("a"), ord("A")}
KEY_FORWARD = {ord("s"), ord("S")}


def print_usage():
    print(
        "Usage: python annotate_video.py <video_path> [--step N|step] [--auto-zero]\n"
        "Controls: click+drag to draw a box, A/S move by step, u=undo last box, ESC=finish\n"
        "Options: --auto-zero  Use id 0 for all boxes (no prompt)\n"
    )


def parse_args(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print_usage()
        return None, None, False

    video_path = None
    step = 1
    auto_zero = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            print_usage()
            return None, None, False
        if arg in ("--auto-zero",):
            auto_zero = True
            i += 1
            continue
        if arg in ("-s", "--step") and i + 1 < len(argv):
            try:
                step = int(argv[i + 1])
            except ValueError:
                print("Step must be a number.")
                return None, None, False
            i += 2
            continue
        elif video_path is None:
            video_path = arg
            i += 1
            continue
        elif step == 1:
            try:
                step = int(arg)
            except ValueError:
                print(f"Unknown argument: {arg}")
                return None, None, False
            i += 1
            continue
        else:
            print(f"Unknown argument: {arg}")
            return None, None, False

    if step < 1:
        step = 1

    return video_path, step, auto_zero


def count_frames(path: str) -> int:
    probe = cv2.VideoCapture(path)
    if not probe.isOpened():
        return 0
    total = 0
    while True:
        ok, _ = probe.read()
        if not ok:
            break
        total += 1
    probe.release()
    return total


def load_annotations(path: str):
    annotations = {}
    if not os.path.isfile(path):
        return annotations, 0
    loaded = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip()]
            if len(parts) < 6:
                continue
            try:
                frame_idx = int(parts[0])
            except ValueError:
                continue
            boxes = []
            nums = []
            try:
                nums = [int(p) for p in parts[1:]]
            except ValueError:
                nums = []
            if not nums:
                continue
            for i in range(0, len(nums), 5):
                chunk = nums[i : i + 5]
                if len(chunk) != 5:
                    continue
                person, x, y, w, h = chunk
                boxes.append((x, y, w, h, person))
            if boxes:
                annotations[frame_idx] = boxes
                loaded += len(boxes)
    return annotations, loaded


def main():
    video_path, step, auto_zero = parse_args(sys.argv[1:])
    if not video_path:
        return

    if not os.path.isfile(video_path):
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = reported_frames if reported_frames > 0 else None
    actual_frames = count_frames(video_path)
    if actual_frames > 0:
        total_frames = actual_frames

    output_csv = os.path.join(
        os.path.dirname(video_path),
        f"{os.path.splitext(os.path.basename(video_path))[0]}.csv",
    )

    annotations, preloaded = load_annotations(output_csv)
    if preloaded:
        print(f"Loaded {preloaded} boxes from {output_csv}")

    current_frame = 0
    drawing = False
    start_point = (0, 0)
    preview_box = None
    quitting = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, _userdata):
        nonlocal drawing, start_point, preview_box, annotations
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            preview_box = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x0, y0 = start_point
            preview_box = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x0, y0 = start_point
            box = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            preview_box = None
            if box[2] == 0 or box[3] == 0:
                return
            if auto_zero:
                person_id = 0
            else:
                person = input("Person id: ").strip()
                if not person:
                    print("Skipped box (empty person id).")
                    return
                try:
                    person_id = int(person)
                except ValueError:
                    print("Skipped box (person id must be a number).")
                    return
            annotations.setdefault(current_frame, []).append(
                (box[0], box[1], box[2], box[3], person_id)
            )

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"Reached end of readable frames at {current_frame}.")
            break

        while True:
            display = frame.copy()
            for x, y, w, h, person in annotations.get(current_frame, []):
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    str(person),
                    (x, max(10, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            if preview_box:
                px, py, pw, ph = preview_box
                cv2.rectangle(display, (px, py), (px + pw, py + ph), (0, 255, 255), 1)

            label = f"Frame {current_frame}"
            if total_frames:
                label += f"/{total_frames}"
            cv2.putText(
                display,
                label,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(20) & 0xFF

            if key == KEY_ESC:
                quitting = True
                break
            if key in KEY_FORWARD:
                next_frame = current_frame + step
                if total_frames is not None and next_frame >= total_frames:
                    quitting = True
                    break
                current_frame = next_frame
                preview_box = None
                break
            if key in KEY_BACK:
                next_frame = max(0, current_frame - step)
                if next_frame != current_frame:
                    current_frame = next_frame
                    preview_box = None
                    break
            if key == ord("u") or key == ord("U"):
                boxes = annotations.get(current_frame)
                if boxes:
                    boxes.pop()
                    if not boxes:
                        annotations.pop(current_frame, None)

        if quitting:
            break

    cap.release()
    cv2.destroyAllWindows()

    rows_written = 0
    with open(output_csv, "w", encoding="utf-8") as f:
        for frame_idx in sorted(annotations.keys()):
            parts = [str(frame_idx)]
            for x, y, w, h, person in annotations[frame_idx]:
                parts.extend([str(person), str(x), str(y), str(w), str(h)])
                rows_written += 1
            f.write(",".join(parts) + "\n")

    print(f"Saved {rows_written} annotations to {output_csv}")


if __name__ == "__main__":
    main()
