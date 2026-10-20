"""
Evaluate YOLO at confidence levels [0.10, 0.12, 0.15, 0.18] on:
1. The 205 missed images from the 720 _testing dataset.
2. The empty-chamber (negative/) images to count false alarms.
"""
import os
import sys
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from eval.experiment_crop_tuning import get_testing_images, split_data, AEGYPTI_DIR, ALBOPICTUS_DIR
from chamber_detector import ChamberDetector, MosquitoClassifier

CHAMBER_DIR_1 = r"C:\Users\LAPTOP\Downloads\Chamber_Dataset"
CHAMBER_DIR_2 = r"C:\Users\LAPTOP\Downloads\Negative_Chamber-20261018T020128Z-1-001\Negative_Chamber"

SWEEP_THRESHOLDS = [0.10, 0.12, 0.15, 0.18]

def get_chamber_images():
    """Collect all unique empty chamber images."""
    paths = []
    seen = set()
    for d in [CHAMBER_DIR_1, CHAMBER_DIR_2]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_p = os.path.join(root, f)
                        base_name = os.path.basename(full_p)
                        if base_name not in seen:
                            seen.add(base_name)
                            paths.append(full_p)
    return paths

def main():
    print("=" * 80)
    print("  YOLO THRESHOLD SWEEP EVALUATION: [0.10, 0.12, 0.15, 0.18]")
    print("=" * 80)

    detector = ChamberDetector()
    detector.worker_running = False
    yolo_model = detector.yolo_model

    # 1. Collect all 720 _testing images
    aeg_items = get_testing_images(AEGYPTI_DIR, "Aedes aegypti")
    albo_items = get_testing_images(ALBOPICTUS_DIR, "Aedes albopictus")
    aeg_tune, aeg_final = split_data(aeg_items, seed=42)
    albo_tune, albo_final = split_data(albo_items, seed=42)
    all_720 = aeg_tune + albo_tune + aeg_final + albo_final

    print(f"Total testing images: {len(all_720)}")

    # 2. Identify the exact 205 missed images at baseline conf = 0.18
    print("\n[Step 1] Identifying baseline missed images at conf = 0.18...")
    missed_images = []
    for item in all_720:
        path = item["path"]
        try:
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = cv2.imread(path)
        if frame is None:
            continue

        res = yolo_model(frame, conf=0.18, imgsz=640, agnostic_nms=True, iou=0.45, verbose=False)
        has_detection = False
        for r in res:
            if len(r.boxes) > 0:
                has_detection = True
                break

        del frame
        if not has_detection:
            missed_images.append(item)

    print(f"Identified {len(missed_images)} missed images at conf = 0.18 (Expected ~205).")

    # 3. Sweep thresholds on the missed images
    print("\n[Step 2] Sweeping thresholds on missed images...")
    missed_sweep_results = {}

    for conf_thresh in SWEEP_THRESHOLDS:
        detected_images = []
        total_boxes = 0
        boxes_on_mosquito = 0
        boxes_not_on_mosquito = 0

        for item in missed_images:
            path = item["path"]
            try:
                frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                frame = cv2.imread(path)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            res = yolo_model(frame, conf=conf_thresh, imgsz=640, agnostic_nms=True, iou=0.45, verbose=False)
            boxes = []
            for r in res:
                for b in r.boxes:
                    coords = b.xyxy.cpu().numpy()[0]
                    bx1, by1, bx2, by2 = map(int, coords[:4])
                    bconf = float(b.conf[0])
                    boxes.append((bx1, by1, bx2, by2, bconf))

            if boxes:
                detected_images.append(item)
                total_boxes += len(boxes)

                # Inspect each box: verify if it contains insect pixels or lands on blank background
                for (bx1, by1, bx2, by2, bconf) in boxes:
                    crop = frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
                    if crop.size > 0:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        # Check variance / dark contrast against background
                        contrast = np.std(gray)
                        dark_pixels = np.count_nonzero(gray < (np.percentile(gray, 85) - 20))
                        # In mosquito macro photos, an insect has distinct structure and dark pixels
                        if contrast > 15 and dark_pixels > 30:
                            boxes_on_mosquito += 1
                        else:
                            boxes_not_on_mosquito += 1
                    else:
                        boxes_not_on_mosquito += 1

            del frame

        missed_sweep_results[conf_thresh] = {
            "recovered_images": len(detected_images),
            "total_boxes": total_boxes,
            "boxes_on_mosquito": boxes_on_mosquito,
            "boxes_not_on_mosquito": boxes_not_on_mosquito
        }
        print(f"  Conf {conf_thresh:.2f}: {len(detected_images)}/{len(missed_images)} images found ({total_boxes} boxes: {boxes_on_mosquito} on mosquito, {boxes_not_on_mosquito} off)")

    # 4. Sweep thresholds on Empty Chamber (Negative) Images
    chamber_images = get_chamber_images()
    print(f"\n[Step 3] Evaluating False Alarms on Empty Chamber Images (N = {len(chamber_images)} images)...")

    chamber_sweep_results = {}
    for conf_thresh in SWEEP_THRESHOLDS:
        false_alarm_frames = 0
        total_false_boxes = 0

        # Also test with ROI applied (live pipeline behavior)
        roi_false_alarm_frames = 0
        roi_false_boxes = 0

        for path in chamber_images:
            try:
                frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                frame = cv2.imread(path)
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Full frame test
            res = yolo_model(frame, conf=conf_thresh, imgsz=640, agnostic_nms=True, iou=0.45, verbose=False)
            boxes = [b for r in res for b in r.boxes]
            if len(boxes) > 0:
                false_alarm_frames += 1
                total_false_boxes += len(boxes)

            # With Chamber ROI test
            xmin = int(w * config.ROI_MARGIN_X)
            xmax = int(w * (1.0 - config.ROI_MARGIN_X))
            ymin = int(h * config.ROI_MARGIN_Y)
            ymax = int(h * (1.0 - config.ROI_MARGIN_Y))
            proc_frame = frame[ymin:ymax, xmin:xmax]

            res_roi = yolo_model(proc_frame, conf=conf_thresh, imgsz=640, agnostic_nms=True, iou=0.45, verbose=False)
            roi_boxes = [b for r in res_roi for b in r.boxes]
            if len(roi_boxes) > 0:
                roi_false_alarm_frames += 1
                roi_false_boxes += len(roi_boxes)

            del frame

        chamber_sweep_results[conf_thresh] = {
            "full_frame_false_images": false_alarm_frames,
            "full_frame_false_boxes": total_false_boxes,
            "roi_false_images": roi_false_alarm_frames,
            "roi_false_boxes": roi_false_boxes
        }
        print(f"  Conf {conf_thresh:.2f}: Full Frame False Alarms = {false_alarm_frames}/{len(chamber_images)} ({total_false_boxes} boxes) | ROI False Alarms = {roi_false_alarm_frames}/{len(chamber_images)} ({roi_false_boxes} boxes)")

    # 5. Output Summary Table
    print("\n" + "=" * 90)
    print("                    YOLO CONFIDENCE SWEEP REPORT")
    print("=" * 90)
    print(f"{'Conf Thresh':<12} | {'Missed Found':<14} | {'Boxes On Mosq':<14} | {'Boxes Not Mosq':<15} | {'Empty Chamber FA (ROI)':<24}")
    print("-" * 90)
    for conf_thresh in SWEEP_THRESHOLDS:
        m = missed_sweep_results[conf_thresh]
        c = chamber_sweep_results[conf_thresh]
        print(f"{conf_thresh:<12.2f} | {m['recovered_images']:>4} / {len(missed_images):<7} | {m['boxes_on_mosquito']:>14} | {m['boxes_not_on_mosquito']:>15} | {c['roi_false_images']:>4} / {len(chamber_images)} ({c['roi_false_boxes']} boxes)")

if __name__ == "__main__":
    main()
