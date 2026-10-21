import os
import sys
import glob
import time
import csv
import random
import shutil
import cv2
import numpy as np

# Add workspace root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from chamber_detector import MosquitoClassifier, ChamberDetector

# ─────────────────────────────────────────────────────────────────────────────
# Paths & Setup
# ─────────────────────────────────────────────────────────────────────────────
AEGYPTI_DIR = r"C:\Users\LAPTOP\Downloads\archive\aegypti"
ALBOPICTUS_DIR = r"C:\Users\LAPTOP\Downloads\archive\albopictus"

EVAL_DIR = os.path.join(BASE_DIR, "eval")
MISSED_DIR = os.path.join(EVAL_DIR, "missed_yolo_examples")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(MISSED_DIR, exist_ok=True)


def get_testing_images(base_path, label_str):
    """Collects only images in folders containing '_testing'."""
    items = []
    for root, dirs, files in os.walk(base_path):
        if "_testing" in root.lower():
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    items.append({
                        "path": os.path.join(root, f),
                        "filename": f,
                        "true_label": label_str
                    })
    return sorted(items, key=lambda x: x["filename"])


def split_data(items, seed=42):
    """Deterministically splits 50/50 into tune and final."""
    random.seed(seed)
    shuffled = list(items)
    random.shuffle(shuffled)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:]


def crop_rect(image, box, padding):
    """Rectangular crop with padding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(w, x2 + pad_x)
    ry2 = min(h, y2 + pad_y)
    return image[ry1:ry2, rx1:rx2]


def crop_square(image, box, padding):
    """
    Square crop keeping 1:1 aspect ratio using real surrounding pixels.
    Expands by padding, centers on the detected box, and clamps to image edges.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1

    side = int(max(bw, bh) * (1.0 + 2.0 * padding))
    half = side // 2

    sx1 = int(cx - half)
    sy1 = int(cy - half)
    sx2 = sx1 + side
    sy2 = sy1 + side

    # Shift if out of bounds
    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0
    if sy1 < 0:
        sy2 -= sy1
        sy1 = 0
    if sx2 > w:
        sx1 -= (sx2 - w)
        sx2 = w
    if sy2 > h:
        sy1 -= (sy2 - h)
        sy2 = h

    sx1 = max(0, sx1)
    sy1 = max(0, sy1)
    sx2 = min(w, sx2)
    sy2 = min(h, sy2)

    return image[sy1:sy2, sx1:sx2]


def detect_and_cache_boxes(detector, items):
    """
    Runs YOLO once per image and caches box metadata.
    Does NOT store full 3280x2464 image in RAM to prevent OOM.
    """
    cached = []
    missed = []
    for item in items:
        path = item["path"]
        try:
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = cv2.imread(path)
        if frame is None:
            continue

        orig_h, orig_w = frame.shape[:2]
        frame_std = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        results = detector.yolo_model(frame_std, conf=0.18, imgsz=config.YOLO_IMGSZ,
                                      agnostic_nms=True, iou=0.45, verbose=False)

        best_box = None
        best_conf = -1.0
        best_cls = None

        for r in results:
            for b in r.boxes:
                c = float(b.conf[0])
                if c > best_conf:
                    coords = b.xyxy.cpu().numpy()[0]
                    x1, y1, x2, y2 = map(int, coords[:4])
                    scale_x = orig_w / float(config.FRAME_WIDTH)
                    scale_y = orig_h / float(config.FRAME_HEIGHT)
                    best_box = (
                        max(0, int(x1 * scale_x)),
                        max(0, int(y1 * scale_y)),
                        min(orig_w, int(x2 * scale_x)),
                        min(orig_h, int(y2 * scale_y))
                    )
                    best_conf = c
                    best_cls = int(b.cls[0])

        cached.append({
            "item": item,
            "path": path,
            "detected": (best_box is not None),
            "box": best_box,
            "yolo_conf": best_conf,
            "yolo_cls": best_cls,
            "orig_shape": (orig_w, orig_h)
        })
        if best_box is None:
            missed.append((item, path))

        del frame
        del frame_std

    return cached, missed


def evaluate_setting(cached_data, clf, padding, is_square):
    """Evaluates a single crop setting across cached detections, reading on demand."""
    total_images = len(cached_data)
    detected_count = sum(1 for d in cached_data if d["detected"])
    detection_rate = (detected_count / total_images) * 100.0

    cm = {"aeg_aeg": 0, "aeg_albo": 0, "albo_aeg": 0, "albo_albo": 0}
    aeg_detected_total = 0
    albo_detected_total = 0

    crop_fn = crop_square if is_square else crop_rect

    for d in cached_data:
        if not d["detected"]:
            continue

        try:
            frame = cv2.imdecode(np.fromfile(d["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = cv2.imread(d["path"])
        if frame is None:
            continue

        crop = crop_fn(frame, d["box"], padding)
        del frame

        if crop.size == 0:
            continue

        pred_label, conf, probs = clf.classify_crop(crop)
        true_label = d["item"]["true_label"]

        if true_label == "Aedes aegypti":
            aeg_detected_total += 1
            if pred_label == "Aedes aegypti":
                cm["aeg_aeg"] += 1
            else:
                cm["aeg_albo"] += 1
        else:
            albo_detected_total += 1
            if pred_label == "Aedes albopictus":
                cm["albo_albo"] += 1
            else:
                cm["albo_aeg"] += 1

    aeg_acc_on_det = (cm["aeg_aeg"] / aeg_detected_total * 100.0) if aeg_detected_total > 0 else 0.0
    albo_acc_on_det = (cm["albo_albo"] / albo_detected_total * 100.0) if albo_detected_total > 0 else 0.0
    overall_acc_on_det = ((cm["aeg_aeg"] + cm["albo_albo"]) / detected_count * 100.0) if detected_count > 0 else 0.0

    # Overall accuracy counting miss as wrong (over total_images)
    total_correct = cm["aeg_aeg"] + cm["albo_albo"]
    overall_acc_all = (total_correct / total_images) * 100.0

    return {
        "padding": padding,
        "is_square": is_square,
        "mode_str": "square" if is_square else "rect",
        "detection_rate": detection_rate,
        "detected_count": detected_count,
        "total_images": total_images,
        "aeg_acc_det": aeg_acc_on_det,
        "albo_acc_det": albo_acc_on_det,
        "overall_acc_det": overall_acc_on_det,
        "overall_acc_all": overall_acc_all,
        "cm": cm
    }


def main():
    print("=" * 75)
    print("  PHASE 4 STEP 2: EXPANDED PADDING & ASPECT PRESERVATION EXPERIMENT")
    print("=" * 75)

    # 1. Collect and split testing dataset
    aeg_items = get_testing_images(AEGYPTI_DIR, "Aedes aegypti")
    albo_items = get_testing_images(ALBOPICTUS_DIR, "Aedes albopictus")

    print(f"Total '_testing' images found: {len(aeg_items)} aegypti, {len(albo_items)} albopictus (Total: {len(aeg_items) + len(albo_items)})")

    aeg_tune, aeg_final = split_data(aeg_items, seed=42)
    albo_tune, albo_final = split_data(albo_items, seed=42)

    tune_items = aeg_tune + albo_tune
    final_items = aeg_final + albo_final

    print(f"Stratified Split (seed=42):")
    print(f"  Tune Half  : {len(tune_items)} images (180 aegypti, 180 albopictus)")
    print(f"  Final Half : {len(final_items)} images (180 aegypti, 180 albopictus)")

    # 2. Initialize Models (Read-only)
    clf = MosquitoClassifier()
    detector = ChamberDetector()
    detector.worker_running = False

    # 3. Detect and cache boxes for Tune Half (YOLO runs once)
    print("\n[Step 1] Running YOLO on Tune Half (360 images)...")
    t0 = time.time()
    tune_cached, tune_missed = detect_and_cache_boxes(detector, tune_items)
    print(f"YOLO detection completed in {time.time()-t0:.1f}s.")
    detected_in_tune = sum(1 for d in tune_cached if d["detected"])
    print(f"Tune Half YOLO Detections: {detected_in_tune} / {len(tune_cached)} ({detected_in_tune/len(tune_cached)*100:.1f}%)")

    # 4. Run grid search over 10 settings on Tune Half
    paddings = [0.20, 0.40, 0.60, 0.80, 1.00]
    settings_results = []

    print("\n[Step 2] Evaluating 10 crop settings on Tune Half...")
    for pad in paddings:
        for is_sq in [False, True]:
            res = evaluate_setting(tune_cached, clf, pad, is_sq)
            settings_results.append(res)

    # Print Tune Half Results Table
    print("\n" + "=" * 90)
    print("                    TUNE HALF RESULTS (N = 360 images: 180 Aeg, 180 Albo)")
    print("=" * 90)
    print(f"{'Setting':<16} | {'Det Rate':<9} | {'Aeg Acc (Det)':<14} | {'Albo Acc (Det)':<15} | {'Acc (Detected)':<15} | {'Acc (All 360)':<14}")
    print("-" * 90)

    for r in settings_results:
        setting_name = f"pad {r['padding']:.2f} {r['mode_str']}"
        print(f"{setting_name:<16} | {r['detection_rate']:>7.1f}% | {r['aeg_acc_det']:>12.2f}% | {r['albo_acc_det']:>13.2f}% | {r['overall_acc_det']:>13.2f}% | {r['overall_acc_all']:>12.2f}%")

    # Confusion matrix breakdown for each setting
    print("\n[Confusion Matrices on Tune Half - Detected Images]")
    for r in settings_results:
        setting_name = f"pad {r['padding']:.2f} {r['mode_str']}"
        cm = r["cm"]
        print(f"--- {setting_name} ---")
        print(f"  True Aegypti   : [Pred Aeg: {cm['aeg_aeg']:>3}, Pred Albo: {cm['aeg_albo']:>3}]")
        print(f"  True Albopictus: [Pred Aeg: {cm['albo_aeg']:>3}, Pred Albo: {cm['albo_albo']:>3}]")

    # Pick best setting on Tune Half based on Acc (All 360) and species balance
    best_setting = max(settings_results, key=lambda x: (x["overall_acc_all"], min(x["aeg_acc_det"], x["albo_acc_det"])))
    print("\n" + "=" * 90)
    print(f"  BEST SETTING ON TUNE HALF: pad {best_setting['padding']:.2f} {best_setting['mode_str'].upper()}")
    print(f"  - Overall Accuracy (All 360): {best_setting['overall_acc_all']:.2f}%")
    print(f"  - Overall Accuracy (Detected): {best_setting['overall_acc_det']:.2f}%")
    print(f"  - Aedes aegypti Accuracy    : {best_setting['aeg_acc_det']:.2f}%")
    print(f"  - Aedes albopictus Accuracy : {best_setting['albo_acc_det']:.2f}%")
    print("=" * 90)

    # 5. Evaluate the BEST setting ONCE on Final Half (Held-out validation)
    print("\n[Step 3] Evaluating Best Setting ONCE on Final Half (N = 360 images)...")
    final_cached, final_missed = detect_and_cache_boxes(detector, final_items)
    final_res = evaluate_setting(final_cached, clf, best_setting["padding"], best_setting["is_square"])

    print("\n" + "=" * 90)
    print(f"  FINAL HALF EVALUATION (UNSEEN HELD-OUT SPLIT, N = 360 images)")
    print("=" * 90)
    print(f"Setting                       : pad {final_res['padding']:.2f} {final_res['mode_str']}")
    print(f"YOLO Detection Rate           : {final_res['detection_rate']:.1f}% ({final_res['detected_count']}/{final_res['total_images']})")
    print(f"Overall Accuracy (All 360)    : {final_res['overall_acc_all']:.2f}%")
    print(f"Overall Accuracy (Detected)   : {final_res['overall_acc_det']:.2f}%")
    print(f"  - Aedes aegypti Accuracy    : {final_res['aeg_acc_det']:.2f}%")
    print(f"  - Aedes albopictus Accuracy : {final_res['albo_acc_det']:.2f}%")
    cm_f = final_res["cm"]
    print(f"Confusion Matrix (Detected):")
    print(f"  True Aegypti   : [Pred Aeg: {cm_f['aeg_aeg']:>3}, Pred Albo: {cm_f['aeg_albo']:>3}]")
    print(f"  True Albopictus: [Pred Aeg: {cm_f['albo_aeg']:>3}, Pred Albo: {cm_f['albo_albo']:>3}]")

    # Baseline comparison on Final Half (pad 0.20 rect)
    baseline_final = evaluate_setting(final_cached, clf, 0.20, False)
    print("\n[Final Half Comparison: Baseline (pad 0.20 rect) vs Best Setting]")
    print(f"{'Metric':<30} | {'Baseline (0.20 rect)':<22} | {'Best Setting (' + best_setting['mode_str'] + ' ' + str(best_setting['padding']) + ')':<22}")
    print("-" * 78)
    print(f"{'Overall Accuracy (All 360)':<30} | {baseline_final['overall_acc_all']:>20.2f}% | {final_res['overall_acc_all']:>20.2f}%")
    print(f"{'Overall Accuracy (Detected)':<30} | {baseline_final['overall_acc_det']:>20.2f}% | {final_res['overall_acc_det']:>20.2f}%")
    print(f"{'Aedes aegypti Accuracy':<30} | {baseline_final['aeg_acc_det']:>20.2f}% | {final_res['aeg_acc_det']:>20.2f}%")
    print(f"{'Aedes albopictus Accuracy':<30} | {baseline_final['albo_acc_det']:>20.2f}% | {final_res['albo_acc_det']:>20.2f}%")

    # 6. Analyze Missed YOLO Images & Save 10 Examples
    all_missed = tune_missed + final_missed
    print(f"\n[Step 4] Analyzing YOLO Missed Images (Total Missed: {len(all_missed)} across 720 images)...")
    print(f"Saving 10 representative missed images to: {MISSED_DIR}...")

    saved_count = 0
    for item, path in all_missed[:10]:
        try:
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = cv2.imread(path)
        if frame is not None:
            fname = f"missed_{item['true_label'].replace(' ', '_')}_{item['filename']}"
            cv2.imwrite(os.path.join(MISSED_DIR, fname), frame)
            saved_count += 1

    # Check reasons for misses on a sample of missed images
    print("\nDiagnosing Missed Images:")
    sample_reasons = []
    for item, path in all_missed[:15]:
        try:
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = cv2.imread(path)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        # Try raw YOLO at conf=0.05
        frame_std = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        r_raw = detector.yolo_model(frame_std, conf=0.05, imgsz=config.YOLO_IMGSZ, verbose=False)
        raw_confs = [float(b.conf[0]) for r in r_raw for b in r.boxes]
        max_raw = max(raw_confs) if raw_confs else 0.0
        sample_reasons.append({
            "file": item["filename"],
            "species": item["true_label"],
            "blur_variance": blur_val,
            "brightness": brightness,
            "raw_max_yolo_conf": max_raw
        })
        print(f"  {item['filename']} ({item['true_label']}): Max Raw YOLO Conf={max_raw:.3f}, Blur Var={blur_val:.1f}, Brightness={brightness:.1f}")

    print("\n" + "=" * 75)
    print("  EXPERIMENT FINISHED")
    print("=" * 75)


if __name__ == "__main__":
    main()
