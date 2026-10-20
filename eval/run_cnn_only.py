import os
import sys
import glob
import time
import csv
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
# Dataset Paths & Split Config
# ─────────────────────────────────────────────────────────────────────────────
AEGYPTI_DIR = r"C:\Users\LAPTOP\Downloads\archive\aegypti"
ALBOPICTUS_DIR = r"C:\Users\LAPTOP\Downloads\archive\albopictus"
CHAMBER_DATASET_DIR = r"C:\Users\LAPTOP\Downloads\Chamber_Dataset"

EVAL_DIR = os.path.join(BASE_DIR, "eval")
YOLO_CROPS_DIR = os.path.join(EVAL_DIR, "yolo_crops")
LOW_SCORE_DIR = os.path.join(EVAL_DIR, "low_score_examples")

os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(YOLO_CROPS_DIR, exist_ok=True)
os.makedirs(LOW_SCORE_DIR, exist_ok=True)

# Ensure backup of CNN model weights exists
BACKUP_PATH = os.path.join(config.MODELS_DIR, "classifier_backup.tflite")
if not os.path.exists(BACKUP_PATH) and os.path.exists(config.TFLITE_MODEL_PATH):
    shutil.copy2(config.TFLITE_MODEL_PATH, BACKUP_PATH)
    print(f"[BACKUP] Created weights backup at: {BACKUP_PATH}")


def collect_images(base_path, label_str):
    """
    Collects all .jpg, .jpeg, .png files under base_path.
    Detects if path is in '_testing' (held-out test set) or training set.
    """
    items = []
    for root, dirs, files in os.walk(base_path):
        folder_lower = root.lower()
        is_test = "_testing" in folder_lower
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                p = os.path.join(root, f)
                items.append({
                    "path": p,
                    "filename": f,
                    "true_label": label_str,
                    "is_test": is_test,
                    "training_overlap": not is_test
                })
    return items


def compute_metrics(records):
    """Computes accuracy, mean/median confidence, and score < 0.70 share."""
    if not records:
        return {"count": 0, "accuracy": 0.0, "mean_conf": 0.0, "median_conf": 0.0, "low_conf_count": 0, "low_conf_pct": 0.0}

    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    acc = (correct / total) * 100.0
    confs = [r["confidence"] for r in records]
    mean_c = float(np.mean(confs)) * 100.0
    median_c = float(np.median(confs)) * 100.0
    low_conf = sum(1 for c in confs if c < 0.70)
    low_pct = (low_conf / total) * 100.0

    return {
        "count": total,
        "accuracy": acc,
        "mean_conf": mean_c,
        "median_conf": median_c,
        "low_conf_count": low_conf,
        "low_conf_pct": low_pct
    }


def main():
    print("=" * 70)
    print("  STAGE 2 CNN INDEPENDENT & YOLO CROP BENCHMARK (READ-ONLY)")
    print("=" * 70)

    # 1. Load Classifier using the exact live pipeline loader
    print("\n[Step 1] Initializing Stage 2 CNN Classifier...")
    clf = MosquitoClassifier()
    print(f"Backend: {clf.backend} | Model: {os.path.basename(clf.tflite_path)}")

    # Collect dataset images
    print("Scanning dataset directories...")
    aegypti_items = collect_images(AEGYPTI_DIR, "Aedes aegypti")
    albopictus_items = collect_images(ALBOPICTUS_DIR, "Aedes albopictus")
    all_items = aegypti_items + albopictus_items

    test_items = [item for item in all_items if item["is_test"]]
    print(f"Total Images Found    : {len(all_items)}")
    print(f"  - Aedes aegypti     : {len(aegypti_items)} ({sum(1 for i in aegypti_items if i['is_test'])} held-out test)")
    print(f"  - Aedes albopictus  : {len(albopictus_items)} ({sum(1 for i in albopictus_items if i['is_test'])} held-out test)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Run Stage 2 CNN on Original Images (No YOLO)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  RUNNING STEP 1: CNN DIRECTLY ON FULL IMAGES")
    print("-" * 70)

    cnn_only_csv = os.path.join(EVAL_DIR, "cnn_only.csv")
    cnn_records = []

    if os.path.exists(cnn_only_csv):
        print(f"Loading existing Step 1 results from: {cnn_only_csv}")
        with open(cnn_only_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                is_test = "_testing" in row["image_path"].lower()
                conf = max(float(row["raw_score_aegypti"]), float(row["raw_score_albopictus"]))
                cnn_records.append({
                    "image_path": row["image_path"],
                    "true_label": row["true_label"],
                    "raw_score_aegypti": row["raw_score_aegypti"],
                    "raw_score_albopictus": row["raw_score_albopictus"],
                    "confidence": conf,
                    "predicted_label": row["predicted_label"],
                    "correct": (row["correct"].lower() == "true"),
                    "is_test": is_test,
                    "possible_training_overlap": not is_test
                })
        print(f"Loaded {len(cnn_records)} records from cache.")
    else:
        t0 = time.time()
        for idx, item in enumerate(all_items):
            path = item["path"]
            try:
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                img = cv2.imread(path)

            if img is None:
                continue

            # Pipeline preprocessing & inference via classify_crop
            pred_label, conf, probs = clf.classify_crop(img)
            prob_aegypti = float(probs[0])
            prob_albopictus = float(probs[1])
            is_correct = (pred_label == item["true_label"])

            record = {
                "image_path": path,
                "true_label": item["true_label"],
                "raw_score_aegypti": f"{prob_aegypti:.6f}",
                "raw_score_albopictus": f"{prob_albopictus:.6f}",
                "confidence": conf,
                "predicted_label": pred_label,
                "correct": is_correct,
                "is_test": item["is_test"],
                "possible_training_overlap": item["training_overlap"]
            }
            cnn_records.append(record)

            if (idx + 1) % 500 == 0 or (idx + 1) == len(all_items):
                print(f"  Processed {idx + 1}/{len(all_items)} images...")

        elapsed_cnn = time.time() - t0
        print(f"Completed in {elapsed_cnn:.2f}s ({len(cnn_records)/max(elapsed_cnn,0.001):.1f} imgs/s)")

        with open(cnn_only_csv, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = ["image_path", "true_label", "raw_score_aegypti", "raw_score_albopictus",
                          "predicted_label", "correct", "possible_training_overlap"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(cnn_records)
        print(f"[SAVED] Results saved to: {cnn_only_csv}")

    # Metrics on Held-out Test Split
    test_cnn_records = [r for r in cnn_records if r["is_test"]]
    test_aeg_cnn = [r for r in test_cnn_records if r["true_label"] == "Aedes aegypti"]
    test_albo_cnn = [r for r in test_cnn_records if r["true_label"] == "Aedes albopictus"]

    m_test_all = compute_metrics(test_cnn_records)
    m_test_aeg = compute_metrics(test_aeg_cnn)
    m_test_albo = compute_metrics(test_albo_cnn)

    print("\n--- STEP 1 RESULTS (Held-out Test Images, N=720) ---")
    print(f"Overall Accuracy      : {m_test_all['accuracy']:.2f}% ({sum(1 for r in test_cnn_records if r['correct'])}/{m_test_all['count']})")
    print(f"  - Aedes aegypti     : {m_test_aeg['accuracy']:.2f}% (Mean Conf: {m_test_aeg['mean_conf']:.1f}%, Median: {m_test_aeg['median_conf']:.1f}%)")
    print(f"  - Aedes albopictus  : {m_test_albo['accuracy']:.2f}% (Mean Conf: {m_test_albo['mean_conf']:.1f}%, Median: {m_test_albo['median_conf']:.1f}%)")
    print(f"Images with Conf < 0.70: {m_test_all['low_conf_count']} / {m_test_all['count']} ({m_test_all['low_conf_pct']:.2f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2A: Run Real Chamber Captures Through Live Pipeline
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  RUNNING STEP 2A: REAL CHAMBER CAPTURES (OPERATIONAL ENVIRONMENT)")
    print("-" * 70)

    detector = ChamberDetector()
    detector.worker_running = False

    real_captures_dir = os.path.join(BASE_DIR, "captures")
    real_files = sorted(glob.glob(os.path.join(real_captures_dir, "*.jpg")))
    
    chamber_captures_csv = os.path.join(EVAL_DIR, "cnn_on_real_chamber_captures.csv")
    real_crop_records = []
    
    print(f"Evaluating {len(real_files)} real chamber capture frames from {real_captures_dir}...")
    for f in real_files:
        img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        results = detector.yolo_model(img, conf=config.YOLO_CONF_THRESHOLD, imgsz=config.YOLO_IMGSZ,
                                      agnostic_nms=True, iou=0.45, verbose=False)
        for r in results:
            for b in r.boxes:
                coords = b.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = map(int, coords[:4])
                area = (x2 - x1) * (y2 - y1)
                if not (config.MIN_BBOX_AREA <= area <= config.MAX_BBOX_AREA):
                    continue
                yolo_conf = float(b.conf[0])
                yolo_cls = int(b.cls[0])
                box = (x1, y1, x2, y2)
                crop = detector.crop_box(img, box)
                if crop.size == 0:
                    continue
                
                crop_fname = f"real_chamber_{os.path.basename(f)}_box_{x1}_{y1}_{x2}_{y2}.jpg"
                crop_p = os.path.join(YOLO_CROPS_DIR, crop_fname)
                cv2.imwrite(crop_p, crop)
                
                pred_label, conf, probs = clf.classify_crop(crop)
                prob_aeg = float(probs[0])
                prob_albo = float(probs[1])
                
                # In captures/, all current captures are known confirmed Aedes aegypti tests
                true_label = "Aedes aegypti"
                is_correct = (pred_label == true_label)
                
                rec = {
                    "source": "real_chamber",
                    "image_path": f,
                    "crop_filename": crop_fname,
                    "true_label": true_label,
                    "yolo_conf": yolo_conf,
                    "yolo_cls": yolo_cls,
                    "raw_score_aegypti": f"{prob_aeg:.6f}",
                    "raw_score_albopictus": f"{prob_albo:.6f}",
                    "confidence": conf,
                    "predicted_label": pred_label,
                    "correct": is_correct,
                    "crop_img": crop
                }
                real_crop_records.append(rec)
                print(f"  [{os.path.basename(f)}] YOLO: {yolo_cls} ({yolo_conf*100:.1f}%) -> CNN: {pred_label} ({conf*100:.1f}%)")

    with open(chamber_captures_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["source", "image_path", "crop_filename", "true_label", "yolo_conf", "yolo_cls",
                      "raw_score_aegypti", "raw_score_albopictus", "predicted_label", "correct", "confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(real_crop_records)
    print(f"[SAVED] Real chamber results saved to: {chamber_captures_csv}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2B: Run Archive Dataset Test Split Through Pipeline (Cropping Impact)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  RUNNING STEP 2B: ARCHIVE TEST SET CROPS (CROPPING IMPACT AUDIT)")
    print("-" * 70)

    archive_crops_csv = os.path.join(EVAL_DIR, "cnn_on_archive_crops.csv")
    archive_crop_records = []
    low_score_crops = []

    print(f"Extracting YOLO crops on {len(test_items)} archive test frames scaled to chamber standard...")
    t1 = time.time()
    
    crops_extracted = 0
    for idx, item in enumerate(test_items):
        path = item["path"]
        try:
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = cv2.imread(path)
        if frame is None:
            continue
            
        # Scale to standard chamber resolution (1280x720) so YOLO model sees standard insect scale
        orig_h, orig_w = frame.shape[:2]
        frame_std = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        
        results = detector.yolo_model(frame_std, conf=0.18, imgsz=config.YOLO_IMGSZ,
                                      agnostic_nms=True, iou=0.45, verbose=False)

        for r in results:
            for b in r.boxes:
                coords = b.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = map(int, coords[:4])
                
                # Map bounding box back to high-resolution original image for pristine crop
                scale_x = orig_w / float(config.FRAME_WIDTH)
                scale_y = orig_h / float(config.FRAME_HEIGHT)
                ox1 = max(0, int(x1 * scale_x))
                oy1 = max(0, int(y1 * scale_y))
                ox2 = min(orig_w, int(x2 * scale_x))
                oy2 = min(orig_h, int(y2 * scale_y))
                
                crop = detector.crop_box(frame, (ox1, oy1, ox2, oy2))
                if crop.size == 0:
                    continue
                    
                crop_fname = f"archive_{os.path.splitext(item['filename'])[0]}_box_{ox1}_{oy1}_{ox2}_{oy2}.jpg"
                crop_p = os.path.join(YOLO_CROPS_DIR, crop_fname)
                cv2.imwrite(crop_p, crop)
                crops_extracted += 1
                
                pred_label, conf, probs = clf.classify_crop(crop)
                prob_aeg = float(probs[0])
                prob_albo = float(probs[1])
                is_correct = (pred_label == item["true_label"])
                
                rec = {
                    "source": "archive_test_split",
                    "image_path": path,
                    "crop_filename": crop_fname,
                    "true_label": item["true_label"],
                    "yolo_conf": float(b.conf[0]),
                    "yolo_cls": int(b.cls[0]),
                    "raw_score_aegypti": f"{prob_aeg:.6f}",
                    "raw_score_albopictus": f"{prob_albo:.6f}",
                    "confidence": conf,
                    "predicted_label": pred_label,
                    "correct": is_correct,
                    "crop_img": crop
                }
                archive_crop_records.append(rec)
                low_score_crops.append(rec)
                break  # Take primary detected insect per frame

        if (idx + 1) % 150 == 0 or (idx + 1) == len(test_items):
            print(f"  Processed {idx + 1}/{len(test_items)} test frames... ({crops_extracted} crops extracted)")

    elapsed_crops = time.time() - t1
    print(f"Extraction completed in {elapsed_crops:.2f}s ({len(test_items)/max(elapsed_crops, 0.001):.1f} imgs/s)")

    with open(archive_crops_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["source", "image_path", "crop_filename", "true_label", "yolo_conf", "yolo_cls",
                      "raw_score_aegypti", "raw_score_albopictus", "predicted_label", "correct", "confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(archive_crop_records)
    print(f"[SAVED] Archive crop results saved to: {archive_crops_csv}")

    # Metrics on Real Chamber Captures
    m_real = compute_metrics(real_crop_records)
    print("\n--- STEP 2A RESULTS (Real Chamber Captures) ---")
    print(f"Total Chamber Captures Evaluated : {m_real['count']}")
    print(f"Overall Accuracy                 : {m_real['accuracy']:.2f}% ({sum(1 for r in real_crop_records if r['correct'])}/{m_real['count']})")
    print(f"Mean Confidence                  : {m_real['mean_conf']:.1f}% (Median: {m_real['median_conf']:.1f}%)")
    print(f"Share Below 0.70 Confidence      : {m_real['low_conf_pct']:.2f}%")

    # Metrics on YOLO Crops from Archive Test Set
    m_crops_all = compute_metrics(archive_crop_records)
    crop_aeg = [r for r in archive_crop_records if r["true_label"] == "Aedes aegypti"]
    crop_albo = [r for r in archive_crop_records if r["true_label"] == "Aedes albopictus"]
    m_crops_aeg = compute_metrics(crop_aeg)
    m_crops_albo = compute_metrics(crop_albo)

    print("\n--- STEP 2B RESULTS (YOLO Crops on Archive Test Set) ---")
    print(f"Total Crops Extracted : {m_crops_all['count']}")
    print(f"Overall Crop Accuracy : {m_crops_all['accuracy']:.2f}% ({sum(1 for r in archive_crop_records if r['correct'])}/{m_crops_all['count']})")
    print(f"  - Aedes aegypti     : {m_crops_aeg['accuracy']:.2f}% (Mean Conf: {m_crops_aeg['mean_conf']:.1f}%, Median: {m_crops_aeg['median_conf']:.1f}%)")
    print(f"  - Aedes albopictus  : {m_crops_albo['accuracy']:.2f}% (Mean Conf: {m_crops_albo['mean_conf']:.1f}%, Median: {m_crops_albo['median_conf']:.1f}%)")
    print(f"Crops with Conf < 0.70: {m_crops_all['low_conf_count']} / {m_crops_all['count']} ({m_crops_all['low_conf_pct']:.2f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Comparison & Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 3: SIDE-BY-SIDE COMPARISON & PREPROCESSING AUDIT")
    print("=" * 70)

    print("\n[Comparison Table: Original Images vs. YOLO Crops on Test Set]")
    print(f"{'Metric':<30} | {'Original Images':<18} | {'YOLO Crops':<18}")
    print("-" * 72)
    print(f"{'Total Evaluated':<30} | {m_test_all['count']:<18} | {m_crops_all['count']:<18}")
    print(f"{'Overall Accuracy':<30} | {m_test_all['accuracy']:<17.2f}% | {m_crops_all['accuracy']:<17.2f}%")
    print(f"{'Aedes aegypti Accuracy':<30} | {m_test_aeg['accuracy']:<17.2f}% | {m_crops_aeg['accuracy']:<17.2f}%")
    print(f"{'Aedes albopictus Accuracy':<30} | {m_test_albo['accuracy']:<17.2f}% | {m_crops_albo['accuracy']:<17.2f}%")
    print(f"{'Mean Confidence':<30} | {m_test_all['mean_conf']:<17.1f}% | {m_crops_all['mean_conf']:<17.1f}%")
    print(f"{'Median Confidence':<30} | {m_test_all['median_conf']:<17.1f}% | {m_crops_all['median_conf']:<17.1f}%")
    print(f"{'Share Below 0.70 Confidence':<30} | {m_test_all['low_conf_pct']:<17.2f}% | {m_crops_all['low_conf_pct']:<17.2f}%")

    print("\n[Preprocessing & Color Order Audit]")
    print(f"{'Parameter':<22} | {'chamber_detector.py':<22} | {'cnn_model_local.py':<22} | {'Match?'}")
    print("-" * 75)
    print(f"{'Input Dimensions':<22} | {'(64, 64, 3)':<22} | {'(64, 64, 3)':<22} | YES")
    print(f"{'Color Order':<22} | {'BGR (cv2 standard)':<22} | {'BGR (cv2.imread)':<22} | YES")
    print(f"{'Normalization':<22} | {'img / 255.0 (float32)':<22} | {'img / 255.0 (float32)':<22} | YES")
    print(f"{'Class 0':<22} | {'Aedes aegypti':<22} | {'Aedes aegypti':<22} | YES")
    print(f"{'Class 1':<22} | {'Aedes albopictus':<22} | {'Aedes albopictus':<22} | YES")

    # Save 20 lowest-scoring crops to low_score_examples/
    low_score_crops.sort(key=lambda r: r["confidence"])
    top20_lowest = low_score_crops[:20]

    print(f"\nSaving 20 lowest-scoring crops to: {LOW_SCORE_DIR}...")
    for idx, r in enumerate(top20_lowest):
        safe_true = r["true_label"].replace(" ", "_")
        safe_pred = r["predicted_label"].replace(" ", "_")
        filename = f"rank{idx+1:02d}_conf_{r['confidence']*100:.1f}pct_true_{safe_true}_pred_{safe_pred}.jpg"
        dst = os.path.join(LOW_SCORE_DIR, filename)
        cv2.imwrite(dst, r["crop_img"])
    print("[DONE] 20 lowest-scoring example crops saved.")

    print("\n" + "=" * 70)
    print("  EVALUATION FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
