"""
Inspect the 8 screw boxes detected inside the ROI on empty-chamber images.
Verify the live pipeline's end-to-end processing (morphology check, classification,
final label, and display color) to confirm they end up as 'Other Species', not Aedes.
"""
import os
import sys
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from chamber_detector import ChamberDetector

CHAMBER_DIR = r"C:\Users\LAPTOP\Downloads\Chamber_Dataset"
SCREWS_OUTPUT_DIR = os.path.join(BASE_DIR, "eval", "screw_crops")
os.makedirs(SCREWS_OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print("  VERIFYING PIPELINE OUTPUT ON EMPTY-CHAMBER SCREWS (INSIDE ROI)")
    print("=" * 80)

    detector = ChamberDetector()
    detector.worker_running = False
    yolo_model = detector.yolo_model

    # Collect chamber images
    chamber_files = sorted([os.path.join(CHAMBER_DIR, f) for f in os.listdir(CHAMBER_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    screw_results = []

    for path in chamber_files:
        try:
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = cv2.imread(path)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        xmin = int(w * config.ROI_MARGIN_X)
        xmax = int(w * (1.0 - config.ROI_MARGIN_X))
        ymin = int(h * config.ROI_MARGIN_Y)
        ymax = int(h * (1.0 - config.ROI_MARGIN_Y))
        proc_frame = frame[ymin:ymax, xmin:xmax]

        # YOLO inference inside ROI at conf=0.18
        res = yolo_model(proc_frame, conf=0.18, imgsz=config.YOLO_IMGSZ, agnostic_nms=True, iou=0.45, verbose=False)
        boxes = []
        for r in res:
            for b in r.boxes:
                coords = b.xyxy.cpu().numpy()[0]
                bx1, by1, bx2, by2 = map(int, coords[:4])
                conf_val = float(b.conf[0])
                cls_id = int(b.cls[0])
                # Map back to full frame
                full_box = (bx1 + xmin, by1 + ymin, bx2 + xmin, by2 + ymin)
                boxes.append((full_box, conf_val, cls_id))

        if boxes:
            fname = os.path.basename(path)
            for b_idx, (box, yolo_conf, yolo_cls) in enumerate(boxes):
                # 1. Crop using live pipeline crop_box
                crop = detector.crop_box(frame, box)

                # 2. Check morphology
                is_insect, morph_reason = detector.check_morphology(crop)

                # 3. Full pipeline classification logic (matches _ai_worker_loop)
                if not is_insect:
                    label = "Other Species"
                    color_bgr = (255, 200, 0)
                    color_name = "Cyan (BGR 255, 200, 0)"
                else:
                    species, clf_conf, probs = detector.classifier.classify_crop(crop)
                    is_confirmed_aedes = (clf_conf >= config.CLASSIFICATION_THRESHOLD) or (yolo_conf >= 0.45)
                    if is_confirmed_aedes:
                        label = f"{species}: {clf_conf*100:.1f}%"
                        color_bgr = (0, 0, 255) if "aegypti" in species.lower() else (0, 140, 255)
                        color_name = "Red" if "aegypti" in species.lower() else "Orange"
                    else:
                        label = "Other Species"
                        color_bgr = (255, 200, 0)
                        color_name = "Cyan (BGR 255, 200, 0)"

                # Save crop
                crop_fname = f"screw_{len(screw_results)+1}_{fname}"
                crop_path = os.path.join(SCREWS_OUTPUT_DIR, crop_fname)
                cv2.imwrite(crop_path, crop)

                screw_results.append({
                    "id": len(screw_results) + 1,
                    "image": fname,
                    "box": box,
                    "yolo_conf": yolo_conf,
                    "is_insect": is_insect,
                    "morph_reason": morph_reason,
                    "final_label": label,
                    "color_name": color_name,
                    "color_bgr": color_bgr,
                    "crop_path": crop_path
                })

        del frame

    print(f"\nTotal Screw Detections Inside ROI: {len(screw_results)}")
    print("-" * 105)
    print(f"{'#':<3} | {'Image':<22} | {'YOLO Conf':<10} | {'Morphology Result':<40} | {'Final Label':<14} | {'Color'}")
    print("-" * 105)
    for s in screw_results:
        print(f"{s['id']:<3} | {s['image']:<22} | {s['yolo_conf']:>8.3f}   | {s['morph_reason']:<40} | {s['final_label']:<14} | {s['color_name']}")

if __name__ == "__main__":
    main()
