"""
Extract and visualize the extra boxes detected at conf 0.12 and 0.15
on the images that were missed at baseline conf 0.18.
Saves annotated full frames and high-res crops for visual inspection.
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
from chamber_detector import ChamberDetector

OUTPUT_DIR = os.path.join(BASE_DIR, "eval", "sweep_extra_boxes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("[INFO] Initializing YOLO...")
    detector = ChamberDetector()
    detector.worker_running = False
    yolo_model = detector.yolo_model

    # Collect 720 testing images
    aeg_items = get_testing_images(AEGYPTI_DIR, "Aedes aegypti")
    albo_items = get_testing_images(ALBOPICTUS_DIR, "Aedes albopictus")
    aeg_tune, aeg_final = split_data(aeg_items, seed=42)
    albo_tune, albo_final = split_data(albo_items, seed=42)
    all_720 = aeg_tune + albo_tune + aeg_final + albo_final

    print("[INFO] Scanning for images missed at 0.18 but detected at 0.12 or 0.15...")
    extra_detected = []

    for item in all_720:
        path = item["path"]
        try:
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = cv2.imread(path)
        if frame is None:
            continue

        # Check if missed at 0.18
        res_018 = yolo_model(frame, conf=0.18, imgsz=640, agnostic_nms=True, iou=0.45, verbose=False)
        has_018 = any(len(r.boxes) > 0 for r in res_018)

        if not has_018:
            # Check if detected at 0.15
            res_015 = yolo_model(frame, conf=0.15, imgsz=640, agnostic_nms=True, iou=0.45, verbose=False)
            boxes_015 = [b for r in res_015 for b in r.boxes]

            # Check if detected at 0.12
            res_012 = yolo_model(frame, conf=0.12, imgsz=640, agnostic_nms=True, iou=0.45, verbose=False)
            boxes_012 = [b for r in res_012 for b in r.boxes]

            if boxes_015 or boxes_012:
                best_boxes = boxes_015 if boxes_015 else boxes_012
                conf_level = "0.15" if boxes_015 else "0.12"
                extra_detected.append({
                    "item": item,
                    "frame": frame,
                    "conf_level": conf_level,
                    "boxes": best_boxes
                })

                if len(extra_detected) >= 12:
                    break
        del frame

    print(f"\n[INFO] Found {len(extra_detected)} extra detected images. Saving visualizations...")
    saved_info = []

    for idx, d in enumerate(extra_detected[:10]):
        item = d["item"]
        frame = d["frame"]
        conf_level = d["conf_level"]
        boxes = d["boxes"]
        h, w = frame.shape[:2]

        vis_frame = frame.copy()
        crop_paths = []

        for b_idx, b in enumerate(boxes):
            coords = b.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, coords[:4])
            conf_val = float(b.conf[0])
            cls_id = int(b.cls[0])
            cls_name = "aegypti" if cls_id == 0 else "albopictus"

            # Draw box on frame
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{cls_name} conf:{conf_val:.3f}"
            cv2.putText(vis_frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Crop box with pad 0.60 square
            crop = detector.crop_box(frame, (x1, y1, x2, y2))
            crop_name = f"crop_{idx+1}_conf{conf_level}_{item['filename']}"
            crop_path = os.path.join(OUTPUT_DIR, crop_name)
            cv2.imwrite(crop_path, crop)
            crop_paths.append(crop_path)

        vis_name = f"vis_{idx+1}_conf{conf_level}_{item['filename']}"
        vis_path = os.path.join(OUTPUT_DIR, vis_name)
        cv2.imwrite(vis_path, vis_frame)

        saved_info.append({
            "idx": idx + 1,
            "filename": item["filename"],
            "species": item["true_label"],
            "conf_level": conf_level,
            "box_conf": float(boxes[0].conf[0]),
            "box_coords": (x1, y1, x2, y2),
            "vis_path": vis_path,
            "crop_path": crop_paths[0] if crop_paths else None
        })

    print("\nSummary of 10 Extra Boxes:")
    for s in saved_info:
        print(f"  #{s['idx']}: {s['filename']} ({s['species']}) | Conf Level: {s['conf_level']} (Actual: {s['box_conf']:.3f}) | BBox: {s['box_coords']}")
        print(f"      Crop saved: {s['crop_path']}")

if __name__ == "__main__":
    main()
