"""
Extract and visualize 10 extra boxes detected at conf 0.12 and 0.15
specifically for Aedes albopictus images that were missed at baseline conf 0.18.
"""
import os
import sys
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from eval.experiment_crop_tuning import get_testing_images, split_data, ALBOPICTUS_DIR
from chamber_detector import ChamberDetector

OUTPUT_DIR = os.path.join(BASE_DIR, "eval", "sweep_extra_boxes_albo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("[INFO] Initializing YOLO for Albopictus extra box extraction...")
    detector = ChamberDetector()
    detector.worker_running = False
    yolo_model = detector.yolo_model

    albo_items = get_testing_images(ALBOPICTUS_DIR, "Aedes albopictus")
    print(f"Total Albopictus testing images: {len(albo_items)}")

    extra_detected = []

    for item in albo_items:
        path = item["path"]
        try:
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = cv2.imread(path)
        if frame is None:
            continue

        # Check if missed at 0.18 (full res)
        res_018 = yolo_model(frame, conf=0.18, imgsz=640, agnostic_nms=True, iou=0.45, verbose=False)
        has_018 = any(len(r.boxes) > 0 for r in res_018)

        if not has_018:
            res_015 = yolo_model(frame, conf=0.15, imgsz=640, agnostic_nms=True, iou=0.45, verbose=False)
            boxes_015 = [b for r in res_015 for b in r.boxes]

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

                if len(extra_detected) >= 10:
                    break
        del frame

    print(f"[INFO] Found {len(extra_detected)} extra detected Albopictus images. Saving...")
    saved_info = []

    for idx, d in enumerate(extra_detected[:10]):
        item = d["item"]
        frame = d["frame"]
        conf_level = d["conf_level"]
        boxes = d["boxes"]

        vis_frame = frame.copy()
        crop_paths = []

        b = boxes[0]
        coords = b.xyxy.cpu().numpy()[0]
        x1, y1, x2, y2 = map(int, coords[:4])
        conf_val = float(b.conf[0])
        cls_id = int(b.cls[0])
        cls_name = "aegypti" if cls_id == 0 else "albopictus"

        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 140, 255), 3)
        label = f"{cls_name} conf:{conf_val:.3f}"
        cv2.putText(vis_frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

        crop = detector.crop_box(frame, (x1, y1, x2, y2))
        crop_name = f"crop_albo_{idx+1}_conf{conf_level}_{item['filename']}"
        crop_path = os.path.join(OUTPUT_DIR, crop_name)
        cv2.imwrite(crop_path, crop)
        crop_paths.append(crop_path)

        vis_name = f"vis_albo_{idx+1}_conf{conf_level}_{item['filename']}"
        vis_path = os.path.join(OUTPUT_DIR, vis_name)
        cv2.imwrite(vis_path, vis_frame)

        saved_info.append({
            "idx": idx + 1,
            "filename": item["filename"],
            "species": item["true_label"],
            "conf_level": conf_level,
            "box_conf": conf_val,
            "box_coords": (x1, y1, x2, y2),
            "vis_path": vis_path,
            "crop_path": crop_path
        })

    print("\nSummary of 10 Albopictus Extra Boxes:")
    for s in saved_info:
        print(f"  #{s['idx']}: {s['filename']} ({s['species']}) | Conf Level: {s['conf_level']} (Actual: {s['box_conf']:.3f}) | BBox: {s['box_coords']}")
        print(f"      Crop saved: {s['crop_path']}")

if __name__ == "__main__":
    main()
