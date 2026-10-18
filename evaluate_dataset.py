import os
import glob
import argparse
import time
import cv2
import numpy as np

import config
from chamber_detector import ChamberDetector

def evaluate_dataset(dataset_dir, is_negative=True, mode=None, save_debug=True):
    print("=" * 65)
    print(" Mosquito Detector Batch Benchmark & Evaluation Tool")
    print(f" Dataset Directory : {dataset_dir}")
    print(f" Dataset Type      : {'Negative (Empty Chamber)' if is_negative else 'Positive (Mosquitoes)'}")
    print(f" Mode              : {mode if mode else config.DETECTION_MODE.upper()}")
    print(f" ROI Margin X      : {config.ROI_MARGIN_X * 100:.1f}%")
    print(f" ROI Margin Y      : {config.ROI_MARGIN_Y * 100:.1f}%")
    print("=" * 65)

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Directory not found: {dataset_dir}")
        return

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        print("[ERROR] No image files found in the specified directory.")
        return

    print(f"[INFO] Found {len(image_paths)} images. Initializing detector...")

    # Initialize detector with image folder as dummy source to avoid camera polling
    detector = ChamberDetector(source=dataset_dir, mode=mode)

    out_dir = os.path.join(config.BASE_DIR, "evaluation_results")
    if save_debug:
        os.makedirs(out_dir, exist_ok=True)

    total_images = len(image_paths)
    images_with_detections = 0
    total_detected_boxes = 0
    inference_times = []

    print("\n--- Running Evaluation ---")
    for idx, img_path in enumerate(image_paths):
        try:
            frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Resize to standard frame resolution for consistent scale
        fh, fw = frame.shape[:2]
        if (fw, fh) != (config.FRAME_WIDTH, config.FRAME_HEIGHT):
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        t0 = time.time()
        boxes = detector.detect_boxes(frame)
        dt = time.time() - t0
        inference_times.append(dt)

        num_boxes = len(boxes)
        if num_boxes > 0:
            images_with_detections += 1
            total_detected_boxes += num_boxes

            if save_debug:
                debug_frame = frame.copy()
                for (bx1, by1, bx2, by2) in boxes:
                    cv2.rectangle(debug_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(debug_frame, f"FP ({bx2-bx1}x{by2-by1})", (bx1, max(15, by1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                debug_name = f"detection_{os.path.basename(img_path)}"
                cv2.imwrite(os.path.join(out_dir, debug_name), debug_frame)

        if (idx + 1) % 25 == 0 or (idx + 1) == total_images:
            print(f"[{idx + 1}/{total_images}] images processed... (Detections so far: {total_detected_boxes})")

    avg_time = np.mean(inference_times) if inference_times else 0.0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0

    print("\n" + "=" * 65)
    print("                    BENCHMARK RESULTS")
    print("=" * 65)
    print(f" Total Images Processed       : {total_images}")
    print(f" Average Processing Speed     : {avg_fps:.1f} FPS ({avg_time*1000:.1f} ms/frame)")

    if is_negative:
        clean_images = total_images - images_with_detections
        tnr = (clean_images / total_images) * 100.0
        fpr = (images_with_detections / total_images) * 100.0

        print(f" Clean Images (Zero False Positives) : {clean_images} / {total_images} ({tnr:.2f}%)")
        print(f" Images with False Positives         : {images_with_detections} / {total_images} ({fpr:.2f}%)")
        print(f" Total False Detection Boxes         : {total_detected_boxes}")
        print("-" * 65)
        if images_with_detections == 0:
            print("  EXCELLENT: 100% True Negative Rate! Zero false alarms in empty chamber.")
        else:
            print(f"  Note: Debug images saved to: {out_dir}")
    else:
        print(f" Images with Mosquito Detected: {images_with_detections} / {total_images}")
        print(f" Total Detected Boxes         : {total_detected_boxes}")

    print("=" * 65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mosquito Detector on Image Datasets")
    parser.add_argument("--dir", type=str, default=r"C:\Users\LAPTOP\Downloads\Chamber_Dataset",
                        help="Path to folder containing test images")
    parser.add_argument("--mode", type=str, choices=["chamber_contrast", "yolo"], default="chamber_contrast",
                        help="Detection engine to benchmark ('chamber_contrast' or 'yolo')")
    parser.add_argument("--positive", action="store_true",
                        help="Set if dataset contains mosquitoes (default: False / Negative dataset)")
    parser.add_argument("--no_debug", action="store_true",
                        help="Disable saving debug images of detected boxes")

    args = parser.parse_args()
    evaluate_dataset(
        dataset_dir=args.dir,
        is_negative=not args.positive,
        mode=args.mode,
        save_debug=not args.no_debug
    )
