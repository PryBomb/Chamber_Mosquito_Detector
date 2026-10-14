import os
import sys
import time
import csv
import argparse
from datetime import datetime
import cv2
import numpy as np

import config

# ─────────────────────────────────────────────────────────────────────────────
# 1. Camera Abstraction (Picamera2 for Pi HQ Camera, OpenCV VideoCapture fallback)
# ─────────────────────────────────────────────────────────────────────────────
class ChamberCamera:
    def __init__(self, source=None, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT, fps=config.FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.is_picam2 = False
        self.cap = None

        # If a specific source is given (e.g. camera index or video file)
        if source is not None:
            # Check if source is a digit string (e.g. "0" -> 0)
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            self.cap = cv2.VideoCapture(source)
            if self.cap.isOpened():
                if isinstance(source, int):
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                print(f"[INFO] Camera/Video stream initialized via source: {source}")
            else:
                print(f"[WARNING] Could not open video/camera source: {source}")
            return

        # Try initializing Picamera2 first (standard on Raspberry Pi OS 64-bit)
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            cam_config = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(cam_config)
            self.picam2.start()
            self.is_picam2 = True
            print("[INFO] Camera initialized via Picamera2 (Raspberry Pi HQ Camera)")
        except Exception as e:
            print(f"[INFO] Picamera2 not available ({e}). Falling back to OpenCV VideoCapture.")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                print("[INFO] Camera initialized via OpenCV VideoCapture")
            else:
                print("[WARNING] No camera device found. Please check connections.")

    def read_frame(self):
        """Returns BGR frame for OpenCV processing, or None on failure."""
        if self.is_picam2:
            rgb_frame = self.picam2.capture_array()
            # Convert RGB (Picamera2 default) to BGR for OpenCV
            return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        elif self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def release(self):
        if self.is_picam2:
            self.picam2.stop()
        elif self.cap is not None:
            self.cap.release()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Stage 2 Classifier (TFLite Model)
# ─────────────────────────────────────────────────────────────────────────────
class MosquitoClassifier:
    def __init__(self, model_path=config.TFLITE_MODEL_PATH):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.classes = config.SPECIES_CLASSES

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found!")

        with open(self.model_path, 'rb') as f:
            model_content = f.read()

        # Try lightweight tflite_runtime first (Raspberry Pi preferred)
        try:
            from tflite_runtime.interpreter import Interpreter
            self.interpreter = Interpreter(model_content=model_content)
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
            self.interpreter = Interpreter(model_content=model_content)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"[INFO] Classifier loaded: {os.path.basename(self.model_path)}")

    def classify_crop(self, bgr_crop):
        """
        Takes a BGR image crop of a detected mosquito.
        Preprocesses to (64, 64), normalizes, and returns (label, confidence, probabilities).
        """
        img_resized = cv2.resize(bgr_crop, (64, 64))
        img_normalized = (img_resized.astype(np.float32) / 255.0).reshape(1, 64, 64, 3)

        self.interpreter.set_tensor(self.input_details[0]['index'], img_normalized)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        class_idx = int(np.argmax(output))
        confidence = float(output[class_idx])
        label = self.classes[class_idx]
        return label, confidence, output

# ─────────────────────────────────────────────────────────────────────────────
# 3. Two-Stage Chamber Detector Engine
# ─────────────────────────────────────────────────────────────────────────────
class ChamberDetector:
    def __init__(self, source=None, mode=None):
        self.camera = ChamberCamera(source=source)
        self.classifier = MosquitoClassifier()
        self.yolo_model = None
        self.mode = mode if mode else config.DETECTION_MODE
        self.last_snapshot_time = 0.0

        # Initialize CSV log if not present
        if not os.path.exists(config.LOG_FILE):
            with open(config.LOG_FILE, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Species", "Confidence", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2", "Snapshot_File"])

        # Try loading YOLOv8
        try:
            from ultralytics import YOLO
            print(f"[INFO] Loading YOLOv8 model ({config.YOLO_MODEL_PATH})...")
            self.yolo_model = YOLO(config.YOLO_MODEL_PATH)
            print("[INFO] YOLOv8 initialized successfully.")
        except ImportError:
            print("[NOTICE] 'ultralytics' not installed. Running in 'chamber_contrast' detection mode.")
            self.mode = "chamber_contrast"
        except Exception as e:
            print(f"[WARNING] Could not load YOLO: {e}. Falling back to 'chamber_contrast' mode.")
            self.mode = "chamber_contrast"

    def detect_boxes(self, frame):
        """
        Stage 1: Detect candidate bounding boxes for insects.
        Returns a list of boxes [(x1, y1, x2, y2), ...]
        """
        boxes = []
        h, w, _ = frame.shape

        if self.mode == "yolo" and self.yolo_model is not None:
            results = self.yolo_model(frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    area = (x2 - x1) * (y2 - y1)
                    if config.MIN_BBOX_AREA <= area <= config.MAX_BBOX_AREA:
                        boxes.append((x1, y1, x2, y2))
        else:
            # Chamber Contrast Detection:
            # In a white chamber, mosquitoes are dark objects with high contrast.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Threshold to isolate dark objects on white background
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 25, 10
            )
            # Morphological cleaning
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if config.MIN_BBOX_AREA <= area <= config.MAX_BBOX_AREA:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    # Exclude extreme aspect ratios (lines/scratches)
                    aspect = float(bw) / float(bh)
                    if 0.25 <= aspect <= 4.0:
                        boxes.append((x, y, x + bw, y + bh))

        return boxes

    def crop_box(self, frame, box):
        """Adds padding around bounding box and crops safely within frame boundaries."""
        h, w, _ = frame.shape
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * config.BBOX_PADDING)
        pad_y = int(bh * config.BBOX_PADDING)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        return crop

    def log_detection(self, species, confidence, box, frame):
        """Logs detection to CSV and saves snapshot."""
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        snapshot_filename = ""

        # Auto-save snapshot with cooldown
        current_time = time.time()
        if config.AUTO_SAVE_SNAPSHOT and (current_time - self.last_snapshot_time >= config.SNAPSHOT_COOLDOWN_SEC):
            clean_species = species.replace(" ", "_")
            file_name = f"{now.strftime('%Y%m%d_%H%M%S')}_{clean_species}_{int(confidence*100)}pct.jpg"
            snapshot_path = os.path.join(config.CAPTURES_DIR, file_name)
            cv2.imwrite(snapshot_path, frame)
            snapshot_filename = file_name
            self.last_snapshot_time = current_time
            print(f"[ALERT] Snapshot saved: {file_name}")

        with open(config.LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, species, f"{confidence*100:.2f}%", box[0], box[1], box[2], box[3], snapshot_filename])

    def run(self):
        print("\n=======================================================")
        print(" Mosquito Chamber Detection System (Raspi HQ Camera)")
        print(f" Mode: {self.mode.upper()}  |  Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        print(" Controls: [Q] Quit  |  [S] Save Snapshot  |  [M] Toggle Mode")
        print("=======================================================\n")

        window_name = "Chamber Mosquito Detector - Live Stream"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 768)

        prev_time = time.time()
        fps = 0.0

        while True:
            frame = self.camera.read_frame()
            if frame is None:
                print("[ERROR] Failed to grab camera frame.")
                break

            # Calculate FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-5))
            prev_time = now

            display = frame.copy()

            # Stage 1: Detect candidate insect bounding boxes
            boxes = self.detect_boxes(frame)

            detected_count = len(boxes)

            # Stage 2 & 3: Crop and Classify each detected insect
            for box in boxes:
                x1, y1, x2, y2 = box
                crop = self.crop_box(frame, box)
                if crop.size == 0:
                    continue

                species, conf, probs = self.classifier.classify_crop(crop)

                # Determine display color and alert status
                if conf >= config.CLASSIFICATION_THRESHOLD:
                    if "aegypti" in species.lower():
                        color = (0, 0, 255)  # Red for Aedes aegypti (Dengue/Zika vector)
                    else:
                        color = (255, 140, 0)  # Orange for Aedes albopictus
                    label = f"{species}: {conf*100:.1f}%"
                    self.log_detection(species, conf, box, frame)
                else:
                    color = (200, 200, 200)
                    label = f"Insect ({species} {conf*100:.0f}%)"

                # Draw bounding box & tag
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # Draw HUD bar
            h, w, _ = display.shape
            cv2.rectangle(display, (0, 0), (w, 55), (20, 20, 20), -1)
            hud_text = f"FPS: {fps:.1f}  |  Mode: {self.mode.upper()}  |  Insects in Chamber: {detected_count}"
            cv2.putText(display, hud_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Bottom help line
            cv2.rectangle(display, (0, h - 25), (w, h), (20, 20, 20), -1)
            cv2.putText(display, "[Q] Quit  |  [S] Save Snapshot  |  [M] Toggle Mode (YOLO / Contrast)",
                        (15, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                snap_path = os.path.join(config.CAPTURES_DIR, f"manual_snap_{now_str}.jpg")
                cv2.imwrite(snap_path, display)
                print(f"[INFO] Manual snapshot saved: {snap_path}")
            elif key == ord('m'):
                self.mode = "chamber_contrast" if self.mode == "yolo" else "yolo"
                print(f"[INFO] Detection mode toggled to: {self.mode.upper()}")

        self.camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released. Exiting clean.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Chamber Mosquito Detector")
    parser.add_argument("--source", type=str, default=None,
                        help="Camera index (e.g. 0, 1) or video file path. Default: auto-detect Picamera2 / Webcam 0")
    parser.add_argument("--mode", type=str, choices=["yolo", "chamber_contrast"], default=None,
                        help="Detection mode: 'yolo' or 'chamber_contrast' (default from config: 'yolo')")
    args = parser.parse_args()

    detector = ChamberDetector(source=args.source, mode=args.mode)
    detector.run()
