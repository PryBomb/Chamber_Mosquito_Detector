import os
import sys
import time
import csv
import argparse
from datetime import datetime
import cv2
import numpy as np

import config

# Ensure console output handles non-ASCII folder paths safely on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

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

        # If a specific source is given (e.g. folder of images, single image, camera index, or video file)
        if source is not None:
            # Check if source is a folder of images
            if isinstance(source, str) and os.path.isdir(source):
                import glob
                extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                self.img_files = []
                for ext in extensions:
                    self.img_files.extend(glob.glob(os.path.join(source, ext)))
                self.img_files = sorted(self.img_files)
                self.img_idx = 0
                self.is_image_folder = True
                print(f"[INFO] Initialized image dataset stream from: {source} ({len(self.img_files)} images)")
                return

            # Check if source is a single image file
            if isinstance(source, str) and os.path.isfile(source) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.img_files = [source]
                self.img_idx = 0
                self.is_image_folder = True
                print(f"[INFO] Initialized single image input: {source}")
                return

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
        if getattr(self, 'is_image_folder', False):
            if not self.img_files:
                return None
            if self.img_idx >= len(self.img_files):
                self.img_idx = 0  # Loop dataset playback
            path = self.img_files[self.img_idx]
            self.current_filename = os.path.basename(path)
            self.img_idx += 1
            try:
                frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                frame = cv2.imread(path)
            if frame is not None:
                # Resize if needed to match standard chamber resolution
                fh, fw = frame.shape[:2]
                if (fw, fh) != (self.width, self.height):
                    frame = cv2.resize(frame, (self.width, self.height))
            return frame

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
# 2. Stage 2 Classifier (LiteRT / ONNX / TFLite Multi-Backend)
# ─────────────────────────────────────────────────────────────────────────────
class MosquitoClassifier:
    def __init__(self, tflite_path=config.TFLITE_MODEL_PATH, onnx_path=config.ONNX_MODEL_PATH):
        self.tflite_path = tflite_path
        self.onnx_path = onnx_path
        self.classes = config.SPECIES_CLASSES
        self.backend = None
        self.interpreter = None
        self.session = None

        # 1. Try modern Google LiteRT (Official TFLite successor; supports Python 3.12 / 3.13)
        if os.path.exists(self.tflite_path):
            with open(self.tflite_path, 'rb') as f:
                model_content = f.read()

            try:
                from ai_edge_litert.interpreter import Interpreter
                self.interpreter = Interpreter(model_content=model_content)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.backend = "litert"
                print(f"[INFO] Classifier loaded via Google AI Edge LiteRT: {os.path.basename(self.tflite_path)}")
                return
            except ImportError:
                pass

            # 2. Try legacy tflite_runtime (for Python <= 3.11)
            try:
                from tflite_runtime.interpreter import Interpreter
                self.interpreter = Interpreter(model_content=model_content)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.backend = "tflite_runtime"
                print(f"[INFO] Classifier loaded via tflite-runtime: {os.path.basename(self.tflite_path)}")
                return
            except ImportError:
                pass

        # 3. Try ONNX Runtime (official pre-built wheels for Python 3.13 on aarch64 / ARM64)
        if os.path.exists(self.onnx_path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
                self.onnx_input_name = self.session.get_inputs()[0].name
                self.backend = "onnx"
                print(f"[INFO] Classifier loaded via ONNX Runtime: {os.path.basename(self.onnx_path)}")
                return
            except ImportError:
                pass

        # 4. Fallback to full TensorFlow Lite (PC / Development)
        if os.path.exists(self.tflite_path):
            try:
                from tensorflow.lite.python.interpreter import Interpreter
                self.interpreter = Interpreter(model_content=model_content)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.backend = "tensorflow"
                print(f"[INFO] Classifier loaded via TensorFlow Lite: {os.path.basename(self.tflite_path)}")
                return
            except ImportError:
                pass

        raise RuntimeError(
            "Could not load any AI inference backend for mosquito classifier!\n"
            "On Python 3.13 (Debian Trixie / Raspberry Pi), install either:\n"
            "  pip install ai-edge-litert\n"
            "or\n"
            "  pip install onnxruntime"
        )

    def classify_crop(self, bgr_crop):
        """
        Takes a BGR image crop of a detected mosquito.
        Preprocesses to (64, 64), normalizes, and returns (label, confidence, probabilities).
        """
        img_resized = cv2.resize(bgr_crop, (64, 64))
        img_normalized = (img_resized.astype(np.float32) / 255.0).reshape(1, 64, 64, 3)

        if self.backend in ["litert", "tflite_runtime", "tensorflow"]:
            self.interpreter.set_tensor(self.input_details[0]['index'], img_normalized)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        elif self.backend == "onnx":
            outputs = self.session.run(None, {self.onnx_input_name: img_normalized})
            output = outputs[0][0]
        else:
            raise RuntimeError("Classifier backend not initialized.")

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
        Respects calibrated Chamber ROI to exclude acrylic wall seams.
        Returns a list of boxes [(x1, y1, x2, y2), ...]
        """
        boxes = []
        h, w, _ = frame.shape

        # Calculate Chamber ROI boundaries
        if getattr(config, 'USE_CHAMBER_ROI', False):
            xmin = int(w * config.ROI_MARGIN_X)
            xmax = int(w * (1.0 - config.ROI_MARGIN_X))
            ymin = int(h * config.ROI_MARGIN_Y)
            ymax = int(h * (1.0 - config.ROI_MARGIN_Y))
            proc_frame = frame[ymin:ymax, xmin:xmax]
            offset_x, offset_y = xmin, ymin
        else:
            proc_frame = frame
            offset_x, offset_y = 0, 0

        # Initialize background model if not present
        if not hasattr(self, 'bg_gray') or self.bg_gray is None:
            gray_proc = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            self.bg_gray = cv2.GaussianBlur(gray_proc, (9, 9), 0).astype(np.float32)
            print("[INFO] Background baseline captured. Chamber calibrated.")

        if self.mode == "yolo" and self.yolo_model is not None:
            # Accelerated YOLO for Raspberry Pi
            results = self.yolo_model(proc_frame, conf=config.YOLO_CONF_THRESHOLD, imgsz=getattr(config, 'YOLO_IMGSZ', 320), verbose=False)
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    area = (x2 - x1) * (y2 - y1)
                    if config.MIN_BBOX_AREA <= area <= config.MAX_BBOX_AREA:
                        boxes.append((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y))
        else:
            # Chamber Baseline Subtraction Detection (100% Glue-Immune, 30+ FPS)
            # Subtracts the static chamber background (glue, walls, shadows)
            curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            curr_blur = cv2.GaussianBlur(curr_gray, (9, 9), 0)

            # Difference between current frame and empty chamber baseline
            # Mosquito is darker than the white floor, so baseline - current > threshold
            diff = cv2.subtract(self.bg_gray.astype(np.uint8), curr_blur)

            # Only genuine new dark objects (mosquito body) trigger the threshold
            _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)

            # Morphological noise filter
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if config.MIN_BBOX_AREA <= area <= config.MAX_BBOX_AREA:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = float(bw) / float(bh)
                    if 0.25 <= aspect <= 4.0:
                        boxes.append((x + offset_x, y + offset_y, x + bw + offset_x, y + bh + offset_y))

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
            h, w, _ = display.shape

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

            # Draw Chamber ROI overlay if enabled
            if getattr(config, 'SHOW_ROI_OVERLAY', False) and getattr(config, 'USE_CHAMBER_ROI', False):
                rx1 = int(w * config.ROI_MARGIN_X)
                rx2 = int(w * (1.0 - config.ROI_MARGIN_X))
                ry1 = int(h * config.ROI_MARGIN_Y)
                ry2 = int(h * (1.0 - config.ROI_MARGIN_Y))
                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(display, "CHAMBER ROI ACTIVE", (rx1 + 8, ry1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw HUD bar
            cv2.rectangle(display, (0, 0), (w, 55), (20, 20, 20), -1)
            src_info = f"File: {self.camera.current_filename} | " if getattr(self.camera, 'is_image_folder', False) else ""
            hud_text = f"{src_info}FPS: {fps:.1f} | Mode: {self.mode.upper()} | Insects: {detected_count}"
            cv2.putText(display, hud_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            # Bottom help line
            cv2.rectangle(display, (0, h - 25), (w, h), (20, 20, 20), -1)
            cv2.putText(display, "[C] Calibrate Baseline (Zero Noise) | [M] Toggle Mode | [S] Save | [Q] Quit",
                        (15, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow(window_name, display)

            # Adjust wait time for image dataset vs live camera
            delay = 100 if getattr(self.camera, 'is_image_folder', False) else 1
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                self.bg_gray = None  # Resets and re-captures current empty chamber as clean baseline
                print("[INFO] Recalibrating chamber baseline... Captured clean background!")
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
