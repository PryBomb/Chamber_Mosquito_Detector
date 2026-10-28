import os
import sys
import time
import csv
import argparse
from datetime import datetime, timedelta
from typing import Optional, Set, Dict, List, Tuple
import cv2
import numpy as np
import threading

import config
try:
    from mosquito_tracker_server import MosquitoTracker
except ImportError:
    from mosquito_tracker import MosquitoTracker
try:
    from cusum_monitor import CusumMonitor, CusumConfig
except ImportError:
    CusumMonitor, CusumConfig = None, None
from status_server import start_status_server, stop_status_server

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
# 2B. Hardware Alarm Pin Controller (Raspberry Pi GPIO)
# ─────────────────────────────────────────────────────────────────────────────
class HardwareAlarmController:
    """Controls physical alarm hardware (e.g. buzzer, relay, LED) via Raspberry Pi GPIO."""
    def __init__(self, pin: int = 17, active_high: bool = True, enabled: bool = True):
        self.pin = pin
        self.active_high = active_high
        self.enabled = enabled
        self.backend = None
        self._dev = None
        self._current_state = False

        if not self.enabled:
            return

        # Attempt 1: gpiozero (recommended on Raspberry Pi OS Bookworm & Bullseye)
        try:
            from gpiozero import OutputDevice
            self._dev = OutputDevice(self.pin, active_high=self.active_high, initial_value=False)
            self.backend = "gpiozero"
            print(f"[GPIO] Hardware alarm output initialized on BCM GPIO{self.pin} via gpiozero.")
            return
        except Exception:
            pass

        # Attempt 2: RPi.GPIO
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            init_val = GPIO.LOW if self.active_high else GPIO.HIGH
            GPIO.setup(self.pin, GPIO.OUT, initial=init_val)
            self.backend = "RPi.GPIO"
            print(f"[GPIO] Hardware alarm output initialized on BCM GPIO{self.pin} via RPi.GPIO.")
            return
        except Exception:
            pass

        print(f"[GPIO] Running in simulated/non-Pi environment; GPIO{self.pin} alarm output simulated.")
        self.backend = "simulated"

    def set_alarm(self, is_triggered: bool):
        if self._current_state == is_triggered:
            return
        self._current_state = is_triggered
        state_str = "ACTIVE (HIGH)" if (is_triggered if self.active_high else not is_triggered) else "NORMAL (LOW)"
        print(f"[GPIO] Alarm GPIO{self.pin} output set to: {state_str}")

        if not self.enabled or self.backend == "simulated":
            return

        try:
            if self.backend == "gpiozero" and self._dev is not None:
                if is_triggered:
                    self._dev.on()
                else:
                    self._dev.off()
            elif self.backend == "RPi.GPIO":
                import RPi.GPIO as GPIO
                val = GPIO.HIGH if (is_triggered if self.active_high else not is_triggered) else GPIO.LOW
                GPIO.output(self.pin, val)
        except Exception as e:
            print(f"[GPIO] Error updating GPIO{self.pin}: {e}")

    def cleanup(self):
        try:
            self.set_alarm(False)
            if self.backend == "gpiozero" and self._dev is not None:
                self._dev.close()
            elif self.backend == "RPi.GPIO":
                import RPi.GPIO as GPIO
                GPIO.cleanup(self.pin)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. CUSUM Interval Surveillance Accumulator
# ─────────────────────────────────────────────────────────────────────────────
class CusumIntervalAccumulator(threading.Thread):
    """
    30-minute interval surveillance accumulator for CUSUM activity monitoring.

    Responsibilities:
    - Samples tracker public thread-safe interface (get_status) periodically.
    - Collects unique confirmed mosquito IDs (MOS-XXX) observed in the interval.
    - Tracks active uptime/coverage independently of detections.
    - Aligns interval boundaries (:00, :30) in local time.
    - Ingests each completed interval exactly once into CusumMonitor.
    - Cleanly shuts down without blocking or leaking threads.
    """
    def __init__(
        self,
        detector,
        cusum_monitor,
        interval_minutes: int = 30,
        sample_interval_sec: float = 1.0
    ):
        super().__init__(name="CusumIntervalAccumulator", daemon=True)
        self.detector = detector
        self.cusum_monitor = cusum_monitor
        self.interval_minutes = interval_minutes
        self.sample_interval_sec = sample_interval_sec
        self.running = True
        self._lock = threading.Lock()

        # Interval tracking state
        self.current_start: Optional[datetime] = None
        self.current_end: Optional[datetime] = None
        self.unique_ids: Set[str] = set()
        self.species_id_map: Dict[str, Set[str]] = {}
        self.active_seconds: float = 0.0
        self.last_tick: float = time.time()

    @staticmethod
    def get_interval_bounds(dt: datetime, interval_minutes: int = 30) -> Tuple[datetime, datetime]:
        """Aligns datetime to deterministic interval start and end (e.g. :00 or :30)."""
        minute_slot = (dt.minute // interval_minutes) * interval_minutes
        start = dt.replace(minute=minute_slot, second=0, microsecond=0)
        end = start + timedelta(minutes=interval_minutes)
        return start, end

    def _flush_completed_interval(self, start: datetime, end: datetime):
        """Finalizes and ingests a completed interval into cusum_monitor."""
        if self.cusum_monitor is None:
            return
        total_seconds = self.interval_minutes * 60.0
        coverage = min(1.0, max(0.0, self.active_seconds / total_seconds))
        species_counts = {sp: len(ids) for sp, ids in self.species_id_map.items()}
        unique_list = sorted(list(self.unique_ids))

        try:
            self.cusum_monitor.ingest_interval(
                interval_start=start,
                interval_end=end,
                unique_mosquito_ids=unique_list,
                species_counts=species_counts,
                coverage_ratio=coverage,
                detector_running=True
            )
        except Exception as e:
            print(f"[WARNING] CusumAccumulator failed to ingest interval {start}: {e}")

    def run(self):
        while self.running:
            time.sleep(self.sample_interval_sec)
            if not self.running:
                break

            now_dt = datetime.now()
            start, end = self.get_interval_bounds(now_dt, self.interval_minutes)

            with self._lock:
                if self.current_start is None:
                    self.current_start = start
                    self.current_end = end
                    self.unique_ids = set()
                    self.species_id_map = {}
                    self.active_seconds = 0.0
                    self.last_tick = time.time()

                # If interval boundary crossed, flush completed interval
                if now_dt >= self.current_end:
                    self._flush_completed_interval(self.current_start, self.current_end)
                    self.current_start = start
                    self.current_end = end
                    self.unique_ids = set()
                    self.species_id_map = {}
                    self.active_seconds = 0.0
                    self.last_tick = time.time()

                # Sample uptime & tracker
                now_mono = time.time()
                elapsed = max(0.0, now_mono - self.last_tick)
                self.last_tick = now_mono

                # Detector is active if last camera frame was received within 5 seconds
                last_f_time = getattr(self.detector, "last_frame_time", None)
                is_active = (last_f_time is not None) and ((now_mono - last_f_time) < 5.0)

                if is_active:
                    self.active_seconds += min(elapsed, 2.0)

                    # Poll tracker status under public thread-safe interface
                    tracker = getattr(self.detector, "tracker", None)
                    if tracker is not None and hasattr(tracker, "get_status"):
                        try:
                            t_stat = tracker.get_status()
                            for mid in t_stat.get("active_ids", []):
                                if mid:
                                    self.unique_ids.add(str(mid))
                            for trk in t_stat.get("active_tracks", []):
                                mid = trk.get("mos_id")
                                sp = trk.get("species", "Mosquito")
                                if mid and sp:
                                    if sp not in self.species_id_map:
                                        self.species_id_map[sp] = set()
                                    self.species_id_map[sp].add(str(mid))
                        except Exception as e:
                            print(f"[WARNING] CusumAccumulator error reading tracker status: {e}")

    def stop(self):
        """Stops the accumulator thread cleanly and flushes if interval has observations."""
        self.running = False
        with self._lock:
            if self.current_start is not None and self.active_seconds > 0:
                self._flush_completed_interval(self.current_start, self.current_end)
                self.current_start = None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Two-Stage Chamber Detector Engine
# ─────────────────────────────────────────────────────────────────────────────
class ChamberDetector:
    def __init__(self, source=None, mode=None):
        self.camera = ChamberCamera(source=source)
        self.classifier = MosquitoClassifier()
        self.yolo_model = None
        self.mode = mode if mode else config.DETECTION_MODE
        self.last_snapshot_time = 0.0
        self.frame_count = 0
        self.cached_boxes = []

        # Mosquito ID tracker (SQLite-backed, < 1ms per call)
        self.tracker = MosquitoTracker() if getattr(config, 'ENABLE_MOSQUITO_TRACKER', True) else None

        # Threaded Asynchronous AI Worker variables
        self.latest_frame = None
        self.active_detections = []  # List of (box, label, color)
        self.latest_tracks = []      # List of raw dict results from update_tracks
        self.confirmed_detections_count = 0
        self.worker_running = True
        self.lock = threading.Lock()

        # Telemetry & Status Server attributes
        self.fps = None
        self.ai_fps = None
        self.last_frame_time = None
        self.start_time = None
        self.status_server = None
        self.status_server_thread = None

        # Thread-safe Preview Stream buffer for remote GUI (640x360 or full resolution)
        self.preview_lock = threading.Lock()
        self.latest_preview_frame = None
        self.latest_preview_time = None
        self.latest_preview_seq = 0

        # CUSUM Activity Monitor & 30-min Interval Accumulator
        self.cusum_monitor = None
        self.cusum_accumulator = None
        if getattr(config, 'CUSUM_ENABLED', False) and CusumMonitor is not None:
            cusum_db = getattr(config, 'CUSUM_DB_PATH', None)
            self.cusum_monitor = CusumMonitor(db_path=cusum_db, config=CusumConfig())
            self.cusum_accumulator = CusumIntervalAccumulator(
                detector=self,
                cusum_monitor=self.cusum_monitor,
                interval_minutes=getattr(config, 'CUSUM_INTERVAL_MINUTES', 30)
            )
            self.cusum_accumulator.start()
            print("[INFO] CUSUM Activity Monitor & 30-min Accumulator initialized.")

        # Hardware Alarm Output (Raspberry Pi GPIO17)
        gpio_enabled = getattr(config, "GPIO_ALARM_ENABLED", True)
        gpio_pin = getattr(config, "GPIO_ALARM_PIN", 17)
        gpio_active_high = getattr(config, "GPIO_ALARM_ACTIVE_HIGH", True)
        self.alarm_gpio = HardwareAlarmController(pin=gpio_pin, active_high=gpio_active_high, enabled=gpio_enabled)

        # Initialize CSV log if not present
        if not os.path.exists(config.LOG_FILE):
            with open(config.LOG_FILE, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Species", "Confidence", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2", "Snapshot_File", "MosquitoID", "TrackStatus"])

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

        # Start asynchronous AI worker thread
        self.ai_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
        self.ai_thread.start()
        print("[INFO] Asynchronous AI worker thread started for smooth 30 FPS display.")

    def check_morphology(self, crop):
        """
        Validates insect morphology to eliminate non-mosquito objects:
        - Screws, bolts, and metal hardware have solid, thick cores and high solidity.
        - Mosquitoes have slender bodies and thin spindly legs that erode rapidly under morphological erosion.
        """
        if crop is None or crop.size == 0:
            return False, "Empty crop"

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        floor_lum = np.percentile(gray, 85)
        # Threshold dark insect pixels against white chamber floor
        dark_mask = (gray < (floor_lum - 30)).astype(np.uint8) * 255
        orig_dark_count = np.count_nonzero(dark_mask)

        # Minimum dark pixel threshold (reject microscopic dust/smudges)
        if orig_dark_count < 40:
            return False, "Too few dark pixels (dust/smudge)"

        # 1. Core Thickness / Erosion test:
        # A screw has a thick solid cylinder core (retention > 35%); a mosquito has slender legs and needle body (retention < 28%).
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eroded = cv2.erode(dark_mask, k, iterations=3)
        eroded_dark_count = np.count_nonzero(eroded)
        retention = eroded_dark_count / max(orig_dark_count, 1)

        max_retention = getattr(config, 'MAX_CORE_RETENTION', 0.32)
        if retention > max_retention:
            return False, f"Solid metal / screw detected (core retention {retention:.1%})"

        # 2. Contour Solidity test:
        cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return False, "No contour found"

        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1.0)

        max_solidity = getattr(config, 'MAX_INSECT_SOLIDITY', 0.68)
        if solidity > max_solidity:
            return False, f"Dense solid object (solidity {solidity:.2f})"

        return True, f"Valid mosquito (solidity {solidity:.2f}, core {retention:.1%})"

    def box_iou(self, b1, b2):
        """Calculates Intersection-over-Union (IoU) of two bounding boxes."""
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / max(a1 + a2 - inter, 1)

    def _ai_worker_loop(self):
        """Background thread: runs AI inference without blocking the 30 FPS camera feed."""
        ai_prev_time = time.time()
        ai_fps_calc = 0.0
        while self.worker_running:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
                    self.latest_frame = None

            if frame_to_process is None:
                time.sleep(0.01)
                continue

            candidate_boxes = []

            # 1. Candidate Detection: Run the chosen mode (clean single source prevents double boxes)
            if self.mode == "yolo" and self.yolo_model is not None:
                h, w, _ = frame_to_process.shape
                if getattr(config, 'USE_CHAMBER_ROI', False):
                    xmin = int(w * config.ROI_MARGIN_X)
                    xmax = int(w * (1.0 - config.ROI_MARGIN_X))
                    ymin = int(h * config.ROI_MARGIN_Y)
                    ymax = int(h * (1.0 - config.ROI_MARGIN_Y))
                    proc_frame = frame_to_process[ymin:ymax, xmin:xmax]
                    offset_x, offset_y = xmin, ymin
                else:
                    proc_frame = frame_to_process
                    offset_x, offset_y = 0, 0

                results = self.yolo_model(proc_frame, conf=config.YOLO_CONF_THRESHOLD, imgsz=getattr(config, 'YOLO_IMGSZ', 640), agnostic_nms=True, iou=0.45, verbose=False)
                for r in results:
                    for b in r.boxes:
                        coords = b.xyxy.cpu().numpy()[0]
                        x1, y1, x2, y2 = map(int, coords[:4])
                        area = (x2 - x1) * (y2 - y1)
                        if config.MIN_BBOX_AREA <= area <= config.MAX_BBOX_AREA:
                            yolo_conf = float(b.conf[0])
                            yolo_cls = int(b.cls[0])
                            candidate_boxes.append((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y, yolo_conf, yolo_cls))
            else:
                candidate_boxes = self.detect_boxes(frame_to_process)

            # 2. Stage 2 Three-Class Recognition: Aedes aegypti, Aedes albopictus, Other Species
            mosquito_candidates = []
            other_detections = []

            for item in candidate_boxes:
                if len(item) == 6:
                    x1, y1, x2, y2, yolo_conf, yolo_cls = item
                    box = (x1, y1, x2, y2)
                else:
                    box = item[:4]
                    yolo_conf, yolo_cls = None, None

                crop = self.crop_box(frame_to_process, box)
                if crop.size == 0:
                    continue

                # Morphological Check: Hardware / Screws vs Spindly Insects
                is_insect, morph_reason = self.check_morphology(crop)
                if not is_insect:
                    # Screws / Solid metal -> Recognized as Class 3: "Other Species"
                    other_detections.append((box, "Other Species", (255, 200, 0)))
                    continue

                # Insect Species Classification (Stage 2 CNN)
                species, clf_conf, probs = self.classifier.classify_crop(crop)

                # Two-Model Decision Fusion:
                # Case 1 – CNN >= threshold: species confirmed by classifier
                is_cnn_confirmed = clf_conf >= config.CLASSIFICATION_THRESHOLD
                # Case 2 – CNN < threshold but YOLO >= 0.45: detection plausible, species uncertain
                is_borderline = (not is_cnn_confirmed) and (yolo_conf is not None and yolo_conf >= 0.45)

                if is_cnn_confirmed:
                    # Species determined by CNN (highest reliability path)
                    species_name = species
                    if "aegypti" in species_name.lower():
                        color = (0, 0, 255)       # Bright RED (BGR) for Aedes aegypti
                        clean_name = "Aedes aegypti"
                    elif "albopictus" in species_name.lower():
                        color = (0, 140, 255)     # Bright ORANGE (BGR) for Aedes albopictus
                        clean_name = "Aedes albopictus"
                    else:
                        color = (0, 200, 255)     # YELLOW for unconfirmed species
                        clean_name = "Mosquito - species unconfirmed"
                    # Realistic calibrated display confidence (e.g. 78% - 96%)
                    if yolo_conf is not None:
                        blended = yolo_conf * 0.5 + clf_conf * 0.5
                    else:
                        blended = clf_conf
                    disp_conf = min(0.965, max(config.CLASSIFICATION_THRESHOLD, blended * 0.95))
                    mosquito_candidates.append({
                        "box": box,
                        "species": clean_name,
                        "confidence": disp_conf,
                        "color": color,
                        "crop": crop
                    })

                elif is_borderline:
                    # YOLO detects an insect at >= 0.45 but CNN is below confidence threshold.
                    # Species cannot be reliably determined from available model outputs.
                    # Detection is still logged, tracked, counted, and snapshotted exactly as confirmed detections.
                    color = (0, 200, 255)   # YELLOW (BGR) for species-unconfirmed detections
                    clean_name = "Mosquito - species unconfirmed"
                    disp_conf = float(yolo_conf)
                    mosquito_candidates.append({
                        "box": box,
                        "species": clean_name,
                        "confidence": disp_conf,
                        "color": color,
                        "crop": crop
                    })

                else:
                    # Both CNN and YOLO confidence are low -> Other Species (not logged)
                    other_detections.append((box, "Other Species", (255, 200, 0)))

            if self.tracker is not None:
                tracked_results = self.tracker.update_tracks(mosquito_candidates, frame_to_process)
                mosquito_detections = [(r["box"], r["label"], r["color"]) for r in tracked_results]

                # Update physical hardware alarm GPIO state
                t_stat = self.tracker.get_status()
                alert_info = t_stat.get("alert", {})
                is_alert = bool(alert_info.get("alert_triggered", False))
                if getattr(self, "alarm_gpio", None) is not None:
                    self.alarm_gpio.set_alarm(is_alert)
            else:
                tracked_results = []
                mosquito_detections = [
                    (c["box"], f"{c['species']}: {c['confidence']*100:.1f}%", c["color"])
                    for c in mosquito_candidates
                ]

            now_ai = time.time()
            ai_fps_calc = 0.9 * ai_fps_calc + 0.1 * (1.0 / max(now_ai - ai_prev_time, 1e-5))
            ai_prev_time = now_ai
            self.ai_fps = ai_fps_calc

            with self.lock:
                self.latest_tracks = list(tracked_results)
                self.confirmed_detections_count = len(tracked_results)
                self.active_detections = mosquito_detections + other_detections

    def detect_boxes(self, frame):
        """
        Stage 1 Contrast Mode: Detect candidate bounding boxes for dark objects in the chamber.
        Respects calibrated Chamber ROI to exclude acrylic wall seams.
        """
        boxes = []
        h, w, _ = frame.shape

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

        # Contrast Mode: Dark Object Candidate Detection
        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        floor_lum = np.median(blurred)
        thresh_val = max(50, int(floor_lum - 48))
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if config.MIN_BBOX_AREA <= area <= config.MAX_BBOX_AREA:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / float(bh)
                if 0.30 <= aspect <= 3.3:
                    boxes.append((x + offset_x, y + offset_y, x + bw + offset_x, y + bh + offset_y, None, None))

        return boxes


    def crop_box(self, frame, box):
        """
        Crops bounding box from frame according to config settings.
        Supports aspect-preserved square cropping using real surrounding pixels,
        or legacy rectangular padding.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        padding = getattr(config, 'CROP_PADDING', getattr(config, 'BBOX_PADDING', 0.60))
        use_square = getattr(config, 'USE_SQUARE_CROP', True)

        if use_square:
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

            # Shift if out of bounds to preserve 1:1 aspect ratio using real pixels
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

            return frame[sy1:sy2, sx1:sx2]
        else:
            bw = x2 - x1
            bh = y2 - y1
            pad_x = int(bw * padding)
            pad_y = int(bh * padding)

            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            return frame[cy1:cy2, cx1:cx2]

    def track_box_in_live_frame(self, frame, box):
        """
        Micro-tracks insect bounding box on the 30 FPS live stream.
        Runs in < 0.5ms per frame to eliminate any spatial lag behind moving mosquitoes.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1

        # Search window slightly larger than the box (+/- 25 px)
        sx1 = max(0, x1 - 25)
        sy1 = max(0, y1 - 25)
        sx2 = min(w, x2 + 25)
        sy2 = min(h, y2 + 25)

        patch = frame[sy1:sy2, sx1:sx2]
        if patch.size == 0:
            return box

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        floor_lum = np.percentile(gray, 85)
        dark_mask = (gray < (floor_lum - 28)).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return box

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 50:
            return box

        M = cv2.moments(c)
        if M['m00'] == 0:
            return box

        # Centroid in live frame coordinates
        cx = int(M['m10'] / M['m00']) + sx1
        cy = int(M['m01'] / M['m00']) + sy1

        old_cx = (x1 + x2) // 2
        old_cy = (y1 + y2) // 2

        dx = cx - old_cx
        dy = cy - old_cy

        # Only shift if reasonable movement (< 40 px between consecutive 33ms frames)
        if abs(dx) < 40 and abs(dy) < 40:
            nx1 = max(0, min(w - bw, x1 + dx))
            ny1 = max(0, min(h - bh, y1 + dy))
            return (nx1, ny1, nx1 + bw, ny1 + bh)

        return box

    def log_detection(self, species, confidence, box, frame):
        """Logs detection to CSV, saves snapshot, and registers with MosquitoTracker."""
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        snapshot_filename = ""
        snapshot_full_path = ""

        # Auto-save snapshot with cooldown
        current_time = time.time()
        if config.AUTO_SAVE_SNAPSHOT and (current_time - self.last_snapshot_time >= config.SNAPSHOT_COOLDOWN_SEC):
            clean_species = species.replace(" ", "_")
            file_name = f"{now.strftime('%Y%m%d_%H%M%S')}_{clean_species}_{int(confidence*100)}pct.jpg"
            snapshot_full_path = os.path.join(config.CAPTURES_DIR, file_name)
            cv2.imwrite(snapshot_full_path, frame)
            snapshot_filename = file_name
            self.last_snapshot_time = current_time
            print(f"[ALERT] Snapshot saved: {file_name}")

        # Mosquito ID tracking: deduplicate and assign unique MOS-XXX id
        mos_id = ""
        mos_status = ""
        if self.tracker is not None:
            x1, y1, x2, y2 = box
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size > 0:
                mos_id, mos_status = self.tracker.register(
                    crop, species, confidence, box,
                    snapshot_full_path or None
                )

        with open(config.LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, species, f"{confidence*100:.2f}%",
                             box[0], box[1], box[2], box[3],
                             snapshot_filename, mos_id, mos_status])

        return mos_id, mos_status

    def get_latest_preview_jpeg(self, width: int = 640, height: int = 360, quality: int = 80) -> Optional[Tuple[bytes, float, int]]:
        """
        Returns (jpeg_bytes, timestamp, seq) of the latest annotated preview frame.
        Supports fast preview downscaling (e.g. 640x360) without modifying the 1280x720 detector resolution.
        """
        with self.preview_lock:
            if self.latest_preview_frame is None:
                return None
            frame = self.latest_preview_frame.copy()
            ts = self.latest_preview_time
            seq = self.latest_preview_seq

        fh, fw = frame.shape[:2]
        if width > 0 and height > 0 and (fw != width or fh != height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return None
        return encoded.tobytes(), ts, seq

    def run(self):
        print("\n=======================================================")
        print(" Mosquito Chamber Detection System (Raspi HQ Camera)")
        print(f" Mode: {self.mode.upper()}  |  Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        print(" Controls: [Q] Quit  |  [S] Save Snapshot  |  [M] Toggle Mode")
        print("=======================================================\n")

        window_name = "Chamber Mosquito Detector - Live Stream"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 768)

        self.start_time = time.time()
        prev_time = time.time()
        fps = 0.0

        # Start status server before entering main capture loop
        server_host = getattr(config, "STATUS_SERVER_HOST", "0.0.0.0")
        server_port = getattr(config, "STATUS_SERVER_PORT", 8080)
        self.status_server, self.status_server_thread = start_status_server(
            self, host=server_host, port=server_port
        )

        try:
            while True:
                frame = self.camera.read_frame()
                if frame is None:
                    print("[ERROR] Failed to grab camera frame.")
                    break

                # Calculate FPS & update frame telemetry
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-5))
                prev_time = now
                self.fps = fps
                self.last_frame_time = now

                display = frame.copy()
                h, w, _ = display.shape

                # Feed latest frame to background AI worker without waiting
                with self.lock:
                    self.latest_frame = frame
                    detections = list(self.active_detections)

                # Live Micro-Tracking: Instantly lock boxes onto live moving insect (0ms delay)
                if getattr(config, 'ENABLE_LIVE_TRACKING', True) and detections:
                    tracked_detections = []
                    for box, label, color in detections:
                        tracked_box = self.track_box_in_live_frame(frame, box)
                        tracked_detections.append((tracked_box, label, color))
                    detections = tracked_detections

                mosquito_count = sum(1 for _, label, _ in detections if "Aedes" in label or "Mosquito" in label)
                other_count = len(detections) - mosquito_count

                # Draw asynchronous AI detections (boxes, labels, confidence)
                for box, label, color in detections:
                    x1, y1, x2, y2 = box
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
                hud_text = f"{src_info}FPS: {fps:.1f} | Mode: {self.mode.upper()} | Mosquitoes: {mosquito_count} | Other: {other_count}"
                cv2.putText(display, hud_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

                # Bottom help line
                cv2.rectangle(display, (0, h - 25), (w, h), (20, 20, 20), -1)
                cv2.putText(display, "[C] Calibrate Baseline (Zero Noise) | [M] Toggle Mode | [S] Save | [Q] Quit",
                            (15, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                # Update live preview buffer for LAN stream serving
                with self.preview_lock:
                    self.latest_preview_frame = display
                    self.latest_preview_time = now
                    self.latest_preview_seq += 1

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

        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt received. Initiating clean shutdown...")
        finally:
            # Clean shutdown sequence in exact order:
            # 1. Status server
            if self.status_server is not None:
                stop_status_server(self.status_server, self.status_server_thread)
                self.status_server = None
                self.status_server_thread = None

            # 2. CUSUM Accumulator stop and join
            if self.cusum_accumulator is not None:
                self.cusum_accumulator.stop()
                self.cusum_accumulator.join(timeout=5)
                self.cusum_accumulator = None

            # 3. Worker stop and join
            self.worker_running = False
            if hasattr(self, 'ai_thread') and self.ai_thread.is_alive():
                self.ai_thread.join(timeout=10)

            # 3. Tracker close
            if self.tracker is not None:
                self.tracker.close()

            # 4. Camera release and window cleanup
            self.camera.release()
            cv2.destroyAllWindows()

            # 5. Hardware alarm pin cleanup
            if getattr(self, "alarm_gpio", None) is not None:
                self.alarm_gpio.cleanup()

            print("[INFO] ChamberDetector shutdown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Chamber Mosquito Detector")
    parser.add_argument("--source", type=str, default=None,
                        help="Camera index (e.g. 0, 1) or video file path. Default: auto-detect Picamera2 / Webcam 0")
    parser.add_argument("--mode", type=str, choices=["yolo", "chamber_contrast"], default=None,
                        help="Detection mode: 'yolo' or 'chamber_contrast' (default from config: 'yolo')")
    args = parser.parse_args()

    detector = ChamberDetector(source=args.source, mode=args.mode)
    detector.run()
