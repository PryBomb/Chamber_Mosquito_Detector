import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
LOG_FILE = os.path.join(BASE_DIR, "detection_log.csv")

# Ensure required directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)

# Model paths
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "mosquito_model.tflite")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "mosquito_model.onnx")
PKL_MODEL_PATH = os.path.join(MODELS_DIR, "mosquito_model.pkl")
YOLO_MODEL_PATH = "yolov8n.pt"  # Will automatically download on first run

# Class labels for the 2nd-stage classifier (matches mosquito_model.tflite training)
SPECIES_CLASSES = ["Aedes aegypti", "Aedes albopictus"]

# Chamber Region of Interest (ROI) Settings
# Trims outer edges (silicone glue seams and purple sidewalls) to achieve 0 false positives
USE_CHAMBER_ROI = True
ROI_MARGIN_X = 0.08  # 8% margin from left and right edges
ROI_MARGIN_Y = 0.05  # 5% margin from top and bottom edges
SHOW_ROI_OVERLAY = True  # Draw ROI boundary on the preview stream

# Camera Configuration
# Raspberry Pi HQ Camera native modes:
# - 1280x720  @ 30-60 fps (Fast, recommended for real-time tracking)
# - 1920x1080 @ 30 fps (High clarity)
# - 4056x3040 @ 10 fps (Full 12.3MP resolution)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Detection & Classification Parameters
# Detection mode: 'yolo' (YOLOv8) or 'chamber_contrast' (Fast background contrast for white chamber)
DETECTION_MODE = "yolo"

# YOLO confidence threshold
YOLO_CONF_THRESHOLD = 0.25

# Classification confidence threshold (below this is labeled 'Uncertain / Insect')
CLASSIFICATION_THRESHOLD = 0.80

# Minimum & maximum bounding box area (in pixels) to filter noise/dust
MIN_BBOX_AREA = 100
MAX_BBOX_AREA = 60000

# Bounding box padding factor (adds margin around the insect before 64x64 crop)
BBOX_PADDING = 0.15

# Auto-save snapshots on detection
AUTO_SAVE_SNAPSHOT = True
SNAPSHOT_COOLDOWN_SEC = 2.0  # Avoid saving duplicate photos of the same mosquito
