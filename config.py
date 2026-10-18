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
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "mosquito_yolo11.pt")

# Class labels for the 2nd-stage classifier (matches mosquito_model.tflite training)
SPECIES_CLASSES = ["Aedes aegypti", "Aedes albopictus"]

# Chamber Region of Interest (ROI) Settings
# Trims outer edges (silicone glue seams and purple sidewalls) to achieve 0 false positives
USE_CHAMBER_ROI = True
ROI_MARGIN_X = 0.15  # 15% margin from left and right edges (cuts out side walls & glue beads)
ROI_MARGIN_Y = 0.10  # 10% margin from top and bottom edges
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
# Detection mode: 'yolo' (Accurate AI insect detection) or 'chamber_contrast'
DETECTION_MODE = "yolo"

# YOLO confidence threshold (0.35 ensures all mosquitoes are detected; morphology rejects screws)
YOLO_CONF_THRESHOLD = 0.35
YOLO_IMGSZ = 512  # Optimal resolution (512x512) for capturing small insect features accurately
YOLO_FRAME_INTERVAL = 1  # Continuous background inference

# Classification confidence threshold (below 80% is labeled 'Uncertain / Non-Target' and suppressed)
CLASSIFICATION_THRESHOLD = 0.80

# Minimum & maximum bounding box area (in pixels) to filter noise/dust/glue beads
MIN_BBOX_AREA = 250
MAX_BBOX_AREA = 45000

# Insect Morphology Calibration (Eliminates screws, bolts, and dense non-insects)
# Mosquitoes have thin legs and slender bodies (solidity < 0.65, core retention < 0.28)
# Screws and bolts are solid dense metal (solidity > 0.70, core retention > 0.35)
MAX_INSECT_SOLIDITY = 0.68
MAX_CORE_RETENTION = 0.32

# Live Visual Micro-Tracking (Locks onto moving mosquito at 30 FPS with 0ms delay)
ENABLE_LIVE_TRACKING = True

# Bounding box padding factor (adds margin around the insect before 64x64 crop)
BBOX_PADDING = 0.15

# Auto-save snapshots on detection
AUTO_SAVE_SNAPSHOT = True
SNAPSHOT_COOLDOWN_SEC = 2.0  # Avoid saving duplicate photos of the same mosquito

