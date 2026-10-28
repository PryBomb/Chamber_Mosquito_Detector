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
YOLO_IMGSZ = 640  # Native training resolution (640x640) preserves small mosquito legs and features
YOLO_FRAME_INTERVAL = 1  # Continuous background inference

# Classification confidence threshold (below 70% is labeled 'Other Species' and suppressed)
CLASSIFICATION_THRESHOLD = 0.70

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

# Bounding box crop settings for Stage 2 CNN input
# Square crop preserves natural 1:1 aspect ratio without distorting or clipping mosquito legs
USE_SQUARE_CROP = True
CROP_PADDING = 0.60  # Default padding factor (0.60 square provides optimal species balance)
BBOX_PADDING = CROP_PADDING  # Backward compatibility alias

# Auto-save snapshots on detection
AUTO_SAVE_SNAPSHOT = True
SNAPSHOT_COOLDOWN_SEC = 2.0  # Avoid saving duplicate photos of the same mosquito

# ─────────────────────────────────────────────────────────────────────────────
# Mosquito ID Tracker (mosquito_tracker.py)
# ─────────────────────────────────────────────────────────────────────────────
ENABLE_MOSQUITO_TRACKER = True      # Set False to disable ID tracking entirely

# SQLite local registry path
TRACKER_DB_PATH = os.path.join(BASE_DIR, "mosquito_registry.db")

# Similarity thresholds for duplicate detection (0.0 – 1.0)
# Increase SIMILARITY_HIGH to be stricter (fewer duplicates marked as new)
# Decrease SIMILARITY_LOW  to be more lenient (fewer unverified)
SIMILARITY_HIGH = 0.80   # >= 80%  -> same mosquito (duplicate)
SIMILARITY_LOW  = 0.55   # 55–80%  -> uncertain match (unverified)

# Feature weights (must sum to 1.0)
COLOR_WEIGHT = 0.65      # HSV colour histogram contribution
MORPH_WEIGHT = 0.35      # Body shape / aspect-ratio contribution

# Colour histogram bins per channel (H and S channels of HSV)
COLOR_HIST_BINS = 32

# Multi-Mosquito Chamber Spatial Tracking
TRACK_MAX_DISTANCE = 120    # Max pixel distance for matching moving mosquito across frames
TRACK_MAX_MISSED   = 15     # Number of frames to retain lost track before removing (~1-2 seconds)
TRACK_IOU_THRESH   = 0.15   # Minimum IoU for spatial overlap match
LOG_INTERVAL_SEC   = 5.0    # Minimum interval (seconds) between periodic CSV logs for same mosquito

# ─────────────────────────────────────────────────────────────────────────────
# Supabase Cloud Database (Main Database)
# ─────────────────────────────────────────────────────────────────────────────
SUPABASE_ENABLED = os.environ.get("SUPABASE_ENABLED", "true").strip().lower() in ("true", "1", "yes")
SUPABASE_URL     = os.environ.get("SUPABASE_URL", "https://wvudxhppveaeoitjcnrp.supabase.co").strip()    # e.g. "https://xyzcompany.supabase.co"
SUPABASE_KEY     = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind2dWR4aHBwdmVhZW9pdGpjbnJwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTI1ODkwODgsImV4cCI6MjEwODE2NTA4OH0.qMqaovi8P_DUG_Bm38e7TAgDQIy3A4ngGXeXlxfchQQ").strip()    # service-role or anon key
SUPABASE_TABLE   = "mosquito_registry"
SUPABASE_BUCKET  = "mosquito-images"

# ─────────────────────────────────────────────────────────────────────────────
# Confirmed Aedes aegypti Threshold Alert Configuration
# ─────────────────────────────────────────────────────────────────────────────
ALERT_ENABLED                 = True
ALERT_TARGET_SPECIES          = "Aedes aegypti"
ALERT_THRESHOLD_COUNT         = 3      # Minimum qualifying unique mosquitoes to trigger alert
ALERT_MIN_CONFIDENCE          = 0.80   # Minimum 80% confidence requirement
ALERT_CONFIRM_FRAMES          = 3      # Must satisfy existing tracking confirmation requirement

# ─────────────────────────────────────────────────────────────────────────────
# CUSUM Activity Monitor Configuration (Decommissioned from active alerts)
# ─────────────────────────────────────────────────────────────────────────────
CUSUM_ENABLED                 = False  # Decommissioned from active alerts; historical DB preserved
CUSUM_DB_PATH                 = os.path.join(BASE_DIR, "cusum_history.db")
CUSUM_INTERVAL_MINUTES        = 30     # Discrete evaluation interval duration (minutes)
CUSUM_BASELINE_DAYS           = 7      # Duration of initial baseline collection period (valid days)
CUSUM_MIN_BASELINE_OBSERVATIONS = 12   # Minimum valid 30-min observations before baseline is ready
CUSUM_K                       = 0.5    # Reference allowance parameter (slack)
CUSUM_H                       = 5.0    # CUSUM alert decision threshold
CUSUM_MIN_STD                 = 0.5    # Minimum standard deviation floor to prevent division by zero
CUSUM_MORNING_START_HOUR      = 6      # Morning window start (06:00)
CUSUM_MORNING_END_HOUR        = 12     # Morning window end (12:00)
CUSUM_AFTERNOON_START_HOUR    = 12     # Afternoon window start (12:00)
CUSUM_AFTERNOON_END_HOUR      = 18     # Afternoon window end (18:00)
CUSUM_MIN_COVERAGE_RATIO      = 0.70   # Required active monitoring ratio for valid observation


