# ============================================================
# config.py — SmartGym AI
# ALL settings live here. Never hardcode values in other files.
# ============================================================

import os
import torch

# ── Paths ──────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
FACES_DIR       = os.path.join(BASE_DIR, "faces")
ATTENDANCE_DIR  = os.path.join(BASE_DIR, "attendance")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
DB_PATH         = os.path.join(DATA_DIR, "smartgym.db")
FACE_DB_PATH    = os.path.join(DATA_DIR, "face_database.pkl")

for _dir in [DATA_DIR, FACES_DIR, ATTENDANCE_DIR, MODEL_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Camera ─────────────────────────────────────────────────
CAMERA_INDEX  = 2
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
FPS_TARGET    = 30

# ── Person Detection (YOLOv8-pose) ─────────────────────────
YOLO_MODEL      = "yolov8n-pose.pt"
YOLO_CONFIDENCE = 0.35

_torch_cuda = torch.cuda.is_available()
YOLO_DEVICE = "cuda" if _torch_cuda else "cpu"
FACE_DEVICE = "cuda" if _torch_cuda else "cpu"
print(f"[Config] Device: {YOLO_DEVICE}")

# ── Face Recognition ───────────────────────────────────────
FACE_RECOG_EVERY_N_FRAMES = 60
FACE_SIMILARITY_THRESHOLD = 0.60

# ── Identity Tracking ──────────────────────────────────────
IDENTITY_TIMEOUT_SECONDS = 300
IOU_THRESHOLD            = 0.15
TRACK_LOST_PATIENCE      = 120

# ── Pose Estimation ────────────────────────────────────────
POSE_MIN_DETECTION_CONF = 0.5
POSE_MIN_TRACKING_CONF  = 0.5

# ── Rep Counting ───────────────────────────────────────────
# SIDE-VIEW CAMERA THRESHOLDS
# These are tuned for a camera perpendicular to the person's body.
# If reps are not counting: lower the DOWN thresholds by 5-10°
# If reps are double-counting: raise the DOWN thresholds by 5-10°

# Pushup thresholds (elbow angle in degrees)
# At bottom of pushup:  elbow ~70-90°  → must reach PUSHUP_ELBOW_DOWN
# At top of pushup:     elbow ~155-165° → must reach PUSHUP_ELBOW_UP
PUSHUP_ELBOW_DOWN = 100   # elbow angle at bottom (lower = stricter)
PUSHUP_ELBOW_UP   = 145   # elbow angle at top    (higher = stricter)

# Squat thresholds (knee angle in degrees)  
# At bottom of squat:   knee ~70-90°  → must reach SQUAT_KNEE_DOWN
# At top (standing):    knee ~165-175° → must reach SQUAT_KNEE_UP
SQUAT_KNEE_DOWN   = 100   # knee angle at bottom  (lower = stricter)
SQUAT_KNEE_UP     = 165   # knee angle at standing (higher = stricter)

# Body angle threshold for pushup validation
# Pushup rep only counts if body_angle (degrees from vertical) > this value
# 40° means body must be more than 40° tilted (fairly horizontal)
# Lower this if pushups aren't being counted: e.g. 25
# Raise this if squats are being falsely counted as pushups: e.g. 55
PUSHUP_MIN_BODY_ANGLE = 40   # NEW — was missing from old config

# Horizontal body check for pushup (legacy — kept for compatibility)
HORIZONTAL_THRESHOLD = 0.08

# Cooldown between reps — prevents double-counting from noise
# 10 frames @ 30fps = 0.33 seconds minimum between reps
REP_COOLDOWN_FRAMES = 10

# ── Exercise Detection ─────────────────────────────────────
# Thresholds for exercise_detector.py (body_angle based)
# body_angle: 0° = fully upright, 90° = fully horizontal
EXERCISE_PUSHUP_BODY_MIN  = 50   # body must be > 50° tilted to confirm pushup
EXERCISE_SQUAT_BODY_MAX   = 45   # body must be < 45° tilted to confirm squat
EXERCISE_STANDING_KNEE_MIN = 155  # FIX: was 165 — too strict, caused "unknown" stuck

# ── Exercise Model (your friend's model) ───────────────────
USE_MOCK_MODEL    = True
PUSHUP_MODEL_PATH = os.path.join(MODEL_DIR, "pushup_model.pkl")
SQUAT_MODEL_PATH  = os.path.join(MODEL_DIR, "squat_model.pkl")

# ── Display ────────────────────────────────────────────────
SHOW_FPS          = True
SHOW_SKELETON     = True
SHOW_BOUNDING_BOX = True
SHOW_IDENTITY     = True
SHOW_ANGLES       = True

COLOR_BOX_KNOWN   = (0, 255, 0)
COLOR_BOX_UNKNOWN = (0, 0, 255)
COLOR_SKELETON    = (0, 255, 255)
COLOR_TEXT        = (255, 255, 255)
COLOR_REP_GOOD    = (0, 255, 0)
COLOR_REP_BAD     = (0, 80, 255)
COLOR_HUD         = (255, 255, 0)

FONT           = 0
FONT_SCALE     = 0.65
FONT_THICKNESS = 2