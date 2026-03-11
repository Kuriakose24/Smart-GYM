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
CAMERA_INDEX  = 1
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
# Lowered 60 → 20: at ~15 FPS, 60 frames = 4s before first ID attempt
FACE_RECOG_EVERY_N_FRAMES = 20
FACE_SIMILARITY_THRESHOLD = 0.60

# ── Identity Tracking ──────────────────────────────────────
IDENTITY_TIMEOUT_SECONDS = 300
IOU_THRESHOLD            = 0.15
TRACK_LOST_PATIENCE      = 120

# ── Pose Estimation ────────────────────────────────────────
POSE_MIN_DETECTION_CONF = 0.5
POSE_MIN_TRACKING_CONF  = 0.5

# ── Rep Counting ───────────────────────────────────────────
# YOLO-TUNED THRESHOLDS
# YOLO + bilateral avg reads ~10° higher than MediaPipe.
# In horizontal (pushup) position YOLO reads even higher due to camera angle.
#
# HOW TO TUNE:
#   Reps not counting   → lower DOWN threshold (e.g. 115 → 110)
#   Reps double-count   → raise DOWN threshold (e.g. 115 → 120)
#   Top never registers → lower UP threshold   (e.g. 150 → 145)

# Pushup thresholds (elbow angle in degrees)
# FIX: raised DOWN 110 → 115 — YOLO reads higher when horizontal
PUSHUP_ELBOW_DOWN = 115   # elbow at bottom of pushup
PUSHUP_ELBOW_UP   = 150   # elbow at top of pushup

# Squat thresholds (knee angle in degrees)
SQUAT_KNEE_DOWN   = 110   # knee at bottom of squat
SQUAT_KNEE_UP     = 160   # knee at standing

# Body angle validation — only used by ExerciseDetector now
# RepCounter no longer gates on this (ExerciseDetector is the gate)
PUSHUP_MIN_BODY_ANGLE = 40

# Front-view squat: hip-drop ratio thresholds
SQUAT_HIP_DROP_DOWN = 0.55
SQUAT_HIP_DROP_UP   = 0.75

# Legacy
HORIZONTAL_THRESHOLD = 0.08

# Cooldown between reps (frames) — prevents double-counting
REP_COOLDOWN_FRAMES = 10

# ── Exercise Detection ─────────────────────────────────────
EXERCISE_PUSHUP_BODY_MIN   = 50   # body > 50° tilt → pushup signal
EXERCISE_SQUAT_BODY_MAX    = 45   # body < 45° tilt → squat signal
EXERCISE_STANDING_KNEE_MIN = 155  # knee > 155° → standing signal

# ── Exercise Model ─────────────────────────────────────────
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