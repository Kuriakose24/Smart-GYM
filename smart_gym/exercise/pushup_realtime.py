import sys
import os

# ── Path fix: works whether run from smart_gym/ root or from any subfolder ──
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# Also support running directly from the folder this file lives in
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# FeedbackEngine import — works from root or from prototypes/ subfolder
try:
    from utils.feedback_engine import FeedbackEngine
except ImportError:
    from feedback_engine import FeedbackEngine


# ─────────────────────────────────────────────────────────────────────────────
# Paths — resolve relative to THIS file so it works from any working directory
# ─────────────────────────────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_THIS_DIR, '..'))

# Look for models/ folder next to this file first, then in project root
def _find_model(filename):
    candidates = [
        os.path.join(_THIS_DIR,       "models", filename),
        os.path.join(_ROOT,           "models", filename),
        os.path.join(_THIS_DIR,       filename),
        os.path.join(_ROOT,           filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find {filename}. Searched:\n" + "\n".join(f"  {p}" for p in candidates)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load ML Model
# ─────────────────────────────────────────────────────────────────────────────
model_path = _find_model("pushup_final_model.pkl")
print(f"[Pushup] Loading model: {model_path}")
model = joblib.load(model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Pose Detector
# ─────────────────────────────────────────────────────────────────────────────
task_path = _find_model("pose_landmarker.task")
print(f"[Pushup] Loading pose model: {task_path}")

base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)


# ─────────────────────────────────────────────────────────────────────────────
# Rep Thresholds
# ─────────────────────────────────────────────────────────────────────────────
BOTTOM_THRESHOLD = 120   # elbow angle → counted as DOWN
TOP_THRESHOLD    = 135   # elbow angle → counted as UP


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


def smooth(prev, current, alpha=0.7):
    return alpha * prev + (1 - alpha) * current


# ─────────────────────────────────────────────────────────────────────────────
# Feedback Engine
# ─────────────────────────────────────────────────────────────────────────────
feedback_engine = FeedbackEngine()


# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

timestamp = 0

rep_count     = 0
correct_reps  = 0
incorrect_reps = 0

stage    = "UP"
feedback = "Get into pushup position"

bottom_reached   = False

# ── FIX 1: No longer wait for elbow > 160 before starting ───────────────────
# OLD: exercise_started = False → required elbow > 160 to begin
#      This meant the first rep's DOWN phase was missed while waiting for
#      the "exercise started" signal.
# FIX: Start immediately as soon as body is horizontal.
#      We still require horizontal_body as the gate — that's enough.
exercise_started = False

# bottom posture storage
bottom_elbow = 0
bottom_body  = 0
bottom_hip   = 0
bottom_back  = 0

# smoothing memory
prev_elbow = 0
prev_body  = 0
prev_hip   = 0
prev_back  = 0

print("Press Q to exit")


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    timestamp += 1

    image    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result   = detector.detect_for_video(mp_image, timestamp)

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        def get_point(i):
            return [landmarks[i].x, landmarks[i].y]

        shoulder = get_point(11)
        elbow    = get_point(13)
        wrist    = get_point(15)
        hip      = get_point(23)
        knee     = get_point(25)
        ankle    = get_point(27)
        ear      = get_point(7)

        # ── Raw Angles ────────────────────────────────────────────────────────
        raw_elbow = calculate_angle(shoulder, elbow, wrist)
        raw_body  = calculate_angle(shoulder, hip, ankle)
        raw_hip   = calculate_angle(shoulder, hip, knee)
        raw_back  = calculate_angle(ear, shoulder, hip)

        # ── Smoothed Angles ───────────────────────────────────────────────────
        elbow_angle = smooth(prev_elbow, raw_elbow)
        body_angle  = smooth(prev_body,  raw_body)
        hip_angle   = smooth(prev_hip,   raw_hip)
        back_angle  = smooth(prev_back,  raw_back)

        prev_elbow = elbow_angle
        prev_body  = body_angle
        prev_hip   = hip_angle
        prev_back  = back_angle

        # ── Orientation Filter ────────────────────────────────────────────────
        # shoulder[1] and hip[1] are normalised Y coords (0=top, 1=bottom)
        # When horizontal, shoulder and hip have nearly the same Y value
        horizontal_body = abs(shoulder[1] - hip[1]) < 0.20

        # ── FIX 1: Detect exercise start ──────────────────────────────────────
        # OLD: required elbow_angle > 160 (arms fully extended) BEFORE starting.
        #      If you went straight into a rep, this was never triggered,
        #      so the first DOWN was missed entirely.
        # FIX: Start as soon as the body is horizontal — no elbow gate needed.
        #      The rep counter handles the UP→DOWN→UP logic correctly regardless
        #      of which phase you enter in.
        if horizontal_body and not exercise_started:
            exercise_started = True
            feedback = "Pushup detected — go!"

        # ── Rep Detection ─────────────────────────────────────────────────────
        if horizontal_body and exercise_started:

            # DOWN phase: elbow bends below threshold
            if elbow_angle < BOTTOM_THRESHOLD + 5:
                stage          = "DOWN"
                bottom_reached = True
                # Always keep the lowest (most-bent) elbow angle captured
                bottom_elbow = elbow_angle
                bottom_body  = body_angle
                bottom_hip   = hip_angle
                bottom_back  = back_angle

            # UP phase: elbow extends above threshold AND we were DOWN
            if elbow_angle > TOP_THRESHOLD and stage == "DOWN" and bottom_reached:
                stage          = "UP"
                bottom_reached = False
                rep_count     += 1

                # ── FIX 2: ML Model integrated ────────────────────────────────
                # Column names must match training CSV exactly.
                # pushup_final_model was trained with:
                #   elbow_angle, back_angle, hip_angle, knee_angle
                # (knee_angle here is actually body alignment angle from training)
                features = pd.DataFrame(
                    [[bottom_elbow, bottom_back, bottom_hip, bottom_body]],
                    columns=["elbow_angle", "back_angle", "hip_angle", "knee_angle"]
                )
                prediction = model.predict(features)[0]

                # ── Rule Overrides (take priority over ML) ────────────────────
                rule_violation = False

                if bottom_body < 150 and bottom_hip < 160:
                    prediction     = "incorrect"
                    feedback       = "Hips too high"
                    rule_violation = True

                elif bottom_body < 150 and bottom_hip > 160:
                    prediction     = "incorrect"
                    feedback       = "Hips sagging"
                    rule_violation = True

                elif bottom_elbow > 115:
                    prediction     = "incorrect"
                    feedback       = "Go lower"
                    rule_violation = True

                # ── Rep Score ─────────────────────────────────────────────────
                score = feedback_engine.calculate_rep_score(
                    bottom_elbow, bottom_body, bottom_hip, bottom_back
                )
                feedback_engine.add_rep_score(score)

                # ── Final Feedback ────────────────────────────────────────────
                if prediction == "correct":
                    correct_reps += 1
                    feedback      = f"Perfect Rep!  Score: {score}"
                else:
                    incorrect_reps += 1
                    if not rule_violation:
                        feedback = feedback_engine.generate_feedback(
                            bottom_elbow, bottom_body, bottom_hip, bottom_back
                        )

        else:
            if not exercise_started:
                feedback = "Get into pushup position"
            stage = "UP"

        # ── Draw Landmarks ────────────────────────────────────────────────────
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

        # ── Debug Angles ──────────────────────────────────────────────────────
        cv2.putText(frame, f"Elbow:{int(elbow_angle)}", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Body:{int(body_angle)}",  (30, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Hip:{int(hip_angle)}",    (30, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Back:{int(back_angle)}",  (30, 275),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Stage: {stage}",          (30, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.putText(frame, f"Reps: {rep_count}",         (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.putText(frame, f"Correct: {correct_reps}",   (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Incorrect: {incorrect_reps}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Feedback: {feedback}",      (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        avg = feedback_engine.get_average_score()
        if avg > 0:
            cv2.putText(frame, f"Avg Score: {avg:.0f}/100", (30, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("SmartGym — Pushup", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ─────────────────────────────────────────────────────────────────────────────
# Session Summary
# ─────────────────────────────────────────────────────────────────────────────
summary = feedback_engine.workout_summary()

print("\n" + "=" * 40)
print("  Pushup Session Summary")
print("=" * 40)
print(f"  Total reps    : {rep_count}")
print(f"  Correct reps  : {correct_reps}")
print(f"  Incorrect reps: {incorrect_reps}")
print(f"  Average score : {summary['average_score']}/100")
if summary["trainer_alert"]:
    print("  ⚠  Trainer Alert: posture needs work")
else:
    print("  ✅ Great workout!")
print("=" * 40)

cap.release()
cv2.destroyAllWindows()