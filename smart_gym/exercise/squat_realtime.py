import sys
import os

# ── Path fix ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ─────────────────────────────────────────────────────────────────────────────
# Path resolver — finds models/ folder wherever the script is run from
# ─────────────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_THIS_DIR, '..'))

def _find_model(filename):
    candidates = [
        os.path.join(_THIS_DIR, "models", filename),
        os.path.join(_ROOT,     "models", filename),
        os.path.join(_THIS_DIR, filename),
        os.path.join(_ROOT,     filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find {filename}. Searched:\n" + "\n".join(f"  {p}" for p in candidates)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load Squat ML Model
# ─────────────────────────────────────────────────────────────────────────────
model_path = _find_model("squat_final_model.pkl")
print(f"[Squat] Loading model: {model_path}")
model = joblib.load(model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Pose Detector
# ─────────────────────────────────────────────────────────────────────────────
task_path = _find_model("pose_landmarker.task")
print(f"[Squat] Loading pose model: {task_path}")

base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle   = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


def smooth(prev, current, alpha=0.7):
    return alpha * prev + (1 - alpha) * current


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
prev_knee = 0

bottom_knee = 180
bottom_hip  = 0
bottom_back = 0

# ── FIX 1: First-rep miss ────────────────────────────────────────────────────
# OLD: exercise_started = False, required is_standing_straight BEFORE any rep
#      could be counted. If you walked in and immediately squatted, you were
#      still in "not started" state — first rep missed.
# FIX: Start immediately. We check is_standing_straight ONCE, but if the
#      person never stands straight first (e.g. walked in mid-squat), we
#      also start after a short down_frames window so no rep is lost.
exercise_started = False
down_frames      = 0
stage            = "top"

rep_count      = 0
correct_reps   = 0
incorrect_reps = 0

feedback = "Stand straight to begin"
rep_lock = 0

# ── FIX 1b: Fallback start counter ───────────────────────────────────────────
# If person goes straight into a squat without standing first,
# trigger exercise_started after 3 frames of going down.
_fallback_down_frames = 0


# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

timestamp = 0

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

        def get_visibility(i):
            return landmarks[i].visibility if hasattr(landmarks[i], 'visibility') else 0.0

        # ── Auto-detect visible side ──────────────────────────────────────────
        left_vis = (get_visibility(11) + get_visibility(23) +
                    get_visibility(25) + get_visibility(27))
        right_vis = (get_visibility(12) + get_visibility(24) +
                     get_visibility(26) + get_visibility(28))

        if left_vis >= right_vis:
            shoulder = get_point(11);  hip     = get_point(23)
            knee_pt  = get_point(25);  ankle_pt= get_point(27)
            ear      = get_point(7)
        else:
            shoulder = get_point(12);  hip     = get_point(24)
            knee_pt  = get_point(26);  ankle_pt= get_point(28)
            ear      = get_point(8)

        wrist_l = get_point(15)
        wrist_r = get_point(16)

        # ── Bilateral knee angle (weighted by visibility) ─────────────────────
        left_knee_pt   = get_point(25);  right_knee_pt = get_point(26)
        left_ankle     = get_point(27);  right_ankle   = get_point(28)
        left_hip_pt    = get_point(23);  right_hip_pt  = get_point(24)
        left_shoulder  = get_point(11);  right_shoulder= get_point(12)

        left_knee_angle  = calculate_angle(left_hip_pt,  left_knee_pt,  left_ankle)
        right_knee_angle = calculate_angle(right_hip_pt, right_knee_pt, right_ankle)

        lv = get_visibility(25) + 0.01
        rv = get_visibility(26) + 0.01
        raw_knee = (left_knee_angle * lv + right_knee_angle * rv) / (lv + rv)

        hip_angle  = calculate_angle(shoulder, hip, knee_pt)
        back_angle = calculate_angle(ear, shoulder, hip)

        knee_angle = smooth(prev_knee, raw_knee)
        prev_knee  = knee_angle

        # ── View detection ────────────────────────────────────────────────────
        lsv = get_visibility(11);  rsv = get_visibility(12)
        vis_balance    = min(lsv, rsv) / (max(lsv, rsv) + 0.01)
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        is_front_view  = (vis_balance > 0.65) and (shoulder_width > 0.15)

        # ── Hip-drop ratio (for front view) ───────────────────────────────────
        avg_hip_y   = (left_hip_pt[1]   + right_hip_pt[1])   / 2
        avg_knee_y  = (left_knee_pt[1]  + right_knee_pt[1])  / 2
        avg_ankle_y = (left_ankle[1]    + right_ankle[1])    / 2
        leg_height      = abs(avg_ankle_y - avg_knee_y) + 0.001
        hip_drop_ratio  = (avg_knee_y - avg_hip_y) / leg_height
        is_squatting_front = hip_drop_ratio < 0.55

        # ── Standing / going-down signals ─────────────────────────────────────
        is_standing_straight = (
            (not is_front_view and knee_angle > 160) or
            (is_front_view and hip_drop_ratio > 0.80)
        )

        is_going_down = (
            (not is_front_view and knee_angle < 150) or
            (is_front_view and is_squatting_front)
        )

        is_standing_up = (
            (not is_front_view and knee_angle > 160) or
            (is_front_view and hip_drop_ratio > 0.75)
        )

        # ── FIX 1: Exercise start logic ───────────────────────────────────────
        # Primary: person stands straight first (normal case)
        if is_standing_straight and not exercise_started:
            exercise_started    = True
            _fallback_down_frames = 0
            feedback = "Squat detected — go!"

        # FIX 1b: Fallback — person walks in already squatting
        # After 3 frames of going-down motion, start anyway so no rep is lost
        if not exercise_started and is_going_down:
            _fallback_down_frames += 1
            if _fallback_down_frames >= 3:
                exercise_started = True
                feedback = "Squat detected — go!"
        else:
            _fallback_down_frames = 0

        # ── FIX 1c: stage must start as "down" if we started mid-squat ───────
        # If exercise just started and person is already going down,
        # set stage to "down" directly so the UP detection counts the rep.
        if exercise_started and is_going_down:
            down_frames += 1
            # Reduced from > 4 to > 1 — catches the first rep faster
            if down_frames > 1:
                stage = "down"

            # Track bottom posture (keep the lowest knee angle seen)
            if knee_angle < bottom_knee:
                bottom_knee = knee_angle
                bottom_hip  = hip_angle
                bottom_back = back_angle
        else:
            down_frames = 0

        # ── Rep complete: person stands back up ───────────────────────────────
        if exercise_started and is_standing_up and stage == "down" and rep_lock == 0:
            rep_count += 1
            stage      = "top"
            rep_lock   = 15

            # ── FIX 2: ML Model prediction ────────────────────────────────────
            # Column names must match what the squat model was trained on exactly
            features = pd.DataFrame(
                [[bottom_knee, bottom_hip, bottom_back]],
                columns=["knee_angle", "hip_angle", "back_angle"]
            )
            prediction = model.predict(features)[0]
            feedback   = ""

            # ── Rule Overrides ────────────────────────────────────────────────
            if bottom_knee > 145:
                prediction = "incorrect"
                feedback   = "Go deeper"

            elif bottom_back < 115:
                prediction = "incorrect"
                feedback   = "Keep chest up"

            elif bottom_hip > 175:
                prediction = "incorrect"
                feedback   = "Sit back more"

            # ── Final classification ──────────────────────────────────────────
            if prediction == "correct":
                correct_reps += 1
                if not feedback:
                    feedback = "Good squat!"
            else:
                incorrect_reps += 1
                if not feedback:
                    feedback = "Adjust posture"

            # Reset bottom tracking for next rep
            bottom_knee = 180
            bottom_hip  = 0
            bottom_back = 0

        # ── Rep cooldown ──────────────────────────────────────────────────────
        if rep_lock > 0:
            rep_lock -= 1

        # ── Draw Landmarks ────────────────────────────────────────────────────
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

        # ── Debug Info ────────────────────────────────────────────────────────
        if is_front_view:
            view_label = f"View: FRONT  Hip-drop: {hip_drop_ratio:.2f}"
        else:
            side_label = "LEFT" if left_vis >= right_vis else "RIGHT"
            view_label = f"View: SIDE ({side_label})"

        cv2.putText(frame, view_label,           (30, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        cv2.putText(frame, f"Knee: {int(knee_angle)}", (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2)
        cv2.putText(frame, f"Hip: {int(hip_angle)}",   (30, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Back: {int(back_angle)}", (30, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Stage: {stage}",          (30, 295),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.putText(frame, f"Reps: {rep_count}",          (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.putText(frame, f"Correct: {correct_reps}",    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Incorrect: {incorrect_reps}",(30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Feedback: {feedback}",       (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("SmartGym — Squat", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ─────────────────────────────────────────────────────────────────────────────
# Session Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 40)
print("  Squat Session Summary")
print("=" * 40)
print(f"  Total reps    : {rep_count}")
print(f"  Correct reps  : {correct_reps}")
print(f"  Incorrect reps: {incorrect_reps}")
if rep_count > 0:
    pct = correct_reps / rep_count * 100
    print(f"  Form accuracy : {pct:.0f}%")
    if pct < 75:
        print("  ⚠  Trainer Alert: posture needs work")
    else:
        print("  ✅ Great workout!")
print("=" * 40)

cap.release()
cv2.destroyAllWindows()