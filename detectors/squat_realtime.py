import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------
# Load Squat ML Model
# --------------------------

model = joblib.load("models/squat_final_model.pkl")


# --------------------------
# Pose Detector
# --------------------------

base_options = python.BaseOptions(
    model_asset_path="models/pose_landmarker.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)


# --------------------------
# Angle Calculation
# --------------------------

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# --------------------------
# Angle Smoothing
# --------------------------

def smooth(prev, current, alpha=0.7):
    return alpha * prev + (1 - alpha) * current


prev_knee = 0


# --------------------------
# Bottom posture tracking
# --------------------------

bottom_knee = 180
bottom_hip = 0
bottom_back = 0


# --------------------------
# Rep logic variables
# --------------------------

exercise_started = False
down_frames = 0
stage = "top"

rep_count = 0
correct_reps = 0
incorrect_reps = 0

feedback = "Stand straight"

rep_lock = 0


# --------------------------
# Camera
# --------------------------

cap = cv2.VideoCapture(0)

timestamp = 0

print("Press Q to exit")


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape
    timestamp += 1

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image
    )

    result = detector.detect_for_video(mp_image, timestamp)

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        def get_point(i):
            return [landmarks[i].x, landmarks[i].y]

        def get_visibility(i):
            return landmarks[i].visibility if hasattr(landmarks[i], 'visibility') else 0.0

        # --------------------------
        # FIX 3: Auto-detect visible side
        # Compare visibility of left vs right landmarks to pick the
        # side currently facing the camera (works for side switching)
        # --------------------------

        left_vis = (
            get_visibility(11) +   # left shoulder
            get_visibility(23) +   # left hip
            get_visibility(25) +   # left knee
            get_visibility(27)     # left ankle
        )
        right_vis = (
            get_visibility(12) +   # right shoulder
            get_visibility(24) +   # right hip
            get_visibility(26) +   # right knee
            get_visibility(28)     # right ankle
        )

        if left_vis >= right_vis:
            # Use left-side landmarks
            shoulder  = get_point(11)
            hip       = get_point(23)
            knee_pt   = get_point(25)
            ankle_pt  = get_point(27)
            ear       = get_point(7)
            wrist_l   = get_point(15)
            wrist_r   = get_point(16)
        else:
            # Use right-side landmarks
            shoulder  = get_point(12)
            hip       = get_point(24)
            knee_pt   = get_point(26)
            ankle_pt  = get_point(28)
            ear       = get_point(8)
            wrist_l   = get_point(15)
            wrist_r   = get_point(16)

        # --------------------------
        # Arms Forward Detection
        # --------------------------

        hands_forward = (
            abs(wrist_l[1] - shoulder[1]) < 0.25 and
            abs(wrist_r[1] - shoulder[1]) < 0.25 and
            wrist_l[0] > shoulder[0] - 0.05 and
            wrist_r[0] > shoulder[0] - 0.05
        )

        left_knee_pt  = get_point(25)
        right_knee_pt = get_point(26)
        left_ankle    = get_point(27)
        right_ankle   = get_point(28)
        left_hip      = get_point(23)
        right_hip     = get_point(24)
        left_shoulder  = get_point(11)
        right_shoulder = get_point(12)

        left_knee_angle  = calculate_angle(left_hip,  left_knee_pt,  left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee_pt, right_ankle)

        # Weight by visibility so the more visible side dominates
        lv = get_visibility(25) + 0.01
        rv = get_visibility(26) + 0.01
        raw_knee = (left_knee_angle * lv + right_knee_angle * rv) / (lv + rv)

        # Side-specific angles for posture classification
        hip_angle  = calculate_angle(shoulder, hip, knee_pt)
        back_angle = calculate_angle(ear, shoulder, hip)

        knee_angle = smooth(prev_knee, raw_knee)
        prev_knee  = knee_angle

        # --------------------------
        # View mode detection
        # From the front, both shoulders are similarly visible AND
        # the horizontal gap between shoulders is large relative to
        # the gap between hips. From the side, one shoulder dominates.
        # --------------------------

        lsv = get_visibility(11)
        rsv = get_visibility(12)
        vis_balance = min(lsv, rsv) / (max(lsv, rsv) + 0.01)  # 1.0 = perfectly front-on

        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        hip_width      = abs(left_hip[0]      - right_hip[0])

        # Front view: both shoulders visible + wide shoulder span
        is_front_view = (vis_balance > 0.65) and (shoulder_width > 0.15)

        # --------------------------
        # Hip-drop ratio for front view
        # Standing: hip_y is well above knee_y (smaller value = higher up).
        # Squatting: hip descends toward knee level.
        # hip_to_knee_ratio close to 1.0 means hips are near knee height = deep squat.
        # We track normalised drop: how much has hip Y grown since standing baseline.
        # --------------------------

        avg_hip_y   = (left_hip[1]      + right_hip[1])      / 2
        avg_knee_y  = (left_knee_pt[1]  + right_knee_pt[1])  / 2
        avg_ankle_y = (left_ankle[1]    + right_ankle[1])    / 2

        # Ratio: 0 = hips at knee height (full squat), 1 = hips at standing height
        leg_height    = abs(avg_ankle_y - avg_knee_y) + 0.001
        hip_drop_ratio = (avg_knee_y - avg_hip_y) / leg_height  # positive = hips above knees

        # is_squatting_front: hips have dropped close to or below knee level
        is_squatting_front = hip_drop_ratio < 0.55   # tune: lower = require deeper squat

        # --------------------------
        # Detect squat start
        # --------------------------

        is_standing_straight = (
            (not is_front_view and knee_angle > 160) or
            (is_front_view and hip_drop_ratio > 0.80)
        )

        if is_standing_straight and not exercise_started:
            exercise_started = True
            feedback = "Squat position detected"

        # --------------------------
        # Squat down detection — works for ALL views
        # Side/diagonal: use knee angle (reliable from the side)
        # Front: use hip drop ratio (knee angle stays ~180 from front)
        # --------------------------

        is_going_down = (
            (not is_front_view and knee_angle < 150) or
            (is_front_view and is_squatting_front)
        )

        if exercise_started and is_going_down:

            down_frames += 1

            if down_frames > 4:
                stage = "down"

            # Track worst-case posture at bottom of squat
            if knee_angle < bottom_knee:
                bottom_knee = knee_angle
                bottom_hip  = hip_angle
                bottom_back = back_angle

        else:
            down_frames = 0

        # --------------------------
        # Stand up detection
        # --------------------------

        is_standing_up = (
            (not is_front_view and knee_angle > 160) or
            (is_front_view and hip_drop_ratio > 0.75)
        )

        if exercise_started and is_standing_up and stage == "down" and rep_lock == 0:

            rep_count += 1
            stage    = "top"
            rep_lock = 15

            features = pd.DataFrame(
                [[bottom_knee, bottom_hip, bottom_back]],
                columns=["knee_angle", "hip_angle", "back_angle"]
            )

            prediction = model.predict(features)[0]
            feedback   = ""

            # --------------------------
            # Strong Rule Violations
            # --------------------------

            if bottom_knee > 145:
                prediction = "incorrect"
                feedback   = "Go deeper"

            elif bottom_back < 115:
                prediction = "incorrect"
                feedback   = "Keep chest up"

            elif bottom_hip > 175:
                prediction = "incorrect"
                feedback   = "Sit back more"

            # --------------------------
            # Final classification
            # --------------------------

            if prediction == "correct":
                correct_reps += 1
                if feedback == "":
                    feedback = "Good squat"
            else:
                incorrect_reps += 1
                if feedback == "":
                    feedback = "Adjust posture"

            # --------------------------
            # FIX 2: Reset ALL bottom values after each rep
            # Previously only bottom_knee was reset, causing stale
            # hip/back values to make every rep appear correct.
            # --------------------------

            bottom_knee = 180
            bottom_hip  = 0
            bottom_back = 0

        # --------------------------
        # Rep cooldown
        # --------------------------

        if rep_lock > 0:
            rep_lock -= 1

        # --------------------------
        # Draw Landmarks
        # --------------------------

        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Show view mode and tracking info
        if is_front_view:
            view_label = f"View: FRONT  Hip-drop: {hip_drop_ratio:.2f}"
        else:
            side_label = "LEFT" if left_vis >= right_vis else "RIGHT"
            view_label = f"View: SIDE ({side_label})"

        cv2.putText(frame, view_label, (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        # --------------------------
        # Display
        # --------------------------

        cv2.putText(frame, f"Reps: {rep_count}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(frame, f"Correct: {correct_reps}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"Incorrect: {incorrect_reps}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Feedback: {feedback}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Knee: {int(knee_angle)}", (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2)

    cv2.imshow("Smart Gym Squat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()