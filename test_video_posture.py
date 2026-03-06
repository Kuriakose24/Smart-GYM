import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load posture model
model = joblib.load("pushup_posture_model.pkl")

# Load pose detector
base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)

# Angle calculation
def calculate_angle(a,b,c):

    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180/np.pi)

    if angle>180:
        angle=360-angle

    return angle


# Open video
cap = cv2.VideoCapture("pushup_test_video.mp4")

frame_id = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    result = detector.detect_for_video(mp_image, frame_id)

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        def get_point(i):
            return [landmarks[i].x, landmarks[i].y]

        shoulder = get_point(11)
        elbow = get_point(13)
        wrist = get_point(15)
        hip = get_point(23)
        knee = get_point(25)
        ankle = get_point(27)

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        back_angle = calculate_angle(shoulder, hip, knee)
        hip_angle = calculate_angle(shoulder, hip, knee)
        knee_angle = calculate_angle(hip, knee, ankle)

        sample = pd.DataFrame(
            [[elbow_angle,back_angle,hip_angle,knee_angle]],
            columns=["elbow_angle","back_angle","hip_angle","knee_angle"]
        )

        prediction = model.predict(sample)[0]

        # Display result
        cv2.putText(frame,
                    f"Posture: {prediction}",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

    cv2.imshow("Pushup Posture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()