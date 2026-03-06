import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------------------------
# Load trained model
# -------------------------

model = joblib.load("pushup_final_model.pkl")


# -------------------------
# Load MediaPipe pose
# -------------------------

base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.PoseLandmarker.create_from_options(options)


# -------------------------
# Angle calculation
# -------------------------

def calculate_angle(a,b,c):

    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180/np.pi)

    if angle>180:
        angle=360-angle

    return angle


# -------------------------
# Load test image
# -------------------------

image_path = "test_correct_image.webp"

frame = cv2.imread(image_path)

if frame is None:
    print("Image not found")
    exit()

h,w,_ = frame.shape


# Convert to mediapipe format

image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=image
)


# -------------------------
# Pose detection
# -------------------------

result = detector.detect(mp_image)


if result.pose_landmarks:

    landmarks = result.pose_landmarks[0]


    def get_point(i):
        return [landmarks[i].x,landmarks[i].y]


    shoulder = get_point(11)
    elbow = get_point(13)
    wrist = get_point(15)

    hip = get_point(23)
    knee = get_point(25)
    ankle = get_point(27)

    ear = get_point(7)


    # -------------------------
    # Calculate angles
    # -------------------------

    elbow_angle = calculate_angle(shoulder,elbow,wrist)

    back_angle = calculate_angle(ear,shoulder,hip)

    hip_angle = calculate_angle(shoulder,hip,knee)

    knee_angle = calculate_angle(hip,knee,ankle)

    # NEW: Body alignment angle
    body_angle = calculate_angle(shoulder,hip,knee)


    print("\nAngles:")
    print("Elbow:",elbow_angle)
    print("Back:",back_angle)
    print("Hip:",hip_angle)
    print("Knee:",knee_angle)
    print("Body Alignment:",body_angle)


    # -------------------------
    # Rule-based error check
    # -------------------------

    if body_angle < 160:
        posture = "incorrect"
        reason = "Loose core / arched back"

    else:

        # -------------------------
        # ML model prediction
        # -------------------------

        features = pd.DataFrame(
            [[elbow_angle,back_angle,hip_angle,knee_angle]],
            columns=["elbow_angle","back_angle","hip_angle","knee_angle"]
        )

        prediction = model.predict(features)[0]

        posture = prediction
        reason = "ML prediction"


    print("\nPrediction:", posture)
    print("Reason:", reason)


    # -------------------------
    # Draw landmarks
    # -------------------------

    for lm in landmarks:

        x=int(lm.x*w)
        y=int(lm.y*h)

        cv2.circle(frame,(x,y),4,(0,255,0),-1)


    # -------------------------
    # Display result
    # -------------------------

    cv2.putText(frame,f"Posture: {posture}",
                (30,50),cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,0,255),2)

    cv2.putText(frame,f"Body Angle: {int(body_angle)}",
                (30,90),cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(255,255,255),2)

else:

    print("No pose detected")


cv2.imshow("Pushup Image Test",frame)

cv2.waitKey(0)

cv2.destroyAllWindows()