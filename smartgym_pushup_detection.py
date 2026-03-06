import cv2
import numpy as np
import mediapipe as mp
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------------------------------
# Load trained ML model
# -------------------------------

model = joblib.load("pushup_final_model.pkl")


# -------------------------------
# Load pose detector
# -------------------------------

base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)


# -------------------------------
# Pose skeleton connections
# -------------------------------

POSE_CONNECTIONS = [
(11,13),(13,15),
(12,14),(14,16),
(11,12),
(11,23),(12,24),
(23,24),
(23,25),(25,27),
(24,26),(26,28)
]


# -------------------------------
# Angle calculation
# -------------------------------

def calculate_angle(a,b,c):

    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180/np.pi)

    if angle>180:
        angle=360-angle

    return angle


# -------------------------------
# Angle smoothing
# -------------------------------

def smooth_angle(prev,current,alpha=0.6):
    return alpha*prev + (1-alpha)*current


# -------------------------------
# Rep variables
# -------------------------------

rep_count = 0
stage = "top"

min_elbow = 180

feedback = ""
posture = "Unknown"

initialized=False
rep_started=False

prev_elbow_angle=0
prev_back_angle=0
prev_hip_angle=0


# -------------------------------
# Prediction smoothing
# -------------------------------

prediction_history=[]
history_size=5


# -------------------------------
# Video input
# -------------------------------

cap = cv2.VideoCapture(0)

timestamp=0

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()


# -------------------------------
# Main loop
# -------------------------------

while cap.isOpened():

    ret,frame=cap.read()

    if not ret:
        break

    h,w,_=frame.shape

    timestamp+=33


    image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image=mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image
    )

    result=detector.detect_for_video(mp_image,timestamp)


    if result.pose_landmarks:

        landmarks=result.pose_landmarks[0]


        # draw landmarks
        for lm in landmarks:

            x=int(lm.x*w)
            y=int(lm.y*h)

            cv2.circle(frame,(x,y),4,(0,255,0),-1)


        # draw skeleton
        for connection in POSE_CONNECTIONS:

            start,end=connection

            x1=int(landmarks[start].x*w)
            y1=int(landmarks[start].y*h)

            x2=int(landmarks[end].x*w)
            y2=int(landmarks[end].y*h)

            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)


        def get_point(i):
            return [landmarks[i].x,landmarks[i].y]


        shoulder=get_point(11)
        elbow=get_point(13)
        wrist=get_point(15)

        hip=get_point(23)
        knee=get_point(25)
        ankle=get_point(27)

        ear=get_point(7)


        # -------------------------------
        # Raw angles
        # -------------------------------

        raw_elbow=calculate_angle(shoulder,elbow,wrist)
        raw_back=calculate_angle(ear,shoulder,hip)
        raw_hip=calculate_angle(shoulder,hip,knee)
        knee_angle=calculate_angle(hip,knee,ankle)


        # -------------------------------
        # Smooth angles
        # -------------------------------

        elbow_angle=smooth_angle(prev_elbow_angle,raw_elbow)
        back_angle=smooth_angle(prev_back_angle,raw_back)
        hip_angle=smooth_angle(prev_hip_angle,raw_hip)

        prev_elbow_angle=elbow_angle
        prev_back_angle=back_angle
        prev_hip_angle=hip_angle


        # -------------------------------
        # Initialization
        # -------------------------------

        if not initialized:

            initialized=True

            if elbow_angle>130:
                stage="top"
            else:
                stage="down"


        # track minimum elbow
        if elbow_angle < min_elbow:
            min_elbow = elbow_angle


        # detect lowering
        if elbow_angle < 120 and stage=="top":

            stage="down"
            rep_started=True


        # detect bottom
        if elbow_angle < 90 and stage=="down":

            stage="bottom"


        # -------------------------------
        # ML posture prediction
        # -------------------------------

        if stage=="bottom":

            features=[[elbow_angle,back_angle,hip_angle,knee_angle]]

            prediction=model.predict(features)[0]

            prediction_history.append(prediction)

            if len(prediction_history)>history_size:
                prediction_history.pop(0)

            posture=max(set(prediction_history),key=prediction_history.count)


        # -------------------------------
        # Rep completion
        # -------------------------------

        if elbow_angle>140 and stage=="bottom":

            rep_count+=1
            stage="top"

            if posture=="correct":
                feedback="Good push-up"
            else:
                feedback="Incorrect posture"


            print("Rep",rep_count,":",feedback)

            min_elbow=180
            rep_started=False


        # -------------------------------
        # UI
        # -------------------------------

        cv2.putText(frame,f"Reps: {rep_count}",
                    (30,80),cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        cv2.putText(frame,f"Stage: {stage}",
                    (30,120),cv2.FONT_HERSHEY_SIMPLEX,
                    1,(255,0,0),2)

        cv2.putText(frame,f"Posture: {posture}",
                    (30,160),cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,0,255),2)

        cv2.putText(frame,f"Feedback: {feedback}",
                    (30,200),cv2.FONT_HERSHEY_SIMPLEX,
                    1,(255,255,0),2)


    cv2.imshow("Smart Gym Push-up Detection",frame)

    if cv2.waitKey(1)&0xFF==27:
        break


cap.release()
cv2.destroyAllWindows()