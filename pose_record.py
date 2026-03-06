import cv2
import mediapipe as mp

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)

# Video writer to save recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('pose_record.avi', fourcc, 20.0, (640,480))

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect pose
    results = pose.process(image)

    # Convert back RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Show window
    cv2.imshow('Pose Detection Camera', image)

    # Save video
    out.write(image)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()