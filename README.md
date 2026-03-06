# Smart Gym AI – Push-Up Pose Detection

## Project Description

This project is a computer vision–based system that analyzes human posture during push-up exercises.
The system uses pose estimation and machine learning to evaluate body alignment, detect posture errors, and assist in improving workout form.

The goal is to build a **Smart Gym Platform** that can automatically monitor exercises and provide feedback using AI.

---

# Features Implemented

* Real-time human pose detection using **MediaPipe**
* Push-up posture analysis using body joint angles
* Machine learning–based posture classification
* Push-up repetition counter
* Video-based posture testing
* Image-based posture testing
* Detection of common push-up mistakes
* Stabilized posture prediction using frame voting and angle smoothing

---

# Detected Push-Up Mistakes

The system detects several common push-up errors:

1. Sagging lower back
2. Hips too high
3. Flared elbows
4. Incomplete range of motion
5. Head dropping forward
6. Hands too wide or too narrow
7. Moving too fast / using momentum
8. Not engaging the core
9. Uneven hand placement
10. Body not kept in a straight line

---

# Technologies Used

* Python
* OpenCV
* MediaPipe Pose
* Scikit-learn (Random Forest Classifier)
* NumPy
* Pandas
* YOLOv8 (for dataset preparation)

---

# System Pipeline

Camera / Video
↓
MediaPipe Pose Detection
↓
Body Landmark Extraction
↓
Joint Angle Calculation
↓
Biomechanical Rule Checking
↓
Machine Learning Model Prediction
↓
Frame Voting Stabilization
↓
Posture Feedback

---

# Dataset Creation

A custom dataset was created for training the posture classification model.

Steps followed:

1. Push-up images and videos were collected.
2. MediaPipe Pose was used to extract body landmarks.
3. Important joint angles were calculated:

   * elbow_angle
   * back_angle
   * hip_angle
   * knee_angle
4. The dataset was labeled as **correct** or **incorrect** posture.
5. The dataset was cleaned and balanced before training.

---

# Model Training

A **Random Forest classifier** was trained using the extracted joint angles.

Training features:

* elbow_angle
* back_angle
* hip_angle
* knee_angle

Model performance:

Accuracy ≈ **89%**

The trained model is saved as:

pushup_final_model.pkl

---

# Project Files

pose_record.py
Records body pose landmarks using MediaPipe.

pushup_rep_counter.py
Detects push-up movement and counts repetitions.

test_pushup_image.py
Tests posture detection using a single image.

test_pushup_video.py
Analyzes push-up posture frame-by-frame using video input.

train_final_model.py
Trains the machine learning model using the prepared dataset.

---

# Installation

Install required libraries:

pip install opencv-python mediapipe numpy pandas scikit-learn

---

# Run the Project

Run the push-up detection system:

python pushup_rep_counter.py

For posture testing using video:

python test_pushup_video.py

For image posture testing:

python test_pushup_image.py

---

# Current Project Status

Completed:

* Dataset creation and preprocessing
* Machine learning model training
* Image posture testing
* Video posture analysis
* Posture error detection

Next Steps:

* Improve rep counter stability
* Real-time webcam posture detection
* Multi-person detection using YOLO
* Integration with Smart Gym platform

---

# Future Improvements

* Multi-person exercise detection
* More exercises (squat, deadlift, etc.)
* Smart Gym dashboard
* Face recognition for gym attendance
* Gym member activity tracking

---

# Author

BTech Artificial Intelligence and Data Science
Mini Project – Smart Gym Platform
