# Smart Gym AI – Push-Up Pose Detection

## Project Description

This project is a computer vision based system that detects human body posture during push-up exercises.
The system uses pose detection to analyze body landmarks and count repetitions using a webcam.

## Features Implemented

* Real-time pose detection using MediaPipe
* Push-up repetition counter
* Exercise dataset prepared in YOLOv8 format
* Posture testing using trained model
* Webcam-based exercise monitoring using OpenCV

## Technologies Used

* Python
* OpenCV
* MediaPipe
* YOLOv8
* NumPy

## Project Files

* `pose_record.py` – Records body pose landmarks
* `pushup_rep_counter.py` – Detects push-up movement and counts reps
* `test_posture_model.py` – Tests the trained posture model
* `test_image_posture.py` – Detects posture from images

## Installation

Install required libraries:

pip install opencv-python mediapipe numpy

## Run the Project

Run the push-up detection system:

python pushup_rep_counter.py

## Future Improvements

* Multi-person exercise detection
* More exercises (squat, deadlift, etc.)
* Smart Gym dashboard
* Face recognition for gym attendance
