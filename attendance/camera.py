"""
camera.py
---------
Handles webcam access and frame reading.
Encapsulated so you can swap in a video file or IP camera later.
"""

import cv2


class Camera:
    def __init__(self, source=0, width=1280, height=720):
        """
        source : 0 = default webcam, or a video file path
        width/height : capture resolution
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None

    def start(self):
        """Open the camera."""
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError(f"[Camera] Could not open source: {self.source}")

        print(f"[Camera] Started — source={self.source}, "
              f"resolution={self.width}x{self.height}")

    def read_frame(self):
        """
        Read one frame.
        Returns (True, frame) on success, (False, None) on failure.
        """
        if self.cap is None:
            raise RuntimeError("[Camera] Camera not started. Call start() first.")
        ret, frame = self.cap.read()
        return ret, frame

    def stop(self):
        """Release the camera."""
        if self.cap:
            self.cap.release()
            print("[Camera] Stopped.")