"""
camera/video_stream.py
----------------------
Handles webcam access and frame reading.
Easily swappable — change source to a video file path or IP camera URL.
"""

import cv2
import sys
import os

# So this file can import config from the parent smart_gym/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class VideoStream:
    def __init__(self,
                 source=config.CAMERA_INDEX,
                 width=config.FRAME_WIDTH,
                 height=config.FRAME_HEIGHT):
        """
        source : 0 = default webcam
                 1 = second webcam
                 "video.mp4" = video file
        """
        self.source = source
        self.width  = width
        self.height = height
        self.cap    = None
        self._frame_count = 0

    def start(self):
        """Open the camera. Raises RuntimeError if it can't open."""
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_MSMF)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,config.FPS_TARGET)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[Camera] ❌ Could not open source: {self.source}\n"
                f"  → Try changing CAMERA_INDEX in config.py (0, 1, 2...)"
            )

        # Read actual resolution (camera may not support requested size)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] ✅ Started — source={self.source}, "
              f"resolution={actual_w}x{actual_h}")

    def read(self):
        """
        Read one frame.
        Returns (True, frame) on success, (False, None) on failure.
        """
        if self.cap is None:
            raise RuntimeError("[Camera] Not started — call start() first.")

        ret, frame = self.cap.read()
        if ret:
            self._frame_count += 1
        return ret, frame

    @property
    def frame_count(self):
        """Total frames read so far."""
        return self._frame_count

    def stop(self):
        """Release the camera."""
        if self.cap:
            self.cap.release()
            print("[Camera] Stopped.")


# ── Test — run this file directly to verify your webcam ───────────────────────
# python camera/video_stream.py
if __name__ == "__main__":
    import time

    print("=" * 45)
    print("  Camera Test — press Q to quit")
    print("=" * 45)

    cam = VideoStream()
    cam.start()

    fps_start = time.time()
    fps_count = 0
    fps_display = 0.0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[Camera] ❌ Failed to read frame.")
            break

        # FPS counter
        fps_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count   = 0
            fps_start   = time.time()

        # Draw info on frame
        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {w}x{h}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {cam.frame_count}",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Camera OK - press Q to quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        cv2.imshow("SmartGym — Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()
    print(f"\n✅ Camera test complete — {cam.frame_count} frames read.")