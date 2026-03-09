"""
rep_counter.py
--------------
Clean per-person rep counter using YOLOv8-pose keypoints.

Works on FULL FRAME coordinates — no cropping, no coordinate confusion.

For pushups it uses whichever side is more visible:
    left side  → left_shoulder, left_elbow, left_wrist
    right side → right_shoulder, right_elbow, right_wrist
"""

import numpy as np


# ── Angle helper ──────────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    """
    Calculate angle at point b, formed by a-b-c.
    Returns angle in degrees (0-180).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


class RepCounter:
    # Thresholds
    BOTTOM_THRESHOLD = 130   # lowered — catches shallow reps too
    TOP_THRESHOLD    = 140   # elbow angle — UP position
    HORIZONTAL_THRESHOLD = 0.15  # shoulder-hip Y diff for pushup detection

    def __init__(self, name="Unknown"):
        self.name = name

        # Rep state
        self.rep_count        = 0
        self.stage            = "UP"
        self.bottom_reached   = False
        self.exercise_started = False
        self.feedback         = "Get into pushup position"

        # Captured at bottom of each rep — for ML scoring
        self.bottom_elbow = 0
        self.bottom_body  = 0
        self.bottom_hip   = 0
        self.bottom_back  = 0

        # Live angles for display
        self.elbow_angle = 0
        self.body_angle  = 0
        self.hip_angle   = 0
        self.back_angle  = 0
        self.horizontal  = False

    def _pick_side(self, kp):
        """
        Pick the more visible side (left or right).
        Returns (shoulder, elbow, wrist, hip, knee, ankle, ear) or None.
        """
        # Try left side first
        left = [kp.get("left_shoulder"), kp.get("left_elbow"),
                kp.get("left_wrist"),    kp.get("left_hip"),
                kp.get("left_knee"),     kp.get("left_ankle"),
                kp.get("left_ear")]

        right = [kp.get("right_shoulder"), kp.get("right_elbow"),
                 kp.get("right_wrist"),    kp.get("right_hip"),
                 kp.get("right_knee"),     kp.get("right_ankle"),
                 kp.get("right_ear")]

        left_visible  = sum(1 for p in left  if p is not None)
        right_visible = sum(1 for p in right if p is not None)

        if left_visible >= right_visible and left_visible >= 4:
            return left
        elif right_visible >= 4:
            return right
        return None

    def update(self, keypoints):
        """
        Update rep count from keypoints dict.
        Call this every frame with the person's keypoints.

        Returns True if a rep was just completed, False otherwise.
        """
        side = self._pick_side(keypoints)
        if side is None:
            return False

        shoulder, elbow, wrist, hip, knee, ankle, ear = side

        # Need at least shoulder, elbow, wrist, hip for pushup
        if not all([shoulder, elbow, wrist, hip]):
            return False

        # Calculate angles
        self.elbow_angle = calculate_angle(shoulder, elbow, wrist)
        self.body_angle  = calculate_angle(shoulder, hip, ankle) if ankle else 180
        self.hip_angle   = calculate_angle(shoulder, hip, knee)  if knee  else 180
        self.back_angle  = calculate_angle(ear, shoulder, hip)   if ear   else 180

        # Orientation check — is body horizontal? (pushup position)
        self.horizontal = abs(shoulder[1] - hip[1]) < \
                          (self.HORIZONTAL_THRESHOLD * 720)
                          # 720 = expected frame height, scales with frame

        # Detect pushup start position
        if self.horizontal and self.elbow_angle > 160 and not self.exercise_started:
            self.exercise_started = True
            self.feedback = "Pushup position detected!"

        rep_completed = False

        if self.horizontal and self.exercise_started:

            # Going DOWN
            if self.elbow_angle < self.BOTTOM_THRESHOLD:
                self.stage          = "DOWN"
                self.bottom_reached = True
                self.bottom_elbow   = self.elbow_angle
                self.bottom_body    = self.body_angle
                self.bottom_hip     = self.hip_angle
                self.bottom_back    = self.back_angle

            # Coming UP — rep complete
            if (self.elbow_angle > self.TOP_THRESHOLD
                    and self.stage == "DOWN"
                    and self.bottom_reached):
                self.stage          = "UP"
                self.bottom_reached = False
                self.rep_count     += 1
                rep_completed       = True

        else:
            if not self.exercise_started:
                self.feedback = "Get into pushup position"
            self.stage = "UP"

        return rep_completed

    def get_bottom_angles(self):
        """Return bottom position angles for ML scoring."""
        return (self.bottom_elbow, self.bottom_body,
                self.bottom_hip,   self.bottom_back)

    def get_debug_info(self):
        """Return dict of current state for display."""
        return {
            "elbow":      self.elbow_angle,
            "body":       self.body_angle,
            "stage":      self.stage,
            "horizontal": self.horizontal,
            "started":    self.exercise_started,
        }