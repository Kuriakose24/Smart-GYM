"""
pose/pose_estimator.py  (v3 — adds back_angle for ML model compatibility)
--------------------------------------------------------------------------
CHANGES vs v2:

FIX 1 — back_angle added to output
    The pushup ML model was trained with MediaPipe angles including:
        back_angle = ear → shoulder → hip  (posture / spine angle)
    YOLO v2 did not produce this angle, so the model always received 0
    which caused every rep to predict "incorrect".

    v3 computes back_angle from YOLO keypoints:
        Primary:   ear (left_ear or right_ear) → shoulder → hip
        Fallback:  nose → shoulder → hip  (if ears not visible)
    This closely matches the MediaPipe back_angle used in training.

FIX 2 — shoulder_angle renamed to arm_angle for clarity
    The old "shoulder" key measured elbow→shoulder→hip which is NOT
    the same as the shoulder joint angle. Renamed to avoid confusion.
    "shoulder" key still present as alias for backwards compatibility.

ANGLES RETURNED:
    {
        "elbow":      float,   # shoulder→elbow→wrist  (pushup depth)
        "knee":       float,   # hip→knee→ankle        (squat depth)
        "hip":        float,   # shoulder→hip→knee     (torso-to-leg)
        "body_angle": float,   # body tilt from vertical (0=upright, 90=lying)
        "back_angle": float,   # ear→shoulder→hip      (spine / posture)  ← NEW
        "shoulder":   float,   # elbow→shoulder→hip    (arm raise — alias)
        "ankle":      None,    # not computable from YOLO (no toe keypoint)

        # Front-view squat signals
        "_is_front_view":  bool,
        "_hip_drop_ratio": float or None,

        # Per-side debug values
        "_left_elbow":  float or None,
        "_right_elbow": float or None,
        "_left_knee":   float or None,
        "_right_knee":  float or None,
    }
"""

import sys, os, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _angle_between(a, b, c):
    """
    Compute the angle at point B formed by A→B and B→C.
    Returns degrees in [0, 180]. Returns None if any point is missing.
    """
    if a is None or b is None or c is None:
        return None

    ax, ay = a
    bx, by = b
    cx, cy = c

    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)

    dot  = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if mag1 < 1e-6 or mag2 < 1e-6:
        return None

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def _body_inclination(shoulder, hip):
    """
    How horizontal is the torso?
    0° = fully upright (standing), 90° = lying flat (pushup position).
    """
    if shoulder is None or hip is None:
        return None

    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _best(left_val, right_val):
    """Average if both available, otherwise whichever is non-None."""
    if left_val is not None and right_val is not None:
        return (left_val + right_val) / 2.0
    return left_val if left_val is not None else right_val


class PoseEstimator:
    """
    Converts YOLOv8-pose keypoints into joint angles.
    Stateless — smoothing is handled by RepCounter.
    """

    def extract(self, keypoints):
        """
        keypoints : dict from PersonTracker  {"left_shoulder": (x,y) or None, ...}
        Returns   : dict of angles (never None itself; missing angles are None inside)
        """
        kp = keypoints or {}

        l_shoulder = kp.get("left_shoulder")
        r_shoulder = kp.get("right_shoulder")
        l_elbow    = kp.get("left_elbow")
        r_elbow    = kp.get("right_elbow")
        l_wrist    = kp.get("left_wrist")
        r_wrist    = kp.get("right_wrist")
        l_hip      = kp.get("left_hip")
        r_hip      = kp.get("right_hip")
        l_knee     = kp.get("left_knee")
        r_knee     = kp.get("right_knee")
        l_ankle    = kp.get("left_ankle")
        r_ankle    = kp.get("right_ankle")
        l_ear      = kp.get("left_ear")
        r_ear      = kp.get("right_ear")
        nose       = kp.get("nose")

        # ── Elbow angle: shoulder → elbow → wrist ─────────────────────────────
        left_elbow_angle  = _angle_between(l_shoulder, l_elbow, l_wrist)
        right_elbow_angle = _angle_between(r_shoulder, r_elbow, r_wrist)
        elbow_angle       = _best(left_elbow_angle, right_elbow_angle)

        # ── Knee angle: hip → knee → ankle ────────────────────────────────────
        left_knee_angle  = _angle_between(l_hip, l_knee, l_ankle)
        right_knee_angle = _angle_between(r_hip, r_knee, r_ankle)
        knee_angle       = _best(left_knee_angle, right_knee_angle)

        # ── Hip angle: shoulder → hip → knee ──────────────────────────────────
        left_hip_angle  = _angle_between(l_shoulder, l_hip, l_knee)
        right_hip_angle = _angle_between(r_shoulder, r_hip, r_knee)
        hip_angle       = _best(left_hip_angle, right_hip_angle)

        # ── Arm/shoulder angle: elbow → shoulder → hip ────────────────────────
        left_arm_angle  = _angle_between(l_elbow, l_shoulder, l_hip)
        right_arm_angle = _angle_between(r_elbow, r_shoulder, r_hip)
        arm_angle       = _best(left_arm_angle, right_arm_angle)

        # ── Body inclination ──────────────────────────────────────────────────
        mid_shoulder = (
            ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
            if l_shoulder and r_shoulder else (l_shoulder or r_shoulder)
        )
        mid_hip = (
            ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
            if l_hip and r_hip else (l_hip or r_hip)
        )
        body_angle = _body_inclination(mid_shoulder, mid_hip)

        # ── Back angle: ear → shoulder → hip  (FIX 1) ────────────────────────
        # This matches the MediaPipe back_angle used to train the ML models.
        # Measures how upright the upper spine is:
        #   ~160-180° = good posture (ear over shoulder over hip)
        #   < 130°    = hunched / chest dropped
        #
        # Priority: use whichever ear is visible. If no ears, fall back to nose.
        # Average left+right if both ears visible (front-facing person).
        left_back_angle  = _angle_between(l_ear, l_shoulder, l_hip)
        right_back_angle = _angle_between(r_ear, r_shoulder, r_hip)

        if left_back_angle is not None or right_back_angle is not None:
            back_angle = _best(left_back_angle, right_back_angle)
        else:
            # Fallback: nose → mid_shoulder → mid_hip
            back_angle = _angle_between(nose, mid_shoulder, mid_hip)

        # ── Front-view detection + hip-drop ratio ─────────────────────────────
        _is_front_view  = False
        _hip_drop_ratio = None

        if l_shoulder and r_shoulder and l_hip and r_hip and l_knee and r_knee:
            shoulder_span          = abs(l_shoulder[0] - r_shoulder[0])
            both_shoulders_visible = l_shoulder is not None and r_shoulder is not None
            both_hips_visible      = l_hip is not None and r_hip is not None
            _is_front_view         = both_shoulders_visible and both_hips_visible and shoulder_span > 80

            if l_ankle and r_ankle:
                avg_hip_y       = (l_hip[1]   + r_hip[1])   / 2
                avg_knee_y      = (l_knee[1]  + r_knee[1])  / 2
                avg_ankle_y     = (l_ankle[1] + r_ankle[1]) / 2
                leg_height      = abs(avg_ankle_y - avg_knee_y) + 0.001
                _hip_drop_ratio = (avg_knee_y - avg_hip_y) / leg_height

        return {
            "elbow":      elbow_angle,
            "knee":       knee_angle,
            "hip":        hip_angle,
            "shoulder":   arm_angle,   # backwards-compat alias
            "body_angle": body_angle,
            "back_angle": back_angle,  # FIX 1: ear→shoulder→hip
            "ankle":      None,        # no toe keypoint in YOLO

            "_is_front_view":  _is_front_view,
            "_hip_drop_ratio": _hip_drop_ratio,

            # Per-side debug
            "_left_elbow":  left_elbow_angle,
            "_right_elbow": right_elbow_angle,
            "_left_knee":   left_knee_angle,
            "_right_knee":  right_knee_angle,
        }

    def debug_string(self, angles):
        if not angles:
            return "No angles"
        parts = []
        for key, label in [("elbow","E"), ("knee","K"), ("hip","H"),
                            ("body_angle","B"), ("back_angle","Bk")]:
            v = angles.get(key)
            if v is not None:
                parts.append(f"{label}:{v:.0f}")
        return "  ".join(parts) if parts else "No angles"