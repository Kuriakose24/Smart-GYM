"""
pose/pose_estimator.py  (v2 — side-view optimised, bilateral fallback)
-----------------------------------------------------------------------
WHAT THIS FILE DOES:
    Takes the keypoints dict from person_detector.py and converts them
    into joint angles (degrees) that exercise_detector and rep_counter use.

WHY THE OLD VERSION WAS BROKEN:
    The old version likely picked ONE side (e.g. only left elbow) and if
    that keypoint had low confidence it returned None — causing the entire
    exercise detector to see no signal.

    For side-view cameras:
        • The camera-facing side is always more visible
        • YOLOv8 often marks the far side with low confidence (< 0.3)
          so person_detector already returns None for those keypoints
        • We need to pick whichever side IS visible, not always left

    This version:
        1. Tries both sides and picks the best (most confident / non-None)
        2. If both are visible, averages them (reduces noise)
        3. Returns a full angles dict even if some keypoints are missing
        4. Adds a body_angle (how horizontal the torso is) which is critical
           for distinguishing pushup from squat

ANGLES RETURNED:
    {
        "elbow":      float,   # arm bend — key for pushup counting
        "knee":       float,   # leg bend — key for squat counting
        "hip":        float,   # torso-to-leg angle — distinguishes pushup vs squat
        "shoulder":   float,   # shoulder relative to hip (posture)
        "body_angle": float,   # how horizontal the body is (0=lying, 90=standing)
        "ankle":      float,   # ankle dorsiflexion (optional, good for squat depth)
    }
    Any angle that cannot be computed returns None in its key.
    The dict is ALWAYS returned — never None itself.

HOW TO USE:
    estimator = PoseEstimator()
    angles = estimator.extract(person["keypoints"])
    elbow_angle = angles["elbow"]   # e.g. 85.0
"""

import sys
import os
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _angle_between(a, b, c):
    """
    Compute the angle at point B formed by the line A→B and B→C.

    Each point is (x, y) in pixel space.
    Returns angle in degrees: 0 = fully bent, 180 = fully straight.

    Returns None if any point is missing.
    """
    if a is None or b is None or c is None:
        return None

    ax, ay = a
    bx, by = b
    cx, cy = c

    # Vectors from B
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)

    # Dot product and magnitudes
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if mag1 < 1e-6 or mag2 < 1e-6:
        return None  # degenerate — points overlap

    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for floating point safety

    return math.degrees(math.acos(cos_angle))


def _body_inclination(shoulder, hip):
    """
    Returns how horizontal the body is, as an angle from vertical.

    0   degrees = fully upright (standing)
    90  degrees = perfectly horizontal (lying down for pushup)

    This is THE key signal for distinguishing pushup from squat —
    both can have a bent elbow/knee but only pushup has a horizontal body.
    """
    if shoulder is None or hip is None:
        return None

    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]  # positive = hip is below shoulder (normal)

    # Angle from vertical
    angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle_from_vertical


def _best(left_val, right_val):
    """
    Given two angle measurements (left and right side), return the best one.

    Strategy:
        - If both available: average them (reduces noise)
        - If only one available: use that one
        - If neither: None
    """
    if left_val is not None and right_val is not None:
        return (left_val + right_val) / 2.0
    return left_val if left_val is not None else right_val


class PoseEstimator:
    """
    Converts YOLOv8-pose keypoints into joint angles.

    One instance, use .extract(keypoints) every frame.
    Stateless — no smoothing here (smoothing is in RepCounter).
    """

    def extract(self, keypoints):
        """
        Main entry point.

        keypoints : dict from person_detector.py
                    { "left_shoulder": (x,y) or None, ... }

        Returns dict of angles. Always returns a dict (never None).
        Any angle that can't be computed is None in the dict.
        """
        kp = keypoints or {}

        # ── Shortcuts ─────────────────────────────────────────────────────────
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

        # ── Elbow angle ───────────────────────────────────────────────────────
        # shoulder → elbow → wrist
        # For pushup: goes from ~160° (arms extended) to ~70-90° (chest near floor)
        left_elbow_angle  = _angle_between(l_shoulder, l_elbow, l_wrist)
        right_elbow_angle = _angle_between(r_shoulder, r_elbow, r_wrist)
        elbow_angle       = _best(left_elbow_angle, right_elbow_angle)

        # ── Knee angle ────────────────────────────────────────────────────────
        # hip → knee → ankle
        # For squat: goes from ~170° (standing) to ~70-90° (bottom of squat)
        left_knee_angle  = _angle_between(l_hip, l_knee, l_ankle)
        right_knee_angle = _angle_between(r_hip, r_knee, r_ankle)
        knee_angle       = _best(left_knee_angle, right_knee_angle)

        # ── Hip angle ─────────────────────────────────────────────────────────
        # shoulder → hip → knee
        # This measures torso-to-leg angle.
        # For pushup: ~170° (body is straight plank)
        # For squat:  ~70-100° (bending at hip)
        # For standing: ~175-180°
        left_hip_angle  = _angle_between(l_shoulder, l_hip, l_knee)
        right_hip_angle = _angle_between(r_shoulder, r_hip, r_knee)
        hip_angle       = _best(left_hip_angle, right_hip_angle)

        # ── Shoulder angle ────────────────────────────────────────────────────
        # elbow → shoulder → hip
        # Tells us if arms are raised (overhead press) or at sides
        left_shoulder_angle  = _angle_between(l_elbow, l_shoulder, l_hip)
        right_shoulder_angle = _angle_between(r_elbow, r_shoulder, r_hip)
        shoulder_angle       = _best(left_shoulder_angle, right_shoulder_angle)

        # ── Ankle angle ───────────────────────────────────────────────────────
        # knee → ankle → toe  (we use heel as proxy since no toe keypoint)
        # Useful for squat depth assessment
        # (optional — won't break anything if None)
        left_ankle_angle  = _angle_between(l_knee,  l_ankle, None)  # can't compute without toe
        right_ankle_angle = _angle_between(r_knee,  r_ankle, None)
        ankle_angle       = None  # skip — no toe keypoint in YOLOv8

        # ── Body inclination ──────────────────────────────────────────────────
        # How horizontal is the torso?
        # 0° = upright,  90° = lying flat
        # Use the midpoint of shoulders and midpoint of hips for robustness
        if l_shoulder and r_shoulder:
            mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2,
                            (l_shoulder[1] + r_shoulder[1]) / 2)
        else:
            mid_shoulder = l_shoulder or r_shoulder

        if l_hip and r_hip:
            mid_hip = ((l_hip[0] + r_hip[0]) / 2,
                       (l_hip[1] + r_hip[1]) / 2)
        else:
            mid_hip = l_hip or r_hip

        body_angle = _body_inclination(mid_shoulder, mid_hip)

        # ── Front-view detection + hip-drop ratio ────────────────────────────────
        # These are injected into the angles dict so RepCounter can use them
        # for accurate front-view squat counting (knee angle collapses to ~180°
        # from the front — hip-drop ratio is the only reliable signal).
        #
        # is_front_view:
        #   Both shoulders must be similarly visible AND shoulder span wide.
        #   From the side, one shoulder is occluded → span collapses.
        #   From the front → both visible and widely spaced.
        #
        # hip_drop_ratio:
        #   = (knee_y - hip_y) / (ankle_y - knee_y)
        #   Standing: hips well above knees → ratio ~1.0+
        #   Squatting: hips drop toward knee level → ratio falls toward 0
        #   Threshold for DOWN: < 0.55  |  UP: > 0.75

        _is_front_view  = False
        _hip_drop_ratio = None

        if l_shoulder and r_shoulder and l_hip and r_hip and l_knee and r_knee:
            shoulder_span = abs(l_shoulder[0] - r_shoulder[0])

            # Visibility balance (1.0 = perfectly symmetric = front-on)
            # YOLOv8 kp_conf is already filtered in person_detector — we use
            # the presence of both keypoints as a proxy for visibility balance.
            # Both being non-None means both were above the 0.3 conf threshold.
            both_shoulders_visible = (l_shoulder is not None and r_shoulder is not None)
            both_hips_visible      = (l_hip      is not None and r_hip      is not None)

            # Wide shoulder span relative to frame width — front view indicator
            # shoulder_span is in pixels; use > 80px as threshold (works for 640-1280px frames)
            _is_front_view = both_shoulders_visible and both_hips_visible and shoulder_span > 80

            if l_ankle and r_ankle:
                avg_hip_y   = (l_hip[1]   + r_hip[1])   / 2
                avg_knee_y  = (l_knee[1]  + r_knee[1])  / 2
                avg_ankle_y = (l_ankle[1] + r_ankle[1]) / 2
                leg_height  = abs(avg_ankle_y - avg_knee_y) + 0.001
                _hip_drop_ratio = (avg_knee_y - avg_hip_y) / leg_height

        # ── Build result ──────────────────────────────────────────────────────
        return {
            "elbow":      elbow_angle,
            "knee":       knee_angle,
            "hip":        hip_angle,
            "shoulder":   shoulder_angle,
            "body_angle": body_angle,
            "ankle":      ankle_angle,

            # Front-view squat detection signals (used by RepCounter)
            "_is_front_view":  _is_front_view,
            "_hip_drop_ratio": _hip_drop_ratio,

            # Raw per-side values — useful for debugging, not used in logic
            "_left_elbow":  left_elbow_angle,
            "_right_elbow": right_elbow_angle,
            "_left_knee":   left_knee_angle,
            "_right_knee":  right_knee_angle,
        }

    def debug_string(self, angles):
        """Return a compact string for on-screen display."""
        if not angles:
            return "No angles"
        e = angles.get("elbow")
        k = angles.get("knee")
        h = angles.get("hip")
        b = angles.get("body_angle")
        parts = []
        if e is not None: parts.append(f"E:{e:.0f}")
        if k is not None: parts.append(f"K:{k:.0f}")
        if h is not None: parts.append(f"H:{h:.0f}")
        if b is not None: parts.append(f"B:{b:.0f}")
        return "  ".join(parts) if parts else "No angles"


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    estimator = PoseEstimator()

    # Simulate a person standing straight (all joints near 180°)
    fake_standing = {
        "left_shoulder":  (300, 200),
        "right_shoulder": (340, 200),
        "left_elbow":     (280, 300),
        "right_elbow":    (360, 300),
        "left_wrist":     (275, 400),
        "right_wrist":    (365, 400),
        "left_hip":       (305, 420),
        "right_hip":      (335, 420),
        "left_knee":      (308, 560),
        "right_knee":     (332, 560),
        "left_ankle":     (310, 700),
        "right_ankle":    (330, 700),
    }

    angles = estimator.extract(fake_standing)
    print("Standing pose angles:")
    print(f"  Elbow : {angles['elbow']:.1f}°  (expect ~170)")
    print(f"  Knee  : {angles['knee']:.1f}°   (expect ~175)")
    print(f"  Hip   : {angles['hip']:.1f}°    (expect ~175)")
    print(f"  Body  : {angles['body_angle']:.1f}° from vertical  (expect ~0)")
    print()

    # Simulate bottom of pushup (elbows bent, body horizontal)
    fake_pushup_down = {
        "left_shoulder":  (200, 400),
        "right_shoulder": (400, 400),
        "left_elbow":     (180, 480),
        "right_elbow":    (420, 480),
        "left_wrist":     (175, 560),
        "right_wrist":    (425, 560),
        "left_hip":       (250, 410),
        "right_hip":      (350, 410),
        "left_knee":      (280, 420),
        "right_knee":     (320, 420),
        "left_ankle":     (310, 430),
        "right_ankle":    (330, 430),
    }

    angles2 = estimator.extract(fake_pushup_down)
    print("Pushup (down) pose angles:")
    print(f"  Elbow : {angles2['elbow']:.1f}°  (expect < 100)")
    print(f"  Knee  : {angles2['knee']:.1f}°   (expect > 140)")
    print(f"  Body  : {angles2['body_angle']:.1f}° from vertical  (expect ~80-90)")
    print()
    print("✅ PoseEstimator self-test complete.")