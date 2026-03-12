"""
utils/feedback_engine.py  (v2 — YOLO-calibrated thresholds + richer coaching)
------------------------------------------------------------------------------
CHANGES vs v1:

OVERHAUL 1 — Angle thresholds recalibrated for YOLO
    Original thresholds were tuned for MediaPipe normalised coordinates.
    YOLO pose reads ~10-15° higher in horizontal position (pushup).
    Updated accordingly.

OVERHAUL 2 — calculate_rep_score() rewritten per exercise
    Old: one generic function for both pushup and squat.
    New: separate _score_pushup() and _score_squat() with exercise-specific
         weighting. Call calculate_rep_score(angles, exercise).

OVERHAUL 3 — generate_feedback() returns (message, severity)
    severity: "good" | "warning" | "error"
    Callers can use severity to colour the on-screen text.

OVERHAUL 4 — Per-rep history with get_rep_history()
    Each rep score stored with feedback for post-session review.

OVERHAUL 5 — Trainer alert threshold lowered to 60 (was 75)
    More appropriate for YOLO angle readings.

PUBLIC API (backwards compatible):
    engine = FeedbackEngine()
    score  = engine.calculate_rep_score(angles, exercise)   # returns 0-100
    msg, severity = engine.generate_feedback(angles, exercise)
    engine.add_rep_score(score, feedback, severity)
    avg    = engine.get_average_score()
    alert  = engine.trainer_alert_required()
    summary = engine.workout_summary()
    history = engine.get_rep_history()
    engine.reset()
"""


class FeedbackEngine:
    """Scores rep quality and generates coaching feedback."""

    ALERT_THRESHOLD = 60

    def __init__(self):
        self._rep_scores  = []
        self._rep_history = []

    # ─────────────────────────────────────────────────────────────────────────
    # Rep Score  (0–100)
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_rep_score(self, angles, exercise="pushup"):
        """
        Score a completed rep from 0–100.

        angles   : dict — same bottom_angles dict from RepCounter
                   keys: "elbow", "knee", "hip", "body_angle", "back_angle"
        exercise : "pushup" | "squat"
        """
        if exercise == "pushup":
            return self._score_pushup(angles)
        elif exercise == "squat":
            return self._score_squat(angles)
        return 50

    def _score_pushup(self, angles):
        """
        Pushup scoring — YOLO-calibrated.
        Weights: depth 35 | body alignment 30 | hip 20 | back 15
        """
        elbow = angles.get("elbow")      or 90
        body  = angles.get("body_angle") or 0
        hip   = angles.get("hip")        or 170
        back  = angles.get("back_angle") or 150

        score = 0

        # Depth (elbow bend at bottom)
        if elbow <= 90:
            score += 35
        elif elbow <= 105:
            score += 25
        elif elbow <= 115:
            score += 15
        else:
            score += 0

        # Body horizontal (body_angle from vertical — pushup ≈ 80-90°)
        if 70 <= body <= 95:
            score += 30
        elif 60 <= body <= 100:
            score += 18
        elif 50 <= body <= 110:
            score += 8
        else:
            score += 0

        # Hip alignment (shoulder→hip→knee, straight plank ≈ 165-180°)
        if hip >= 165:
            score += 20
        elif hip >= 155:
            score += 13
        elif hip >= 140:
            score += 6
        else:
            score += 0

        # Back posture (ear→shoulder→hip, neutral ≈ 150-170°)
        if back >= 150:
            score += 15
        elif back >= 135:
            score += 9
        elif back >= 120:
            score += 4
        else:
            score += 0

        return min(100, score)

    def _score_squat(self, angles):
        """
        Squat scoring — YOLO-calibrated.
        Weights: depth 35 | hip hinge 25 | back 25 | body control 15
        """
        knee = angles.get("knee")        or 160
        hip  = angles.get("hip")         or 170
        body = angles.get("body_angle")  or 0
        back = angles.get("back_angle")  or 150

        score = 0

        # Depth (knee angle — lower = deeper squat)
        if knee <= 95:
            score += 35
        elif knee <= 110:
            score += 25
        elif knee <= 125:
            score += 15
        elif knee <= 140:
            score += 5
        else:
            score += 0

        # Hip hinge (shoulder→hip→knee — good squat ≈ 60-130°)
        if 60 <= hip <= 130:
            score += 25
        elif 130 < hip <= 150:
            score += 15
        elif 150 < hip <= 165:
            score += 7
        else:
            score += 0

        # Back upright (ear→shoulder→hip — upright ≈ 140°+)
        if back >= 145:
            score += 25
        elif back >= 130:
            score += 16
        elif back >= 115:
            score += 8
        else:
            score += 0

        # Body control (from vertical — squatting ≈ 20-50°)
        if 15 <= body <= 55:
            score += 15
        elif 10 <= body <= 70:
            score += 8
        else:
            score += 0

        return min(100, score)

    # ─────────────────────────────────────────────────────────────────────────
    # Feedback / coaching tips
    # ─────────────────────────────────────────────────────────────────────────

    def generate_feedback(self, angles, exercise="pushup"):
        """
        Returns (message: str, severity: str).
        severity: "good" | "warning" | "error"
        Returns the single most important coaching tip.
        """
        if exercise == "pushup":
            return self._feedback_pushup(angles)
        elif exercise == "squat":
            return self._feedback_squat(angles)
        return ("Good rep!", "good")

    def _feedback_pushup(self, angles):
        elbow = angles.get("elbow")      or 90
        body  = angles.get("body_angle") or 85
        hip   = angles.get("hip")        or 170
        back  = angles.get("back_angle") or 150

        # Errors — most impactful first
        if elbow > 115:
            return ("Go lower — chest closer to floor", "error")
        if body < 55:
            return ("Body not horizontal — get into plank", "error")
        if hip < 140:
            return ("Hips sagging — tighten your core", "error")
        if hip > 175 and body < 75:
            return ("Hips too high — lower your hips", "error")

        # Warnings
        if elbow > 100:
            return ("Push a little lower for full depth", "warning")
        if back < 130:
            return ("Keep your head neutral — don't drop chin", "warning")
        if hip < 155:
            return ("Engage core — hips dipping slightly", "warning")
        if hip > 175:
            return ("Slight hip pike — try to level out", "warning")

        return ("Perfect form — keep it up!", "good")

    def _feedback_squat(self, angles):
        knee = angles.get("knee")        or 160
        hip  = angles.get("hip")         or 170
        body = angles.get("body_angle")  or 10
        back = angles.get("back_angle")  or 150

        # Errors
        if knee > 145:
            return ("Go deeper — squat below parallel", "error")
        if back < 115:
            return ("Keep chest up — don't lean forward", "error")
        if hip > 175:
            return ("Sit back more — push hips back and down", "error")

        # Warnings
        if knee > 120:
            return ("Go a bit deeper for full range", "warning")
        if back < 130:
            return ("Chest slightly forward — stay more upright", "warning")
        if hip > 165:
            return ("Push hips back slightly more", "warning")
        if body > 60:
            return ("Leaning too far forward — chest up", "warning")

        return ("Great squat — full depth and upright!", "good")

    # ─────────────────────────────────────────────────────────────────────────
    # Score tracking
    # ─────────────────────────────────────────────────────────────────────────

    def add_rep_score(self, score, feedback="", severity="good"):
        self._rep_scores.append(score)
        self._rep_history.append({
            "rep":      len(self._rep_scores),
            "score":    score,
            "feedback": feedback,
            "severity": severity,
        })

    def get_average_score(self):
        if not self._rep_scores:
            return 0
        return sum(self._rep_scores) / len(self._rep_scores)

    def get_rep_history(self):
        return list(self._rep_history)

    def trainer_alert_required(self):
        return self.get_average_score() < self.ALERT_THRESHOLD

    def workout_summary(self):
        avg = self.get_average_score()
        return {
            "total_reps":    len(self._rep_scores),
            "average_score": round(avg, 2),
            "trainer_alert": self.trainer_alert_required(),
            "rep_history":   self.get_rep_history(),
        }

    def reset(self):
        self._rep_scores  = []
        self._rep_history = []