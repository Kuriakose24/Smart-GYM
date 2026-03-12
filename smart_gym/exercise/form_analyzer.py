"""
exercise/form_analyzer.py  (v1 — modular form analysis)
--------------------------------------------------------
NEW MODULE — extracted from rep_counter.py

Responsibilities:
    1. Load and cache ML models (pushup_final_model.pkl, squat_final_model.pkl)
    2. Apply rule-based overrides BEFORE ML (catches obvious errors quickly)
    3. Run the ML model if no rule fired
    4. Return a structured FormResult for RepCounter to use

WHY THIS MODULE EXISTS:
    rep_counter.py was doing 4 jobs: counting, ML prediction, rule checking,
    and feedback generation. That made it hard to tune any one part.
    Now each module has a single job:
        form_analyzer.py  → is this rep correct/incorrect + why?
        feedback_engine.py → what score? what coaching tip?
        rep_counter.py     → counting only (UP/DOWN state machine)

IMPORTANT — ANGLE SOURCES:
    The ML models were trained with MediaPipe angles.
    YOLO pose_estimator now produces matching angles:
        elbow_angle  = shoulder→elbow→wrist
        hip_angle    = shoulder→hip→knee
        back_angle   = ear→shoulder→hip    (added in pose_estimator v3)
        knee_angle   = hip→knee→ankle

    Pushup model columns: ["elbow_angle", "back_angle", "hip_angle", "knee_angle"]
    Squat  model columns: ["knee_angle",  "hip_angle",  "back_angle"]

RULE THRESHOLDS — YOLO CALIBRATED:
    All thresholds in this file are calibrated for YOLO pixel-space angles.
    YOLO reads ~10-15° higher than MediaPipe in horizontal position.
    Edit PUSHUP_RULES / SQUAT_RULES dicts at the top to retune without
    touching any other code.

PUBLIC API:
    analyzer = FormAnalyzer()
    result   = analyzer.analyze(bottom_angles, exercise)

    result.prediction   → "correct" | "incorrect" | None (no model)
    result.rule_fired   → bool
    result.rule_reason  → str or None  ("Go lower", "Hips sagging", ...)
    result.ml_used      → bool
    result.angles_used  → dict  (the angles that were fed to the model)
"""

import sys, os
import pandas as pd
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Rule thresholds (edit here to retune — no other file changes needed) ──────

PUSHUP_RULES = {
    # (condition_lambda, feedback_string)
    # Evaluated in order — FIRST match wins
    "go_lower":    lambda e, b, h, bk: e > 115,        # elbow not bent enough
    "hips_high":   lambda e, b, h, bk: b < 55 and h < 150,  # body not flat + hip high
    "hips_sagging":lambda e, b, h, bk: b < 55 and h > 165,  # body not flat + hip drops
    "head_drop":   lambda e, b, h, bk: bk is not None and bk < 110,  # chin to chest
}
PUSHUP_RULE_MESSAGES = {
    "go_lower":     "Go lower",
    "hips_high":    "Hips too high",
    "hips_sagging": "Hips sagging",
    "head_drop":    "Keep head neutral",
}

SQUAT_RULES = {
    "go_deeper":    lambda k, h, bk: k > 145,               # knee not bent enough
    "chest_down":   lambda k, h, bk: bk is not None and bk < 115,  # leaning too far
    "no_hip_hinge": lambda k, h, bk: h > 175,               # hips didn't move
}
SQUAT_RULE_MESSAGES = {
    "go_deeper":    "Go deeper",
    "chest_down":   "Keep chest up",
    "no_hip_hinge": "Sit back more",
}


# ── FormResult ────────────────────────────────────────────────────────────────

class FormResult:
    """Returned by FormAnalyzer.analyze(). Carries all analysis info."""
    __slots__ = ("prediction", "rule_fired", "rule_reason", "ml_used", "angles_used")

    def __init__(self):
        self.prediction  = None    # "correct" | "incorrect" | None
        self.rule_fired  = False
        self.rule_reason = None    # e.g. "Go lower"
        self.ml_used     = False
        self.angles_used = {}

    def __repr__(self):
        return (f"FormResult(prediction={self.prediction!r}, "
                f"rule={self.rule_reason!r}, ml={self.ml_used})")


# ── Model loader ──────────────────────────────────────────────────────────────

def _find_model(filename):
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, ".."))
    candidates = [
        os.path.join(_root, "models", filename),
        os.path.join(_here, "models", filename),
        os.path.join(_root, filename),
        os.path.join(_here, filename),
        os.path.join(config.MODEL_DIR, filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ── FormAnalyzer ──────────────────────────────────────────────────────────────

class FormAnalyzer:
    """
    Single instance — shared across all persons.
    Thread-safe for read (predict) operations.
    """

    def __init__(self):
        self._pushup_model = None
        self._squat_model  = None
        self._load_models()

    def _load_models(self):
        pm = _find_model("pushup_final_model.pkl")
        sm = _find_model("squat_final_model.pkl")

        if pm:
            try:
                self._pushup_model = joblib.load(pm)
                print(f"[FormAnalyzer] ✅ Pushup model: {pm}")
            except Exception as e:
                print(f"[FormAnalyzer] ⚠ Pushup model load failed: {e}")
        else:
            print("[FormAnalyzer] ⚠ pushup_final_model.pkl not found")

        if sm:
            try:
                self._squat_model = joblib.load(sm)
                print(f"[FormAnalyzer] ✅ Squat model:  {sm}")
            except Exception as e:
                print(f"[FormAnalyzer] ⚠ Squat model load failed: {e}")
        else:
            print("[FormAnalyzer] ⚠ squat_final_model.pkl not found")

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, bottom_angles, exercise):
        """
        Analyze form for one completed rep.

        bottom_angles : dict captured at deepest point of the rep
                        keys: "elbow", "knee", "hip", "body_angle", "back_angle"
        exercise      : "pushup" | "squat"

        Returns FormResult.
        """
        if exercise == "pushup":
            return self._analyze_pushup(bottom_angles)
        elif exercise == "squat":
            return self._analyze_squat(bottom_angles)

        result = FormResult()
        result.angles_used = bottom_angles
        return result

    # ── Pushup ────────────────────────────────────────────────────────────────

    def _analyze_pushup(self, angles):
        result = FormResult()
        result.angles_used = angles

        elbow = angles.get("elbow")      or 90
        body  = angles.get("body_angle") or 85
        hip   = angles.get("hip")        or 170
        back  = angles.get("back_angle")            # can be None — rules handle it

        # Step 1: Rule overrides
        for rule_key, condition in PUSHUP_RULES.items():
            try:
                if condition(elbow, body, hip, back):
                    result.prediction  = "incorrect"
                    result.rule_fired  = True
                    result.rule_reason = PUSHUP_RULE_MESSAGES[rule_key]
                    return result
            except Exception:
                pass

        # Step 2: ML model
        if self._pushup_model is not None:
            try:
                back_val = back if back is not None else 150   # neutral fallback
                features = pd.DataFrame(
                    [[elbow, back_val, hip, body]],
                    columns=["elbow_angle", "back_angle", "hip_angle", "knee_angle"]
                )
                result.prediction = self._pushup_model.predict(features)[0]
                result.ml_used    = True
            except Exception as e:
                print(f"[FormAnalyzer] Pushup predict error: {e}")

        return result

    # ── Squat ─────────────────────────────────────────────────────────────────

    def _analyze_squat(self, angles):
        result = FormResult()
        result.angles_used = angles

        knee = angles.get("knee")        or 160
        hip  = angles.get("hip")         or 170
        back = angles.get("back_angle")           # can be None

        # Step 1: Rule overrides
        for rule_key, condition in SQUAT_RULES.items():
            try:
                if condition(knee, hip, back):
                    result.prediction  = "incorrect"
                    result.rule_fired  = True
                    result.rule_reason = SQUAT_RULE_MESSAGES[rule_key]
                    return result
            except Exception:
                pass

        # Step 2: ML model
        if self._squat_model is not None:
            try:
                back_val = back if back is not None else 150
                features = pd.DataFrame(
                    [[knee, hip, back_val]],
                    columns=["knee_angle", "hip_angle", "back_angle"]
                )
                result.prediction = self._squat_model.predict(features)[0]
                result.ml_used    = True
            except Exception as e:
                print(f"[FormAnalyzer] Squat predict error: {e}")

        return result

    @property
    def pushup_model_loaded(self):
        return self._pushup_model is not None

    @property
    def squat_model_loaded(self):
        return self._squat_model is not None