"""
exercise/rep_counter.py  (v9 — modular: counting only)
-------------------------------------------------------
CHANGES vs v8:

ARCHITECTURE — rep_counter now has ONE job: counting.
    Form analysis  → exercise/form_analyzer.FormAnalyzer
    Score + tips   → utils/feedback_engine.FeedbackEngine (v2)
    Rep counting   → this file only

PER-REP HISTORY:
    Every completed rep stored in per_rep_results:
    { rep, exercise, prediction, rule_reason, score, feedback, severity, angles }
    Surfaced in get_stats() and manager summary.
"""

import sys, os, time
from collections import deque
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from utils.feedback_engine  import FeedbackEngine
from exercise.form_analyzer import FormAnalyzer



# Shared FormAnalyzer — models loaded once
_form_analyzer = FormAnalyzer() if FormAnalyzer else None


# ── AngleSmoother ─────────────────────────────────────────────────────────────
class AngleSmoother:
    def __init__(self, window=5, max_velocity=45):
        self.window       = window
        self.max_velocity = max_velocity
        self._history     = deque(maxlen=window)
        self._last        = None

    def update(self, angle):
        if angle is None:
            return self._last
        if self._last is not None and abs(angle - self._last) > self.max_velocity:
            angle = (self._last + angle) / 2.0
        self._history.append(angle)
        self._last = sum(self._history) / len(self._history)
        return self._last

    def reset(self):
        self._history.clear()
        self._last = None


# ── RepCounter ────────────────────────────────────────────────────────────────
class RepCounter:
    """Counts reps for ONE person. Keyed by name — survives track ID changes."""

    HYSTERESIS          = 5.0
    MIN_DOWN_FRAMES     = 1
    SQUAT_MAX_HIP_ANGLE = 170.0

    def __init__(self, name="Unknown", exercise="pushup"):
        self.name     = name
        self.exercise = exercise

        self.rep_count      = 0
        self.correct_reps   = 0
        self.incorrect_reps = 0

        self.stage              = "UP"
        self.feedback           = "Ready — start your exercise!"
        self.feedback_severity  = "good"
        self.last_form_feedback = ""
        self.last_form_severity = "good"
        self.form_score         = None
        self.last_prediction    = None

        # Full per-rep history
        self.per_rep_results = []

        self._elbow_smoother    = AngleSmoother(window=5, max_velocity=45)
        self._knee_smoother     = AngleSmoother(window=5, max_velocity=45)
        self._hip_drop_smoother = AngleSmoother(window=4, max_velocity=0.3)

        self._down_frames     = 0
        self._cooldown        = 0
        self._last_body_angle = None
        self._last_back_angle = None

        self.bottom_angles = {
            "elbow": 0.0, "knee": 0.0, "hip": 0.0,
            "body_angle": 0.0, "back_angle": 0.0
        }
        self.best_depth        = 180.0
        self.total_depth       = 0.0
        self._depth_rep_count  = 0
        self.last_rep_time     = None
        self.per_exercise_reps = {}

        self._feedback_engine = FeedbackEngine() if FeedbackEngine else None

        print(f"[RepCounter] Created for '{name}' — {exercise}")

    # ── Frame update ──────────────────────────────────────────────────────────

    def update(self, angles, current_exercise="unknown"):
        if angles is None:
            return False
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        if current_exercise == "unknown" or current_exercise != self.exercise:
            return False

        ba = angles.get("body_angle")
        if ba is not None:
            self._last_body_angle = ba
        bk = angles.get("back_angle")
        if bk is not None:
            self._last_back_angle = bk

        if self.exercise == "pushup":
            return self._update_pushup(angles)
        elif self.exercise == "squat":
            return self._update_squat(angles)
        return False

    # ── Pushup state machine ──────────────────────────────────────────────────

    def _update_pushup(self, angles):
        raw_elbow = angles.get("elbow")
        if raw_elbow is None:
            return False
        elbow = self._elbow_smoother.update(raw_elbow)
        if elbow is None:
            return False

        body = angles.get("body_angle") or self._last_body_angle
        back = angles.get("back_angle") or self._last_back_angle

        if elbow < config.PUSHUP_ELBOW_DOWN:
            self._down_frames += 1
            self.bottom_angles = {
                "elbow":      elbow,
                "knee":       angles.get("knee",      0.0) or 0.0,
                "hip":        angles.get("hip",        0.0) or 0.0,
                "body_angle": body or 0.0,
                "back_angle": back or 0.0,
            }
            if elbow < self.best_depth:
                self.best_depth = elbow
            if self.stage == "UP":
                self.stage    = "DOWN"
                self.feedback = "Push up!"

        elif elbow > config.PUSHUP_ELBOW_UP - self.HYSTERESIS:
            if self.stage == "DOWN" and self._down_frames >= self.MIN_DOWN_FRAMES:
                self._complete_rep()
                self._down_frames = 0
                return True
            self._down_frames = 0
            if self.stage != "UP":
                self.stage    = "UP"
                self.feedback = f"Rep {self.rep_count} — go again!"

        return False

    # ── Squat state machine ───────────────────────────────────────────────────

    def _update_squat(self, angles):
        is_front_view  = angles.get("_is_front_view",  False)
        hip_drop_ratio = angles.get("_hip_drop_ratio", None)
        if is_front_view and hip_drop_ratio is not None:
            return self._update_squat_front(angles, hip_drop_ratio)
        return self._update_squat_side(angles)

    def _update_squat_front(self, angles, hip_drop_ratio):
        smooth = self._hip_drop_smoother.update(hip_drop_ratio)
        if smooth is None:
            return False

        if smooth < config.SQUAT_HIP_DROP_DOWN:
            self._down_frames += 1
            self.bottom_angles = {
                "elbow":      angles.get("elbow",      0.0) or 0.0,
                "knee":       angles.get("knee",        0.0) or 0.0,
                "hip":        angles.get("hip",         0.0) or 0.0,
                "body_angle": angles.get("body_angle",  0.0) or 0.0,
                "back_angle": angles.get("back_angle",  0.0) or 0.0,
            }
            knee_val = angles.get("knee") or 180.0
            if knee_val < self.best_depth:
                self.best_depth = knee_val
            if self.stage == "UP":
                self.stage    = "DOWN"
                self.feedback = "Stand up!"

        elif smooth > config.SQUAT_HIP_DROP_UP - 0.05:
            if self.stage == "DOWN" and self._down_frames >= self.MIN_DOWN_FRAMES:
                self._complete_rep()
                self._down_frames = 0
                return True
            self._down_frames = 0
            if self.stage != "UP":
                self.stage    = "UP"
                self.feedback = f"Rep {self.rep_count} — go again!"

        return False

    def _update_squat_side(self, angles):
        raw_knee  = angles.get("knee")
        hip_angle = angles.get("hip")
        if raw_knee is None:
            return False
        knee = self._knee_smoother.update(raw_knee)
        if knee is None:
            return False

        hip_flexing  = (hip_angle is None) or (hip_angle < self.SQUAT_MAX_HIP_ANGLE)
        up_threshold = config.SQUAT_KNEE_UP - self.HYSTERESIS

        if knee < config.SQUAT_KNEE_DOWN and hip_flexing:
            self._down_frames += 1
            self.bottom_angles = {
                "elbow":      angles.get("elbow",      0.0) or 0.0,
                "knee":       knee,
                "hip":        angles.get("hip",         0.0) or 0.0,
                "body_angle": angles.get("body_angle",  0.0) or 0.0,
                "back_angle": angles.get("back_angle",  0.0) or 0.0,
            }
            if knee < self.best_depth:
                self.best_depth = knee
            if self.stage == "UP":
                self.stage    = "DOWN"
                self.feedback = "Stand up!"

        elif knee > up_threshold:
            if self.stage == "DOWN" and self._down_frames >= self.MIN_DOWN_FRAMES:
                self._complete_rep()
                self._down_frames = 0
                return True
            self._down_frames = 0
            if self.stage != "UP":
                self.stage    = "UP"
                self.feedback = f"Rep {self.rep_count} — go again!"

        return False

    # ── Rep completion ────────────────────────────────────────────────────────

    def _complete_rep(self):
        self.rep_count        += 1
        self._depth_rep_count += 1
        self.stage             = "UP"
        self._cooldown         = config.REP_COOLDOWN_FRAMES
        self.last_rep_time     = time.time()
        depth_key              = "elbow" if self.exercise == "pushup" else "knee"
        self.total_depth      += self.bottom_angles.get(depth_key, 0)
        self.per_exercise_reps[self.exercise] = \
            self.per_exercise_reps.get(self.exercise, 0) + 1

        # Step 1: Form analysis
        form_result = None
        if _form_analyzer:
            form_result = _form_analyzer.analyze(self.bottom_angles, self.exercise)

        prediction  = form_result.prediction  if form_result else None
        rule_reason = form_result.rule_reason  if form_result else None

        # Step 2: Score + coaching tip
        score    = None
        tip      = ""
        severity = "good"

        if self._feedback_engine:
            score = self._feedback_engine.calculate_rep_score(
                self.bottom_angles, self.exercise
            )
            tip, severity = self._feedback_engine.generate_feedback(
                self.bottom_angles, self.exercise
            )
            self._feedback_engine.add_rep_score(score, tip, severity)

        self.form_score      = score
        self.last_prediction = prediction

        # Step 3: Display feedback text
        if prediction is None:
            fb       = f"Rep {self.rep_count} done!"
            severity = "good"
        elif prediction == "correct":
            self.correct_reps += 1
            score_str = f"  Score:{score}" if score is not None else ""
            verb = "Perfect!" if self.exercise == "pushup" else "Great squat!"
            fb   = f"✅ Rep {self.rep_count} — {verb}{score_str}"
            severity = "good"
        else:
            self.incorrect_reps += 1
            reason = rule_reason or tip or "Check your form"
            fb     = f"❌ Rep {self.rep_count} — {reason}"
            severity = "error"

        self.feedback           = fb
        self.last_form_feedback = fb
        self.last_form_severity = severity

        # Step 4: Store per-rep record
        self.per_rep_results.append({
            "rep":         self.rep_count,
            "exercise":    self.exercise,
            "prediction":  prediction,
            "rule_reason": rule_reason,
            "score":       score,
            "feedback":    fb,
            "severity":    severity,
            "angles": {
                "elbow": round(self.bottom_angles.get("elbow",      0), 1),
                "knee":  round(self.bottom_angles.get("knee",       0), 1),
                "hip":   round(self.bottom_angles.get("hip",        0), 1),
                "body":  round(self.bottom_angles.get("body_angle", 0), 1),
                "back":  round(self.bottom_angles.get("back_angle", 0), 1),
            }
        })

        print(f"[RepCounter] ✅ {self.name} — Rep {self.rep_count} "
              f"({self.exercise} #{self.per_exercise_reps[self.exercise]})  "
              f"pred={prediction}  score={score}  "
              f"✅{self.correct_reps} ❌{self.incorrect_reps}  → {fb}")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_avg_depth(self):
        return (self.total_depth / self._depth_rep_count
                if self._depth_rep_count else 0.0)

    def get_avg_form_score(self):
        scores = [r["score"] for r in self.per_rep_results if r["score"] is not None]
        return round(sum(scores) / len(scores), 1) if scores else None

    def get_form_accuracy(self):
        if self.rep_count == 0:
            return None
        return round(self.correct_reps / self.rep_count * 100, 1)

    def trainer_alert_required(self):
        if self._feedback_engine:
            return self._feedback_engine.trainer_alert_required()
        acc = self.get_form_accuracy()
        return False if acc is None else acc < 60.0

    def switch_exercise(self, exercise):
        if exercise == self.exercise:
            return
        self.exercise           = exercise
        self.stage              = "UP"
        self.feedback           = "Ready — start your exercise!"
        self.last_form_feedback = ""
        self._down_frames       = 0
        self._cooldown          = 0
        self.best_depth         = 180.0
        self.total_depth        = 0.0
        self._depth_rep_count   = 0
        self._last_body_angle   = None
        self._last_back_angle   = None
        self.form_score         = None
        self.last_prediction    = None
        self._elbow_smoother.reset()
        self._knee_smoother.reset()
        self._hip_drop_smoother.reset()
        print(f"[RepCounter] {self.name} switched to {exercise}")

    def get_stats(self):
        return {
            "name":               self.name,
            "exercise":           self.exercise,
            "reps":               self.rep_count,
            "correct_reps":       self.correct_reps,
            "incorrect_reps":     self.incorrect_reps,
            "form_accuracy":      self.get_form_accuracy(),
            "per_exercise_reps":  dict(self.per_exercise_reps),
            "stage":              self.stage,
            "feedback":           self.feedback,
            "last_form_feedback": self.last_form_feedback,
            "last_form_severity": self.last_form_severity,
            "best_depth":         round(self.best_depth, 1),
            "avg_depth":          round(self.get_avg_depth(), 1),
            "avg_form_score":     self.get_avg_form_score(),
            "last_prediction":    self.last_prediction,
            "trainer_alert":      self.trainer_alert_required(),
            "per_rep_results":    list(self.per_rep_results),
        }


# ── RepCounterManager ─────────────────────────────────────────────────────────
class RepCounterManager:

    def __init__(self, default_exercise="pushup"):
        self.default_exercise = default_exercise
        self.counters = {}
        print(f"[RepCounterManager] v9 ready. Default: {default_exercise}")

    def update(self, tracked_with_angles):
        results = []

        for person in tracked_with_angles:
            name     = person.get("name", "Unknown")
            angles   = person.get("angles")
            exercise = person.get("exercise", "unknown")

            if name == "Unknown":
                result = dict(person)
                result.update({
                    "rep_count": 0, "correct_reps": 0, "incorrect_reps": 0,
                    "form_accuracy": None, "stage": "Identifying...",
                    "feedback": "Please face camera", "last_form_feedback": "",
                    "last_form_severity": "good", "rep_completed": False,
                    "bottom_angles": {}, "form_score": None,
                    "last_prediction": None, "trainer_alert": False,
                    "per_rep_results": [],
                })
                results.append(result)
                continue

            if name not in self.counters:
                self.counters[name] = RepCounter(
                    name=name, exercise=self.default_exercise
                )

            counter       = self.counters[name]
            rep_completed = counter.update(angles, current_exercise=exercise)

            result = dict(person)
            result.update({
                "rep_count":          counter.rep_count,
                "correct_reps":       counter.correct_reps,
                "incorrect_reps":     counter.incorrect_reps,
                "form_accuracy":      counter.get_form_accuracy(),
                "stage":              counter.stage,
                "feedback":           counter.feedback,
                "last_form_feedback": counter.last_form_feedback,
                "last_form_severity": counter.last_form_severity,
                "rep_completed":      rep_completed,
                "bottom_angles":      counter.bottom_angles,
                "form_score":         counter.form_score,
                "last_prediction":    counter.last_prediction,
                "trainer_alert":      counter.trainer_alert_required(),
                "per_rep_results":    counter.per_rep_results,
            })
            results.append(result)

        return results

    def set_exercise_for_all(self, exercise):
        for counter in self.counters.values():
            counter.switch_exercise(exercise)
        self.default_exercise = exercise

    def set_exercise_for_person(self, name, exercise):
        if name in self.counters:
            self.counters[name].switch_exercise(exercise)
        else:
            self.counters[name] = RepCounter(name=name, exercise=exercise)

    def get_summary(self):
        return {key: c.get_stats() for key, c in self.counters.items()}

    def get_person_reps(self, name):
        return self.counters[name].rep_count if name in self.counters else 0

    def session_summary(self):
        summaries = {}
        for name, counter in self.counters.items():
            stats    = counter.get_stats()
            ex       = stats["exercise"].capitalize()
            accuracy = stats["form_accuracy"]
            alert    = stats["trainer_alert"]

            print("\n" + "=" * 50)
            print(f"  {ex} Session Summary — {name}")
            print("=" * 50)
            print(f"  Total reps      : {stats['reps']}")
            print(f"  Correct reps    : {stats['correct_reps']}")
            print(f"  Incorrect reps  : {stats['incorrect_reps']}")
            if accuracy is not None:
                print(f"  Form accuracy   : {accuracy:.0f}%")
            if stats["avg_form_score"] is not None:
                print(f"  Avg form score  : {stats['avg_form_score']}/100")

            if stats["per_rep_results"]:
                print(f"\n  Per-rep breakdown:")
                for r in stats["per_rep_results"]:
                    icon      = "✅" if r["prediction"] == "correct" else "❌"
                    score_str = f"  score={r['score']}" if r["score"] is not None else ""
                    a         = r["angles"]
                    angle_str = (f"  [E:{a['elbow']} K:{a['knee']} "
                                 f"H:{a['hip']} B:{a['body']} Bk:{a['back']}]")
                    print(f"    Rep {r['rep']:2d}  {icon}  {r['feedback']}"
                          f"{score_str}{angle_str}")

            if alert:
                print("\n  ⚠  Trainer Alert: posture needs work")
            else:
                print("\n  ✅ Great workout!")
            print("=" * 50)

            summaries[name] = stats
        return summaries