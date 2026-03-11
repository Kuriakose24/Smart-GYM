"""
pushup_scorer.py
----------------
Your friend's ML model + rule-based override — cleanly isolated.
Takes bottom angles, returns prediction + feedback + score.

Requires:
    Smart-GYM/models/pushup_final_model.pkl
    Smart-GYM/utils/feedback_engine.py
"""

import os
import sys
import joblib
import pandas as pd

SMART_GYM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SMART_GYM_ROOT)
from utils.feedback_engine import FeedbackEngine

MODEL_PKL = os.path.join(SMART_GYM_ROOT, "models", "pushup_final_model.pkl")


class PushupScorer:
    def __init__(self):
        print(f"[PushupScorer] Loading ML model...")
        self._model          = joblib.load(MODEL_PKL)
        self.feedback_engine = FeedbackEngine()
        self.correct_reps    = 0
        self.incorrect_reps  = 0
        self.last_feedback   = ""
        print("[PushupScorer] Ready.")

    def score_rep(self, elbow_angle, body_angle, hip_angle, back_angle):
        """
        Score a completed rep using ML + rule-based override.
        """
        print(f"[Scorer] Bottom angles → elbow:{elbow_angle:.1f}  "
              f"body:{body_angle:.1f}  hip:{hip_angle:.1f}  back:{back_angle:.1f}")
        # ML prediction
        features = pd.DataFrame(
            [[elbow_angle, back_angle, hip_angle, body_angle]],
            columns=["elbow_angle", "back_angle", "hip_angle", "knee_angle"]
        )
        prediction = self._model.predict(features)[0]

        # Rule-based override — tuned to Kevin's camera angle
        # Good reps:  hip 164-171, body 168-173
        # Bad reps:   hip 147-151, body 160-161
        rule_violation = False

        if elbow_angle > 110:
            prediction, feedback = "incorrect", "Go deeper!"
            rule_violation = True
        elif hip_angle < 160:
            prediction, feedback = "incorrect", "Hips sagging"
            rule_violation = True
        elif body_angle < 163:
            prediction, feedback = "incorrect", "Keep body straight"
            rule_violation = True

        # Score
        score = self.feedback_engine.calculate_rep_score(
            elbow_angle, body_angle, hip_angle, back_angle
        )
        self.feedback_engine.add_rep_score(score)

        # Final feedback
        if prediction == "correct":
            self.correct_reps += 1
            feedback = "Perfect Rep"
        else:
            self.incorrect_reps += 1
            if not rule_violation:
                feedback = self.feedback_engine.generate_feedback(
                    elbow_angle, body_angle, hip_angle, back_angle
                )

        self.last_feedback = feedback
        return prediction, feedback, score

    def get_summary(self):
        ws = self.feedback_engine.workout_summary()
        return {
            "correct_reps":   self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "average_score":  ws["average_score"],
            "trainer_alert":  ws["trainer_alert"]
        } 