# utils/feedback_engine.py


class FeedbackEngine:
    """
    Module 3: Feedback Engine

    Responsibilities:
    - Calculate rep score
    - Generate prioritized feedback
    - Store rep scores
    - Calculate average workout score
    - Decide if trainer alert is needed
    """

    def __init__(self):
        self.rep_scores = []

    # ---------------------------------
    # Rep Score Calculation
    # ---------------------------------
    def calculate_rep_score(self, elbow_angle, body_angle, hip_angle, back_angle):

        score = 0

        # Body alignment (highest priority)
        if 160 <= body_angle <= 200:
            score += 40
        elif 140 <= body_angle < 160:
            score += 25
        else:
            score += 10

        # Hip alignment
        if hip_angle > 160:
            score += 25
        elif hip_angle > 140:
            score += 15
        else:
            score += 5

        # Back alignment
        if back_angle > 150:
            score += 20
        elif back_angle > 130:
            score += 10
        else:
            score += 5

        # Push-up depth
        if elbow_angle < 90:
            score += 15
        elif elbow_angle < 110:
            score += 10
        else:
            score += 5

        return score

    # ---------------------------------
    # Prioritized Feedback Generator
    # ---------------------------------
    def generate_feedback(self, elbow_angle, body_angle, hip_angle, back_angle):

        # Priority order for corrections

        if body_angle < 140:
            return "Hips sagging"

        elif body_angle > 210:
            return "Hips too high"

        elif elbow_angle > 100:
            return "Lower your body"

        elif back_angle < 120:
            return "Keep back straight"

        else:
            return "Good form"

    # ---------------------------------
    # Store Rep Score
    # ---------------------------------
    def add_rep_score(self, score):

        self.rep_scores.append(score)

    # ---------------------------------
    # Average Workout Score
    # ---------------------------------
    def get_average_score(self):

        if len(self.rep_scores) == 0:
            return 0

        return sum(self.rep_scores) / len(self.rep_scores)

    # ---------------------------------
    # Trainer Alert Decision
    # ---------------------------------
    def trainer_alert_required(self):

        avg_score = self.get_average_score()

        if avg_score < 75:
            return True

        return False

    # ---------------------------------
    # Workout Summary
    # ---------------------------------
    def workout_summary(self):

        avg_score = self.get_average_score()

        summary = {
            "total_reps": len(self.rep_scores),
            "average_score": round(avg_score, 2),
            "trainer_alert": self.trainer_alert_required()
        }

        return summary