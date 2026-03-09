"""
main.py
-------
Gym AI — Attendance + Pushup Tracker (YOLOv8-pose pipeline)

Pipeline:
    Camera
    → YOLOv8-pose (persons + keypoints in one shot)
    → FaceNet identity (locked for 120s)
    → RepCounter (angle-based, per person)
    → PushupScorer (ML + rules, per person)
    → Attendance CSV

Press 'q' to quit.
Press 's' to print session summary.
"""

import cv2
from camera import Camera
from recognizer import FaceRecognizer
from attendance import AttendanceTracker
from tracker import IdentityTracker
from pose_detector import PoseDetector
from rep_counter import RepCounter
from pushup_scorer import PushupScorer


# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_SOURCE    = 0
FACES_DIR        = "faces"
ATTENDANCE_DIR   = "attendance"
POSE_MODEL       = "yolov8n-pose.pt"
POSE_CONFIDENCE  = 0.5
FACE_THRESHOLD   = 0.7
IDENTITY_TIMEOUT = 120.0
# ──────────────────────────────────────────────────────────────────────────────


def run():
    print("\n=== Gym AI — YOLOv8-pose Pipeline ===\n")

    # Init modules
    camera      = Camera(source=CAMERA_SOURCE)
    pose_det    = PoseDetector(model_path=POSE_MODEL,
                               confidence=POSE_CONFIDENCE)
    recognizer  = FaceRecognizer(faces_dir=FACES_DIR,
                                 threshold=FACE_THRESHOLD)
    id_tracker  = IdentityTracker(recognizer, timeout=IDENTITY_TIMEOUT)
    att_tracker = AttendanceTracker(output_dir=ATTENDANCE_DIR)

    # Per-person state — created on first recognition
    rep_counters = {}   # name → RepCounter
    scorers      = {}   # name → PushupScorer

    # Load scorer once (shared ML model)
    print("[Main] Loading PushupScorer...")
    shared_scorer_loaded = False

    camera.start()
    print("\nControls: [q] quit   [s] summary\n")

    # Panel x positions for up to 3 persons side by side
    panel_positions = [20, 360, 700]

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                break

            # ── 1. Detect all persons + keypoints ─────────────
            persons = pose_det.detect(frame)

            # Draw skeleton on all persons
            pose_det.draw_skeleton(frame, persons)

            # Track which names are active this frame
            active = []

            for person in persons:
                box = person["box"]
                kp  = person["keypoints"]

                # ── 2. Identify person ────────────────────────
                name = id_tracker.identify(frame, box)
                active.append(name)

                # ── 3. Mark attendance ────────────────────────
                att_tracker.mark(name)

                if name == "Unknown":
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(frame, "Unknown",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0,0,255), 2)
                    continue

                # ── 4. Create per-person trackers ─────────────
                if name not in rep_counters:
                    rep_counters[name] = RepCounter(name=name)
                    scorers[name]      = PushupScorer()
                    print(f"[Main] Tracking started: {name}")

                # ── 5. Update rep counter ─────────────────────
                rep_completed = rep_counters[name].update(kp)

                # ── 6. Score the rep if completed ─────────────
                if rep_completed:
                    ea, ba, ha, bka = rep_counters[name].get_bottom_angles()
                    _, feedback, score = scorers[name].score_rep(ea, ba, ha, bka)
                    rep_counters[name].feedback = feedback
                    print(f"[{name}] Rep {rep_counters[name].rep_count} — "
                          f"{feedback} (score: {score})")

            # ── 7. Draw stats panels ──────────────────────────
            known = [n for n in active if n != "Unknown"]
            for i, name in enumerate(dict.fromkeys(known)):  # unique, ordered
                px = panel_positions[min(i, len(panel_positions)-1)]
                _draw_panel(frame, name,
                            rep_counters[name],
                            scorers[name], px)

            # ── 8. HUD bar ────────────────────────────────────
            _draw_hud(frame, att_tracker)

            cv2.imshow("Gym AI", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                _print_summary(att_tracker, rep_counters, scorers)

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        _print_summary(att_tracker, rep_counters, scorers)
        print("\n[Main] Session ended.")


def _draw_panel(frame, name, rep_counter, scorer, panel_x):
    """Draw stats panel for one person."""
    panel_y = 60
    w_panel = 320
    h_panel = 210

    # Dark background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x, panel_y),
                  (panel_x + w_panel, panel_y + h_panel),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.rectangle(frame,
                  (panel_x, panel_y),
                  (panel_x + w_panel, panel_y + h_panel),
                  (80, 80, 80), 1)

    rc  = rep_counter
    sc  = scorer
    dbg = rc.get_debug_info()

    lines = [
        (f"{name}",                                    (255, 255, 0)),
        (f"Reps      : {rc.rep_count}",                (255, 255, 255)),
        (f"Correct   : {sc.correct_reps}",             (0, 255, 0)),
        (f"Incorrect : {sc.incorrect_reps}",           (0, 80, 255)),
        (f"Avg Score : {sc.get_summary()['average_score']}", (200, 200, 0)),
        (f"Stage     : {dbg['stage']}",                (200, 200, 200)),
        (f"Feedback  : {rc.feedback}",                 (0, 200, 255)),
        (f"Elbow:{dbg['elbow']:.0f}  Body:{dbg['body']:.0f}  H:{'Y' if dbg['horizontal'] else 'N'}", (150, 150, 150)),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text,
                    (panel_x + 8, panel_y + 25 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, color, 1)


def _draw_hud(frame, tracker):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 50), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    present = sorted(tracker.marked)
    text = f"Present ({len(present)}): " + (", ".join(present) if present else "—")
    cv2.putText(frame, text, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


def _print_summary(tracker, rep_counters, scorers):
    print("\n" + "="*45)
    print(tracker.get_summary())
    for name in rep_counters:
        rc = rep_counters[name]
        sc = scorers[name].get_summary()
        print(f"\n--- {name} ---")
        print(f"  Total Reps    : {rc.rep_count}")
        print(f"  Correct       : {sc['correct_reps']}")
        print(f"  Incorrect     : {sc['incorrect_reps']}")
        print(f"  Avg Score     : {sc['average_score']}")
        if sc["trainer_alert"]:
            print("  ⚠  Needs improvement")
        else:
            print("  ✔  Good workout!")
    print("="*45)


if __name__ == "__main__":
    run()