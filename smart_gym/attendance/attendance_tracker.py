"""
attendance/attendance_tracker.py
----------------------------------
Marks attendance exactly ONCE per person per session.
Saves to a CSV file in attendance/ folder.

Upgraded from your original attendance.py:
    OLD: marks as soon as name appears (even Unknown)
    NEW: only marks after identity is LOCKED (confirmed by FaceNet)
         waits for N consecutive frames of same name before marking
         this prevents false attendance from brief misidentifications

Output: attendance/2026-03-09.csv
    Name,Time,Date
    Kevin,10:05:32,2026-03-09
    Nickson,10:06:14,2026-03-09
"""

import os
import csv
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AttendanceTracker:
    # Number of consecutive frames a name must appear before marking
    CONFIRM_FRAMES = 15

    def __init__(self, output_dir=config.ATTENDANCE_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Today's CSV path
        today = datetime.now().strftime("%Y-%m-%d")
        self.csv_path = os.path.join(output_dir, f"{today}.csv")
        self.today    = today

        # People already marked today (loaded from file if exists)
        self.marked = set()
        self._load_existing()

        # Confirmation buffer — track consecutive frames per name
        # { name: consecutive_frame_count }
        self._confirm_buffer = {}

        print(f"[Attendance] Log: {self.csv_path}")
        if self.marked:
            print(f"[Attendance] Already marked today: {sorted(self.marked)}")

    def _load_existing(self):
        """Load already-marked names from today's CSV (resume after restart)."""
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.marked.add(row["Name"])

    def update(self, names):
        """
        Call every frame with list of currently identified names.
        Only marks attendance after CONFIRM_FRAMES consecutive appearances.

        names : list of strings e.g. ["Kevin", "Nickson", "Unknown"]

        Returns list of newly marked names this frame (usually empty).
        """
        newly_marked = []

        # Filter out Unknown
        known_names = [n for n in names if n != "Unknown"]

        # Increment confirmation buffer for visible names
        for name in known_names:
            if name not in self.marked:
                self._confirm_buffer[name] = \
                    self._confirm_buffer.get(name, 0) + 1

                # Mark once confirmed
                if self._confirm_buffer[name] >= self.CONFIRM_FRAMES:
                    if self._mark(name):
                        newly_marked.append(name)

        # Reset buffer for names no longer visible
        visible = set(known_names)
        for name in list(self._confirm_buffer.keys()):
            if name not in visible:
                self._confirm_buffer[name] = 0

        return newly_marked

    def _mark(self, name):
        """
        Write one attendance record to CSV.
        Returns True if newly marked, False if already marked.
        """
        if name in self.marked:
            return False

        now       = datetime.now()
        time_str  = now.strftime("%H:%M:%S")
        date_str  = now.strftime("%Y-%m-%d")

        # Write header only if file is new
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Time", "Date"])
            writer.writerow([name, time_str, date_str])

        self.marked.add(name)
        del self._confirm_buffer[name]
        print(f"[Attendance] ✅ Marked: {name} at {time_str}")
        return True

    def mark_direct(self, name):
        """
        Mark attendance immediately without confirmation buffer.
        Use this for manual override.
        """
        if name != "Unknown":
            return self._mark(name)
        return False

    def get_summary(self):
        """Return formatted summary string."""
        if not self.marked:
            return "No attendance marked yet today."
        lines = [f"--- Attendance {self.today} ---"]
        for name in sorted(self.marked):
            lines.append(f"  ✅ {name}")
        lines.append(f"  Total: {len(self.marked)} person(s)")
        return "\n".join(lines)

    def is_marked(self, name):
        """Check if a person has been marked today."""
        return name in self.marked

    def get_confirmation_progress(self, name):
        """
        Returns (current_count, required_count) for a name being confirmed.
        Useful for showing a progress indicator on screen.
        """
        current  = self._confirm_buffer.get(name, 0)
        required = self.CONFIRM_FRAMES
        return current, required


# ── Test — run this file directly ─────────────────────────────────────────────
# python attendance/attendance_tracker.py
if __name__ == "__main__":
    import cv2
    import time
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera.video_stream import VideoStream
    from detection.person_detector import PersonDetector
    from tracking.person_tracker import PersonTracker
    from identity.identity_linker import IdentityLinker

    print("=" * 58)
    print("  Attendance Test — press Q to quit")
    print(f"  Stand in frame — attendance marks after {AttendanceTracker.CONFIRM_FRAMES} frames")
    print("  Check attendance/ folder for CSV after test")
    print("=" * 58)

    cam      = VideoStream()
    detector = PersonDetector()
    tracker  = PersonTracker()
    linker   = IdentityLinker()
    attend   = AttendanceTracker()

    cam.start()

    fps_start   = time.time()
    fps_count   = 0
    fps_display = 0.0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        fps_count += 1

        # Full pipeline
        persons  = detector.detect(frame)
        tracked  = tracker.update(persons)
        identity = linker.update(frame, tracked)

        # Get current names and update attendance
        current_names = [p["name"] for p in identity]
        newly_marked  = attend.update(current_names)

        # Draw each person
        for p in identity:
            x1, y1, x2, y2 = p["box"]
            name  = p["name"]
            tid   = p["track_id"]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Show confirmation progress if not yet marked
            if name != "Unknown" and not attend.is_marked(name):
                curr, req = attend.get_confirmation_progress(name)
                progress  = f"Confirming... {curr}/{req}"
                cv2.putText(frame, progress,
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 200, 255), 2)
            elif attend.is_marked(name):
                cv2.putText(frame, "✓ Present",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

            # Name label
            label = f"{name}  ID:{tid}"
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        # Flash "MARKED!" when someone is newly marked
        if newly_marked:
            h, w = frame.shape[:2]
            msg = f"MARKED: {', '.join(newly_marked)}"
            cv2.putText(frame, msg,
                        (w//2 - 150, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

        # FPS + attendance summary HUD
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count   = 0
            fps_start   = time.time()

        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        present = sorted(attend.marked)
        hud = f"FPS:{fps_display:.0f}  |  Present ({len(present)}): " + \
              (', '.join(present) if present else 'none yet')
        cv2.putText(frame, hud,
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(frame, "Press Q to quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 1)

        cv2.imshow("SmartGym - Attendance Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if cv2.getWindowProperty("SmartGym - Attendance Test",
                                  cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.stop()
    cv2.destroyAllWindows()

    print(f"\n{attend.get_summary()}")
    print(f"\n✅ Attendance test complete.")
    print(f"   CSV saved to: {attend.csv_path}")