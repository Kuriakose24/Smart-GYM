"""
database/db_handler.py — SQLite Implementation
------------------------------------------------
Handles:
    - Attendance logging (once per person per day)
    - Workout session management
    - Per-rep logging with angles

Database: data/smartgym.db (SQLite — no server needed)

Tables:
    attendance   — who showed up, when
    sessions     — one row per person/exercise session
    reps         — one row per completed rep
"""

import sqlite3
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DBHandler:

    def __init__(self, db_path=config.DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        print(f"[DB] Connected: {db_path}")

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS attendance (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT    NOT NULL,
                date      TEXT    NOT NULL,
                time      TEXT    NOT NULL,
                UNIQUE(name, date)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                exercise   TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time   TEXT,
                rep_count  INTEGER DEFAULT 0,
                best_depth REAL    DEFAULT 0,
                avg_depth  REAL    DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS reps (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL,
                name        TEXT    NOT NULL,
                exercise    TEXT    NOT NULL,
                rep_number  INTEGER NOT NULL,
                elbow_angle REAL,
                knee_angle  REAL,
                hip_angle   REAL,
                body_angle  REAL,
                form_score  REAL,
                timestamp   TEXT    NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
        """)
        self.conn.commit()

    # ── Attendance ────────────────────────────────────────────────────────────

    def mark_attendance(self, name):
        """
        Mark attendance for today. Returns True if newly marked, False if already marked.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            self.conn.execute(
                "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
                (name, today, datetime.now().strftime("%H:%M:%S"))
            )
            self.conn.commit()
            print(f"[DB] Attendance marked: {name} on {today}")
            return True
        except sqlite3.IntegrityError:
            # UNIQUE constraint → already marked today
            return False

    def get_attendance(self, date=None):
        """Get attendance records. Default: today."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        rows = self.conn.execute(
            "SELECT name, time FROM attendance WHERE date=? ORDER BY time",
            (date,)
        ).fetchall()
        return [{"name": r[0], "time": r[1]} for r in rows]

    # ── Sessions ──────────────────────────────────────────────────────────────

    def start_session(self, name, exercise):
        """Create a new workout session. Returns session_id."""
        cur = self.conn.execute(
            "INSERT INTO sessions (name, exercise, start_time, rep_count) VALUES (?,?,?,0)",
            (name, exercise, datetime.now().isoformat())
        )
        self.conn.commit()
        sid = cur.lastrowid
        print(f"[DB] Session started: {name} — {exercise} (id={sid})")
        return sid

    def update_session_reps(self, session_id, rep_count, best_depth, avg_depth):
        """Update running stats on an open session."""
        self.conn.execute(
            "UPDATE sessions SET rep_count=?, best_depth=?, avg_depth=? WHERE id=?",
            (rep_count, best_depth, avg_depth, session_id)
        )
        self.conn.commit()

    def end_session(self, session_id, rep_count=0, best_depth=0, avg_depth=0):
        """Close a session with final stats."""
        self.conn.execute(
            "UPDATE sessions SET end_time=?, rep_count=?, best_depth=?, avg_depth=? WHERE id=?",
            (datetime.now().isoformat(), rep_count, best_depth, avg_depth, session_id)
        )
        self.conn.commit()
        print(f"[DB] Session ended: id={session_id}  reps={rep_count}  best={best_depth:.0f}°")

    # ── Reps ──────────────────────────────────────────────────────────────────

    def log_rep(self, session_id, name, exercise, rep_number, angles, form_score=None):
        """Log one completed rep with its angles."""
        self.conn.execute(
            """INSERT INTO reps
               (session_id, name, exercise, rep_number,
                elbow_angle, knee_angle, hip_angle, body_angle,
                form_score, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id, name, exercise, rep_number,
                angles.get("elbow"),
                angles.get("knee"),
                angles.get("hip"),
                angles.get("body"),
                form_score,
                datetime.now().isoformat(),
            )
        )
        self.conn.commit()

    # ── Analytics queries ─────────────────────────────────────────────────────

    def get_workout_history(self, name, limit=10):
        """Get recent sessions for a person."""
        rows = self.conn.execute(
            """SELECT exercise, start_time, rep_count, best_depth, avg_depth
               FROM sessions
               WHERE name=?
               ORDER BY start_time DESC
               LIMIT ?""",
            (name, limit)
        ).fetchall()
        return [
            {
                "exercise":   r[0],
                "start_time": r[1],
                "rep_count":  r[2],
                "best_depth": r[3],
                "avg_depth":  r[4],
            }
            for r in rows
        ]

    def get_rep_detail(self, session_id):
        """Get all reps for a session."""
        rows = self.conn.execute(
            "SELECT rep_number, elbow_angle, knee_angle, form_score, timestamp FROM reps WHERE session_id=? ORDER BY rep_number",
            (session_id,)
        ).fetchall()
        return [
            {
                "rep":         r[0],
                "elbow_angle": r[1],
                "knee_angle":  r[2],
                "form_score":  r[3],
                "timestamp":   r[4],
            }
            for r in rows
        ]

    def get_stats_summary(self, name):
        """Quick stats for a person across all time."""
        row = self.conn.execute(
            """SELECT
                COUNT(DISTINCT date(start_time)) as days,
                SUM(rep_count) as total_reps,
                MIN(best_depth) as best_depth
               FROM sessions WHERE name=?""",
            (name,)
        ).fetchone()
        if row:
            return {"days": row[0], "total_reps": row[1], "best_depth": row[2]}
        return {"days": 0, "total_reps": 0, "best_depth": None}


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing DBHandler...")
    db = DBHandler(db_path="data/test_smartgym.db")

    # Attendance
    print(db.mark_attendance("Kevin"))   # True
    print(db.mark_attendance("Kevin"))   # False (already marked)
    print(db.get_attendance())

    # Session + reps
    sid = db.start_session("Kevin", "pushup")
    db.log_rep(sid, "Kevin", "pushup", 1, {"elbow": 85, "knee": 170, "hip": 175, "body": 80}, form_score=92)
    db.log_rep(sid, "Kevin", "pushup", 2, {"elbow": 90, "knee": 168, "hip": 172, "body": 78}, form_score=88)
    db.update_session_reps(sid, rep_count=2, best_depth=85, avg_depth=87.5)
    db.end_session(sid, rep_count=2, best_depth=85, avg_depth=87.5)

    print("\nWorkout history:")
    for row in db.get_workout_history("Kevin"):
        print(" ", row)

    print("\nRep detail:")
    for row in db.get_rep_detail(sid):
        print(" ", row)

    print("\nStats summary:")
    print(db.get_stats_summary("Kevin"))

    os.remove("data/test_smartgym.db")
    print("\n✅ DBHandler test passed.")