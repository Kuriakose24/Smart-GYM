"""
attendance.py
-------------
Tracks who has been seen and marks attendance exactly ONCE per session.
Saves a CSV log in the attendance/ folder.

Output format (attendance/2025-01-15.csv):
    Name,Time
    Kevin,09:14:32
    Rahul,09:15:01
"""

import os
import csv
from datetime import datetime


class AttendanceTracker:
    def __init__(self, output_dir="attendance"):
        """
        output_dir : folder where CSV files are saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Today's CSV file path
        today = datetime.now().strftime("%Y-%m-%d")
        self.csv_path = os.path.join(output_dir, f"{today}.csv")

        # Track who's already been marked this session
        self.marked = set()

        # Load already-marked names if file exists (resume across restarts)
        self._load_existing()

        print(f"[Attendance] Log file: {self.csv_path}")
        if self.marked:
            print(f"[Attendance] Already marked today: {self.marked}")

    def _load_existing(self):
        """Load names already recorded in today's CSV."""
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.marked.add(row["Name"])

    def mark(self, name):
        """
        Mark attendance for a person.
        Does nothing if already marked.

        Returns True if newly marked, False if already marked.
        """
        if name == "Unknown" or name in self.marked:
            return False

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Write header only if file is new
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Time"])
            writer.writerow([name, timestamp])

        self.marked.add(name)
        print(f"[Attendance] ✓ Marked: {name} at {timestamp}")
        return True

    def mark_all(self, names):
        """Mark attendance for a list of names at once."""
        for name in names:
            self.mark(name)

    def get_summary(self):
        """Return a summary string of today's attendance."""
        if not self.marked:
            return "No attendance marked yet."
        lines = ["--- Today's Attendance ---"]
        for name in sorted(self.marked):
            lines.append(f"  ✓ {name}")
        lines.append(f"  Total: {len(self.marked)} person(s)")
        return "\n".join(lines)