"""
fca/core/command_buffer.py — Rolling history of executed commands for rewind.

Stores (angle, speed, dt, timestamp) for the last N seconds. When the human
triggers rewind, we replay the inverse: same steering angle, reversed throttle.
"""
import threading
import time
from collections import deque


class CommandBuffer:
    """Thread-safe FIFO of recent driving commands."""

    def __init__(self, max_seconds=5.0):
        self.max_seconds = max_seconds
        self.buffer = deque()
        self.lock = threading.Lock()
        self.last_time = None

    def add(self, angle_car, speed_car):
        """Record a command that was just executed."""
        now = time.time()
        with self.lock:
            if self.last_time is None:
                dt = 0.05
            else:
                dt = now - self.last_time
            self.last_time = now

            self.buffer.append({
                "angle_car": float(angle_car),
                "speed_car": float(speed_car),
                "dt": float(dt),
                "timestamp": now,
            })

            # Drop entries older than max_seconds
            cutoff = now - self.max_seconds
            while self.buffer and self.buffer[0]["timestamp"] < cutoff:
                self.buffer.popleft()

    def get_inverse_sequence(self):
        """Returns list of inverse commands (most recent first).

        Inverse rule: keep steering angle, negate speed direction.
        A car turning left while moving forward must keep wheels pointed left
        while moving backward to retrace the same arc.
        """
        with self.lock:
            seq = []
            for cmd in reversed(self.buffer):
                seq.append({
                    "angle_car": cmd["angle_car"],     # SAME steering
                    "speed_car": -cmd["speed_car"],    # REVERSED throttle
                    "dt": cmd["dt"],
                })
            return seq

    def clear(self):
        with self.lock:
            self.buffer.clear()
            self.last_time = None

    def __len__(self):
        with self.lock:
            return len(self.buffer)
