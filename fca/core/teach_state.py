"""
fca/core/teach_state.py — Tracks the human's selected steering angle during teach mode.

Selection commands (A/D/S) adjust selected_angle without committing.
Action commands (W/X) capture the current frame and commit selected_angle as
a training label.
"""
import threading


class TeachController:
    """The human's currently-selected angle during teaching."""

    ANGLE_STEP_DEG = 5.0
    MIN_ANGLE_CAR = 50.0
    MAX_ANGLE_CAR = 130.0
    CENTRE_ANGLE_CAR = 90.0

    def __init__(self):
        self.selected_angle_car = self.CENTRE_ANGLE_CAR
        self.lock = threading.Lock()

    def steer_left(self):
        with self.lock:
            self.selected_angle_car = max(
                self.MIN_ANGLE_CAR,
                self.selected_angle_car - self.ANGLE_STEP_DEG,
            )
            return self.selected_angle_car

    def steer_right(self):
        with self.lock:
            self.selected_angle_car = min(
                self.MAX_ANGLE_CAR,
                self.selected_angle_car + self.ANGLE_STEP_DEG,
            )
            return self.selected_angle_car

    def centre(self):
        with self.lock:
            self.selected_angle_car = self.CENTRE_ANGLE_CAR
            return self.selected_angle_car

    def get(self):
        with self.lock:
            return self.selected_angle_car

    def set(self, angle_car):
        """Set directly (e.g., from a UI slider in future)."""
        with self.lock:
            self.selected_angle_car = max(
                self.MIN_ANGLE_CAR,
                min(self.MAX_ANGLE_CAR, float(angle_car)),
            )
            return self.selected_angle_car
