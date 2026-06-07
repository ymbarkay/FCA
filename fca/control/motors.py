"""
fca/control/motors.py — PiCar-V motor abstraction.

Wraps the SunFounder picar library. Single point of contact between our
software and the hardware. Handles graceful shutdown.
"""
import atexit
import signal
import sys


class Motors:
    """PiCar-V motor wrapper.

    Pass already-initialised front_wheels/back_wheels objects from picar setup,
    or None for dry-run.
    """

    def __init__(self, front_wheels=None, back_wheels=None,
                 max_speed=35, dry_run=False):
        self.front_wheels = front_wheels
        self.back_wheels = back_wheels
        self.max_speed = max_speed
        self.dry_run = dry_run or (front_wheels is None or back_wheels is None)

        self._last_direction = None
        self._last_angle = None
        self._last_speed = 0
        self._motors_stopped = True

        if self.dry_run:
            print("[motors] DRY RUN — no actual motor commands will be sent")
        else:
            print(f"[motors] initialised, max_speed={max_speed}")

        atexit.register(self.stop)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.stop(center=True)
        sys.exit(0)

    def _clamp_angle(self, angle_car):
        if self.dry_run or self.front_wheels is None:
            return int(max(50, min(130, angle_car)))

        return max(
            self.front_wheels._min_angle,
            min(self.front_wheels._max_angle, int(angle_car))
        )

    def steer(self, angle_car):
        """Turn only the front wheels. Does not move the car."""
        angle_clamped = self._clamp_angle(angle_car)

        if angle_clamped == self._last_angle:
            return

        self._last_angle = angle_clamped

        if not self.dry_run:
            self.front_wheels.turn(angle_clamped)

    def throttle(self, speed_car):
        """Control only the rear wheels. Positive = forward, negative = backward."""
        speed_int = int(speed_car)

        if abs(speed_int) < 2:
            self._stop_motors()
            return

        if speed_int > 0:
            direction = "forward"
        else:
            direction = "backward"
            speed_int = abs(speed_int)

        speed_int = min(speed_int, self.max_speed)

        if direction == self._last_direction and speed_int == self._last_speed:
            return

        if not self.dry_run:
            if direction != self._last_direction:
                if direction == "forward":
                    self.back_wheels.forward()
                else:
                    self.back_wheels.backward()
                self._last_direction = direction

            if speed_int != self._last_speed:
                self.back_wheels.speed = speed_int

        self._last_direction = direction
        self._last_speed = speed_int
        self._motors_stopped = False

    def set_max_speed(self, max_speed):
        max_speed = int(max(0, min(100, int(max_speed))))
        self.max_speed = max_speed
        print(f"[motors] max_speed -> {self.max_speed}")

    def drive(self, angle_car, speed_car):
        """Send steering + throttle command. angle_car: 50-130°, speed_car: -max..max."""
        self.steer(angle_car)
        self.throttle(speed_car)

    def stop(self, center=True):
        """Stop rear wheels. Optionally centre steering."""
        self._stop_motors()

        if center:
            try:
                self.steer(90)
            except Exception:
                pass

    def _stop_motors(self):
        if self._motors_stopped:
            return

        if not self.dry_run:
            try:
                self.back_wheels.speed = 0
                self.back_wheels.stop()
            except Exception:
                pass

        self._last_direction = None
        self._last_speed = 0
        self._motors_stopped = True