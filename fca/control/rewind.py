"""
fca/control/rewind.py — replay inverse commands at slow/controlled reverse speed.

Called when the human triggers rewind. Plays back the command buffer's inverse
sequence. This is not mathematically exact rollback; it is a practical physical
repositioning step before TEACH mode.
"""
import time


# ─── Tunables ─────────────────────────────────────────────────────────────

# Speed mapping for reverse replay.
# Uses recorded command speed as baseline, with a floor to overcome static friction.
REWIND_MIN_SPEED = 14
REWIND_MAX_SPEED = 28

# Scale down recorded speed during reverse replay.
REWIND_SPEED_GAIN = 0.80

# Replay at longer duration than original because backward motion is weaker
# and motor/friction losses mean exact dt reversal under-rewinds.
REWIND_TIME_SCALE = 1.20

# Maximum total rewind duration.
MAX_REWIND_TIME_S = 4.5

# Minimum total time. Prevents tiny 1mm rewind when buffer has tiny dt values.
MIN_TOTAL_REWIND_TIME_S = 0.35

# Per-command duration clamp.
MIN_DT_PER_STEP_S = 0.04
MAX_DT_PER_STEP_S = 0.30

# Preserve explicit stop windows from forward pass (speed nearly zero).
STOP_SPEED_EPS = 2.0

# Stop pause before/after rewind.
STOP_BEFORE_S = 0.15
STOP_AFTER_S = 0.15


def rewind(command_buffer, motors, abort_check=None):
    """
    Replay inverse commands.

    Strategy:
        - Read recent executed commands in reverse order.
        - Keep steering angle the same.
        - Reverse throttle at fixed REWIND_SPEED.
        - Stretch the duration slightly using REWIND_TIME_SCALE.
    """
    seq = command_buffer.get_inverse_sequence()

    if not seq:
        print("[rewind] empty buffer, nothing to do")
        return

    original_total_dt = sum(float(cmd.get("dt", 0.0)) for cmd in seq)

    print(
        f"[rewind] replaying {len(seq)} inverse commands | "
        f"buffer_dt={original_total_dt:.2f}s | "
        f"scale={REWIND_TIME_SCALE}"
    )

    # Time budget is based on what was actually buffered, not a small fixed cap.
    target_total_time = original_total_dt * REWIND_TIME_SCALE
    target_total_time = max(MIN_TOTAL_REWIND_TIME_S, target_total_time)
    target_total_time = min(MAX_REWIND_TIME_S, target_total_time)

    # Stop first
    motors.stop(center=False)
    time.sleep(STOP_BEFORE_S)

    total_time = 0.0

    for cmd in seq:
        if abort_check is not None and abort_check():
            print("[rewind] aborted by external signal")
            break

        if total_time >= target_total_time:
            print("[rewind] target rewind time reached")
            break

        raw_dt = float(cmd.get("dt", 0.0))
        dt = raw_dt * REWIND_TIME_SCALE
        dt = max(MIN_DT_PER_STEP_S, min(MAX_DT_PER_STEP_S, dt))

        angle = float(cmd["angle_car"])
        recorded_speed = abs(float(cmd.get("speed_car", 0.0)))

        if recorded_speed < STOP_SPEED_EPS:
            # Forward pass was effectively stopped for this slot.
            motors.stop(center=False)
            time.sleep(dt)
            total_time += dt
            continue

        reverse_speed = max(
            REWIND_MIN_SPEED,
            min(REWIND_MAX_SPEED, recorded_speed * REWIND_SPEED_GAIN),
        )

        # Keep recorded steering angle, reverse throttle with matched magnitude.
        motors.drive(angle, -reverse_speed)

        time.sleep(dt)
        total_time += dt

    motors.stop(center=False)
    time.sleep(STOP_AFTER_S)

    print(f"[rewind] complete, total reverse time {total_time:.2f}s")