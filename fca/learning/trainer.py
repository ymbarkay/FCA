"""
fca/learning/trainer.py — OPTIONAL background gradient step thread.

The background trainer now only updates while the runtime mode is TEACH.

Reason:
    For clean experiments, adaptation should only happen during explicit human
    teaching. PAUSED mode should not silently keep training, and AUTOPILOT
    should not change unless the user has entered TEACH and added samples.

Event-triggered training inside driving_loop can still happen immediately after
each W/X commit. This background trainer is only extra replay training during
TEACH mode.
"""
import time

from fca.core.state import MODE_TEACH


# Defaults — tune these
UPDATE_PERIOD_S = 1.5
BATCH_SIZE = 16
MIN_BUFFER_FOR_TRAINING = 8
CHECKPOINT_EVERY_N_UPDATES = 25


def run_trainer(
    state,
    controller,
    replay_buffer,
    update_period_s=UPDATE_PERIOD_S,
    batch_size=BATCH_SIZE,
    checkpoint_every_n=CHECKPOINT_EVERY_N_UPDATES,
):
    """Main loop for the trainer thread."""
    if controller.adapter is None:
        print("[trainer] adapter=none, trainer thread exiting")
        return

    print(
        f"[trainer] started — period={update_period_s}s batch={batch_size} "
        f"min_buffer={MIN_BUFFER_FOR_TRAINING} mode_only={MODE_TEACH}"
    )

    updates_since_checkpoint = 0

    while not state.shutdown:
        time.sleep(update_period_s)

        # Only train during explicit TEACH mode.
        # This prevents silent replay training during PAUSED or AUTOPILOT.
        if state.get_mode() != MODE_TEACH:
            continue

        if len(replay_buffer) < MIN_BUFFER_FOR_TRAINING:
            continue

        if hasattr(replay_buffer, "correction_count"):
            if replay_buffer.correction_count() == 0:
                continue

        sample = replay_buffer.sample(batch_size)
        if sample is None:
            continue

        features, deltas, speeds = sample

        try:
            step_t0 = time.perf_counter()
            loss = controller.gradient_step(
                features,
                deltas,
                speeds,
                train_speed=True,
                historical_blend=float(
                    getattr(controller, "HISTORICAL_GRADIENT_BLEND", 0.0)
                ),
                update_historical=True,
            )
            step_ms = (time.perf_counter() - step_t0) * 1000.0
        except Exception as e:
            print(f"[trainer] gradient step failed: {e}")
            continue

        state.record_learning_steps(step_ms, 1, last_loss=loss)

        updates_since_checkpoint += 1

        if updates_since_checkpoint >= checkpoint_every_n:
            try:
                saved = controller.save_checkpoint()
                if not saved:
                    print("[trainer] checkpoint save skipped")
            except Exception as e:
                print(f"[trainer] checkpoint save failed: {e}")
            updates_since_checkpoint = 0

    # On shutdown
    try:
        saved = controller.save_checkpoint()
        if not saved:
            print("[trainer] final checkpoint save skipped")
    except Exception:
        pass

    print(f"[trainer] stopped, final updates: {state.total_updates}")