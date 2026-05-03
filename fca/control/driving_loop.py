"""
fca/control/driving_loop.py — main control thread.

Implements the runtime mode state machine:
    AUTOPILOT → REWIND_TO_TEACH → TEACH → AUTOPILOT
            \\
             → TEACH directly
             → PAUSED

Reads frames, runs inference, sends to motors based on current mode.
Logs every frame to CSV.
"""
import threading
import time
import random
from collections import deque

import cv2
import torch

from fca.core.state import (
    MODE_AUTOPILOT,
    MODE_TEACH,
    MODE_REVERSE_MANUAL,
    MODE_PAUSED,
    MODE_DATASET_COLLECTION,
)


# ─── Tunables ─────────────────────────────────────────────────────────────
JPEG_WIDTH = 480
JPEG_HEIGHT = 270
JPEG_QUALITY = 70
PREVIEW_EVERY_N_FRAMES = 2
PREVIEW_EVERY_N_FRAMES_AUTOPILOT = 3
LOG_EVERY_N_FRAMES_AUTOPILOT = 3
ENABLE_AUTOPILOT_FRAME_LOGGING = False

TEACH_STEP_DURATION_S = 0.4
TEACH_LONG_STEP_DURATION_S = 0.7
TEACH_BACKWARD_DURATION_S = 0.4
TEACH_BACKWARD_SPEED = 20
REVERSE_MANUAL_SPEED = 34
REVERSE_MANUAL_KICK_SPEED = 35
REVERSE_MANUAL_KICK_DURATION_S = 0.20
TEACH_FORWARD_SPEED = 35
DATASET_COLLECTION_FORWARD_SPEED = 26
ENABLE_COMMAND_BUFFER = False
ENABLE_AUTOPILOT_ANCHORS = False
STORE_TEACH_CORRECTIONS_IN_REPLAY = False
ENABLE_ANTI_FORGETTING_REHEARSAL = True
REHEARSAL_RECENT_CAPACITY = 256
REHEARSAL_PROTECTED_CAPACITY = 512
REHEARSAL_PROTECTED_FRACTION = 0.60
REHEARSAL_BATCH_SIZE = 16
REHEARSAL_STEPS_PER_COMMIT = 3
REHEARSAL_LR_MULTIPLIER = 0.60


class DrivingLoop:
    """The real-time control thread."""

    # Number of gradient steps per teaching event
    UPDATE_STEPS_PER_COMMIT = 1
    BOOST_STEPS_PER_COMMIT = 5
    BOOST_LR_MULTIPLIER = 2.8

    def __init__(
        self,
        state,
        controller,
        motors,
        command_buffer,
        teach_controller,
        replay_buffer,
        session_logger,
        capture_src=0,
    ):
        self.state = state
        self.controller = controller
        self.motors = motors
        self.command_buffer = command_buffer
        self.teach_controller = teach_controller
        self.replay_buffer = replay_buffer
        self.session_logger = session_logger
        self.capture_src = capture_src

        # Pending teach commands queue — set by GUI/keyboard handlers
        self.pending_teach_command = None
        self._teach_lock = threading.Lock()

        # Dataset mode runtime controls
        self._dataset_motion_enabled = True
        self._dataset_capture_stop_pending = False
        self._dataset_capture_global_pending = False
        self._last_mode = None

        # TEACH/manual rewind hold-to-reverse control
        self._teach_backward_hold = False
        self._reverse_hold_started_at = None

        # FPS tracking
        self._fps_window = deque(maxlen=30)
        self._preview_counter = 0
        self._log_counter = 0

        # Hybrid rehearsal memory:
        # - recent: keeps latest corrections for fast adaptation
        # - protected: long-term representative memory via reservoir sampling
        self._rehearsal_recent = deque(maxlen=REHEARSAL_RECENT_CAPACITY)
        self._rehearsal_protected = []
        self._rehearsal_seen = 0

    # ─── Public API for GUI to inject commands ────────────────────────────
    def request_teach_command(self, command, **kwargs):
        """Called by the GUI when the human presses a teach action key."""
        with self._teach_lock:
            self.pending_teach_command = (command, kwargs)

    def clear_pending_teach_command(self):
        with self._teach_lock:
            self.pending_teach_command = None

    def request_dataset_capture_stop_frame(self):
        """Request one stop-labeled dataset frame capture, regardless of mode."""
        with self._teach_lock:
            self._dataset_capture_global_pending = True

    def _consume_dataset_capture_stop_frame(self):
        with self._teach_lock:
            pending = self._dataset_capture_global_pending
            self._dataset_capture_global_pending = False
            return pending

    def _consume_teach_command(self):
        with self._teach_lock:
            cmd = self.pending_teach_command
            self.pending_teach_command = None
            return cmd

    # ─── Main loop ────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.capture_src)

        if not cap.isOpened():
            print(f"[driving] ERROR — could not open camera {self.capture_src}")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        print("[driving] loop started")

        fps_report_time = time.time()

        while not self.state.shutdown:
            loop_t0 = time.time()
            ret, frame = cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()

            # Determine current mode early so mode-specific throttling can be applied.
            mode = self.state.get_mode()

            # FPS tracking
            self._fps_window.append(now)
            if len(self._fps_window) >= 2:
                window_dt = self._fps_window[-1] - self._fps_window[0]
                fps = (len(self._fps_window) - 1) / window_dt if window_dt > 0 else 0.0
            else:
                fps = 0.0

            # Encode browser preview less often to reduce loop overhead.
            self._preview_counter += 1
            preview_stride = (
                PREVIEW_EVERY_N_FRAMES_AUTOPILOT
                if mode == MODE_AUTOPILOT
                else PREVIEW_EVERY_N_FRAMES
            )
            if self._preview_counter >= preview_stride:
                self._preview_counter = 0

                preview = cv2.resize(frame, (JPEG_WIDTH, JPEG_HEIGHT))
                ok, jpeg = cv2.imencode(
                    ".jpg",
                    preview,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )

                if ok:
                    with self.state.lock:
                        self.state.latest_frame_jpeg = jpeg.tobytes()
                        self.state.latest_frame_raw = frame

            if mode != self._last_mode:
                if mode != MODE_REVERSE_MANUAL:
                    self._teach_backward_hold = False
                    self._reverse_hold_started_at = None

                if mode == MODE_DATASET_COLLECTION:
                    self._dataset_motion_enabled = True
                    self._dataset_capture_stop_pending = False

                if mode == MODE_REVERSE_MANUAL:
                    self._teach_backward_hold = False
                    self._reverse_hold_started_at = None

                self._last_mode = mode

            # Run inference every frame for telemetry display
            try:
                pred = self.controller.predict(frame, mode)
            except Exception as e:
                print(f"[driving] inference error: {e}")
                time.sleep(0.05)
                continue

            # Update telemetry
            with self.state.lock:
                self.state.base_angle_norm = pred["base_angle_norm"]
                self.state.base_speed_prob = pred["base_speed_prob"]
                self.state.delta_angle_norm = pred["delta_angle_norm"]
                self.state.delta_speed_logit = pred["delta_speed_logit"]
                self.state.final_angle_car = pred["final_angle_car"]
                self.state.final_speed_car = pred["final_speed_car"]
                self.state.fps = fps
                self.state.feature_ms = pred.get("feature_ms", 0.0)
                self.state.inference_ms = pred["inference_ms"]
                self.state.adapter_ms = pred["adapter_ms"]
                self.state.frames_processed += 1
                self.state.replay_buffer_size = len(self.replay_buffer)
                if ENABLE_COMMAND_BUFFER:
                    self.state.command_buffer_size = len(self.command_buffer)
                else:
                    self.state.command_buffer_size = 0
                self.state.selected_angle_car = self.teach_controller.get()

            dataset_capture_frame = False
            dataset_speed_norm_label = ""
            dataset_angle_car_label = ""

            # Global one-shot stop frame capture works in any mode.
            if self._consume_dataset_capture_stop_frame():
                dataset_capture_frame = True
                dataset_speed_norm_label = "0"
                dataset_angle_car_label = f"{self.teach_controller.get():.2f}"

            # Mode-dependent behaviour
            if mode == MODE_AUTOPILOT:
                self._handle_autopilot(pred, frame)

            elif mode == MODE_TEACH:
                self._handle_teach(pred, frame)

            elif mode == MODE_REVERSE_MANUAL:
                self._handle_reverse_manual()

            elif mode == MODE_PAUSED:
                # Stop motors and centre steering in PAUSED mode.
                self.motors.stop(center=True)

            elif mode == MODE_DATASET_COLLECTION:
                (
                    dataset_capture_frame,
                    dataset_speed_norm_label,
                    dataset_angle_car_label,
                ) = self._handle_dataset_collection()

            # Log to CSV (throttled in AUTOPILOT for lower latency).
            self._log_counter += 1
            should_log = True
            if mode == MODE_AUTOPILOT:
                if ENABLE_AUTOPILOT_FRAME_LOGGING:
                    should_log = (self._log_counter % LOG_EVERY_N_FRAMES_AUTOPILOT) == 0
                else:
                    should_log = False

            if should_log:
                try:
                    snapshot = self.state.telemetry_snapshot()
                    self.session_logger.log_frame(
                        snapshot,
                        frame_bgr=frame,
                        dataset_capture_frame=dataset_capture_frame,
                        dataset_speed_norm_label=dataset_speed_norm_label,
                        dataset_angle_car_label=dataset_angle_car_label,
                    )
                except Exception as e:
                    print(f"[driving] log error: {e}")

            loop_ms = (time.time() - loop_t0) * 1000.0
            model_ms = (
                float(pred.get("feature_ms", 0.0))
                + float(pred.get("adapter_ms", 0.0))
                + float(pred.get("inference_ms", 0.0))
            )
            other_ms = max(0.0, loop_ms - model_ms)

            with self.state.lock:
                self.state.loop_ms = loop_ms
                self.state.other_ms = other_ms

            # FPS report every 5s
            if now - fps_report_time > 5.0:
                print(
                    f"[driving] {fps:.1f} FPS  mode={mode}  "
                    f"replay={len(self.replay_buffer)}  "
                    f"updates={self.state.total_updates}  "
                    f"loss={self.state.last_teach_loss:.4f}"
                )
                fps_report_time = now

        cap.release()
        self.motors.stop(center=True)
        print("[driving] loop stopped")

    # ─── Mode handlers ────────────────────────────────────────────────────
    def _handle_autopilot(self, pred, frame):
        """Drive using base + adapter output. Record commands for rewind."""
        angle = pred["final_angle_car"]
        speed = pred["final_speed_car"]

        self.motors.drive(angle, speed)
        if ENABLE_COMMAND_BUFFER:
            self.command_buffer.add(angle, speed)
        if ENABLE_AUTOPILOT_ANCHORS:
            self._maybe_add_anchor_sample(pred, frame)

    def _maybe_add_anchor_sample(self, pred, frame):
        """Add zero-delta anchor samples during stable autopilot."""
        if self.controller.adapter is None:
            return

        if self.controller.adapter_type != "deep":
            return

        now = time.time()
        if not hasattr(self, "_last_anchor_time"):
            self._last_anchor_time = 0.0

        # Add anchor at most every 0.5s.
        if now - self._last_anchor_time < 0.5:
            return

        self._last_anchor_time = now

        try:
            target = self.controller.teach_step(
                base_angle_norm=pred["base_angle_norm"],
                base_speed_prob=pred["base_speed_prob"],
                human_angle_car=self.controller.angle_norm_to_car(
                    pred["base_angle_norm"]
                ),
                human_speed_norm=1.0 if pred["base_speed_prob"] >= 0.5 else 0.0,
                image=frame,
                add_to_gate=False,
            )

            if target is not None:
                anchor_angle = target["target_delta_angle"]
                if self.controller.adapter_type != "deep":
                    anchor_angle = 0.0

                self.replay_buffer.add(
                    target["input_features"],
                    anchor_angle,
                    target["target_speed_norm"],
                    sample_kind="anchor",
                )

        except Exception as e:
            print(f"[driving] anchor sample failed: {e}")

    def _handle_teach(self, pred, frame):
        """
        Process pending teach action commands.

        In TEACH mode:
          - A/D/S only adjust selected steering and physically turn the servo.
          - No movement happens unless W / Shift+W / B is pressed.
          - If no command is pending, keep the car stopped but hold selected steering.
        """
        selected_angle_car = self.teach_controller.get()
        teach_cmd = self._consume_teach_command()

        if teach_cmd is None:
            if self._teach_backward_hold:
                self.motors.drive(selected_angle_car, -TEACH_BACKWARD_SPEED)
                with self.state.lock:
                    self.state.human_active = True
                return

            # Stay stopped, but do NOT centre the wheels.
            # This lets A/D/S visibly change steering angle.
            self.motors.stop(center=False)
            self.motors.steer(selected_angle_car)
            with self.state.lock:
                self.state.human_active = False
            return

        command, kwargs = teach_cmd

        if command == "steer_update":
            # Selection changed. Show steering physically. No learning, no movement.
            self.motors.stop(center=False)
            self.motors.steer(selected_angle_car)

        elif command == "forward_step":
            self._teach_forward_step(pred, frame, long=False)

        elif command == "long_forward_step":
            self._teach_forward_step(pred, frame, long=True)

        elif command == "stop_teach":
            self._teach_stop_label(pred, frame)

        elif command == "backward":
            self._teach_backward()

        elif command == "backward_hold_start":
            self._teach_backward_hold = True

        elif command == "backward_hold_stop":
            self._teach_backward_hold = False
            self.motors.stop(center=False)
            self.motors.steer(selected_angle_car)
            with self.state.lock:
                self.state.human_active = False

        else:
            print(f"[driving] unknown teach command: {command}")

    def _handle_dataset_collection(self):
        """Dataset mode with move/stop/capture controls."""
        selected_angle_car = self.teach_controller.get()
        cmd = self._consume_teach_command()

        if cmd is not None:
            command, _kwargs = cmd

            if command == "dataset_stop":
                self._dataset_motion_enabled = False
                # Capture one stop frame (speed=0) immediately after stopping.
                self._dataset_capture_stop_pending = True

            elif command == "dataset_resume":
                self._dataset_motion_enabled = True
                self._dataset_capture_stop_pending = False

            elif command == "dataset_capture_stop_frame":
                if not self._dataset_motion_enabled:
                    self._dataset_capture_stop_pending = True

        dataset_capture_frame = False
        dataset_speed_norm_label = ""
        dataset_angle_car_label = f"{selected_angle_car:.2f}"

        if self._dataset_motion_enabled:
            self.motors.drive(selected_angle_car, DATASET_COLLECTION_FORWARD_SPEED)

            dataset_capture_frame = True
            dataset_speed_norm_label = "1"

            with self.state.lock:
                self.state.human_active = True
                self.state.final_angle_car = float(selected_angle_car)
                self.state.final_speed_car = float(DATASET_COLLECTION_FORWARD_SPEED)

            return dataset_capture_frame, dataset_speed_norm_label, dataset_angle_car_label

        # Stopped dataset mode: only capture when requested.
        self.motors.stop(center=False)
        self.motors.steer(selected_angle_car)

        if self._dataset_capture_stop_pending:
            dataset_capture_frame = True
            dataset_speed_norm_label = "0"
            self._dataset_capture_stop_pending = False

        with self.state.lock:
            self.state.human_active = True
            self.state.final_angle_car = float(selected_angle_car)
            self.state.final_speed_car = 0.0

        return dataset_capture_frame, dataset_speed_norm_label, dataset_angle_car_label

    def _handle_reverse_manual(self):
        """Independent reverse mode: human steers and manually backs up."""
        selected_angle_car = self.teach_controller.get()
        cmd = self._consume_teach_command()

        if cmd is not None:
            command, _kwargs = cmd

            if command == "steer_update":
                pass
            elif command == "backward_hold_start":
                self._teach_backward_hold = True
                self._reverse_hold_started_at = time.time()
            elif command == "backward_hold_stop":
                self._teach_backward_hold = False
                self._reverse_hold_started_at = None
            else:
                print(f"[driving] reverse manual ignored command: {command}")

        if self._teach_backward_hold:
            now = time.time()
            speed = REVERSE_MANUAL_SPEED

            if self._reverse_hold_started_at is not None:
                if now - self._reverse_hold_started_at < REVERSE_MANUAL_KICK_DURATION_S:
                    speed = REVERSE_MANUAL_KICK_SPEED

            self.motors.drive(selected_angle_car, -speed)
            with self.state.lock:
                self.state.human_active = True
                self.state.final_angle_car = float(selected_angle_car)
                self.state.final_speed_car = float(-speed)
            return

        self.motors.stop(center=False)
        self.motors.steer(selected_angle_car)
        with self.state.lock:
            self.state.human_active = False
            self.state.final_angle_car = float(selected_angle_car)
            self.state.final_speed_car = 0.0

    # ─── Teach action implementations ─────────────────────────────────────
    def _do_gradient_steps(self, n_steps=None):
        """
        Do N gradient steps on freshly sampled batches from the buffer.

        Called inline from teach handlers so the human sees an immediate
        learning signal.
        """
        if n_steps is None:
            n_steps = self.UPDATE_STEPS_PER_COMMIT

        if self.controller.adapter is None:
            return 0.0

        last_loss = 0.0
        steps_done = 0

        for _ in range(n_steps):
            sample = self.replay_buffer.sample(batch_size=16)

            if sample is None:
                break

            features, deltas, speeds = sample

            try:
                last_loss = self.controller.gradient_step(
                    features,
                    deltas,
                    speeds,
                    train_speed=True,
                )
                steps_done += 1

            except Exception as e:
                print(f"[driving] gradient step failed: {e}")
                break

        with self.state.lock:
            self.state.last_teach_loss = last_loss
            self.state.total_updates += steps_done

        return last_loss

    def _boost_latest_target(self, target, n_steps=None, lr_multiplier=None):
        """Run focused updates on the newest human-labelled target for faster local adaptation."""
        if target is None or self.controller.adapter is None:
            return 0.0

        if n_steps is None:
            n_steps = self.BOOST_STEPS_PER_COMMIT

        if lr_multiplier is None:
            lr_multiplier = self.BOOST_LR_MULTIPLIER

        features = target.get("input_features")
        if features is None:
            return 0.0

        if isinstance(features, torch.Tensor):
            batch_features = features.detach().cpu().clone()
        else:
            batch_features = torch.tensor(features, dtype=torch.float32)

        if batch_features.ndim == 1:
            batch_features = batch_features.unsqueeze(0)

        target_delta = float(target.get("target_delta_angle", 0.0))
        target_speed = float(target.get("target_speed_norm", 0.0))

        batch_target_deltas = torch.tensor([[target_delta]], dtype=torch.float32)
        batch_target_speeds = torch.tensor([[target_speed]], dtype=torch.float32)

        last_loss = 0.0
        steps_done = 0

        optimizer = getattr(self.controller, "optimizer", None)
        old_lrs = None

        if optimizer is not None:
            old_lrs = [group["lr"] for group in optimizer.param_groups]
            for group in optimizer.param_groups:
                group["lr"] = min(group["lr"] * lr_multiplier, 0.05)

        try:
            for _ in range(n_steps):
                try:
                    last_loss = self.controller.gradient_step(
                        batch_features,
                        batch_target_deltas,
                        batch_target_speeds,
                        train_speed=False,
                        delta_penalty_weight=0.0,
                        clip_grad_norm=5.0,
                    )
                    steps_done += 1
                except Exception as e:
                    print(f"[driving] focused teach update failed: {e}")
                    break
        finally:
            if optimizer is not None and old_lrs is not None:
                for group, old_lr in zip(optimizer.param_groups, old_lrs):
                    group["lr"] = old_lr

        with self.state.lock:
            self.state.last_teach_loss = last_loss
            self.state.total_updates += steps_done

        return last_loss

    def _remember_target_for_rehearsal(self, target):
        """Store a compact CPU copy of a correction target for later rehearsal."""
        if target is None:
            return

        features = target.get("input_features")
        if features is None:
            return

        if isinstance(features, torch.Tensor):
            f = features.detach().cpu().clone()
        else:
            f = torch.tensor(features, dtype=torch.float32)

        if f.ndim > 1:
            f = f.squeeze(0)

        item = (
            f,
            float(target.get("target_delta_angle", 0.0)),
            float(target.get("target_speed_norm", 0.0)),
        )

        # Always keep recent corrections for local plasticity.
        self._rehearsal_recent.append(item)

        # Reservoir memory protects old cases from being forgotten over time.
        self._rehearsal_seen += 1

        if len(self._rehearsal_protected) < REHEARSAL_PROTECTED_CAPACITY:
            self._rehearsal_protected.append(item)
        else:
            j = random.randint(0, self._rehearsal_seen - 1)
            if j < REHEARSAL_PROTECTED_CAPACITY:
                self._rehearsal_protected[j] = item

    def _rehearsal_update(self, n_steps=REHEARSAL_STEPS_PER_COMMIT):
        """Rehearse a few older corrections to reduce catastrophic forgetting."""
        if not ENABLE_ANTI_FORGETTING_REHEARSAL:
            return 0.0

        if self.controller.adapter is None:
            return 0.0

        recent_n = len(self._rehearsal_recent)
        protected_n = len(self._rehearsal_protected)
        total_n = recent_n + protected_n

        if total_n < max(4, REHEARSAL_BATCH_SIZE):
            return 0.0

        optimizer = getattr(self.controller, "optimizer", None)
        old_lrs = None

        if optimizer is not None:
            old_lrs = [group["lr"] for group in optimizer.param_groups]
            for group in optimizer.param_groups:
                group["lr"] = max(1e-5, group["lr"] * REHEARSAL_LR_MULTIPLIER)

        last_loss = 0.0
        steps_done = 0

        try:
            for _ in range(max(1, n_steps)):
                batch_n = min(REHEARSAL_BATCH_SIZE, total_n)

                target_protected = int(round(batch_n * REHEARSAL_PROTECTED_FRACTION))
                n_protected = min(target_protected, protected_n)
                n_recent = min(batch_n - n_protected, recent_n)

                # Fill any deficit from whichever pool still has available samples.
                remaining = batch_n - (n_protected + n_recent)
                if remaining > 0:
                    extra_recent = min(remaining, max(0, recent_n - n_recent))
                    n_recent += extra_recent
                    remaining -= extra_recent
                if remaining > 0:
                    extra_protected = min(remaining, max(0, protected_n - n_protected))
                    n_protected += extra_protected

                batch = []
                if n_recent > 0:
                    batch.extend(random.sample(list(self._rehearsal_recent), n_recent))
                if n_protected > 0:
                    batch.extend(random.sample(self._rehearsal_protected, n_protected))

                features = torch.stack([b[0] for b in batch], dim=0)
                deltas = torch.tensor([b[1] for b in batch], dtype=torch.float32).unsqueeze(-1)
                speeds = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(-1)

                try:
                    last_loss = self.controller.gradient_step(
                        features,
                        deltas,
                        speeds,
                        train_speed=True,
                        delta_penalty_weight=0.01,
                        clip_grad_norm=1.0,
                    )
                    steps_done += 1
                except Exception as e:
                    print(f"[driving] rehearsal step failed: {e}")
                    break
        finally:
            if optimizer is not None and old_lrs is not None:
                for group, old_lr in zip(optimizer.param_groups, old_lrs):
                    group["lr"] = old_lr

        if steps_done > 0:
            with self.state.lock:
                self.state.last_teach_loss = last_loss
                self.state.total_updates += steps_done

        return last_loss

    def _compute_teach_boost(self, predicted_angle_car, selected_angle_car, predicted_speed_prob, target_speed_norm):
        """Scale focused teach intensity by current error so hard misses adapt faster."""
        angle_err = abs(float(predicted_angle_car) - float(selected_angle_car))
        speed_err = abs(float(predicted_speed_prob) - float(target_speed_norm))

        # Keep adaptation responsive, but cap aggressiveness to avoid drift/forgetting.
        severity = min(2.0, (angle_err / 18.0) + (1.0 * speed_err))
        scale = 1.0 + severity

        steps = int(round(self.BOOST_STEPS_PER_COMMIT * scale))
        steps = max(self.BOOST_STEPS_PER_COMMIT, min(10, steps))

        lr_mult = min(4.5, self.BOOST_LR_MULTIPLIER * scale)
        return steps, lr_mult

    def _teach_forward_step(self, pred, frame, long=False):
        """Commit: capture frame, log target, gradient step, execute, stop."""
        selected_angle_car = self.teach_controller.get()

        # 1. Compute target and add to replay buffer
        target = self.controller.teach_step(
            base_angle_norm=pred["base_angle_norm"],
            base_speed_prob=pred["base_speed_prob"],
            human_angle_car=selected_angle_car,
            human_speed_norm=1.0,
            image=frame,
        )

        if target is not None and STORE_TEACH_CORRECTIONS_IN_REPLAY:
            self.replay_buffer.add(
                target["input_features"],
                target["target_delta_angle"],
                target["target_speed_norm"],
            )

        # Strengthen immediate effect of this exact teaching event.
        boost_steps, boost_lr_mult = self._compute_teach_boost(
            predicted_angle_car=pred["final_angle_car"],
            selected_angle_car=selected_angle_car,
            predicted_speed_prob=pred["base_speed_prob"],
            target_speed_norm=1.0,
        )
        loss = self._boost_latest_target(target, n_steps=boost_steps, lr_multiplier=boost_lr_mult)

        # Rehearse older teach corrections to preserve previously learned locations.
        rehearsal_loss = self._rehearsal_update()
        if rehearsal_loss > 0:
            loss = rehearsal_loss

        # Add newest correction after rehearsal so sampling focuses on older cases.
        self._remember_target_for_rehearsal(target)

        # 2. Optional replay-batch update (kept off by default to avoid drift).
        if STORE_TEACH_CORRECTIONS_IN_REPLAY:
            loss = self._do_gradient_steps()

        # 3. Log correction
        try:
            self.session_logger.log_correction(
                frame_bgr=frame,
                command="forward_long" if long else "forward",
                base_angle_norm=pred["base_angle_norm"],
                base_speed_prob=pred["base_speed_prob"],
                human_angle_car=selected_angle_car,
                human_speed_norm=1.0,
                target_delta_angle=target["target_delta_angle"] if target else 0.0,
                loss_after_update=loss,
                total_updates=self.state.total_updates,
                session=self.state.session_label,
            )
        except Exception as e:
            print(f"[driving] correction log failed: {e}")

        with self.state.lock:
            self.state.corrections_logged += 1
            self.state.human_active = True

        # 4. Execute selected forward step
        duration = TEACH_LONG_STEP_DURATION_S if long else TEACH_STEP_DURATION_S

        self.motors.drive(selected_angle_car, TEACH_FORWARD_SPEED)
        time.sleep(duration)

        # Stop but keep steering where the human selected it
        self.motors.stop(center=False)
        self.motors.steer(selected_angle_car)

        with self.state.lock:
            self.state.human_active = False

    def _teach_stop_label(self, pred, frame):
        """Commit: log a 'should stop here' label, gradient step, stop."""
        selected_angle_car = self.teach_controller.get()

        target = self.controller.teach_step(
            base_angle_norm=pred["base_angle_norm"],
            base_speed_prob=pred["base_speed_prob"],
            human_angle_car=selected_angle_car,
            human_speed_norm=0.0,
            image=frame,
        )

        if target is not None and STORE_TEACH_CORRECTIONS_IN_REPLAY:
            self.replay_buffer.add(
                target["input_features"],
                target["target_delta_angle"],
                target["target_speed_norm"],
            )

        # Strengthen immediate effect of this exact teaching event.
        boost_steps, boost_lr_mult = self._compute_teach_boost(
            predicted_angle_car=pred["final_angle_car"],
            selected_angle_car=selected_angle_car,
            predicted_speed_prob=pred["base_speed_prob"],
            target_speed_norm=0.0,
        )
        loss = self._boost_latest_target(target, n_steps=boost_steps, lr_multiplier=boost_lr_mult)

        rehearsal_loss = self._rehearsal_update()
        if rehearsal_loss > 0:
            loss = rehearsal_loss

        self._remember_target_for_rehearsal(target)

        if STORE_TEACH_CORRECTIONS_IN_REPLAY:
            loss = self._do_gradient_steps()

        try:
            self.session_logger.log_correction(
                frame_bgr=frame,
                command="stop",
                base_angle_norm=pred["base_angle_norm"],
                base_speed_prob=pred["base_speed_prob"],
                human_angle_car=selected_angle_car,
                human_speed_norm=0.0,
                target_delta_angle=target["target_delta_angle"] if target else 0.0,
                loss_after_update=loss,
                total_updates=self.state.total_updates,
                session=self.state.session_label,
            )
        except Exception as e:
            print(f"[driving] correction log failed: {e}")

        with self.state.lock:
            self.state.corrections_logged += 1

        # Stop but keep selected steering visible
        self.motors.stop(center=False)
        self.motors.steer(selected_angle_car)

    def _teach_backward(self):
        """Backward step for repositioning. NO LEARNING."""
        selected_angle_car = self.teach_controller.get()

        with self.state.lock:
            self.state.human_active = True

        self.motors.drive(selected_angle_car, -TEACH_BACKWARD_SPEED)
        time.sleep(TEACH_BACKWARD_DURATION_S)

        # Stop but keep selected steering visible
        self.motors.stop(center=False)
        self.motors.steer(selected_angle_car)

        with self.state.lock:
            self.state.human_active = False