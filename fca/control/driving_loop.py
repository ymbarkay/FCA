"""
fca/control/driving_loop.py — main control thread.

Implements the runtime mode state machine:
    AUTOPILOT → REVERSE_MANUAL → TEACH → AUTOPILOT
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
import numpy as np
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
ENABLE_AUTOPILOT_ANCHORS = False
STORE_TEACH_CORRECTIONS_IN_REPLAY = False
ENABLE_ANTI_FORGETTING_REHEARSAL = True
REHEARSAL_RECENT_CAPACITY = 256
REHEARSAL_PROTECTED_CAPACITY = 768
REHEARSAL_PROTECTED_FRACTION = 0.72
REHEARSAL_ELDER_CAPACITY = 192
REHEARSAL_ELDER_FRACTION = 0.24
REHEARSAL_ELDER_UPDATE_STRIDE = 6
REHEARSAL_BATCH_SIZE = 20
REHEARSAL_STEPS_PER_COMMIT = 4
REHEARSAL_LR_MULTIPLIER = 0.60
BOOST_TARGET_REPEATS = 4
BOOST_REHEARSAL_BATCH_SIZE = 14
VALIDATED_EXEMPLAR_CAPACITY = 192
VALIDATED_EXEMPLAR_FRACTION = 0.18
LONG_HORIZON_DRIFT_DECAY_START = 7.5e-4
LONG_HORIZON_DRIFT_DECAY_END = 3.0e-3
LONG_HORIZON_MIN_BOOST_SCALE = 0.45
LONG_HORIZON_MAX_REHEARSAL_SCALE = 2.6
ENABLE_TEACH_PHOTOMETRIC_AUGMENTATION = True
TEACH_AUGMENT_PROB = 0.35
CAMERA_FOURCC = "MJPG"
CAMERA_TARGET_FPS = 30


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
        teach_controller,
        replay_buffer,
        session_logger,
        capture_src=0,
    ):
        self.state = state
        self.controller = controller
        self.motors = motors
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
        # - elder: much slower-timescale memory for very long-horizon retention
        # - validated: frozen exemplar bank captured from known-good manual saves
        self._rehearsal_recent = deque(maxlen=REHEARSAL_RECENT_CAPACITY)
        self._rehearsal_protected = []
        self._rehearsal_elder = []
        self._rehearsal_validated = []
        self._rehearsal_protected_bucket_counts = {}
        self._rehearsal_elder_bucket_counts = {}
        self._rehearsal_seen = 0
        self._rehearsal_elder_seen = 0
        self.autopilot_frame_logging_enabled = bool(ENABLE_AUTOPILOT_FRAME_LOGGING)

        with self.state.lock:
            self.state.autopilot_frame_logging = self.autopilot_frame_logging_enabled

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

    def set_autopilot_frame_logging(self, enabled):
        enabled = bool(enabled)
        self.autopilot_frame_logging_enabled = enabled
        with self.state.lock:
            self.state.autopilot_frame_logging = enabled

    def capture_validated_exemplars(self, capacity=VALIDATED_EXEMPLAR_CAPACITY):
        """Freeze a balanced exemplar bank from the current known-good rehearsal memory."""
        capacity = int(max(0, capacity))
        if capacity <= 0:
            self._rehearsal_validated = []
            return 0

        source_pool = self._rehearsal_elder + self._rehearsal_protected + list(self._rehearsal_recent)
        if not source_pool:
            self._rehearsal_validated = []
            return 0

        snapshot = self._sample_bucket_balanced(source_pool, min(capacity, len(source_pool)))
        self._rehearsal_validated = [self._clone_rehearsal_item(item) for item in snapshot]
        return len(self._rehearsal_validated)

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
        if CAMERA_FOURCC:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
            except Exception:
                pass
        if CAMERA_TARGET_FPS > 0:
            try:
                cap.set(cv2.CAP_PROP_FPS, CAMERA_TARGET_FPS)
            except Exception:
                pass

        print("[driving] loop started")

        fps_report_time = time.time()

        while not self.state.shutdown:
            loop_t0 = time.time()
            camera_t0 = loop_t0
            ret, frame = cap.read()
            camera_ms = (time.time() - camera_t0) * 1000.0

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
            with self.state.lock:
                has_video_clients = self.state.video_client_count > 0

            preview_ms = 0.0
            if has_video_clients:
                self._preview_counter += 1
                preview_stride = (
                    PREVIEW_EVERY_N_FRAMES_AUTOPILOT
                    if mode == MODE_AUTOPILOT
                    else PREVIEW_EVERY_N_FRAMES
                )
                if self._preview_counter >= preview_stride:
                    self._preview_counter = 0
                    preview_t0 = time.time()

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

                    preview_ms = (time.time() - preview_t0) * 1000.0
            else:
                self._preview_counter = 0

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
                self.state.replay_buffer_bytes = self.replay_buffer.approximate_bytes()
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
            control_t0 = time.time()
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
            control_ms = (time.time() - control_t0) * 1000.0

            # Log to CSV (throttled in AUTOPILOT for lower latency).
            self._log_counter += 1
            should_log = mode != MODE_PAUSED
            if mode == MODE_AUTOPILOT:
                if self.autopilot_frame_logging_enabled:
                    should_log = (self._log_counter % LOG_EVERY_N_FRAMES_AUTOPILOT) == 0
                else:
                    should_log = False

            logging_ms = 0.0
            if should_log:
                logging_t0 = time.time()
                try:
                    snapshot = self.state.telemetry_snapshot()
                    snapshot.update({
                        "config": pred.get("config", getattr(self.controller, "learning_paradigm", "")),
                        "active_learning_paradigm": pred.get(
                            "active_learning_paradigm",
                            getattr(self.controller, "learning_paradigm", ""),
                        ),
                        "inference_backend": pred.get("inference_backend", getattr(self.controller, "inference_backend", "")),
                        "feature_gate": pred.get("feature_gate", 1.0),
                        "gate_e0": pred.get("gate_e0", 0.0),
                        "gate_e1": pred.get("gate_e1", 0.0),
                        "gate_e2": pred.get("gate_e2", 0.0),
                        "gate_e3": pred.get("gate_e3", 0.0),
                        "gate_entropy": pred.get("gate_entropy", 0.0),
                        "top_expert": pred.get("top_expert", ""),
                        "intent_pred": pred.get("intent_pred", ""),
                        "intent_stop_prob": pred.get("intent_stop_prob", 0.0),
                        "intent_left_prob": pred.get("intent_left_prob", 0.0),
                        "intent_straight_prob": pred.get("intent_straight_prob", 0.0),
                        "intent_right_prob": pred.get("intent_right_prob", 0.0),
                        "final_angle_norm": pred.get("final_angle_norm", pred.get("base_angle_norm", 0.0)),
                        "final_speed_prob": pred.get("final_speed_prob", pred.get("base_speed_prob", 0.0)),
                    })
                    if hasattr(self.controller, "training_metrics_snapshot"):
                        snapshot.update(self.controller.training_metrics_snapshot())
                    self.session_logger.log_frame(
                        snapshot,
                        frame_bgr=frame,
                        dataset_capture_frame=dataset_capture_frame,
                        dataset_speed_norm_label=dataset_speed_norm_label,
                        dataset_angle_car_label=dataset_angle_car_label,
                    )
                except Exception as e:
                    print(f"[driving] log error: {e}")
                logging_ms = (time.time() - logging_t0) * 1000.0

            loop_ms = (time.time() - loop_t0) * 1000.0
            model_ms = (
                float(pred.get("feature_ms", 0.0))
                + float(pred.get("adapter_ms", 0.0))
                + float(pred.get("inference_ms", 0.0))
            )
            measured_non_model_ms = camera_ms + preview_ms + control_ms + logging_ms
            other_ms = max(0.0, loop_ms - model_ms - measured_non_model_ms)

            with self.state.lock:
                self.state.camera_ms = camera_ms
                self.state.preview_ms = preview_ms
                self.state.loop_ms = loop_ms
                self.state.control_ms = control_ms
                self.state.logging_ms = logging_ms
                self.state.other_ms = other_ms

            # FPS report every 5s
            if now - fps_report_time > 5.0:
                print(
                    f"[driving] {fps:.1f} FPS  mode={mode}  "
                    f"cam={camera_ms:.1f}ms  prev={preview_ms:.1f}ms  "
                    f"ctrl={control_ms:.1f}ms  log={logging_ms:.1f}ms  "
                    f"other={other_ms:.1f}ms  "
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

    def _teach_assignment(self, human_angle_car, human_speed_norm):
        target_intent = ""
        assigned_expert = ""
        try:
            if hasattr(self.controller, "target_intent_from_controls"):
                target_intent = self.controller.target_intent_from_controls(
                    human_angle_car,
                    human_speed_norm,
                )
            if target_intent and hasattr(self.controller, "selected_expert_for_intent"):
                assigned_expert = self.controller.selected_expert_for_intent(target_intent)
        except Exception:
            target_intent = ""
            assigned_expert = ""
        return target_intent, assigned_expert

    def _correction_log_fields(
        self,
        pred,
        human_angle_car,
        human_speed_norm,
        target_intent_override="",
        selected_expert_override="",
        train_metrics_override=None,
        focused_teach_loss=None,
        rehearsal_loss=None,
    ):
        train_metrics = {}
        if train_metrics_override is not None:
            train_metrics = dict(train_metrics_override)
        elif hasattr(self.controller, "training_metrics_snapshot"):
            train_metrics = self.controller.training_metrics_snapshot()

        state_snapshot = self.state.telemetry_snapshot()
        target_intent = ""
        try:
            if hasattr(self.controller, "target_intent_from_controls"):
                target_intent = self.controller.target_intent_from_controls(
                    human_angle_car,
                    human_speed_norm,
                )
        except Exception:
            target_intent = ""

        fields = {
            "config": pred.get("config", getattr(self.controller, "learning_paradigm", "")),
            "active_learning_paradigm": pred.get(
                "active_learning_paradigm",
                getattr(self.controller, "learning_paradigm", ""),
            ),
            "inference_backend": pred.get("inference_backend", getattr(self.controller, "inference_backend", "")),
            "feature_gate": pred.get("feature_gate", 1.0),
            "base_angle_norm": pred.get("base_angle_norm", 0.0),
            "final_angle_norm": pred.get("final_angle_norm", pred.get("base_angle_norm", 0.0)),
            "final_angle_car": pred.get("final_angle_car", 0.0),
            "base_speed_prob": pred.get("base_speed_prob", 0.0),
            "final_speed_prob": pred.get("final_speed_prob", pred.get("base_speed_prob", 0.0)),
            "final_speed_car": pred.get("final_speed_car", 0.0),
            "feature_ms": pred.get("feature_ms", 0.0),
            "inference_ms": pred.get("inference_ms", 0.0),
            "adapter_ms": pred.get("adapter_ms", 0.0),
            "human_angle_car": human_angle_car,
            "human_speed_norm": human_speed_norm,
            "target_intent": target_intent,
            "gate_e0": pred.get("gate_e0", 0.0),
            "gate_e1": pred.get("gate_e1", 0.0),
            "gate_e2": pred.get("gate_e2", 0.0),
            "gate_e3": pred.get("gate_e3", 0.0),
            "gate_entropy": pred.get("gate_entropy", 0.0),
            "top_expert": pred.get("top_expert", ""),
            "intent_pred": pred.get("intent_pred", ""),
            "intent_stop_prob": pred.get("intent_stop_prob", 0.0),
            "intent_left_prob": pred.get("intent_left_prob", 0.0),
            "intent_straight_prob": pred.get("intent_straight_prob", 0.0),
            "intent_right_prob": pred.get("intent_right_prob", 0.0),
            "replay_buffer_size": state_snapshot.get("replay_buffer_size", 0),
            "replay_buffer_bytes": state_snapshot.get("replay_buffer_bytes", 0),
            "total_updates": state_snapshot.get("total_updates", 0),
            "last_teach_loss": state_snapshot.get("last_teach_loss", 0.0),
            "focused_teach_loss": state_snapshot.get("last_focused_teach_loss", 0.0),
            "rehearsal_loss": state_snapshot.get("last_rehearsal_loss", 0.0),
            "last_learning_step_ms": state_snapshot.get("last_learning_step_ms", 0.0),
            "avg_learning_step_ms": state_snapshot.get("avg_learning_step_ms", 0.0),
        }
        fields.update(train_metrics)

        if target_intent_override:
            fields["target_intent"] = target_intent_override
        elif target_intent:
            fields["target_intent"] = target_intent

        if selected_expert_override:
            fields["selected_expert_for_teach"] = selected_expert_override

        if focused_teach_loss is not None:
            fields["focused_teach_loss"] = float(max(0.0, float(focused_teach_loss)))

        if rehearsal_loss is not None:
            fields["rehearsal_loss"] = float(max(0.0, float(rehearsal_loss)))

        if not fields.get("selected_expert_for_teach") and getattr(
            self.controller,
            "INTENT_EXPERT_SUPERVISION_ENABLED",
            False,
        ):
            try:
                if hasattr(self.controller, "selected_expert_for_intent"):
                    fields["selected_expert_for_teach"] = self.controller.selected_expert_for_intent(target_intent)
            except Exception:
                pass

        return fields

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
        step_time_total_ms = 0.0

        for _ in range(n_steps):
            sample = self.replay_buffer.sample(batch_size=16)
            if sample is None:
                break

            features, deltas, speeds = sample

            try:
                step_t0 = time.perf_counter()
                last_loss = self.controller.gradient_step(
                    features,
                    deltas,
                    speeds,
                    train_speed=True,
                    training_context="generic",
                )
                step_time_total_ms += (time.perf_counter() - step_t0) * 1000.0
                steps_done += 1

            except Exception as e:
                print(f"[driving] gradient step failed: {e}")
                break

        self.state.record_learning_steps(step_time_total_ms, steps_done, last_loss=last_loss)

        return last_loss

    def _controller_intent_scale(self, attr_name, intent_name, default=1.0, minimum=0.0):
        mapping = getattr(self.controller, attr_name, None)
        value = default

        if isinstance(mapping, dict):
            key = str(intent_name or "").strip().lower()
            if key and key in mapping:
                value = mapping[key]
            elif "*" in mapping:
                value = mapping["*"]

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(default)

        return max(float(minimum), value)

    def _boost_latest_target(
        self,
        target,
        n_steps=None,
        lr_multiplier=None,
        target_intent_name="",
        selected_expert_name="",
    ):
        """Run focused mixed updates so the newest correction does not overwrite older ones."""
        if target is None or self.controller.adapter is None:
            return 0.0, {}

        if n_steps is None:
            n_steps = self.BOOST_STEPS_PER_COMMIT

        if lr_multiplier is None:
            lr_multiplier = self.BOOST_LR_MULTIPLIER

        n_steps = int(round(
            float(n_steps)
            * self._controller_intent_scale(
                "TEACH_FOCUSED_STEP_SCALE_BY_INTENT",
                target_intent_name,
                default=1.0,
                minimum=0.25,
            )
        ))
        n_steps = max(1, min(14, n_steps))
        lr_cap = self._controller_intent_scale(
            "TEACH_FOCUSED_MAX_LR_MULTIPLIER_BY_INTENT",
            target_intent_name,
            default=getattr(self.controller, "TEACH_FOCUSED_MAX_LR_MULTIPLIER", 5.25),
            minimum=0.5,
        )
        lr_multiplier = min(
            lr_cap,
            float(lr_multiplier)
            * self._controller_intent_scale(
                "TEACH_FOCUSED_LR_SCALE_BY_INTENT",
                target_intent_name,
                default=1.0,
                minimum=0.25,
            ),
        )

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

        target_repeats = int(round(
            BOOST_TARGET_REPEATS
            * self._controller_intent_scale(
                "FOCUSED_TARGET_REPEAT_SCALE_BY_INTENT",
                target_intent_name,
                default=1.0,
                minimum=0.25,
            )
        ))
        target_repeats = max(1, min(16, target_repeats))

        batch_features = batch_features.repeat(target_repeats, 1)
        batch_target_deltas = batch_target_deltas.repeat(target_repeats, 1)
        batch_target_speeds = batch_target_speeds.repeat(target_repeats, 1)

        last_loss = 0.0
        steps_done = 0
        step_time_total_ms = 0.0
        focus_metrics = {}
        focused_rehearsal_batch_size = int(round(
            BOOST_REHEARSAL_BATCH_SIZE
            * float(max(0.0, getattr(self.controller, "FOCUSED_REHEARSAL_BATCH_SCALE", 1.0)))
            * self._controller_intent_scale(
                "FOCUSED_REHEARSAL_BATCH_SCALE_BY_INTENT",
                target_intent_name,
                default=1.0,
                minimum=0.0,
            )
        ))
        focused_rehearsal_batch_size = max(0, focused_rehearsal_batch_size)

        optimizer = getattr(self.controller, "optimizer", None)
        old_lrs = None

        if optimizer is not None:
            old_lrs = [group["lr"] for group in optimizer.param_groups]
            for group in optimizer.param_groups:
                group["lr"] = min(group["lr"] * lr_multiplier, 0.05)

        try:
            for _ in range(n_steps):
                rehearsal_batch = []
                if focused_rehearsal_batch_size > 0:
                    rehearsal_batch = self._sample_rehearsal_items(focused_rehearsal_batch_size)

                step_features = batch_features
                step_target_deltas = batch_target_deltas
                step_target_speeds = batch_target_speeds
                supervision_mask = torch.ones(step_features.shape[0], dtype=torch.bool)

                if rehearsal_batch:
                    rehearsal_features = torch.stack([item[0] for item in rehearsal_batch], dim=0)
                    rehearsal_deltas = torch.tensor(
                        [item[1] for item in rehearsal_batch],
                        dtype=torch.float32,
                    ).unsqueeze(-1)
                    rehearsal_speeds = torch.tensor(
                        [item[2] for item in rehearsal_batch],
                        dtype=torch.float32,
                    ).unsqueeze(-1)

                    step_features = torch.cat((step_features, rehearsal_features), dim=0)
                    step_target_deltas = torch.cat((step_target_deltas, rehearsal_deltas), dim=0)
                    step_target_speeds = torch.cat((step_target_speeds, rehearsal_speeds), dim=0)
                    supervision_mask = torch.cat(
                        (
                            supervision_mask,
                            torch.zeros(rehearsal_features.shape[0], dtype=torch.bool),
                        ),
                        dim=0,
                    )

                try:
                    historical_blend = 0.0
                    if rehearsal_batch and getattr(self.controller, "HISTORICAL_GRADIENT_ENABLED", False):
                        historical_blend = 0.5 * float(
                            getattr(self.controller, "HISTORICAL_GRADIENT_BLEND", 0.0)
                        )

                    step_t0 = time.perf_counter()
                    last_loss = self.controller.gradient_step(
                        step_features,
                        step_target_deltas,
                        step_target_speeds,
                        train_speed=True,
                        delta_penalty_weight=0.0,
                        clip_grad_norm=1.0,
                        historical_blend=historical_blend,
                        update_historical=False,
                        training_context="teach_focus",
                        expert_supervision_mask=supervision_mask,
                        target_intent_override=target_intent_name,
                        selected_expert_override=selected_expert_name,
                    )
                    if hasattr(self.controller, "training_metrics_snapshot"):
                        focus_metrics = self.controller.training_metrics_snapshot()
                    step_time_total_ms += (time.perf_counter() - step_t0) * 1000.0
                    steps_done += 1
                except Exception as e:
                    print(f"[driving] focused teach update failed: {e}")
                    break
        finally:
            if optimizer is not None and old_lrs is not None:
                for group, old_lr in zip(optimizer.param_groups, old_lrs):
                    group["lr"] = old_lr

        self.state.record_learning_steps(step_time_total_ms, steps_done, last_loss=last_loss)
        if hasattr(self.state, "record_focused_teach_loss"):
            self.state.record_focused_teach_loss(last_loss if steps_done > 0 else 0.0)

        return last_loss, focus_metrics

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

        delta = float(target.get("target_delta_angle", 0.0))
        speed = float(target.get("target_speed_norm", 0.0))
        bucket = self._correction_bucket(delta, speed)

        item = (f, delta, speed, bucket)

        # Always keep recent corrections for local plasticity.
        self._rehearsal_recent.append(item)

        # Reservoir memory protects old cases from being forgotten over time.
        self._rehearsal_seen += 1

        if len(self._rehearsal_protected) < REHEARSAL_PROTECTED_CAPACITY:
            self._rehearsal_protected.append(item)
            self._bucket_inc(self._rehearsal_protected_bucket_counts, bucket)
        else:
            replace_idx = self._pick_balanced_replace_index(
                self._rehearsal_protected,
                self._rehearsal_protected_bucket_counts,
                bucket,
            )
            if replace_idx is None:
                j = random.randint(0, self._rehearsal_seen - 1)
                if j < REHEARSAL_PROTECTED_CAPACITY:
                    replace_idx = j

            if replace_idx is not None:
                old_bucket = self._rehearsal_protected[replace_idx][3]
                self._rehearsal_protected[replace_idx] = item
                self._bucket_dec(self._rehearsal_protected_bucket_counts, old_bucket)
                self._bucket_inc(self._rehearsal_protected_bucket_counts, bucket)

        # A slower reservoir only sees periodic corrections, so very old cases
        # remain replayable over much longer teaching horizons.
        if (self._rehearsal_seen % REHEARSAL_ELDER_UPDATE_STRIDE) == 0:
            self._rehearsal_elder_seen += 1
            if len(self._rehearsal_elder) < REHEARSAL_ELDER_CAPACITY:
                self._rehearsal_elder.append(item)
                self._bucket_inc(self._rehearsal_elder_bucket_counts, bucket)
            else:
                replace_idx = self._pick_balanced_replace_index(
                    self._rehearsal_elder,
                    self._rehearsal_elder_bucket_counts,
                    bucket,
                )
                if replace_idx is None:
                    j = random.randint(0, self._rehearsal_elder_seen - 1)
                    if j < REHEARSAL_ELDER_CAPACITY:
                        replace_idx = j

                if replace_idx is not None:
                    old_bucket = self._rehearsal_elder[replace_idx][3]
                    self._rehearsal_elder[replace_idx] = item
                    self._bucket_dec(self._rehearsal_elder_bucket_counts, old_bucket)
                    self._bucket_inc(self._rehearsal_elder_bucket_counts, bucket)

    @staticmethod
    def _bucket_inc(counts, bucket):
        counts[bucket] = int(counts.get(bucket, 0)) + 1

    @staticmethod
    def _bucket_dec(counts, bucket):
        cur = int(counts.get(bucket, 0)) - 1
        if cur > 0:
            counts[bucket] = cur
        elif bucket in counts:
            del counts[bucket]

    @staticmethod
    def _correction_bucket(delta, speed):
        """Coarse bucket for diversity-preserving rehearsal memory."""
        angle = float(max(0.0, min(1.0, delta)))
        angle_bin = min(7, int(angle * 8.0))
        speed_bin = 1 if float(speed) >= 0.5 else 0
        return int(angle_bin * 2 + speed_bin)

    def _pick_balanced_replace_index(self, pool, counts, incoming_bucket):
        """Prefer replacing overrepresented buckets to keep long-term diversity."""
        if not pool or not counts:
            return None

        incoming_count = int(counts.get(incoming_bucket, 0))
        max_count = max(int(v) for v in counts.values())
        if incoming_count + 1 >= max_count:
            return None

        candidate_buckets = [k for k, v in counts.items() if int(v) == max_count]
        if not candidate_buckets:
            return None

        candidate_set = set(candidate_buckets)
        candidate_indices = [i for i, item in enumerate(pool) if item[3] in candidate_set]
        if not candidate_indices:
            return None

        return random.choice(candidate_indices)

    @staticmethod
    def _clone_rehearsal_item(item):
        features, delta, speed, bucket = item
        return features.detach().cpu().clone(), float(delta), float(speed), int(bucket)

    def _sample_bucket_balanced(self, pool, sample_n):
        """Round-robin buckets so frozen exemplars keep diverse corrections alive."""
        if sample_n <= 0 or not pool:
            return []

        grouped = {}
        for item in pool:
            grouped.setdefault(item[3], []).append(item)

        active_buckets = list(grouped.keys())
        random.shuffle(active_buckets)
        for bucket in active_buckets:
            random.shuffle(grouped[bucket])

        picked = []
        while len(picked) < sample_n and active_buckets:
            next_active = []
            for bucket in active_buckets:
                items = grouped.get(bucket)
                if not items:
                    continue

                picked.append(items.pop())
                if len(picked) >= sample_n:
                    break
                if items:
                    next_active.append(bucket)

            random.shuffle(next_active)
            active_buckets = next_active

        return picked

    def _sample_rehearsal_items(self, batch_n):
        """Sample a balanced rehearsal batch from recent and protected memories."""
        recent_n = len(self._rehearsal_recent)
        protected_n = len(self._rehearsal_protected)
        elder_n = len(self._rehearsal_elder)
        validated_n = len(self._rehearsal_validated)
        total_n = recent_n + protected_n + elder_n + validated_n

        if batch_n <= 0 or total_n <= 0:
            return []

        batch_n = min(batch_n, total_n)

        target_validated = int(round(batch_n * VALIDATED_EXEMPLAR_FRACTION))
        n_validated = min(target_validated, validated_n)
        if validated_n > 0 and batch_n >= 4:
            n_validated = max(1, n_validated)

        remaining = batch_n - n_validated

        target_elder = int(round(remaining * REHEARSAL_ELDER_FRACTION))
        n_elder = min(target_elder, elder_n)

        remaining_after_elder = remaining - n_elder
        target_protected = int(round(remaining_after_elder * REHEARSAL_PROTECTED_FRACTION))
        n_protected = min(target_protected, protected_n)
        n_recent = min(remaining_after_elder - n_protected, recent_n)

        remaining = batch_n - (n_validated + n_elder + n_protected + n_recent)
        if remaining > 0:
            extra_protected = min(remaining, max(0, protected_n - n_protected))
            n_protected += extra_protected
            remaining -= extra_protected
        if remaining > 0:
            extra_elder = min(remaining, max(0, elder_n - n_elder))
            n_elder += extra_elder
            remaining -= extra_elder
        if remaining > 0:
            extra_recent = min(remaining, max(0, recent_n - n_recent))
            n_recent += extra_recent
            remaining -= extra_recent
        if remaining > 0:
            extra_validated = min(remaining, max(0, validated_n - n_validated))
            n_validated += extra_validated

        batch = []
        if n_validated > 0:
            batch.extend(self._sample_bucket_balanced(self._rehearsal_validated, n_validated))
        if n_elder > 0:
            batch.extend(random.sample(self._rehearsal_elder, n_elder))
        if n_protected > 0:
            batch.extend(random.sample(self._rehearsal_protected, n_protected))
        if n_recent > 0:
            batch.extend(random.sample(list(self._rehearsal_recent), n_recent))

        random.shuffle(batch)
        return batch

    def _rehearsal_update(self, n_steps=REHEARSAL_STEPS_PER_COMMIT):
        """Rehearse a few older corrections to reduce catastrophic forgetting."""
        if not ENABLE_ANTI_FORGETTING_REHEARSAL:
            return 0.0

        if self.controller.adapter is None:
            return 0.0

        recent_n = len(self._rehearsal_recent)
        protected_n = len(self._rehearsal_protected)
        elder_n = len(self._rehearsal_elder)
        validated_n = len(self._rehearsal_validated)
        total_n = recent_n + protected_n + elder_n + validated_n

        if total_n < max(4, REHEARSAL_BATCH_SIZE):
            return 0.0

        optimizer = getattr(self.controller, "optimizer", None)
        old_lrs = None

        if optimizer is not None:
            old_lrs = [group["lr"] for group in optimizer.param_groups]
            for group in optimizer.param_groups:
                group["lr"] = max(1e-5, group["lr"] * REHEARSAL_LR_MULTIPLIER)

        batch_size_scale = float(max(0.5, getattr(self.controller, "REHEARSAL_BATCH_SIZE_SCALE", 1.0)))
        effective_batch_size = max(4, int(round(REHEARSAL_BATCH_SIZE * batch_size_scale)))
        steps_scale = float(max(0.5, getattr(self.controller, "REHEARSAL_STEPS_SCALE", 1.0)))
        effective_steps = max(1, int(round(max(1, n_steps) * steps_scale)))

        last_loss = 0.0
        steps_done = 0
        step_time_total_ms = 0.0

        try:
            for _ in range(effective_steps):
                batch_n = min(effective_batch_size, total_n)
                batch = self._sample_rehearsal_items(batch_n)

                if not batch:
                    break

                features = torch.stack([b[0] for b in batch], dim=0)
                deltas = torch.tensor([b[1] for b in batch], dtype=torch.float32).unsqueeze(-1)
                speeds = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(-1)

                try:
                    step_t0 = time.perf_counter()
                    last_loss = self.controller.gradient_step(
                        features,
                        deltas,
                        speeds,
                        train_speed=True,
                        delta_penalty_weight=0.01,
                        clip_grad_norm=1.0,
                        historical_blend=float(
                            getattr(self.controller, "HISTORICAL_GRADIENT_BLEND", 0.0)
                        ),
                        update_historical=True,
                        training_context="rehearsal",
                    )
                    step_time_total_ms += (time.perf_counter() - step_t0) * 1000.0
                    steps_done += 1
                except Exception as e:
                    print(f"[driving] rehearsal step failed: {e}")
                    break
        finally:
            if optimizer is not None and old_lrs is not None:
                for group, old_lr in zip(optimizer.param_groups, old_lrs):
                    group["lr"] = old_lr

        self.state.record_learning_steps(step_time_total_ms, steps_done, last_loss=last_loss)
        if hasattr(self.state, "record_rehearsal_loss"):
            self.state.record_rehearsal_loss(last_loss if steps_done > 0 else 0.0)

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

    def _compute_teach_update_profile(
        self,
        predicted_angle_car,
        selected_angle_car,
        predicted_speed_prob,
        target_speed_norm,
    ):
        """Shift from plasticity to consolidation as checkpoint drift grows."""
        boost_steps, boost_lr_mult = self._compute_teach_boost(
            predicted_angle_car=predicted_angle_car,
            selected_angle_car=selected_angle_car,
            predicted_speed_prob=predicted_speed_prob,
            target_speed_norm=target_speed_norm,
        )

        drift_rms = 0.0
        if hasattr(self.controller, "checkpoint_drift_rms"):
            try:
                drift_rms = float(self.controller.checkpoint_drift_rms())
            except Exception:
                drift_rms = 0.0

        start = LONG_HORIZON_DRIFT_DECAY_START
        end = max(start * 1.01, LONG_HORIZON_DRIFT_DECAY_END)
        if drift_rms <= start:
            return boost_steps, boost_lr_mult, REHEARSAL_STEPS_PER_COMMIT

        progress = (drift_rms - start) / float(end - start)
        progress = max(0.0, min(1.0, progress))

        boost_scale = 1.0 - progress * (1.0 - LONG_HORIZON_MIN_BOOST_SCALE)
        rehearsal_scale = 1.0 + progress * (LONG_HORIZON_MAX_REHEARSAL_SCALE - 1.0)

        boost_steps = max(2, int(round(boost_steps * boost_scale)))
        boost_lr_mult = max(1.1, boost_lr_mult * boost_scale)

        rehearsal_steps = int(round(REHEARSAL_STEPS_PER_COMMIT * rehearsal_scale))
        rehearsal_steps = max(REHEARSAL_STEPS_PER_COMMIT, min(10, rehearsal_steps))

        return boost_steps, boost_lr_mult, rehearsal_steps

    def _maybe_augment_teach_frame(self, frame):
        """Apply slight photometric-only perturbations to improve lighting robustness."""
        if not ENABLE_TEACH_PHOTOMETRIC_AUGMENTATION:
            return frame

        if random.random() > TEACH_AUGMENT_PROB:
            return frame

        img = frame.astype(np.float32) / 255.0

        # Mild exposure jitter.
        gain = random.uniform(0.92, 1.08)
        bias = random.uniform(-0.05, 0.05)
        img = img * gain + bias

        # Mild global contrast around per-channel mean.
        if random.random() < 0.7:
            contrast = random.uniform(0.92, 1.08)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            img = (img - mean) * contrast + mean

        # Mild gamma perturbation for shading changes.
        if random.random() < 0.6:
            gamma = random.uniform(0.90, 1.10)
            img = np.power(np.clip(img, 0.0, 1.0), gamma)

        # Tiny sensor noise to reduce overfitting to exact pixels.
        if random.random() < 0.25:
            sigma = random.uniform(0.0, 0.01)
            if sigma > 0:
                img = img + np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)

        img = np.clip(img, 0.0, 1.0)
        return (img * 255.0).astype(np.uint8)

    def _teach_forward_step(self, pred, frame, long=False):
        """Commit: capture frame, log target, gradient step, execute, stop."""
        selected_angle_car = self.teach_controller.get()
        teach_image = self._maybe_augment_teach_frame(frame)

        # 1. Compute target and add to replay buffer
        target = self.controller.teach_step(
            base_angle_norm=pred["base_angle_norm"],
            base_speed_prob=pred["base_speed_prob"],
            human_angle_car=selected_angle_car,
            human_speed_norm=1.0,
            image=teach_image,
        )

        if target is not None and STORE_TEACH_CORRECTIONS_IN_REPLAY:
            self.replay_buffer.add(
                target["input_features"],
                target["target_delta_angle"],
                target["target_speed_norm"],
            )

        target_intent, assigned_expert = self._teach_assignment(selected_angle_car, 1.0)

        # Strengthen immediate effect of this exact teaching event.
        boost_steps, boost_lr_mult, rehearsal_steps = self._compute_teach_update_profile(
            predicted_angle_car=pred["final_angle_car"],
            selected_angle_car=selected_angle_car,
            predicted_speed_prob=pred["base_speed_prob"],
            target_speed_norm=1.0,
        )
        focused_loss, focused_metrics = self._boost_latest_target(
            target,
            n_steps=boost_steps,
            lr_multiplier=boost_lr_mult,
            target_intent_name=target_intent,
            selected_expert_name=assigned_expert,
        )
        loss = focused_loss

        # Rehearse older teach corrections to preserve previously learned locations.
        rehearsal_loss = self._rehearsal_update(n_steps=rehearsal_steps)
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
                session=self.state.session_label,
                target_delta_angle=target["target_delta_angle"] if target else 0.0,
                loss_after_update=loss,
                **self._correction_log_fields(
                    pred,
                    selected_angle_car,
                    1.0,
                    target_intent_override=target_intent,
                    selected_expert_override=assigned_expert,
                    train_metrics_override=focused_metrics,
                    focused_teach_loss=focused_loss,
                    rehearsal_loss=rehearsal_loss,
                ),
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
        teach_image = self._maybe_augment_teach_frame(frame)

        target = self.controller.teach_step(
            base_angle_norm=pred["base_angle_norm"],
            base_speed_prob=pred["base_speed_prob"],
            human_angle_car=selected_angle_car,
            human_speed_norm=0.0,
            image=teach_image,
        )

        if target is not None and STORE_TEACH_CORRECTIONS_IN_REPLAY:
            self.replay_buffer.add(
                target["input_features"],
                target["target_delta_angle"],
                target["target_speed_norm"],
            )

        target_intent, assigned_expert = self._teach_assignment(selected_angle_car, 0.0)

        # Strengthen immediate effect of this exact teaching event.
        boost_steps, boost_lr_mult, rehearsal_steps = self._compute_teach_update_profile(
            predicted_angle_car=pred["final_angle_car"],
            selected_angle_car=selected_angle_car,
            predicted_speed_prob=pred["base_speed_prob"],
            target_speed_norm=0.0,
        )
        focused_loss, focused_metrics = self._boost_latest_target(
            target,
            n_steps=boost_steps,
            lr_multiplier=boost_lr_mult,
            target_intent_name=target_intent,
            selected_expert_name=assigned_expert,
        )
        loss = focused_loss

        rehearsal_loss = self._rehearsal_update(n_steps=rehearsal_steps)
        if rehearsal_loss > 0:
            loss = rehearsal_loss

        self._remember_target_for_rehearsal(target)

        if STORE_TEACH_CORRECTIONS_IN_REPLAY:
            loss = self._do_gradient_steps()

        try:
            self.session_logger.log_correction(
                frame_bgr=frame,
                command="stop",
                session=self.state.session_label,
                target_delta_angle=target["target_delta_angle"] if target else 0.0,
                loss_after_update=loss,
                **self._correction_log_fields(
                    pred,
                    selected_angle_car,
                    0.0,
                    target_intent_override=target_intent,
                    selected_expert_override=assigned_expert,
                    train_metrics_override=focused_metrics,
                    focused_teach_loss=focused_loss,
                    rehearsal_loss=rehearsal_loss,
                ),
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