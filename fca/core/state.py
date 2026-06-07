"""
fca/core/state.py — Shared state across all threads.

Single source of truth for what the car is doing. Every thread reads/writes
through this object, always under the lock.
"""
import threading
import time


# ─── Runtime modes ────────────────────────────────────────────────────────
MODE_AUTOPILOT       = "AUTOPILOT"        # Base + adapter drive the car
MODE_TEACH           = "TEACH"            # Stepwise human teaching
MODE_REVERSE_MANUAL  = "REVERSE_MANUAL"   # Human-controlled reverse repositioning
MODE_PAUSED          = "PAUSED"           # Stopped, waiting
MODE_DATASET_COLLECTION = "DATASET_COLLECTION"  # Slow forward drive + continuous image capture

ALL_MODES = (
    MODE_AUTOPILOT,
    MODE_TEACH,
    MODE_REVERSE_MANUAL,
    MODE_PAUSED,
    MODE_DATASET_COLLECTION,
)


class FCAState:
    """Thread-safe shared state object."""

    def __init__(self):
        self.lock = threading.Lock()

        # Runtime mode
        self.mode = MODE_PAUSED
        self.previous_mode = MODE_PAUSED

        # Live frame data
        self.latest_frame_jpeg = None       # browser MJPEG stream
        self.latest_frame_raw = None        # numpy array for inference
        self.video_client_count = 0         # active MJPEG viewers

        # Latest model outputs (populated each inference)
        self.base_angle_norm = 0.5          # [0, 1]
        self.base_speed_prob = 0.0          # [0, 1]
        self.delta_angle_norm = 0.0         # adapter correction
        self.delta_speed_logit = 0.0
        self.final_angle_car = 90.0         # degrees, 50-130
        self.final_speed_car = 0.0          # 0 or 35
        self.max_speed = 35

        # Teaching state
        self.selected_angle_car = 90.0      # degrees, what the human is selecting
        self.last_teach_loss = 0.0
        self.last_focused_teach_loss = 0.0
        self.last_rehearsal_loss = 0.0
        self.total_updates = 0
        self.last_learning_step_ms = 0.0
        self.avg_learning_step_ms = 0.0
        self._learning_step_time_total_ms = 0.0
        self._learning_step_samples = 0
        self.autopilot_frame_logging = False

        # Buffer / counters
        self.replay_buffer_size = 0
        self.replay_buffer_bytes = 0
        self.corrections_logged = 0

        # Performance metrics
        self.fps = 0.0
        self.camera_ms = 0.0
        self.preview_ms = 0.0
        self.feature_ms = 0.0
        self.inference_ms = 0.0
        self.adapter_ms = 0.0
        self.control_ms = 0.0
        self.logging_ms = 0.0
        self.loop_ms = 0.0
        self.other_ms = 0.0
        self.frames_processed = 0

        # Session metadata
        self.session_label = "default"
        self.session_started = time.time()

        # Control flags
        self.shutdown = False
        self.human_active = False           # is human currently overriding?

        # Event log for "interesting" markers
        self.event_markers = []             # list of (timestamp, description)

    def set_mode(self, new_mode):
        """Change runtime mode safely."""
        if new_mode not in ALL_MODES:
            raise ValueError(f"Invalid mode: {new_mode}")
        with self.lock:
            self.previous_mode = self.mode
            self.mode = new_mode

    def get_mode(self):
        with self.lock:
            return self.mode

    def telemetry_snapshot(self):
        """Return a dict suitable for sending to the GUI."""
        with self.lock:
            return {
                'mode': self.mode,
                'session': self.session_label,
                'session_elapsed': time.time() - self.session_started,
                'video_client_count': self.video_client_count,
                'frames': self.frames_processed,
                'fps': round(self.fps, 1),
                'camera_ms': round(self.camera_ms, 2),
                'preview_ms': round(self.preview_ms, 2),
                'feature_ms': round(self.feature_ms, 2),
                'inference_ms': round(self.inference_ms, 2),
                'adapter_ms': round(self.adapter_ms, 2),
                'control_ms': round(self.control_ms, 2),
                'logging_ms': round(self.logging_ms, 2),
                'loop_ms': round(self.loop_ms, 2),
                'other_ms': round(self.other_ms, 2),
                'base_angle_norm': round(self.base_angle_norm, 4),
                'delta_angle_norm': round(self.delta_angle_norm, 4),
                'final_angle_car': round(self.final_angle_car, 1),
                'base_speed_prob': round(self.base_speed_prob, 4),
                'delta_speed_logit': round(self.delta_speed_logit, 4),
                'final_speed_car': round(self.final_speed_car, 1),
                'max_speed': int(self.max_speed),
                'selected_angle_car': round(self.selected_angle_car, 1),
                'last_teach_loss': round(self.last_teach_loss, 6),
                'last_focused_teach_loss': round(self.last_focused_teach_loss, 6),
                'last_rehearsal_loss': round(self.last_rehearsal_loss, 6),
                'total_updates': self.total_updates,
                'last_learning_step_ms': round(self.last_learning_step_ms, 3),
                'avg_learning_step_ms': round(self.avg_learning_step_ms, 3),
                'autopilot_frame_logging': bool(self.autopilot_frame_logging),
                'replay_buffer_size': self.replay_buffer_size,
                'replay_buffer_bytes': self.replay_buffer_bytes,
                'corrections_logged': self.corrections_logged,
                'human_active': self.human_active,
            }

    def record_learning_steps(self, total_ms, step_count, last_loss=None):
        step_count = int(max(0, step_count))
        total_ms = float(max(0.0, total_ms))
        if step_count <= 0:
            return

        avg_ms = total_ms / float(step_count)

        with self.lock:
            self.total_updates += step_count
            self.last_learning_step_ms = avg_ms
            self._learning_step_time_total_ms += total_ms
            self._learning_step_samples += step_count
            self.avg_learning_step_ms = (
                self._learning_step_time_total_ms / float(self._learning_step_samples)
                if self._learning_step_samples > 0 else 0.0
            )
            if last_loss is not None:
                self.last_teach_loss = float(last_loss)

    def record_focused_teach_loss(self, loss):
        with self.lock:
            self.last_focused_teach_loss = float(max(0.0, float(loss or 0.0)))

    def record_rehearsal_loss(self, loss):
        with self.lock:
            self.last_rehearsal_loss = float(max(0.0, float(loss or 0.0)))

    def add_event_marker(self, description=""):
        with self.lock:
            self.event_markers.append((time.time(), description))
