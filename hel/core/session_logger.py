"""
hel/core/session_logger.py — Per-session CSV logging.

Two files per session:
  - frames.csv         every frame's state
  - corrections.csv    only teaching events (with frame paths)

Frames at correction events are saved to disk as JPEGs for offline retraining.
"""
import csv
import os
import time
from datetime import datetime

import cv2

from hel.core.state import MODE_DATASET_COLLECTION


class SessionLogger:
    SAVE_TEACH_CORRECTION_FRAMES = False

    FRAME_COLUMNS = [
        "elapsed_s", "wall_time", "session", "mode",
        "base_angle_norm", "delta_angle_norm", "final_angle_car",
        "base_speed_prob", "delta_speed_logit", "final_speed_car",
        "human_active", "selected_angle_car",
        "dataset_angle_car", "dataset_speed_norm",
        "fps", "inference_ms", "adapter_ms",
        "replay_buffer_size", "command_buffer_size",
        "total_updates", "last_teach_loss",
        "frame_path",
    ]

    CORRECTION_COLUMNS = [
        "elapsed_s", "wall_time", "session", "command",
        "frame_path",
        "base_angle_norm", "base_speed_prob",
        "human_angle_car", "human_speed_norm",
        "target_delta_angle", "loss_after_update", "total_updates",
    ]

    DATASET_COLUMNS = ["image_id", "angle", "speed"]

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        self._frames_fh = None
        self._frames_writer = None
        self._corrections_fh = None
        self._corrections_writer = None
        self._frames_dir = None
        self._dataset_frames_dir = None
        self._dataset_csv_fh = None
        self._dataset_csv_writer = None
        self._dataset_image_id = 0

        self._current_session = None
        self._start_time = None
        self._frame_counter = 0

    def _open_session_files(self, session_label):
        """Open new CSV files for a new session label."""
        self._close()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{ts}_{session_label}"

        self._frames_dir = os.path.join(self.log_dir, f"{session_id}_frames")
        os.makedirs(self._frames_dir, exist_ok=True)

        self._dataset_frames_dir = os.path.join(self.log_dir, f"{session_id}_dataset")
        os.makedirs(self._dataset_frames_dir, exist_ok=True)

        dataset_csv_path = os.path.join(self._dataset_frames_dir, "train.csv")

        frames_path = os.path.join(self.log_dir, f"{session_id}_frames.csv")
        corrections_path = os.path.join(self.log_dir, f"{session_id}_corrections.csv")

        self._frames_fh = open(frames_path, "w", newline="")
        self._frames_writer = csv.writer(self._frames_fh)
        self._frames_writer.writerow(self.FRAME_COLUMNS)

        self._corrections_fh = open(corrections_path, "w", newline="")
        self._corrections_writer = csv.writer(self._corrections_fh)
        self._corrections_writer.writerow(self.CORRECTION_COLUMNS)

        self._dataset_csv_fh = open(dataset_csv_path, "w", newline="")
        self._dataset_csv_writer = csv.writer(self._dataset_csv_fh)
        self._dataset_csv_writer.writerow(self.DATASET_COLUMNS)

        self._current_session = session_label
        self._start_time = time.time()
        self._frame_counter = 0
        self._dataset_image_id = 0

        print(f"[logger] new session → {frames_path}")
        print(f"[logger] dataset CSV → {dataset_csv_path}")

    def log_frame(
        self,
        state_snapshot,
        frame_bgr=None,
        dataset_capture_frame=False,
        dataset_speed_norm_label="",
        dataset_angle_car_label="",
    ):
        """Log one row to frames.csv. Called every inference frame."""
        if state_snapshot["session"] != self._current_session:
            self._open_session_files(state_snapshot["session"])

        elapsed = time.time() - self._start_time
        frame_path = ""
        dataset_angle_car = dataset_angle_car_label
        dataset_speed_norm = dataset_speed_norm_label

        if (
            dataset_capture_frame
            and frame_bgr is not None
            and self._dataset_frames_dir is not None
        ):
            image_id = self._dataset_image_id
            frame_path = os.path.join(self._dataset_frames_dir, f"{image_id}.png")

            try:
                cv2.imwrite(frame_path, frame_bgr)

                if self._dataset_csv_writer is not None:
                    try:
                        angle_car = float(dataset_angle_car_label)
                    except Exception:
                        angle_car = float(state_snapshot.get("selected_angle_car", 90.0))

                    angle_norm = (angle_car - 50.0) / 80.0
                    angle_norm = max(0.0, min(1.0, angle_norm))

                    try:
                        speed_norm = float(dataset_speed_norm_label)
                    except Exception:
                        speed_norm = 0.0

                    self._dataset_csv_writer.writerow([
                        image_id,
                        f"{angle_norm:.4f}",
                        f"{speed_norm:.1f}",
                    ])

                    if image_id % 30 == 0 and self._dataset_csv_fh is not None:
                        self._dataset_csv_fh.flush()

                self._dataset_image_id += 1
            except Exception as e:
                print(f"[logger] WARN — failed to save dataset frame: {e}")
                frame_path = ""

        row = [
            f"{elapsed:.3f}", f"{time.time():.3f}",
            state_snapshot["session"], state_snapshot["mode"],
            f"{state_snapshot['base_angle_norm']:.5f}",
            f"{state_snapshot['delta_angle_norm']:.5f}",
            f"{state_snapshot['final_angle_car']:.2f}",
            f"{state_snapshot['base_speed_prob']:.5f}",
            f"{state_snapshot['delta_speed_logit']:.5f}",
            f"{state_snapshot['final_speed_car']:.2f}",
            int(state_snapshot["human_active"]),
            f"{state_snapshot['selected_angle_car']:.2f}",
            dataset_angle_car,
            dataset_speed_norm,
            f"{state_snapshot['fps']:.2f}",
            f"{state_snapshot['inference_ms']:.3f}",
            f"{state_snapshot['adapter_ms']:.3f}",
            state_snapshot["replay_buffer_size"],
            state_snapshot["command_buffer_size"],
            state_snapshot["total_updates"],
            f"{state_snapshot['last_teach_loss']:.6f}",
            frame_path,
        ]
        self._frames_writer.writerow(row)

        self._frame_counter += 1
        if self._frame_counter % 30 == 0:
            self._frames_fh.flush()

    def log_correction(self, frame_bgr, command, base_angle_norm,
                       base_speed_prob, human_angle_car, human_speed_norm,
                       target_delta_angle, loss_after_update, total_updates,
                       session):
        """Log one row to corrections.csv and save the frame as JPEG."""
        if session != self._current_session:
            self._open_session_files(session)

        elapsed = time.time() - self._start_time

        frame_path = ""
        if self.SAVE_TEACH_CORRECTION_FRAMES and frame_bgr is not None:
            frame_id = f"{int(elapsed * 1000):08d}"
            frame_path = os.path.join(self._frames_dir, f"{frame_id}.jpg")
            try:
                cv2.imwrite(frame_path, frame_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception as e:
                print(f"[logger] WARN — failed to save frame: {e}")
                frame_path = ""

        row = [
            f"{elapsed:.3f}", f"{time.time():.3f}", session, command,
            frame_path,
            f"{base_angle_norm:.5f}", f"{base_speed_prob:.5f}",
            f"{human_angle_car:.2f}", f"{human_speed_norm:.2f}",
            f"{target_delta_angle:.5f}",
            f"{loss_after_update:.6f}", total_updates,
        ]
        self._corrections_writer.writerow(row)
        self._corrections_fh.flush()  # always flush corrections immediately

    def _close(self):
        if self._frames_fh is not None:
            self._frames_fh.close()
            self._frames_fh = None
        if self._corrections_fh is not None:
            self._corrections_fh.close()
            self._corrections_fh = None
        if self._dataset_csv_fh is not None:
            self._dataset_csv_fh.close()
            self._dataset_csv_fh = None
            self._dataset_csv_writer = None

    def close(self):
        self._close()
        print(f"[logger] session closed")
