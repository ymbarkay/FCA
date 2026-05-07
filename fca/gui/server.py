"""
fca/gui/server.py — Flask + WebSocket server for the operator interface.

Serves:
  GET /            : the browser UI (templates/index.html)
  GET /video_feed  : MJPEG stream of latest camera frame
  WS  /ws          : commands in, telemetry out
"""
import json
import os
import threading
import time

from flask import Flask, Response, render_template
from flask_sock import Sock

from fca.core.state import (
    MODE_AUTOPILOT,
    MODE_TEACH,
    MODE_REVERSE_MANUAL,
    MODE_PAUSED,
    MODE_DATASET_COLLECTION,
)


def create_app(state, controller, driving_loop, teach_controller, replay_buffer):
    """Build the Flask app with all dependencies wired in."""
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    app = Flask(__name__, template_folder=template_dir)
    sock = Sock(app)

    # ─── UI ───────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    # ─── MJPEG video stream ───────────────────────────────────────────────
    def generate_mjpeg():
        last_frame_id = None

        while not state.shutdown:
            with state.lock:
                frame = state.latest_frame_jpeg
                frame_id = state.frames_processed

            if frame is not None and frame_id != last_frame_id:
                last_frame_id = frame_id
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

            time.sleep(0.033)

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    # ─── WebSocket: commands in, telemetry out ────────────────────────────
    @sock.route("/ws")
    def ws_handler(ws):
        # Telemetry push thread
        def push_telemetry():
            while not state.shutdown:
                try:
                    payload = json.dumps({
                        "type": "telemetry",
                        **state.telemetry_snapshot(),
                        **controller.checkpoint_status_snapshot(),
                    })
                    ws.send(payload)
                    time.sleep(0.1)
                except Exception:
                    break

        threading.Thread(target=push_telemetry, daemon=True).start()

        # Receive commands
        while not state.shutdown:
            try:
                raw = ws.receive(timeout=1)

                if raw is None:
                    continue

                msg = json.loads(raw)

            except Exception:
                break

            try:
                _handle_command(
                    msg,
                    state,
                    controller,
                    driving_loop,
                    teach_controller,
                    replay_buffer,
                )
            except Exception as e:
                print(f"[server] command error: {e}")

    return app


def _request_steer_update_if_teach(state, driving_loop):
    """Make angle selection physically visible in TEACH mode."""
    if state.get_mode() in (MODE_TEACH, MODE_REVERSE_MANUAL):
        driving_loop.request_teach_command("steer_update")


def _handle_command(msg, state, controller, driving_loop, teach_controller, replay_buffer):
    """Dispatch a command from the browser."""
    t = msg.get("type")

    # ── Mode switching ────────────────────────────────────────────────────
    if t == "mode":
        new_mode = msg.get("value")

        if new_mode in (
            MODE_AUTOPILOT,
            MODE_TEACH,
            MODE_REVERSE_MANUAL,
            MODE_PAUSED,
            MODE_DATASET_COLLECTION,
        ):
            state.set_mode(new_mode)

            # Clear any pending teach command when changing mode
            if hasattr(driving_loop, "clear_pending_teach_command"):
                driving_loop.clear_pending_teach_command()
            else:
                driving_loop.pending_teach_command = None

            print(f"[server] mode → {new_mode}")

        else:
            print(f"[server] ignored invalid mode: {new_mode}")

    elif t == "shutdown":
        state.shutdown = True

    # ── Selection commands: no learning, no movement ──────────────────────
    elif t == "steer_left":
        teach_controller.steer_left()
        _request_steer_update_if_teach(state, driving_loop)

    elif t == "steer_right":
        teach_controller.steer_right()
        _request_steer_update_if_teach(state, driving_loop)

    elif t == "centre":
        teach_controller.centre()
        _request_steer_update_if_teach(state, driving_loop)

    elif t == "set_angle":
        teach_controller.set(msg["value"])
        _request_steer_update_if_teach(state, driving_loop)

    # ── Teach action commands: commit + execute ───────────────────────────
    elif t == "forward_step":
        if state.get_mode() == MODE_TEACH:
            driving_loop.request_teach_command("forward_step")
        else:
            print(f"[server] forward_step ignored — mode={state.get_mode()}")

    elif t == "long_forward_step":
        if state.get_mode() == MODE_TEACH:
            driving_loop.request_teach_command("long_forward_step")
        else:
            print(f"[server] long_forward_step ignored — mode={state.get_mode()}")

    elif t == "stop_teach":
        if state.get_mode() == MODE_TEACH:
            driving_loop.request_teach_command("stop_teach")
        else:
            print(f"[server] stop_teach ignored — mode={state.get_mode()}")

    elif t == "backward":
        if state.get_mode() == MODE_TEACH:
            driving_loop.request_teach_command("backward")
        else:
            print(f"[server] backward ignored — mode={state.get_mode()}")

    elif t == "backward_hold_start":
        if state.get_mode() in (MODE_TEACH, MODE_REVERSE_MANUAL):
            driving_loop.request_teach_command("backward_hold_start")
        else:
            print(f"[server] backward_hold_start ignored — mode={state.get_mode()}")

    elif t == "backward_hold_stop":
        if state.get_mode() in (MODE_TEACH, MODE_REVERSE_MANUAL):
            driving_loop.request_teach_command("backward_hold_stop")
        else:
            print(f"[server] backward_hold_stop ignored — mode={state.get_mode()}")

    elif t == "dataset_stop":
        if state.get_mode() == MODE_DATASET_COLLECTION:
            driving_loop.request_teach_command("dataset_stop")
        else:
            print(f"[server] dataset_stop ignored — mode={state.get_mode()}")

    elif t == "dataset_resume":
        if state.get_mode() == MODE_DATASET_COLLECTION:
            driving_loop.request_teach_command("dataset_resume")
        else:
            print(f"[server] dataset_resume ignored — mode={state.get_mode()}")

    elif t == "dataset_capture_stop_frame":
        if hasattr(driving_loop, "request_dataset_capture_stop_frame"):
            driving_loop.request_dataset_capture_stop_frame()
        else:
            driving_loop.request_teach_command("dataset_capture_stop_frame")

    # ── Recovery ──────────────────────────────────────────────────────────
    elif t == "rewind":
        # Independent manual reverse mode is allowed from any current mode.
        state.set_mode(MODE_REVERSE_MANUAL)

        if hasattr(driving_loop, "clear_pending_teach_command"):
            driving_loop.clear_pending_teach_command()
        else:
            driving_loop.pending_teach_command = None

        print("[server] reverse manual mode entered")

    # ── Session ───────────────────────────────────────────────────────────
    elif t == "session":
        with state.lock:
            state.session_label = str(msg.get("value", "")).strip() or "default"

        print(f"[server] session → {state.session_label}")

    elif t == "event_marker":
        state.add_event_marker(msg.get("description", ""))

    # ── Adapter management ────────────────────────────────────────────────
    elif t == "reset_adapter":
        controller.reset_adapter()
        print("[server] adapter reset")

    elif t == "save_checkpoint":
        saved = controller.save_checkpoint(manual=True)
        if saved:
            exemplar_count = 0
            if hasattr(driving_loop, "capture_validated_exemplars"):
                exemplar_count = int(driving_loop.capture_validated_exemplars())
            print(f"[server] checkpoint saved ({exemplar_count} validated exemplars)")
        else:
            print("[server] checkpoint save skipped")

    elif t == "clear_buffer":
        replay_buffer.clear()
        print("[server] replay buffer cleared")

    else:
        print(f"[server] unknown command: {msg}")