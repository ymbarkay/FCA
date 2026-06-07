"""
run.py — FCA entry point.

Wires every component together: state, controller, motors, driving loop,
trainer thread, GUI server. Handles graceful shutdown.

Usage:
    python3 run.py --mode test --adapter scalar
    python3 run.py --mode drive --adapter scalar
    python3 run.py --mode drive --adapter deep
    python3 run.py --mode drive --adapter none
    python3 run.py --mode drive --no-gui

Edge TPU base examples:
    python3 run.py --mode test --adapter scalar \
      --base-model tflite_models/best_model_finetuned_int8_edgetpu.tflite

Frozen model examples:
        python3 run.py --mode drive --adapter none \
            --base-model tflite_models/best_model_finetuned_int8_edgetpu.tflite

        python3 run.py --mode drive --adapter none \
            --base-model checkpoints/frozen_driver.keras

        python3 run.py --mode drive --adapter none \
            --base-model checkpoints/frozen_driver.h5

Deep adapter example:
    python3 run.py --mode test --adapter deep \
      --base-model tflite_models/best_model_finetuned_int8_edgetpu.tflite \
      --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite

Force CPU:
    python3 run.py --mode test --adapter scalar \
      --no-tpu \
      --base-model tflite_models/best_model_finetuned_int8.tflite
"""
import argparse
import os
import sys
import threading
import time

# Make the package importable when running this script directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fca.core.state import FCAState, MODE_PAUSED
from fca.core.teach_state import TeachController
from fca.core.session_logger import SessionLogger
from fca.learning.controller import AdaptiveController
from fca.learning.replay_buffer import ReplayBuffer
from fca.learning.trainer import run_trainer
from fca.control.motors import Motors
from fca.control.driving_loop import DrivingLoop
from fca.gui.server import create_app


def parse_args():
    p = argparse.ArgumentParser(
        description="FCA — Feedback-based Continual Adaptation operator interface"
    )

    # Runtime mode
    p.add_argument(
        "--mode",
        default="drive",
        choices=["test", "drive"],
        help="test = no motors/dry run; drive = real PiCar control",
    )

    # Adapter mode
    p.add_argument(
        "--adapter",
        default="scalar",
        choices=["scalar", "deep", "none"],
        help="Stage A: scalar | Stage B: deep | none = base model only",
    )

    # Base / frozen model
    p.add_argument(
        "--base-model",
        default="tflite_models/best_model_finetuned_int8.tflite",
        help=(
            "Path to the frozen autopilot model. Supported formats: "
            ".tflite, .keras, .h5"
        ),
    )

    p.add_argument(
        "--cpu-base-model",
        default=None,
        help=(
            "Optional non-compiled INT8 .tflite model for CPU fallback. "
            "Use this when --base-model points to an EdgeTPU-compiled TFLite model."
        ),
    )

    # Deep feature extractor model
    p.add_argument(
        "--feature-model",
        default=None,
        help=(
            "Path to TFLite feature extractor model for --adapter deep. "
            "Example: tflite_models/feature_extractor_dense512_int8_edgetpu.tflite"
        ),
    )

    p.add_argument(
        "--learning-paradigm",
        default="moe_v4_intent_routing",
        help=(
            "Online deep learning paradigm loaded from fca/learning/paradigms/. "
            "Only used when --adapter deep is selected."
        ),
    )

    p.add_argument(
        "--no-tpu",
        action="store_true",
        help="Force CPU TFLite even if Edge TPU delegate is available",
    )

    p.add_argument(
        "--tflite-threads",
        type=int,
        default=4,
        help="CPU TFLite thread count when not using Edge TPU",
    )

    # Adapter checkpoint
    p.add_argument(
        "--checkpoint",
        default="checkpoints/adapter.pt",
        help="Path for adapter weight checkpoint/load/save",
    )

    # Camera
    p.add_argument(
        "--capture-src",
        type=int,
        default=0,
        help="OpenCV camera index, default: 0",
    )

    # Control
    p.add_argument(
        "--max-speed",
        type=int,
        default=35,
        help="Maximum forward speed command",
    )

    # Learning
    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adapter learning rate",
    )

    p.add_argument(
        "--update-period",
        type=float,
        default=1.5,
        help="Seconds between background trainer gradient steps",
    )

    p.add_argument(
        "--no-trainer",
        action="store_true",
        help=(
            "Disable background trainer thread. "
            "Useful if teaching events already perform inline updates."
        ),
    )

    # GUI
    p.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Flask GUI port",
    )

    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Flask GUI host",
    )

    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Headless mode, no Flask server",
    )

    # Session duration
    p.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Max session duration in seconds, 0 = until interrupted",
    )

    # PiCar config
    p.add_argument(
        "--picar-config",
        default=(
            "/home/pi/SunFounder_PiCar-V/remote_control/"
            "remote_control/driver/config"
        ),
        help="PiCar config file path",
    )

    return p.parse_args()


def setup_picar(args):
    """
    Initialise PiCar hardware.

    Returns:
        (front_wheels, back_wheels) or (None, None)
    """
    if args.mode == "test":
        print("[run] mode=test — PiCar hardware skipped")
        return None, None

    try:
        import picar

        picar.setup()

        front = picar.front_wheels.Front_Wheels(
            debug=False,
            db=args.picar_config,
        )
        back = picar.back_wheels.Back_Wheels(
            debug=False,
            db=args.picar_config,
        )

        front.ready()
        back.ready()

        print("[run] picar hardware initialised")
        return front, back

    except ModuleNotFoundError:
        print("[run] WARN — picar module not found, falling back to dry run")
        return None, None

    except Exception as e:
        print(f"[run] WARN — picar setup failed ({e}), falling back to dry run")
        return None, None


def validate_args(args):
    if args.adapter == "deep" and not args.feature_model:
        print("[run] FATAL — --adapter deep requires --feature-model")
        print(
            "[run] example:\n"
            "  python3 run.py --mode test --adapter deep \\\n"
            "    --base-model tflite_models/best_model_finetuned_int8_edgetpu.tflite \\\n"
            "    --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite"
        )
        sys.exit(1)

    if args.adapter != "deep" and args.feature_model:
        print("[run] WARN — --feature-model provided but adapter is not deep; it will be ignored")


def print_config(args):
    print("=" * 60)
    print("  FCA — Feedback-based Continual Adaptation")
    print(f"  Adapter:        {args.adapter}")
    print(f"  Mode:           {args.mode}")
    print(f"  Base model:     {args.base_model}")
    print(f"  Feature model:  {args.feature_model}")
    print(f"  Paradigm:       {args.learning_paradigm}")
    print(f"  CPU fallback:   {args.cpu_base_model}")
    print(f"  Edge TPU:       {'disabled' if args.no_tpu else 'enabled'}")
    print(f"  TFLite threads: {args.tflite_threads}")
    print(f"  Max speed:      {args.max_speed}")
    print(f"  GUI:            {'disabled' if args.no_gui else 'enabled'}")
    print(f"  Trainer:        {'disabled' if args.no_trainer else 'enabled'}")
    print("=" * 60)


def main():
    args = parse_args()
    validate_args(args)
    print_config(args)

    # ─── Build components ─────────────────────────────────────────────────
    state = FCAState()
    with state.lock:
        state.max_speed = int(args.max_speed)
    teach_controller = TeachController()
    replay_buffer = ReplayBuffer(capacity=5000)
    session_logger = SessionLogger(log_dir="logs")

    front_wheels, back_wheels = setup_picar(args)

    motors = Motors(
        front_wheels=front_wheels,
        back_wheels=back_wheels,
        max_speed=args.max_speed,
        dry_run=(args.mode == "test"),
    )

    # Adaptive controller: base TFLite/TPU model + optional feature extractor + adapter
    try:
        controller = AdaptiveController(
            base_model_path=args.base_model,
            adapter_type=args.adapter,
            checkpoint_path=args.checkpoint if args.adapter != "none" else None,
            learning_rate=args.learning_rate,
            max_speed=args.max_speed,
            use_tpu=not args.no_tpu,
            cpu_base_model_path=args.cpu_base_model,
            num_threads=args.tflite_threads,
            feature_model_path=args.feature_model,
            learning_paradigm=args.learning_paradigm,
        )

    except FileNotFoundError as e:
        print(f"[run] FATAL — {e}")
        print(f"[run] check base model path: {args.base_model}")
        if args.cpu_base_model:
            print(f"[run] check CPU fallback model: {args.cpu_base_model}")
        if args.feature_model:
            print(f"[run] check feature model path: {args.feature_model}")
        sys.exit(1)

    except NotImplementedError as e:
        print(f"[run] FATAL — {e}")
        sys.exit(1)

    except Exception as e:
        print(f"[run] FATAL — controller initialisation failed: {e}")
        sys.exit(1)

    # Driving loop
    driving_loop = DrivingLoop(
        state=state,
        controller=controller,
        motors=motors,
        teach_controller=teach_controller,
        replay_buffer=replay_buffer,
        session_logger=session_logger,
        capture_src=args.capture_src,
    )

    # ─── Start threads ────────────────────────────────────────────────────
    drive_thread = threading.Thread(
        target=driving_loop.run,
        daemon=True,
        name="driving_loop",
    )
    drive_thread.start()

    trainer_thread = None

    if args.adapter == "none":
        print("[run] adapter=none — trainer thread skipped")

    elif args.no_trainer:
        print("[run] --no-trainer — background trainer skipped")

    else:
        trainer_thread = threading.Thread(
            target=run_trainer,
            args=(state, controller, replay_buffer),
            kwargs={"update_period_s": args.update_period},
            daemon=True,
            name="trainer",
        )
        trainer_thread.start()

    # Initial mode
    state.set_mode(MODE_PAUSED)
    print("[run] initial mode: PAUSED — switch to AUTOPILOT, TEACH, REVERSE_MANUAL, or DATASET_COLLECTION from GUI")

    # ─── Run server or headless blocker ───────────────────────────────────
    server_thread = None

    try:
        if args.no_gui:
            print("[run] headless mode — Ctrl+C to exit")

            if args.duration > 0:
                time.sleep(args.duration)
                state.shutdown = True
            else:
                while not state.shutdown:
                    time.sleep(1)

        else:
            app = create_app(
                state,
                controller,
                driving_loop,
                teach_controller,
                replay_buffer,
            )

            print(f"[run] GUI live at http://{args.host}:{args.port}")
            print("[run] open it in your laptop browser to drive\n")

            server_thread = threading.Thread(
                target=lambda: app.run(
                    host=args.host,
                    port=args.port,
                    threaded=True,
                    debug=False,
                    use_reloader=False,
                ),
                daemon=True,
                name="server",
            )
            server_thread.start()

            if args.duration > 0:
                time.sleep(args.duration)
                state.shutdown = True
            else:
                while not state.shutdown:
                    time.sleep(1)

    except KeyboardInterrupt:
        print("\n[run] keyboard interrupt — shutting down")
        state.shutdown = True

    # ─── Shutdown ─────────────────────────────────────────────────────────
    print("[run] cleanup...")

    try:
        motors.stop()
    except Exception as e:
        print(f"[run] motor stop failed: {e}")

    if controller.adapter is not None:
        try:
            saved = controller.save_checkpoint()
            if saved:
                print("[run] final checkpoint saved")
            else:
                print("[run] final checkpoint save skipped")
        except Exception as e:
            print(f"[run] checkpoint save failed: {e}")

    try:
        session_logger.close()
    except Exception as e:
        print(f"[run] session_logger close failed: {e}")

    drive_thread.join(timeout=2.0)

    if trainer_thread is not None:
        trainer_thread.join(timeout=2.0)

    if server_thread is not None:
        server_thread.join(timeout=0.5)

    print("[run] done")


if __name__ == "__main__":
    main()