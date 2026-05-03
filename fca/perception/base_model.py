"""
fca/perception/base_model.py — Wraps the existing INT8/EdgeTPU TFLite MobileNetV2.

Supports:
  - tflite_runtime first, TensorFlow fallback
  - Edge TPU delegate if available
  - CPU fallback for non-EdgeTPU .tflite models
  - predict_raw() exposes angle_probs, angle_norm, speed_raw for adapter losses

Important:
  If model_path is an EdgeTPU-compiled .tflite file, CPU fallback may fail.
  Keep both files if you want fallback:
      best_model_finetuned_int8_edgetpu.tflite
      best_model_finetuned_int8.tflite
"""
import os
import platform

import cv2
import numpy as np


try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    TFLITE_BACKEND = "tflite_runtime"
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    load_delegate = tf.lite.experimental.load_delegate
    TFLITE_BACKEND = "tensorflow"


class BaseModel:
    """INT8/EdgeTPU TFLite MobileNetV2 dual-head: 17 angle classes + binary speed."""

    # Preprocessing
    CROP_TOP = 80
    CROP_BOTTOM = 240
    CROP_LEFT = 0
    CROP_RIGHT = 320
    ORIGINAL_SIZE = (320, 240)  # cv2 format: width, height
    TARGET_SIZE = (224, 224)

    NUM_ANGLE_CLASSES = 17
    ANGLE_VALUES_NORM = np.linspace(0.0, 1.0, NUM_ANGLE_CLASSES).astype(np.float32)
    ANGLE_CLASSES_CAR = np.arange(NUM_ANGLE_CLASSES).astype(np.float32) * 5.0 + 50.0

    # Match your working PiCar camera pipeline.
    # Set True only if frames come directly from raw cv2.VideoCapture BGR.
    INPUT_IS_BGR = False

    USE_EXPECTED_ANGLE = True

    # TPU options
    USE_TPU_IF_AVAILABLE = True

    def __init__(self, model_path, use_tpu=None, cpu_model_path=None, num_threads=4):
        """
        Args:
            model_path:
                Main model path. Can be EdgeTPU-compiled or normal INT8 TFLite.

            use_tpu:
                True/False override. If None, uses USE_TPU_IF_AVAILABLE.

            cpu_model_path:
                Optional non-compiled INT8 model for CPU fallback if model_path is EdgeTPU-compiled.

            num_threads:
                CPU TFLite thread count.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        self.model_path = model_path
        self.cpu_model_path = cpu_model_path
        self.num_threads = num_threads

        self.using_tpu = False

        if use_tpu is None:
            use_tpu = self.USE_TPU_IF_AVAILABLE

        print(f"[base_model] TFLite backend: {TFLITE_BACKEND}")
        print(f"[base_model] model path: {model_path}")

        self.interpreter = self._create_interpreter(
            model_path=model_path,
            use_tpu=use_tpu,
            cpu_model_path=cpu_model_path,
            num_threads=num_threads,
        )

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(f"[base_model] loaded: {model_path}")
        print(f"[base_model] runtime: {'Edge TPU' if self.using_tpu else 'CPU'}")

        print("[base_model] input details:")
        for d in self.input_details:
            print(
                " ",
                d["name"],
                d["shape"],
                d["dtype"],
                d["quantization"],
            )

        print("[base_model] output details:")
        for d in self.output_details:
            print(
                " ",
                d["name"],
                d["shape"],
                d["dtype"],
                d["quantization"],
            )

    # ─── Interpreter creation ─────────────────────────────────────────────
    @staticmethod
    def _get_edgetpu_delegate_library():
        system = platform.system().lower()

        if system == "linux":
            return "libedgetpu.so.1"

        if system == "darwin":
            return "libedgetpu.1.dylib"

        if system == "windows":
            return "edgetpu.dll"

        return None

    def _create_interpreter(self, model_path, use_tpu=True, cpu_model_path=None, num_threads=4):
        """
        Priority:
          1. Edge TPU delegate if enabled and available.
          2. CPU fallback.

        If the main model is EdgeTPU-compiled, CPU fallback on that same file may fail.
        In that case, pass cpu_model_path to a non-compiled INT8 model.
        """
        force_cpu = os.environ.get("FORCE_CPU", "0") == "1"

        if use_tpu and not force_cpu:
            delegate_lib = self._get_edgetpu_delegate_library()

            if delegate_lib is not None:
                try:
                    print(f"[base_model] trying Edge TPU delegate: {delegate_lib}")
                    delegate = load_delegate(delegate_lib)

                    interpreter = Interpreter(
                        model_path=model_path,
                        experimental_delegates=[delegate],
                    )
                    interpreter.allocate_tensors()

                    self.using_tpu = True
                    print("[base_model] using Edge TPU")
                    return interpreter

                except Exception as e:
                    print("[base_model] Edge TPU failed:")
                    print("[base_model]", repr(e))
                    print("[base_model] falling back to CPU...")
            else:
                print("[base_model] no known Edge TPU delegate for this platform.")
                print("[base_model] falling back to CPU...")
        else:
            print("[base_model] FORCE_CPU=1 or use_tpu=False. Using CPU.")

        # CPU fallback path
        fallback_path = cpu_model_path if cpu_model_path is not None else model_path

        if not os.path.exists(fallback_path):
            raise FileNotFoundError(f"CPU fallback model not found: {fallback_path}")

        try:
            print(f"[base_model] trying CPU TFLite model: {fallback_path}")
            interpreter = Interpreter(
                model_path=fallback_path,
                num_threads=num_threads,
            )
            interpreter.allocate_tensors()

            self.using_tpu = False
            print("[base_model] using CPU TFLite runtime")
            return interpreter

        except Exception as e:
            raise RuntimeError(
                "Could not load model on CPU.\n"
                "If the main file is EdgeTPU-compiled, CPU fallback may not work.\n"
                "Pass a non-compiled INT8 model as cpu_model_path.\n\n"
                f"CPU error: {repr(e)}"
            )

    # ─── Preprocessing ────────────────────────────────────────────────────
    def preprocess(self, image):
        """Camera frame → (1, 224, 224, 3) float32 in [-1, 1]."""
        if image is None:
            raise ValueError("Input image is None.")

        image = np.asarray(image)

        if image.ndim == 4:
            image = image[0]

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3), got {image.shape}")

        image = image.astype(np.float32)

        if image.max() <= 1.5:
            image = image * 255.0

        if self.INPUT_IS_BGR:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)

        if image.shape[1] != 320 or image.shape[0] != 240:
            image = cv2.resize(
                image,
                self.ORIGINAL_SIZE,
                interpolation=cv2.INTER_AREA,
            )

        image = image[
            self.CROP_TOP:self.CROP_BOTTOM,
            self.CROP_LEFT:self.CROP_RIGHT,
            :,
        ]

        image = cv2.resize(
            image,
            self.TARGET_SIZE,
            interpolation=cv2.INTER_AREA,
        )

        image = image.astype(np.float32)
        image = (image / 127.5) - 1.0

        return np.expand_dims(image, axis=0).astype(np.float32)

    # ─── Quantization helpers ─────────────────────────────────────────────
    @staticmethod
    def _quantize_input(x_float, input_detail):
        dtype = input_detail["dtype"]

        if dtype == np.float32:
            return x_float.astype(np.float32)

        scale, zero_point = input_detail["quantization"]

        if scale == 0:
            raise ValueError("Invalid input quantization scale = 0.")

        x_q = np.round(x_float / scale + zero_point)

        if dtype == np.int8:
            return np.clip(x_q, -128, 127).astype(np.int8)

        if dtype == np.uint8:
            return np.clip(x_q, 0, 255).astype(np.uint8)

        raise TypeError(f"Unsupported input dtype: {dtype}")

    @staticmethod
    def _dequantize_output(y, output_detail):
        dtype = output_detail["dtype"]

        if dtype == np.float32:
            return y.astype(np.float32)

        scale, zero_point = output_detail["quantization"]

        if scale == 0:
            return y.astype(np.float32)

        return scale * (y.astype(np.float32) - zero_point)

    def _extract_outputs(self):
        """Pull outputs from interpreter. Returns (angle_probs, speed_raw)."""
        outputs = []

        for out_detail in self.output_details:
            raw = self.interpreter.get_tensor(out_detail["index"])
            out = self._dequantize_output(raw, out_detail)
            outputs.append(out)

        angle_probs = None
        speed_raw = None

        for out in outputs:
            if out.ndim >= 2 and out.shape[-1] == self.NUM_ANGLE_CLASSES:
                angle_probs = out[0].astype(np.float32)
            elif out.size == 1:
                speed_raw = float(out.reshape(-1)[0])

        if angle_probs is None or speed_raw is None:
            raise RuntimeError(
                f"Could not identify outputs. Shapes: {[o.shape for o in outputs]}"
            )

        # Normalise softmax probs. INT8 dequantization can create small errors.
        angle_probs = np.maximum(angle_probs, 0.0)
        s = float(angle_probs.sum())

        if s > 0:
            angle_probs = angle_probs / s
        else:
            angle_probs = (
                np.ones(self.NUM_ANGLE_CLASSES, dtype=np.float32)
                / self.NUM_ANGLE_CLASSES
            )

        speed_raw = float(np.clip(speed_raw, 0.0, 1.0))

        return angle_probs, speed_raw

    # ─── Public API ───────────────────────────────────────────────────────
    def predict_raw(self, image):
        """
        Run inference.

        Returns:
            angle_probs: ndarray (17,) softmax over angle classes
            angle_norm:  float in [0, 1]
            speed_raw:   float in [0, 1] sigmoid speed probability
        """
        x = self.preprocess(image)
        x = self._quantize_input(x, self.input_details[0])

        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()

        angle_probs, speed_raw = self._extract_outputs()

        if self.USE_EXPECTED_ANGLE:
            angle_norm = float(angle_probs @ self.ANGLE_VALUES_NORM)
            angle_norm = round(angle_norm * 16) / 16
            angle_norm = float(np.clip(angle_norm, 0.0, 1.0))
        else:
            angle_class = int(np.argmax(angle_probs))
            angle_norm = angle_class / 16.0

        return angle_probs, angle_norm, speed_raw

    def predict(self, image, max_speed=35):
        """
        Compatibility method.

        Returns:
            angle_car: 50..130
            speed_car: 0 or max_speed
        """
        _angle_probs, angle_norm, speed_raw = self.predict_raw(image)

        angle_car = self.angle_norm_to_car(angle_norm)
        speed_car = self.speed_prob_to_car(speed_raw, max_speed=max_speed)

        return angle_car, speed_car

    # ─── Conversion helpers ───────────────────────────────────────────────
    @staticmethod
    def angle_norm_to_car(angle_norm):
        return float(np.clip(angle_norm, 0.0, 1.0)) * 80.0 + 50.0

    @staticmethod
    def car_to_angle_norm(angle_car):
        return float(np.clip((float(angle_car) - 50.0) / 80.0, 0.0, 1.0))

    @staticmethod
    def speed_prob_to_car(speed_prob, max_speed=35):
        return int(float(speed_prob) >= 0.5) * max_speed