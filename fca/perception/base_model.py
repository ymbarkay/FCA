"""
fca/perception/base_model.py — Frozen inference model wrapper.

Supports frozen steering/speed models saved as:
    - .tflite      via tflite_runtime first, TensorFlow fallback
    - .keras/.h5   via tf.keras when TensorFlow is installed

The runtime API stays the same regardless of backend:
    - predict_raw() -> angle_probs, angle_norm, speed_raw
    - predict()     -> angle_car, speed_car

Important:
    If model_path is an EdgeTPU-compiled .tflite file, CPU fallback may fail.
    Keep both files if you want fallback:
            best_model_finetuned_int8_edgetpu.tflite
            best_model_finetuned_int8.tflite
"""
import importlib
import os
import platform

import cv2
import numpy as np


try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    TFLITE_BACKEND = "tflite_runtime"
except ImportError:
    try:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
        load_delegate = tf.lite.experimental.load_delegate
        TFLITE_BACKEND = "tensorflow"
    except ImportError:
        Interpreter = None
        load_delegate = None
        TFLITE_BACKEND = "unavailable"


class BaseModel:
    """Frozen dual-head model: 17 angle classes + binary speed."""

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

    SUPPORTED_EXTENSIONS = (".tflite", ".keras", ".h5")

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
            raise FileNotFoundError(f"Frozen model not found: {model_path}")

        self.model_path = model_path
        self.cpu_model_path = cpu_model_path
        self.num_threads = num_threads
        self.model_format = self._detect_model_format(model_path)
        self.backend_name = self.model_format

        self.using_tpu = False
        self.keras_model = None
        self.interpreter = None
        self.input_details = []
        self.output_details = []
        self.target_size = self.TARGET_SIZE

        if use_tpu is None:
            use_tpu = self.USE_TPU_IF_AVAILABLE

        print(f"[base_model] model path: {model_path}")

        if self.model_format == ".tflite":
            if Interpreter is None:
                raise ImportError(
                    "No TFLite runtime available. Install tflite-runtime or tensorflow."
                )

            print(f"[base_model] TFLite backend: {TFLITE_BACKEND}")
            self.interpreter = self._create_interpreter(
                model_path=model_path,
                use_tpu=use_tpu,
                cpu_model_path=cpu_model_path,
                num_threads=num_threads,
            )

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.target_size = self._detect_target_size_from_tflite()
            self.backend_name = "edgetpu" if self.using_tpu else TFLITE_BACKEND

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
        else:
            self.keras_model = self._load_keras_model(model_path)
            self.target_size = self._detect_target_size_from_keras()
            self.backend_name = "tensorflow.keras"
            print(f"[base_model] loaded keras model: {model_path}")
            print(f"[base_model] runtime: {self.backend_name}")
            print(f"[base_model] input size: {self.target_size[0]}x{self.target_size[1]}")

    @classmethod
    def _detect_model_format(cls, model_path):
        ext = os.path.splitext(model_path)[1].lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported frozen model format '{ext}'. "
                f"Expected one of: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )
        return ext

    @staticmethod
    def _load_tensorflow():
        try:
            return importlib.import_module("tensorflow")
        except ImportError as e:
            raise ImportError(
                "TensorFlow is required for .keras/.h5 frozen models. "
                "Install tensorflow in the runtime environment to use this backend."
            ) from e

    def _load_keras_model(self, model_path):
        tf = self._load_tensorflow()
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            raise RuntimeError(f"Could not load keras model '{model_path}': {e}") from e

    def _detect_target_size_from_tflite(self):
        if not self.input_details:
            return self.TARGET_SIZE

        shape = tuple(int(v) for v in self.input_details[0].get("shape", []))
        if len(shape) == 4 and shape[1] > 0 and shape[2] > 0:
            return (int(shape[2]), int(shape[1]))
        return self.TARGET_SIZE

    def _detect_target_size_from_keras(self):
        input_shape = getattr(self.keras_model, "input_shape", None)
        if isinstance(input_shape, list) and input_shape:
            input_shape = input_shape[0]

        if input_shape is None or len(input_shape) < 3:
            return self.TARGET_SIZE

        height = input_shape[1]
        width = input_shape[2]
        if height is None or width is None:
            return self.TARGET_SIZE

        return (int(width), int(height))

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
            self.target_size,
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

    @staticmethod
    def _sigmoid(x):
        x = float(np.clip(x, -50.0, 50.0))
        return 1.0 / (1.0 + np.exp(-x))

    def _normalise_angle_output(self, values):
        angle_logits = np.asarray(values, dtype=np.float32).reshape(-1)
        if angle_logits.size != self.NUM_ANGLE_CLASSES:
            return None

        if np.all(np.isfinite(angle_logits)) and np.all(angle_logits >= 0.0):
            s = float(angle_logits.sum())
            if 0.99 <= s <= 1.01:
                probs = angle_logits
            else:
                probs = None
        else:
            probs = None

        if probs is None:
            shifted = angle_logits - float(np.max(angle_logits))
            exp_logits = np.exp(np.clip(shifted, -50.0, 50.0))
            s = float(exp_logits.sum())
            if s > 0:
                probs = exp_logits / s
            else:
                probs = np.ones(self.NUM_ANGLE_CLASSES, dtype=np.float32) / self.NUM_ANGLE_CLASSES

        probs = np.maximum(probs, 0.0)
        s = float(probs.sum())
        if s > 0:
            return (probs / s).astype(np.float32)
        return np.ones(self.NUM_ANGLE_CLASSES, dtype=np.float32) / self.NUM_ANGLE_CLASSES

    def _normalise_speed_output(self, value):
        speed_value = float(np.asarray(value).reshape(-1)[0])
        if 0.0 <= speed_value <= 1.0:
            return speed_value
        return self._sigmoid(speed_value)

    def _collect_output_arrays(self, outputs):
        arrays = []

        def visit(obj):
            if obj is None:
                return
            if isinstance(obj, dict):
                for value in obj.values():
                    visit(value)
                return
            if isinstance(obj, (list, tuple)):
                for value in obj:
                    visit(value)
                return

            value = obj
            if hasattr(value, "numpy"):
                value = value.numpy()
            arrays.append(np.asarray(value))

        visit(outputs)
        return arrays

    def _parse_outputs(self, outputs):
        arrays = self._collect_output_arrays(outputs)
        angle_probs = None
        speed_raw = None

        for arr in arrays:
            if arr.size == 0:
                continue

            flat = arr
            if flat.ndim >= 2 and flat.shape[0] == 1:
                flat = flat[0]

            flat = np.asarray(flat)

            if flat.ndim == 1 and flat.shape[-1] == self.NUM_ANGLE_CLASSES + 1:
                if angle_probs is None:
                    angle_probs = self._normalise_angle_output(flat[:self.NUM_ANGLE_CLASSES])
                if speed_raw is None:
                    speed_raw = self._normalise_speed_output(flat[self.NUM_ANGLE_CLASSES])
                continue

            if flat.ndim == 1 and flat.shape[-1] == self.NUM_ANGLE_CLASSES and angle_probs is None:
                angle_probs = self._normalise_angle_output(flat)
                continue

            if flat.size == 1 and speed_raw is None:
                speed_raw = self._normalise_speed_output(flat)

        if angle_probs is None or speed_raw is None:
            raise RuntimeError(
                "Could not identify frozen model outputs. "
                f"Observed shapes: {[tuple(a.shape) for a in arrays]}"
            )

        return angle_probs, float(np.clip(speed_raw, 0.0, 1.0))

    def _extract_tflite_outputs(self):
        outputs = []
        for out_detail in self.output_details:
            raw = self.interpreter.get_tensor(out_detail["index"])
            outputs.append(self._dequantize_output(raw, out_detail))
        return self._parse_outputs(outputs)

    def _extract_keras_outputs(self, x):
        outputs = self.keras_model(x, training=False)
        return self._parse_outputs(outputs)

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
        if self.model_format == ".tflite":
            x_q = self._quantize_input(x, self.input_details[0])
            self.interpreter.set_tensor(self.input_details[0]["index"], x_q)
            self.interpreter.invoke()
            angle_probs, speed_raw = self._extract_tflite_outputs()
        else:
            angle_probs, speed_raw = self._extract_keras_outputs(x.astype(np.float32))

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