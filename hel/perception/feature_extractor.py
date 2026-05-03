"""
hel/perception/feature_extractor.py

TFLite/EdgeTPU feature extractor wrapper.

Takes the same camera frame format as BaseModel and returns a float32 feature
vector, e.g. 512-d from dense_2.
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


class FeatureExtractor:
    # Same preprocessing as training/base model
    CROP_TOP = 80
    CROP_BOTTOM = 240
    CROP_LEFT = 0
    CROP_RIGHT = 320
    ORIGINAL_SIZE = (320, 240)
    TARGET_SIZE = (224, 224)

    INPUT_IS_BGR = False
    USE_TPU_IF_AVAILABLE = True

    def __init__(self, model_path, use_tpu=True, num_threads=4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Feature extractor model not found: {model_path}")

        self.model_path = model_path
        self.using_tpu = False

        print(f"[feature_extractor] TFLite backend: {TFLITE_BACKEND}")
        print(f"[feature_extractor] model path: {model_path}")

        self.interpreter = self._create_interpreter(
            model_path=model_path,
            use_tpu=use_tpu,
            num_threads=num_threads,
        )

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        if len(self.output_details) != 1:
            print(f"[feature_extractor] WARN — expected 1 output, got {len(self.output_details)}")

        print(f"[feature_extractor] runtime: {'Edge TPU' if self.using_tpu else 'CPU'}")

        print("[feature_extractor] input details:")
        for d in self.input_details:
            print(" ", d["name"], d["shape"], d["dtype"], d["quantization"])

        print("[feature_extractor] output details:")
        for d in self.output_details:
            print(" ", d["name"], d["shape"], d["dtype"], d["quantization"])

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

    def _create_interpreter(self, model_path, use_tpu=True, num_threads=4):
        force_cpu = os.environ.get("FORCE_CPU", "0") == "1"

        if use_tpu and not force_cpu:
            delegate_lib = self._get_edgetpu_delegate_library()

            if delegate_lib:
                try:
                    print(f"[feature_extractor] trying Edge TPU delegate: {delegate_lib}")
                    delegate = load_delegate(delegate_lib)

                    interpreter = Interpreter(
                        model_path=model_path,
                        experimental_delegates=[delegate],
                    )
                    interpreter.allocate_tensors()

                    self.using_tpu = True
                    print("[feature_extractor] using Edge TPU")
                    return interpreter

                except Exception as e:
                    print("[feature_extractor] Edge TPU failed:")
                    print("[feature_extractor]", repr(e))
                    print("[feature_extractor] falling back to CPU...")

        print("[feature_extractor] using CPU TFLite runtime")
        interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        interpreter.allocate_tensors()

        self.using_tpu = False
        return interpreter

    def preprocess(self, image):
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

    def extract(self, image):
        """
        Returns:
            feature: np.ndarray shape (D,), dtype float32
        """
        x = self.preprocess(image)
        input_detail = self.input_details[0]
        x = self._quantize_input(x, input_detail)

        self.interpreter.set_tensor(input_detail["index"], x)
        self.interpreter.invoke()

        out_detail = self.output_details[0]
        raw = self.interpreter.get_tensor(out_detail["index"])
        feat = self._dequantize_output(raw, out_detail)

        feat = np.asarray(feat, dtype=np.float32).reshape(-1)

        return feat