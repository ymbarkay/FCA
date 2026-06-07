"""
fca/learning/controller.py

Controller with two runtime paths:
  - scalar/none: legacy base-model + residual adapter path
  - deep: EdgeTPU feature extractor + trainable PyTorch policy head (CPU)
"""
import os
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from fca.learning.paradigms import angle_expected_value, get_learning_paradigm, list_learning_paradigm_snapshots
from fca.perception.base_model import BaseModel
from fca.learning.adapter_scalar import ScalarAdapter, ANGLE_DELTA_BOUND


SMOOTHING_BY_MODE = {
    "AUTOPILOT": 0.65,
    "TEACH": 0.0,
    "REVERSE_MANUAL": 0.0,
    "PAUSED": 0.0,
    "DATASET_COLLECTION": 0.0,
}


class AdaptiveController:
    ANGLE_MIN_CAR = 50.0
    ANGLE_MAX_CAR = 130.0
    NUM_ANGLE_CLASSES = 17
    INTENT_NAMES = ("stop", "left", "straight", "right")

    # Online-EWC defaults for anti-forgetting.
    EWC_ENABLED = True
    EWC_LAMBDA = 1.5e-3
    EWC_FISHER_DECAY = 0.99
    EWC_WARMUP_STEPS = 8
    VALIDATED_REINFORCEMENT_ENABLED = True
    VALIDATED_FISHER_BOOST = 0.02
    VALIDATED_STABLE_QUANTILE = 0.60
    VALIDATED_FISHER_MAX = 10.0
    VALIDATED_REINFORCEMENT_COUNT_GAIN = 0.15
    VALIDATED_REINFORCEMENT_MAX_MULT = 3.0
    VALIDATED_RETENTION_EWC_GAIN = 0.12
    VALIDATED_RETENTION_ANCHOR_GAIN = 0.16
    VALIDATED_RETENTION_MAX_MULT = 2.5
    FIXED_ANCHOR_ENABLED = True
    FIXED_ANCHOR_LAMBDA = 4e-4
    FIXED_ANCHOR_WARMUP_STEPS = 8
    HISTORICAL_GRADIENT_ENABLED = True
    HISTORICAL_GRADIENT_BLEND = 0.30
    HISTORICAL_GRADIENT_MOMENTUM = 0.90
    MOE_BALANCING_ENABLED = True
    MOE_BALANCE_WARMUP_STEPS = 4
    MOE_LOAD_BALANCE_WEIGHT = 0.03
    MOE_GATE_ENTROPY_WEIGHT = 0.004
    INTENT_ROUTING_ENABLED = True
    INTENT_LOSS_WEIGHT = 0.22
    INTENT_EXPERT_SUPERVISION_ENABLED = False
    INTENT_EXPERT_SUPERVISION_TEACH_ONLY = False
    INTENT_EXPERT_DIRECT_LOSS_WEIGHT = 1.0
    INTENT_EXPERT_GATE_LOSS_WEIGHT = 0.35
    INTENT_EXPERT_DIRECT_LOSS_WEIGHT_BY_INTENT = {}
    INTENT_EXPERT_GATE_LOSS_WEIGHT_BY_INTENT = {}
    INTENT_STOP_THRESHOLD = 0.5
    INTENT_CENTER_MARGIN_NORM = 0.12
    INFERENCE_GATE_TEMPERATURE = 1.0
    TRAIN_GATE_TEMPERATURE = 1.0
    TEACH_GATE_TEMPERATURE = 1.0
    REHEARSAL_GATE_TEMPERATURE = 1.0
    TEACH_GATE_TEMPERATURE_BY_INTENT = {}
    TEACH_LOAD_BALANCE_WEIGHT_MULT = 1.0
    TEACH_GATE_ENTROPY_WEIGHT_MULT = 1.0
    REHEARSAL_LOAD_BALANCE_WEIGHT_MULT = 1.0
    REHEARSAL_GATE_ENTROPY_WEIGHT_MULT = 1.0
    TEACH_LOAD_BALANCE_WEIGHT_MULT_BY_INTENT = {}
    TEACH_GATE_ENTROPY_WEIGHT_MULT_BY_INTENT = {}
    GATE_LR_SCALE = 1.0
    FOCUSED_REHEARSAL_BATCH_SCALE = 1.0
    FOCUSED_REHEARSAL_BATCH_SCALE_BY_INTENT = {}
    FOCUSED_TARGET_REPEAT_SCALE_BY_INTENT = {}
    TEACH_FOCUSED_STEP_SCALE_BY_INTENT = {}
    TEACH_FOCUSED_LR_SCALE_BY_INTENT = {}
    TEACH_FOCUSED_MAX_LR_MULTIPLIER = 5.25
    TEACH_FOCUSED_MAX_LR_MULTIPLIER_BY_INTENT = {}
    REHEARSAL_BATCH_SIZE_SCALE = 1.0
    REHEARSAL_STEPS_SCALE = 1.0
    DEEP_ADAPTIVE_CLASS_WEIGHTING_ENABLED = False
    DEEP_CLASS_STATS_MOMENTUM = 0.97
    DEEP_CLASS_FREQ_GAIN = 0.55
    DEEP_CLASS_HARDNESS_GAIN = 0.75
    DEEP_CLASS_WEIGHT_MIN = 0.65
    DEEP_CLASS_WEIGHT_MAX = 2.6
    DEFAULT_DEEP_LEARNING_PARADIGM = "moe_v4_intent_routing"
    CONTEXT_TASK_ROUTING_ENABLED = False
    CONTEXT_TASK_LOSS_WEIGHT = 0.18

    def __init__(
        self,
        base_model_path,
        adapter_type="scalar",
        checkpoint_path=None,
        learning_rate=1e-3,
        max_speed=35,
        device="cpu",
        use_tpu=True,
        cpu_base_model_path=None,
        num_threads=4,
        feature_model_path=None,
        learning_paradigm=None,
    ):
        self.device = device
        self.max_speed = max_speed
        self.adapter_type = adapter_type
        self.feature_extractor = None
        self.base_model = None
        self.use_tpu = use_tpu
        self.cpu_base_model_path = cpu_base_model_path
        self.num_threads = num_threads
        self.feature_model_path = feature_model_path
        self.initial_base_model_path = base_model_path
        self.base_model_path = base_model_path
        self.configured_frozen_model_path = str(base_model_path or "")
        self.inference_backend = "main" if adapter_type != "none" else "frozen"
        self.learning_paradigm = ""
        self.learning_paradigm_label = ""
        self.learning_paradigm_description = ""
        self._requested_learning_paradigm = (
            str(learning_paradigm or self.DEFAULT_DEEP_LEARNING_PARADIGM).strip()
            or self.DEFAULT_DEEP_LEARNING_PARADIGM
        )

        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = self._determine_checkpoint_dir(checkpoint_path)
        self.learning_rate = learning_rate
        self.weights_lock = threading.Lock()
        self._validated_save_count = 0
        self._ewc_steps = 0
        self._ewc_fisher = {}
        self._ewc_theta_star = {}
        self._fixed_anchor = {}
        self._fixed_anchor_active = False
        self._manual_checkpoint_locked = False
        self._last_manual_checkpoint_saved_at = None
        self._historical_grad_ema = {}
        self._deep_class_count_ema = None
        self._deep_class_loss_ema = None
        self._last_training_metrics = self._empty_training_metrics()

        if adapter_type == "scalar":
            # Legacy scalar mode still uses base model inference.
            self.base_model = BaseModel(
                base_model_path,
                use_tpu=use_tpu,
                cpu_model_path=cpu_base_model_path,
                num_threads=num_threads,
            )
            self.adapter = ScalarAdapter().to(device)
            self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=learning_rate)

        elif adapter_type == "deep":
            if feature_model_path is None:
                raise ValueError(
                    "adapter_type='deep' requires feature_model_path. "
                    "Pass --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite"
                )

            from fca.perception.feature_extractor import FeatureExtractor

            self.feature_extractor = FeatureExtractor(
                feature_model_path,
                use_tpu=use_tpu,
                num_threads=num_threads,
            )

            (
                _learning_paradigm_spec,
                self.adapter,
                self.optimizer,
            ) = self._build_deep_adapter(self._requested_learning_paradigm)

        elif adapter_type == "none":
            # Base-model only legacy mode.
            self.base_model = BaseModel(
                base_model_path,
                use_tpu=use_tpu,
                cpu_model_path=cpu_base_model_path,
                num_threads=num_threads,
            )
            self.adapter = None
            self.optimizer = None

        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

        checkpoint_loaded = False
        if self.adapter is not None and checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint_loaded = self._load_checkpoint(checkpoint_path)

        if self.adapter is not None:
            self._reset_ewc_state()
            self._reset_historical_grad_state()
            if checkpoint_loaded and self.FIXED_ANCHOR_ENABLED:
                self._refresh_fixed_anchor()

        self._smoothed_angle_car = None
        self._last_mode_for_smoothing = None

    def _load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                state = checkpoint["model_state"]
                self._validated_save_count = int(max(0, checkpoint.get("validated_save_count", 0)))
            else:
                state = checkpoint
                self._validated_save_count = 0

            inferred_paradigm = self._infer_learning_paradigm(checkpoint, state)
            if inferred_paradigm and inferred_paradigm != self.learning_paradigm:
                self._replace_deep_adapter(inferred_paradigm)

            load_kind = self._load_adapter_state(state)
            if self.adapter is not None:
                self.adapter.eval()
            print(f"[controller] loaded adapter from {checkpoint_path} ({load_kind})")
            return True

        except Exception as e:
            print(f"[controller] WARN — could not load adapter: {e}")
            return False

    def set_max_speed(self, max_speed):
        with self.weights_lock:
            self.max_speed = int(max(0, min(100, int(max_speed))))
        print(f"[controller] max_speed -> {self.max_speed}")

    def _determine_checkpoint_dir(self, checkpoint_path=None):
        candidate = checkpoint_path if checkpoint_path is not None else self.checkpoint_path
        if candidate:
            directory = os.path.dirname(str(candidate))
            if directory:
                return directory
        return "checkpoints"

    def _deep_paradigm_controller_defaults(self):
        return {
            "MOE_BALANCING_ENABLED": True,
            "MOE_LOAD_BALANCE_WEIGHT": 0.03,
            "MOE_GATE_ENTROPY_WEIGHT": 0.004,
            "INTENT_ROUTING_ENABLED": True,
            "INTENT_LOSS_WEIGHT": 0.22,
            "INTENT_EXPERT_SUPERVISION_ENABLED": False,
            "INTENT_EXPERT_SUPERVISION_TEACH_ONLY": False,
            "INTENT_EXPERT_DIRECT_LOSS_WEIGHT": 1.0,
            "INTENT_EXPERT_GATE_LOSS_WEIGHT": 0.35,
            "INTENT_EXPERT_DIRECT_LOSS_WEIGHT_BY_INTENT": {},
            "INTENT_EXPERT_GATE_LOSS_WEIGHT_BY_INTENT": {},
            "INFERENCE_GATE_TEMPERATURE": 1.0,
            "TRAIN_GATE_TEMPERATURE": 1.0,
            "TEACH_GATE_TEMPERATURE": 1.0,
            "REHEARSAL_GATE_TEMPERATURE": 1.0,
            "TEACH_GATE_TEMPERATURE_BY_INTENT": {},
            "TEACH_LOAD_BALANCE_WEIGHT_MULT": 1.0,
            "TEACH_GATE_ENTROPY_WEIGHT_MULT": 1.0,
            "REHEARSAL_LOAD_BALANCE_WEIGHT_MULT": 1.0,
            "REHEARSAL_GATE_ENTROPY_WEIGHT_MULT": 1.0,
            "TEACH_LOAD_BALANCE_WEIGHT_MULT_BY_INTENT": {},
            "TEACH_GATE_ENTROPY_WEIGHT_MULT_BY_INTENT": {},
            "GATE_LR_SCALE": 1.0,
            "FOCUSED_REHEARSAL_BATCH_SCALE": 1.0,
            "FOCUSED_REHEARSAL_BATCH_SCALE_BY_INTENT": {},
            "FOCUSED_TARGET_REPEAT_SCALE_BY_INTENT": {},
            "TEACH_FOCUSED_STEP_SCALE_BY_INTENT": {},
            "TEACH_FOCUSED_LR_SCALE_BY_INTENT": {},
            "TEACH_FOCUSED_MAX_LR_MULTIPLIER": 5.25,
            "TEACH_FOCUSED_MAX_LR_MULTIPLIER_BY_INTENT": {},
            "REHEARSAL_BATCH_SIZE_SCALE": 1.0,
            "REHEARSAL_STEPS_SCALE": 1.0,
            "DEEP_ADAPTIVE_CLASS_WEIGHTING_ENABLED": False,
            "CONTEXT_TASK_ROUTING_ENABLED": False,
            "CONTEXT_TASK_LOSS_WEIGHT": 0.18,
        }

    def list_available_learning_paradigms(self):
        return list_learning_paradigm_snapshots()

    def _configure_deep_learning_paradigm(self, paradigm_id):
        selected_id = (
            str(paradigm_id or self.DEFAULT_DEEP_LEARNING_PARADIGM).strip()
            or self.DEFAULT_DEEP_LEARNING_PARADIGM
        )
        spec = get_learning_paradigm(selected_id)

        for name, value in self._deep_paradigm_controller_defaults().items():
            setattr(self, name, value)
        for name, value in spec.controller_overrides.items():
            setattr(self, name, value)

        self.learning_paradigm = spec.paradigm_id
        self.learning_paradigm_label = spec.label
        self.learning_paradigm_description = spec.description
        return spec

    def _build_deep_adapter(self, paradigm_id):
        spec = self._configure_deep_learning_paradigm(paradigm_id)
        kwargs = {
            "feature_dim": 512,
            "hidden1": 256,
            "hidden2": 128,
            "num_angle_classes": self.NUM_ANGLE_CLASSES,
        }

        if spec.family == "moe":
            kwargs["num_experts"] = 4
        if (
            bool(spec.controller_overrides.get("INTENT_ROUTING_ENABLED"))
            or bool(spec.controller_overrides.get("CONTEXT_TASK_ROUTING_ENABLED"))
        ):
            kwargs["num_intents"] = 4

        adapter = spec.build_adapter(**kwargs).to(self.device)
        optimizer = torch.optim.AdamW(
            self._deep_optimizer_param_groups(adapter),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        return spec, adapter, optimizer

    def _deep_optimizer_param_groups(self, adapter):
        gate_scale = float(max(0.05, getattr(self, "GATE_LR_SCALE", 1.0)))
        if abs(gate_scale - 1.0) < 1e-6:
            return adapter.parameters()

        gate_keywords = (
            "gate",
            "intent_head",
            "intent_to_gate",
            "task_head",
            "context_to_gate",
            "task_to_gate",
            "intent_gate_scale",
        )
        gate_params = []
        other_params = []
        for name, param in adapter.named_parameters():
            if not param.requires_grad:
                continue
            if any(keyword in name for keyword in gate_keywords):
                gate_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if other_params:
            param_groups.append({"params": other_params})
        if gate_params:
            param_groups.append({"params": gate_params, "lr": self.learning_rate * gate_scale})
        return param_groups or adapter.parameters()

    def _replace_deep_adapter(self, paradigm_id):
        _spec, adapter, optimizer = self._build_deep_adapter(paradigm_id)
        self.adapter = adapter
        self.optimizer = optimizer
        self._deep_class_count_ema = None
        self._deep_class_loss_ema = None
        self._last_training_metrics = self._empty_training_metrics()

    def _default_checkpoint_path_for_learning_paradigm(self, paradigm_id):
        return os.path.abspath(
            os.path.join(self._determine_checkpoint_dir(), f"{paradigm_id}.pt")
        )

    def _infer_learning_paradigm(self, checkpoint, state):
        if self.adapter_type != "deep":
            return ""

        raw_id = ""
        if isinstance(checkpoint, dict):
            raw_id = str(checkpoint.get("paradigm_id", "")).strip()
        if raw_id:
            try:
                return get_learning_paradigm(raw_id).paradigm_id
            except ValueError:
                pass

        if isinstance(state, dict):
            if "intent_gate_scale" in state:
                return "moe_v6_intent_supervised_plastic"
            if "task_head.weight" in state or "task_to_gate.weight" in state:
                return "moe_v5_contextual_task_route"
            if "intent_head.weight" in state or "intent_to_gate.weight" in state:
                return "moe_v4_intent_routing"
            if "experts.0.hidden.weight" in state:
                if self.learning_paradigm in {
                    "moe_v1_baseline",
                    "moe_v2_gate_balance",
                    "moe_v3_adaptive_class_weight",
                }:
                    return self.learning_paradigm
                return "moe_v2_gate_balance"
            if "net.4.weight" in state:
                return "dense_single_head"

        return ""

    def list_available_policy_heads(self):
        directory = self._determine_checkpoint_dir()
        if not os.path.isdir(directory):
            return []

        try:
            names = [
                entry.name
                for entry in os.scandir(directory)
                if entry.is_file() and entry.name.lower().endswith(".pt")
            ]
        except OSError:
            return []

        names.sort(key=str.lower)
        return names

    def switch_policy_head(self, checkpoint_name_or_path):
        if self.adapter is None:
            raise ValueError("No online model is available in this run.")

        raw_path = str(checkpoint_name_or_path or "").strip()
        if not raw_path:
            raise ValueError("Policy head selection is required.")

        if os.path.isabs(raw_path):
            checkpoint_path = raw_path
        else:
            checkpoint_path = os.path.join(self._determine_checkpoint_dir(), raw_path)

        checkpoint_path = os.path.abspath(checkpoint_path)
        if not checkpoint_path.lower().endswith(".pt"):
            raise ValueError("Policy head must be a .pt file.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Policy head not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state = checkpoint["model_state"]
            validated_save_count = int(max(0, checkpoint.get("validated_save_count", 0)))
        else:
            state = checkpoint
            validated_save_count = 0

        inferred_paradigm = self._infer_learning_paradigm(checkpoint, state)

        with self.weights_lock:
            if inferred_paradigm and inferred_paradigm != self.learning_paradigm:
                self._replace_deep_adapter(inferred_paradigm)
            load_kind = self._load_adapter_state(state)
            self.adapter.eval()
            self.checkpoint_path = checkpoint_path
            self.checkpoint_dir = self._determine_checkpoint_dir(checkpoint_path)
            self._validated_save_count = validated_save_count
            self._manual_checkpoint_locked = False
            self._last_manual_checkpoint_saved_at = None
            self._smoothed_angle_car = None
            self._last_mode_for_smoothing = None
            self._deep_class_count_ema = None
            self._deep_class_loss_ema = None
            self._reset_ewc_state()
            self._reset_historical_grad_state()
            if self.FIXED_ANCHOR_ENABLED:
                self._refresh_fixed_anchor()
            else:
                self._fixed_anchor = {}
                self._fixed_anchor_active = False

        print(f"[controller] switched policy head -> {checkpoint_path} ({load_kind})")
        return checkpoint_path

    def switch_learning_paradigm(self, paradigm_id):
        if self.adapter_type != "deep" or self.adapter is None:
            raise ValueError("Learning paradigm switching requires --adapter deep.")

        checkpoint_path = self._default_checkpoint_path_for_learning_paradigm(paradigm_id)
        checkpoint_loaded = False

        with self.weights_lock:
            self._replace_deep_adapter(paradigm_id)
            self.adapter.eval()
            self.checkpoint_path = checkpoint_path
            self.checkpoint_dir = self._determine_checkpoint_dir(checkpoint_path)
            self._validated_save_count = 0
            self._manual_checkpoint_locked = False
            self._last_manual_checkpoint_saved_at = None
            self._smoothed_angle_car = None
            self._last_mode_for_smoothing = None

            if os.path.exists(checkpoint_path):
                checkpoint_loaded = self._load_checkpoint(checkpoint_path)

            self._reset_ewc_state()
            self._reset_historical_grad_state()
            if checkpoint_loaded and self.FIXED_ANCHOR_ENABLED:
                self._refresh_fixed_anchor()
            else:
                self._fixed_anchor = {}
                self._fixed_anchor_active = False

        print(
            f"[controller] learning paradigm -> {self.learning_paradigm}"
            f" (checkpoint={'loaded' if checkpoint_loaded else 'fresh'})"
        )
        return self.learning_paradigm

    def _make_base_model(self, model_path):
        model_path = str(model_path or "").strip()
        if not model_path:
            raise ValueError("Frozen model path is required.")

        cpu_fallback = None
        if (
            str(model_path).lower().endswith(".tflite")
            and self.cpu_base_model_path
            and os.path.abspath(model_path) == os.path.abspath(self.initial_base_model_path)
        ):
            cpu_fallback = self.cpu_base_model_path

        return BaseModel(
            model_path,
            use_tpu=self.use_tpu,
            cpu_model_path=cpu_fallback,
            num_threads=self.num_threads,
        )

    def set_inference_backend(self, backend, frozen_model_path=None):
        backend = str(backend or "").strip().lower()
        if backend not in {"main", "frozen"}:
            raise ValueError(f"Unsupported inference backend: {backend}")

        provided_path = str(frozen_model_path or "").strip()
        if provided_path:
            with self.weights_lock:
                self.configured_frozen_model_path = provided_path

        if backend == "main":
            if self.adapter is None:
                raise ValueError("Main online model is not available in this run.")

            with self.weights_lock:
                self.inference_backend = "main"
                self._smoothed_angle_car = None
            print("[controller] inference backend -> main")
            return

        model_path = str(
            provided_path
            or self.configured_frozen_model_path
            or self.base_model_path
            or ""
        ).strip()
        if not model_path:
            raise ValueError("Frozen model path is required for frozen inference.")

        reuse_existing = (
            self.base_model is not None
            and os.path.abspath(model_path) == os.path.abspath(self.base_model_path)
        )
        frozen_model = self.base_model if reuse_existing else self._make_base_model(model_path)

        with self.weights_lock:
            self.base_model = frozen_model
            self.base_model_path = model_path
            self.configured_frozen_model_path = model_path
            self.inference_backend = "frozen"
            self._smoothed_angle_car = None

        print(f"[controller] inference backend -> frozen ({model_path})")

    def set_frozen_model_path(self, frozen_model_path):
        path = str(frozen_model_path or "").strip()
        if not path:
            raise ValueError("Frozen model path cannot be empty.")

        with self.weights_lock:
            self.configured_frozen_model_path = path

        print(f"[controller] configured frozen model path -> {path}")

    @staticmethod
    def angle_norm_to_car(angle_norm):
        angle_norm = float(np.clip(angle_norm, 0.0, 1.0))
        return AdaptiveController.ANGLE_MIN_CAR + angle_norm * (
            AdaptiveController.ANGLE_MAX_CAR - AdaptiveController.ANGLE_MIN_CAR
        )

    @staticmethod
    def car_to_angle_norm(angle_car):
        angle_car = float(np.clip(
            angle_car,
            AdaptiveController.ANGLE_MIN_CAR,
            AdaptiveController.ANGLE_MAX_CAR,
        ))
        return (angle_car - AdaptiveController.ANGLE_MIN_CAR) / (
            AdaptiveController.ANGLE_MAX_CAR - AdaptiveController.ANGLE_MIN_CAR
        )

    @staticmethod
    def speed_prob_to_car(speed_prob, max_speed=35):
        return int(max_speed if float(speed_prob) >= 0.5 else 0)

    def _empty_training_metrics(self):
        return {
            "target_intent": "",
            "selected_expert_for_teach": "",
            "selected_expert_angle_norm": 0.0,
            "selected_expert_speed_prob": 0.0,
            "train_batch_size": 0,
            "train_total_loss": 0.0,
            "train_angle_loss": 0.0,
            "train_speed_loss": 0.0,
            "train_intent_loss": 0.0,
            "train_task_loss": 0.0,
            "train_expert_direct_loss": 0.0,
            "train_gate_supervision_loss": 0.0,
            "train_load_balance_loss": 0.0,
            "train_entropy_penalty": 0.0,
            "train_gate_mean_max": 0.0,
            "train_gate_mean_margin": 0.0,
            "train_gate_mean_entropy": 0.0,
        }

    def training_metrics_snapshot(self):
        with self.weights_lock:
            return dict(self._last_training_metrics)

    def _intent_name_from_index(self, index):
        idx = int(index)
        if 0 <= idx < len(self.INTENT_NAMES):
            return self.INTENT_NAMES[idx]
        return ""

    @staticmethod
    def _expert_name_from_index(index):
        idx = int(index)
        if idx < 0:
            return ""
        return f"gate_e{idx}"

    @staticmethod
    def _normalize_intent_name(intent_name):
        return str(intent_name or "").strip().lower()

    def _intent_override_value(self, attr_name, intent_name, default):
        mapping = getattr(self, attr_name, None)
        if not isinstance(mapping, dict):
            return default

        intent_name = self._normalize_intent_name(intent_name)
        if intent_name and intent_name in mapping:
            return mapping[intent_name]
        if "*" in mapping:
            return mapping["*"]
        return default

    def _intent_override_float(self, attr_name, intent_name, default, minimum=None):
        value = self._intent_override_value(attr_name, intent_name, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(default)

        if minimum is not None:
            value = max(float(minimum), value)
        return value

    def _summarize_index_labels(self, indices, kind="intent"):
        if indices is None:
            return ""

        values = indices.detach().view(-1).to("cpu")
        if values.numel() <= 0:
            return ""

        unique, counts = torch.unique(values, return_counts=True)
        top_index = int(unique[int(torch.argmax(counts).item())].item())
        if kind == "expert":
            label = self._expert_name_from_index(top_index)
        else:
            label = self._intent_name_from_index(top_index)

        if unique.numel() == 1:
            return label
        return f"mixed:{label}"

    def target_intent_from_controls(self, angle_car, speed_norm):
        angle_norm = float(np.clip(self.car_to_angle_norm(angle_car), 0.0, 1.0))
        speed = float(np.clip(speed_norm, 0.0, 1.0))
        angle_targets = torch.tensor([[angle_norm]], dtype=torch.float32, device=self.device)
        speed_targets = torch.tensor([[speed]], dtype=torch.float32, device=self.device)
        intent_target = self._derive_intent_targets(angle_targets, speed_targets)
        return self._intent_name_from_index(int(intent_target.item()))

    def selected_expert_for_intent(self, intent_name, num_experts=None):
        intent_name = str(intent_name or "").strip().lower()
        if not intent_name:
            return ""

        try:
            intent_index = self.INTENT_NAMES.index(intent_name)
        except ValueError:
            return ""

        if num_experts is None:
            num_experts = int(getattr(getattr(self, "adapter", None), "num_experts", 4) or 4)
        if int(num_experts) <= 0:
            return ""

        expert_index = max(0, min(intent_index, int(num_experts) - 1))
        return self._expert_name_from_index(expert_index)

    def _routing_defaults(self, feature_gate):
        return {
            "config": str(self.learning_paradigm or self.adapter_type),
            "active_learning_paradigm": str(self.learning_paradigm or ""),
            "inference_backend": str(self.inference_backend),
            "feature_gate": float(feature_gate),
            "gate_e0": 0.0,
            "gate_e1": 0.0,
            "gate_e2": 0.0,
            "gate_e3": 0.0,
            "gate_entropy": 0.0,
            "top_expert": "",
            "intent_pred": "",
            "intent_stop_prob": 0.0,
            "intent_left_prob": 0.0,
            "intent_straight_prob": 0.0,
            "intent_right_prob": 0.0,
        }

    def _gate_temperature_for_context(self, training_context, target_intent_name=""):
        if training_context == "teach_focus":
            base_temp = getattr(self, "TEACH_GATE_TEMPERATURE", 1.0)
            return self._intent_override_float(
                "TEACH_GATE_TEMPERATURE_BY_INTENT",
                target_intent_name,
                base_temp,
                minimum=0.25,
            )
        if training_context == "rehearsal":
            return float(max(0.25, getattr(self, "REHEARSAL_GATE_TEMPERATURE", 1.0)))
        return float(max(0.25, getattr(self, "TRAIN_GATE_TEMPERATURE", 1.0)))

    def _routing_regularization_multipliers(self, training_context, target_intent_name=""):
        if training_context == "teach_focus":
            return (
                self._intent_override_float(
                    "TEACH_LOAD_BALANCE_WEIGHT_MULT_BY_INTENT",
                    target_intent_name,
                    getattr(self, "TEACH_LOAD_BALANCE_WEIGHT_MULT", 1.0),
                    minimum=0.0,
                ),
                self._intent_override_float(
                    "TEACH_GATE_ENTROPY_WEIGHT_MULT_BY_INTENT",
                    target_intent_name,
                    getattr(self, "TEACH_GATE_ENTROPY_WEIGHT_MULT", 1.0),
                    minimum=0.0,
                ),
            )
        if training_context == "rehearsal":
            return (
                float(max(0.0, getattr(self, "REHEARSAL_LOAD_BALANCE_WEIGHT_MULT", 1.0))),
                float(max(0.0, getattr(self, "REHEARSAL_GATE_ENTROPY_WEIGHT_MULT", 1.0))),
            )
        return 1.0, 1.0

    def _routing_snapshot(self, gate_probs=None, intent_probs=None, feature_gate=1.0):
        snapshot = self._routing_defaults(feature_gate)

        if gate_probs is not None:
            gate_values = gate_probs.detach().cpu().reshape(-1).tolist()
            padded = list(gate_values[:4])
            while len(padded) < 4:
                padded.append(0.0)

            for index, value in enumerate(padded[:4]):
                snapshot[f"gate_e{index}"] = float(value)

            effective = np.asarray(gate_values, dtype=np.float32)
            if effective.size > 0:
                snapshot["gate_entropy"] = float(-np.sum(effective * np.log(np.clip(effective, 1e-9, 1.0))))
                snapshot["top_expert"] = f"gate_e{int(np.argmax(effective))}"

        if intent_probs is not None:
            intent_values = intent_probs.detach().cpu().reshape(-1).tolist()
            padded = list(intent_values[:4])
            while len(padded) < 4:
                padded.append(0.0)

            snapshot["intent_stop_prob"] = float(padded[0])
            snapshot["intent_left_prob"] = float(padded[1])
            snapshot["intent_straight_prob"] = float(padded[2])
            snapshot["intent_right_prob"] = float(padded[3])

            intent_names = ["stop", "left", "straight", "right"]
            if len(intent_values) > 0:
                snapshot["intent_pred"] = intent_names[int(np.argmax(intent_values[:4]))]

        return snapshot

    def _predict_scalar_or_none(self, image, mode):
        _angle_probs, base_angle_norm, base_speed_prob = self.base_model.predict_raw(image)

        delta_angle = 0.0
        delta_speed_logit = 0.0
        adapter_ms = 0.0
        feature_ms = 0.0

        if self.adapter is not None:
            t0 = time.time()
            with self.weights_lock:
                self.adapter.eval()
                x = self.adapter.get_input_features(base_angle_norm, base_speed_prob).to(self.device)
                with torch.no_grad():
                    da, ds = self.adapter(x)
                delta_angle = float(da.item())
                delta_speed_logit = float(ds.item())
            adapter_ms = (time.time() - t0) * 1000.0

        final_angle_norm = float(np.clip(base_angle_norm + delta_angle, 0.0, 1.0))
        base_speed_logit = self._prob_to_logit(base_speed_prob)
        final_speed_logit = base_speed_logit + delta_speed_logit
        final_speed_prob = float(self._sigmoid(final_speed_logit))

        final_angle_car = self.base_model.angle_norm_to_car(final_angle_norm)
        final_speed_car = self.base_model.speed_prob_to_car(final_speed_prob, max_speed=self.max_speed)

        return {
            "base_angle_norm": float(base_angle_norm),
            "base_speed_prob": float(base_speed_prob),
            "delta_angle_norm": float(delta_angle),
            "delta_speed_logit": float(delta_speed_logit),
            "final_angle_norm": float(final_angle_norm),
            "final_angle_car": float(final_angle_car),
            "final_speed_prob": float(final_speed_prob),
            "final_speed_car": int(final_speed_car),
            "feature_ms": float(feature_ms),
            "adapter_ms": float(adapter_ms),
            **self._routing_defaults(feature_gate=1.0),
        }

    def _predict_frozen_model(self, image, mode):
        if self.base_model is None:
            raise RuntimeError("Frozen inference requested but no frozen model is loaded.")

        _angle_probs, angle_norm, speed_prob = self.base_model.predict_raw(image)
        final_angle_car = self.base_model.angle_norm_to_car(angle_norm)
        final_speed_car = self.base_model.speed_prob_to_car(speed_prob, max_speed=self.max_speed)

        return {
            "base_angle_norm": float(angle_norm),
            "base_speed_prob": float(speed_prob),
            "delta_angle_norm": 0.0,
            "delta_speed_logit": 0.0,
            "final_angle_norm": float(angle_norm),
            "final_angle_car": float(final_angle_car),
            "final_speed_prob": float(speed_prob),
            "final_speed_car": int(final_speed_car),
            "feature_ms": 0.0,
            "adapter_ms": 0.0,
            **self._routing_defaults(feature_gate=0.0),
        }

    def _predict_deep_policy(self, image, mode):
        t0 = time.time()
        deep_features = self.feature_extractor.extract(image)
        t_feature = time.time()

        x = torch.tensor(deep_features, dtype=torch.float32, device=self.device).view(1, -1)

        gate_probs = None
        intent_probs = None

        with self.weights_lock:
            self.adapter.eval()
            with torch.no_grad():
                if hasattr(self.adapter, "forward_with_gate"):
                    gate_outputs = self.adapter.forward_with_gate(
                        x,
                        gate_temperature=float(getattr(self, "INFERENCE_GATE_TEMPERATURE", 1.0)),
                    )
                    angle_logits = gate_outputs[0]
                    speed_logit = gate_outputs[1]
                    if len(gate_outputs) > 2:
                        gate_probs = gate_outputs[2]
                    if len(gate_outputs) > 4:
                        intent_probs = gate_outputs[4]
                else:
                    angle_logits, speed_logit = self.adapter(x)

        angle_norm = float(torch.clamp(angle_expected_value(angle_logits), 0.0, 1.0).item())
        speed_prob = float(torch.sigmoid(speed_logit).item())

        final_angle_car = self.angle_norm_to_car(angle_norm)
        final_speed_car = self.speed_prob_to_car(speed_prob, max_speed=self.max_speed)

        adapter_ms = (time.time() - t_feature) * 1000.0
        feature_ms = (t_feature - t0) * 1000.0

        return {
            "base_angle_norm": float(angle_norm),
            "base_speed_prob": float(speed_prob),
            "delta_angle_norm": 0.0,
            "delta_speed_logit": 0.0,
            "final_angle_norm": float(angle_norm),
            "final_angle_car": float(final_angle_car),
            "final_speed_prob": float(speed_prob),
            "final_speed_car": int(final_speed_car),
            "feature_ms": float(feature_ms),
            "adapter_ms": float(adapter_ms),
            **self._routing_snapshot(
                gate_probs=gate_probs,
                intent_probs=intent_probs,
                feature_gate=1.0,
            ),
        }

    def predict(self, image, mode):
        t_start = time.time()

        if self.inference_backend == "frozen":
            out = self._predict_frozen_model(image, mode)
        elif self.adapter_type == "deep":
            out = self._predict_deep_policy(image, mode)
        else:
            out = self._predict_scalar_or_none(image, mode)

        final_angle_car = out["final_angle_car"]
        alpha = SMOOTHING_BY_MODE.get(mode, 0.0)

        if mode != self._last_mode_for_smoothing:
            self._smoothed_angle_car = None
            self._last_mode_for_smoothing = mode

        if alpha > 0:
            if self._smoothed_angle_car is None:
                self._smoothed_angle_car = final_angle_car
            else:
                self._smoothed_angle_car = (
                    alpha * self._smoothed_angle_car + (1.0 - alpha) * final_angle_car
                )
            final_angle_car = self._smoothed_angle_car

        out["final_angle_car"] = float(final_angle_car)
        out["inference_ms"] = float((time.time() - t_start) * 1000.0 - out["adapter_ms"] - out["feature_ms"])
        if out["inference_ms"] < 0:
            out["inference_ms"] = 0.0

        return out

    def teach_step(
        self,
        base_angle_norm,
        base_speed_prob,
        human_angle_car,
        human_speed_norm,
        image=None,
        add_to_gate=True,
    ):
        if self.adapter is None:
            return None

        if self.adapter_type == "deep":
            if image is None:
                raise ValueError("teach_step with deep policy requires image.")

            deep_features = self.feature_extractor.extract(image)
            x = torch.tensor(deep_features, dtype=torch.float32).view(1, -1)

            target_angle_norm = float(np.clip(self.car_to_angle_norm(human_angle_car), 0.0, 1.0))

            return {
                "input_features": x,
                "target_delta_angle": target_angle_norm,  # kept key name for compatibility
                "target_speed_norm": float(human_speed_norm),
                "base_speed_prob": float(base_speed_prob),
            }

        human_angle_norm = self.base_model.car_to_angle_norm(human_angle_car)
        target_delta_angle = float(np.clip(
            human_angle_norm - float(base_angle_norm),
            -ANGLE_DELTA_BOUND,
            ANGLE_DELTA_BOUND,
        ))

        with self.weights_lock:
            x = self.adapter.get_input_features(base_angle_norm, base_speed_prob)

        return {
            "input_features": x,
            "target_delta_angle": target_delta_angle,
            "target_speed_norm": float(human_speed_norm),
            "base_speed_prob": float(base_speed_prob),
        }

    def gradient_step(
        self,
        batch_features,
        batch_target_deltas,
        batch_target_speeds,
        batch_base_speed_probs=None,
        train_speed=True,
        delta_penalty_weight=0.01,
        clip_grad_norm=1.0,
        historical_blend=0.0,
        update_historical=False,
        training_context="generic",
        gate_temperature=None,
        expert_supervision_mask=None,
        target_intent_override="",
        selected_expert_override="",
    ):
        if self.adapter is None:
            return 0.0

        with self.weights_lock:
            self.adapter.train()

            target_intent_name = self._normalize_intent_name(target_intent_override)

            batch_features = batch_features.to(self.device)
            batch_target_deltas = batch_target_deltas.to(self.device)
            batch_target_speeds = batch_target_speeds.to(self.device)

            if batch_target_deltas.ndim == 1:
                batch_target_deltas = batch_target_deltas.unsqueeze(1)
            if batch_target_speeds.ndim == 1:
                batch_target_speeds = batch_target_speeds.unsqueeze(1)

            supervision_mask = None
            if expert_supervision_mask is not None:
                supervision_mask = expert_supervision_mask.to(self.device)
                if supervision_mask.ndim > 1:
                    supervision_mask = supervision_mask.view(-1)
                supervision_mask = supervision_mask.bool()
                if supervision_mask.shape[0] != batch_target_deltas.shape[0]:
                    supervision_mask = None

            gate_temperature_value = (
                float(gate_temperature)
                if gate_temperature is not None else self._gate_temperature_for_context(training_context, target_intent_name)
            )
            load_balance_mult, entropy_mult = self._routing_regularization_multipliers(
                training_context,
                target_intent_name,
            )
            expert_direct_loss_weight = self._intent_override_float(
                "INTENT_EXPERT_DIRECT_LOSS_WEIGHT_BY_INTENT",
                target_intent_name,
                getattr(self, "INTENT_EXPERT_DIRECT_LOSS_WEIGHT", 1.0),
                minimum=0.0,
            )
            gate_supervision_loss_weight = self._intent_override_float(
                "INTENT_EXPERT_GATE_LOSS_WEIGHT_BY_INTENT",
                target_intent_name,
                getattr(self, "INTENT_EXPERT_GATE_LOSS_WEIGHT", 0.35),
                minimum=0.0,
            )

            training_metrics = self._empty_training_metrics()
            training_metrics["train_batch_size"] = int(batch_target_deltas.shape[0])

            if self.adapter_type == "deep":
                gate_probs = None
                intent_logits = None
                task_logits = None
                expert_angle_logits = None
                expert_speed_logits = None
                gate_logits = None
                if hasattr(self.adapter, "forward_with_gate"):
                    gate_outputs = self.adapter.forward_with_gate(
                        batch_features,
                        gate_temperature=gate_temperature_value,
                    )
                    angle_logits = gate_outputs[0]
                    speed_logit = gate_outputs[1]
                    if len(gate_outputs) > 2:
                        gate_probs = gate_outputs[2]
                    if len(gate_outputs) > 3:
                        intent_logits = gate_outputs[3]
                    if len(gate_outputs) > 5:
                        task_logits = gate_outputs[5]
                    if len(gate_outputs) > 9:
                        expert_angle_logits = gate_outputs[7]
                        expert_speed_logits = gate_outputs[8]
                        gate_logits = gate_outputs[9]
                else:
                    angle_logits, speed_logit = self.adapter(batch_features)

                angle_targets = torch.clamp(batch_target_deltas, 0.0, 1.0)
                angle_class = torch.round(angle_targets * (self.NUM_ANGLE_CLASSES - 1)).long().squeeze(1)
                angle_ce = F.cross_entropy(angle_logits, angle_class, reduction="none")
                logged_intent_targets = self._derive_intent_targets(angle_targets, batch_target_speeds)

                if self.DEEP_ADAPTIVE_CLASS_WEIGHTING_ENABLED:
                    class_weights = self._deep_angle_class_weights(angle_class, angle_ce.detach())
                    sample_weights = class_weights[angle_class]
                    angle_loss = torch.mean(angle_ce * sample_weights)
                else:
                    angle_loss = torch.mean(angle_ce)

                if train_speed:
                    speed_loss = F.binary_cross_entropy_with_logits(speed_logit, batch_target_speeds)
                else:
                    speed_loss = torch.tensor(0.0, device=self.device)

                loss = 2.0 * angle_loss + speed_loss
                intent_targets = None
                if target_intent_override:
                    training_metrics["target_intent"] = str(target_intent_override)
                else:
                    label_targets = logged_intent_targets
                    if supervision_mask is not None and torch.any(supervision_mask):
                        label_targets = logged_intent_targets[supervision_mask]
                    training_metrics["target_intent"] = self._summarize_index_labels(label_targets, kind="intent")
                training_metrics["train_angle_loss"] = float(angle_loss.detach().item())
                training_metrics["train_speed_loss"] = float(speed_loss.detach().item())

                if self.INTENT_ROUTING_ENABLED and intent_logits is not None:
                    intent_targets = logged_intent_targets
                    intent_loss = F.cross_entropy(intent_logits, intent_targets)
                    loss = loss + float(self.INTENT_LOSS_WEIGHT) * intent_loss
                    training_metrics["train_intent_loss"] = float(intent_loss.detach().item())

                if self.CONTEXT_TASK_ROUTING_ENABLED and task_logits is not None:
                    task_targets = self._derive_context_task_targets(angle_targets, batch_target_speeds)
                    task_loss = F.cross_entropy(task_logits, task_targets)
                    loss = loss + float(self.CONTEXT_TASK_LOSS_WEIGHT) * task_loss
                    training_metrics["train_task_loss"] = float(task_loss.detach().item())

                if (
                    self.INTENT_EXPERT_SUPERVISION_ENABLED
                    and expert_angle_logits is not None
                    and expert_speed_logits is not None
                ):
                    apply_expert_supervision = True
                    if self.INTENT_EXPERT_SUPERVISION_TEACH_ONLY and training_context != "teach_focus":
                        apply_expert_supervision = False

                    if intent_targets is None:
                        intent_targets = self._derive_intent_targets(angle_targets, batch_target_speeds)

                    if apply_expert_supervision:
                        expert_targets = self._intent_targets_to_expert_targets(
                            intent_targets,
                            expert_angle_logits.shape[1],
                        )
                        supervised_mask = supervision_mask
                        if supervised_mask is None:
                            supervised_mask = torch.ones_like(expert_targets, dtype=torch.bool, device=self.device)

                        if torch.any(supervised_mask):
                            sample_index = torch.arange(expert_targets.shape[0], device=self.device)[supervised_mask]
                            supervised_expert_targets = expert_targets[supervised_mask]
                            selected_angle_logits = expert_angle_logits[sample_index, supervised_expert_targets]
                            selected_speed_logit = expert_speed_logits[sample_index, supervised_expert_targets]
                            if selected_speed_logit.ndim == 1:
                                selected_speed_logit = selected_speed_logit.unsqueeze(1)

                            expert_angle_loss = F.cross_entropy(
                                selected_angle_logits,
                                angle_class[supervised_mask],
                            )
                            if train_speed:
                                expert_speed_loss = F.binary_cross_entropy_with_logits(
                                    selected_speed_logit,
                                    batch_target_speeds[supervised_mask],
                                )
                            else:
                                expert_speed_loss = torch.tensor(0.0, device=self.device)

                            expert_direct_loss = 2.0 * expert_angle_loss + expert_speed_loss
                            loss = loss + expert_direct_loss_weight * expert_direct_loss
                            if selected_expert_override:
                                training_metrics["selected_expert_for_teach"] = str(selected_expert_override)
                            else:
                                training_metrics["selected_expert_for_teach"] = self._summarize_index_labels(
                                    supervised_expert_targets,
                                    kind="expert",
                                )
                            training_metrics["selected_expert_angle_norm"] = float(
                                torch.mean(angle_expected_value(selected_angle_logits)).detach().item()
                            )
                            training_metrics["selected_expert_speed_prob"] = float(
                                torch.mean(torch.sigmoid(selected_speed_logit)).detach().item()
                            )
                            training_metrics["train_expert_direct_loss"] = float(
                                expert_direct_loss.detach().item()
                            )

                            if gate_logits is not None and gate_logits.shape[-1] == expert_angle_logits.shape[1]:
                                gate_target_loss = F.cross_entropy(
                                    gate_logits[supervised_mask],
                                    supervised_expert_targets,
                                )
                                loss = loss + gate_supervision_loss_weight * gate_target_loss
                                training_metrics["train_gate_supervision_loss"] = float(
                                    gate_target_loss.detach().item()
                                )

                if (
                    self.MOE_BALANCING_ENABLED
                    and gate_probs is not None
                    and self._ewc_steps >= self.MOE_BALANCE_WARMUP_STEPS
                    and gate_probs.shape[-1] > 1
                ):
                    num_experts = gate_probs.shape[-1]
                    mean_gate = torch.mean(gate_probs, dim=0)
                    uniform = torch.full_like(mean_gate, 1.0 / float(num_experts))

                    # Keep batch-level expert usage close to uniform.
                    load_balance_loss = torch.mean((mean_gate - uniform) ** 2) * float(num_experts)

                    # Prevent early single-expert collapse while still allowing specialization.
                    gate_entropy = -torch.sum(
                        gate_probs * torch.log(torch.clamp(gate_probs, 1e-8, 1.0)),
                        dim=-1,
                    ).mean()
                    target_entropy = float(np.log(float(num_experts)))
                    entropy_deficit = torch.clamp(target_entropy - gate_entropy, min=0.0)
                    training_metrics["train_load_balance_loss"] = float(
                        load_balance_loss.detach().item()
                    )
                    training_metrics["train_entropy_penalty"] = float(
                        entropy_deficit.detach().item()
                    )

                    loss = (
                        loss
                        + float(self.MOE_LOAD_BALANCE_WEIGHT) * load_balance_mult * load_balance_loss
                        + float(self.MOE_GATE_ENTROPY_WEIGHT) * entropy_mult * entropy_deficit
                    )

                if gate_probs is not None:
                    gate_values = gate_probs.detach()
                    sorted_gate = torch.sort(gate_values, dim=-1, descending=True).values
                    training_metrics["train_gate_mean_max"] = float(
                        torch.mean(sorted_gate[:, 0]).item()
                    )
                    training_metrics["train_gate_mean_entropy"] = float(
                        torch.mean(
                            -torch.sum(
                                gate_values * torch.log(torch.clamp(gate_values, 1e-8, 1.0)),
                                dim=-1,
                            )
                        ).item()
                    )
                    if sorted_gate.shape[-1] > 1:
                        training_metrics["train_gate_mean_margin"] = float(
                            torch.mean(sorted_gate[:, 0] - sorted_gate[:, 1]).item()
                        )

            else:
                delta_angle, delta_speed_logit = self.adapter(batch_features)
                angle_loss = F.mse_loss(delta_angle, batch_target_deltas)
                delta_penalty = torch.mean(delta_angle ** 2)

                if train_speed:
                    if batch_base_speed_probs is not None:
                        batch_base_speed_probs = batch_base_speed_probs.to(self.device)
                        if batch_base_speed_probs.ndim == 1:
                            batch_base_speed_probs = batch_base_speed_probs.unsqueeze(1)

                        p = torch.clamp(batch_base_speed_probs, 1e-4, 1.0 - 1e-4)
                        base_speed_logit = torch.log(p / (1.0 - p))
                        final_speed_logit = base_speed_logit + delta_speed_logit

                        speed_loss = F.binary_cross_entropy_with_logits(
                            final_speed_logit,
                            batch_target_speeds,
                        )
                    else:
                        speed_loss = F.binary_cross_entropy_with_logits(
                            delta_speed_logit,
                            batch_target_speeds,
                        )
                else:
                    speed_loss = torch.tensor(0.0, device=self.device)

                loss = angle_loss + 0.5 * speed_loss + delta_penalty_weight * delta_penalty
                training_metrics["train_angle_loss"] = float(angle_loss.detach().item())
                training_metrics["train_speed_loss"] = float(speed_loss.detach().item())

            if self.EWC_ENABLED and self._ewc_steps >= self.EWC_WARMUP_STEPS:
                ewc_penalty = self._ewc_penalty()
                ewc_mult, _anchor_mult = self._validated_retention_multipliers()
                loss = loss + (self.EWC_LAMBDA * ewc_mult) * ewc_penalty

            if (
                self.FIXED_ANCHOR_ENABLED
                and self._fixed_anchor_active
                and self._ewc_steps >= self.FIXED_ANCHOR_WARMUP_STEPS
            ):
                anchor_penalty = self._fixed_anchor_penalty()
                _ewc_mult, anchor_mult = self._validated_retention_multipliers()
                loss = loss + (self.FIXED_ANCHOR_LAMBDA * anchor_mult) * anchor_penalty

            self.optimizer.zero_grad()
            loss.backward()
            if historical_blend > 0:
                self._blend_with_historical_gradients(historical_blend)
            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), clip_grad_norm)
            if update_historical:
                self._update_historical_gradients()
            if self.EWC_ENABLED:
                self._ewc_update_after_backward()
            self.optimizer.step()
            self.adapter.eval()
            self._ewc_steps += 1
            training_metrics["train_total_loss"] = float(loss.detach().item())
            self._last_training_metrics = training_metrics

        return float(loss.item())

    def _ensure_deep_class_stats(self):
        if self._deep_class_count_ema is None or self._deep_class_loss_ema is None:
            self._deep_class_count_ema = torch.ones(
                self.NUM_ANGLE_CLASSES,
                dtype=torch.float32,
                device=self.device,
            )
            self._deep_class_loss_ema = torch.ones(
                self.NUM_ANGLE_CLASSES,
                dtype=torch.float32,
                device=self.device,
            )

    def _deep_angle_class_weights(self, angle_class, angle_ce_detached):
        self._ensure_deep_class_stats()

        momentum = float(np.clip(self.DEEP_CLASS_STATS_MOMENTUM, 0.0, 0.9999))

        with torch.no_grad():
            class_one_hot = F.one_hot(
                angle_class,
                num_classes=self.NUM_ANGLE_CLASSES,
            ).float()

            batch_count = class_one_hot.sum(dim=0)
            batch_loss_sum = (class_one_hot * angle_ce_detached.unsqueeze(1)).sum(dim=0)
            batch_loss_mean = batch_loss_sum / torch.clamp(batch_count, min=1.0)
            seen_mask = batch_count > 0

            self._deep_class_count_ema.mul_(momentum).add_(batch_count, alpha=(1.0 - momentum))
            updated_loss = self._deep_class_loss_ema * momentum + batch_loss_mean * (1.0 - momentum)
            self._deep_class_loss_ema = torch.where(seen_mask, updated_loss, self._deep_class_loss_ema)

            class_freq = self._deep_class_count_ema / torch.clamp(
                torch.sum(self._deep_class_count_ema),
                min=1e-6,
            )
            inv_freq = 1.0 / torch.sqrt(torch.clamp(class_freq, min=1e-6))
            inv_freq = inv_freq / torch.clamp(torch.mean(inv_freq), min=1e-6)

            class_hardness = self._deep_class_loss_ema / torch.clamp(
                torch.mean(self._deep_class_loss_ema),
                min=1e-6,
            )

            weights = (
                1.0
                + float(self.DEEP_CLASS_FREQ_GAIN) * (inv_freq - 1.0)
                + float(self.DEEP_CLASS_HARDNESS_GAIN) * (class_hardness - 1.0)
            )

            w_min = float(max(0.1, self.DEEP_CLASS_WEIGHT_MIN))
            w_max = float(max(w_min, self.DEEP_CLASS_WEIGHT_MAX))
            weights = torch.clamp(weights, min=w_min, max=w_max)
            return weights.to(angle_ce_detached.device)

    def _derive_intent_targets(self, angle_targets, speed_targets):
        speed = speed_targets.squeeze(1)
        angle = angle_targets.squeeze(1)

        stop_thr = float(np.clip(self.INTENT_STOP_THRESHOLD, 0.0, 1.0))
        margin = float(np.clip(self.INTENT_CENTER_MARGIN_NORM, 0.02, 0.35))

        stop_mask = speed < stop_thr
        left_mask = angle < (0.5 - margin)
        right_mask = angle > (0.5 + margin)

        intent = torch.full_like(speed, 2, dtype=torch.long)
        intent = torch.where(left_mask, torch.ones_like(intent), intent)
        intent = torch.where(right_mask, torch.full_like(intent, 3), intent)
        intent = torch.where(stop_mask, torch.zeros_like(intent), intent)
        return intent

    def _derive_context_task_targets(self, angle_targets, speed_targets):
        return self._derive_intent_targets(angle_targets, speed_targets)

    def _intent_targets_to_expert_targets(self, intent_targets, num_experts):
        if int(num_experts) <= 1:
            return torch.zeros_like(intent_targets)
        return torch.clamp(intent_targets, min=0, max=int(num_experts) - 1)

    def save_checkpoint(self, manual=False):
        if self.adapter is None or self.checkpoint_path is None:
            return False

        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
        tmp_path = f"{self.checkpoint_path}.tmp"

        with self.weights_lock:
            if self._manual_checkpoint_locked and not manual:
                return False

            if self.adapter_type == "deep":
                payload = {
                    "model_state": self.adapter.state_dict(),
                    "feature_dim": 512,
                    "num_angle_classes": self.NUM_ANGLE_CLASSES,
                    "architecture": getattr(
                        self.adapter,
                        "architecture_name",
                        "LivePolicyHead-512-256-128",
                    ),
                    "paradigm_id": str(
                        self.learning_paradigm or self.DEFAULT_DEEP_LEARNING_PARADIGM
                    ),
                    "validated_save_count": int(max(0, self._validated_save_count)),
                }
            else:
                payload = {
                    "model_state": self.adapter.state_dict(),
                    "validated_save_count": int(max(0, self._validated_save_count)),
                }

            if manual:
                self._validated_save_count += 1
                payload["validated_save_count"] = int(self._validated_save_count)

            torch.save(payload, tmp_path)
            os.replace(tmp_path, self.checkpoint_path)

            if manual and self.VALIDATED_REINFORCEMENT_ENABLED:
                self._reinforce_validated_weights(self._validated_save_count)

            if manual and self.FIXED_ANCHOR_ENABLED:
                self._refresh_fixed_anchor()
            if manual:
                self._manual_checkpoint_locked = True
                self._last_manual_checkpoint_saved_at = time.time()
                self._reset_historical_grad_state()

        return True

    def checkpoint_status_snapshot(self):
        checkpoint_path = self.checkpoint_path or ""
        active_policy_head = os.path.basename(checkpoint_path) if checkpoint_path else ""
        with self.weights_lock:
            return {
                "checkpoint_path": self.checkpoint_path or "",
                "checkpoint_dir": str(self.checkpoint_dir or ""),
                "checkpoint_locked": self._manual_checkpoint_locked,
                "last_manual_checkpoint_saved_at": self._last_manual_checkpoint_saved_at,
                "validated_save_count": int(max(0, self._validated_save_count)),
                "max_speed": int(self.max_speed),
                "adapter_type": str(self.adapter_type),
                "active_policy_head": active_policy_head,
                "available_policy_heads": self.list_available_policy_heads(),
                "active_learning_paradigm": str(self.learning_paradigm),
                "active_learning_paradigm_label": str(self.learning_paradigm_label),
                "available_learning_paradigms": self.list_available_learning_paradigms(),
                "inference_backend": str(self.inference_backend),
                "main_model_available": bool(self.adapter is not None),
                "feature_model_path": str(self.feature_model_path or ""),
                "frozen_model_path": str(self.base_model_path or ""),
                "configured_frozen_model_path": str(self.configured_frozen_model_path or ""),
                "frozen_model_backend": str(getattr(self.base_model, "backend_name", "")),
            }

    def checkpoint_drift_rms(self):
        if self.adapter is None or not self._fixed_anchor_active:
            return 0.0

        total_sq = 0.0
        total_params = 0

        with self.weights_lock:
            for name, param in self.adapter.named_parameters():
                if not param.requires_grad:
                    continue

                anchor = self._fixed_anchor.get(name)
                if anchor is None:
                    continue

                diff = (param.detach() - anchor).float()
                total_sq += float(torch.sum(diff * diff).item())
                total_params += int(diff.numel())

        if total_params <= 0:
            return 0.0

        return float(np.sqrt(total_sq / float(total_params)))

    def reset_adapter(self):
        if self.adapter is None:
            return

        checkpoint_loaded = False

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                with self.weights_lock:
                    checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

                    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                        state = checkpoint["model_state"]
                    else:
                        state = checkpoint

                    self._load_adapter_state(state)
                    self.adapter.eval()

                checkpoint_loaded = True
                print(f"[controller] reset adapter from checkpoint: {self.checkpoint_path}")

            except Exception as e:
                print(f"[controller] WARN — reset from checkpoint failed: {e}")

        if checkpoint_loaded:
            self._reset_ewc_state()
            self._reset_historical_grad_state()
            self._refresh_fixed_anchor()
            return

        # Fallback: if no checkpoint is available, reinitialize parameters.
        with self.weights_lock:
            for _name, module in self.adapter.named_modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            self.adapter.eval()

        self._reset_ewc_state()
        self._fixed_anchor = {}
        self._fixed_anchor_active = False
        self._reset_historical_grad_state()

        print("[controller] reset adapter by reinitializing parameters (no checkpoint)")

    def _load_adapter_state(self, state):
        if self.adapter is None:
            return "none"

        if hasattr(self.adapter, "load_compatible_state_dict"):
            return str(self.adapter.load_compatible_state_dict(state))

        self.adapter.load_state_dict(state)
        return "native"

    @staticmethod
    def _sigmoid(x):
        x = float(np.clip(x, -50.0, 50.0))
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _prob_to_logit(p):
        p = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        return float(np.log(p / (1.0 - p)))

    def _reset_ewc_state(self):
        """Initialise EWC reference weights and Fisher buffers from current adapter."""
        self._ewc_steps = 0
        self._ewc_fisher = {}
        self._ewc_theta_star = {}

        if self.adapter is None:
            return

        with torch.no_grad():
            for name, param in self.adapter.named_parameters():
                if not param.requires_grad:
                    continue
                p = param.detach().clone().to(self.device)
                self._ewc_theta_star[name] = p
                self._ewc_fisher[name] = torch.zeros_like(p)

    def _reset_historical_grad_state(self):
        self._historical_grad_ema = {}

    def _validated_retention_multipliers(self):
        """Increase consolidation pressure as validated saves accumulate."""
        count = int(max(0, self._validated_save_count))
        if count <= 0:
            return 1.0, 1.0

        max_mult = float(max(1.0, self.VALIDATED_RETENTION_MAX_MULT))
        count_factor = float(np.log1p(count))

        ewc_gain = float(max(0.0, self.VALIDATED_RETENTION_EWC_GAIN))
        anchor_gain = float(max(0.0, self.VALIDATED_RETENTION_ANCHOR_GAIN))

        ewc_mult = min(max_mult, 1.0 + ewc_gain * count_factor)
        anchor_mult = min(max_mult, 1.0 + anchor_gain * count_factor)
        return ewc_mult, anchor_mult

    def _reinforce_validated_weights(self, validation_count=1):
        """Strengthen retention on validated weights when operator manually saves."""
        if self.adapter is None:
            return

        base_boost = float(max(0.0, self.VALIDATED_FISHER_BOOST))
        gain = float(max(0.0, self.VALIDATED_REINFORCEMENT_COUNT_GAIN))
        max_mult = float(max(1.0, self.VALIDATED_REINFORCEMENT_MAX_MULT))
        count = int(max(1, validation_count))
        boost_mult = min(max_mult, 1.0 + gain * float(count - 1))
        boost = base_boost * boost_mult
        if boost <= 0.0:
            return

        q = float(np.clip(self.VALIDATED_STABLE_QUANTILE, 0.0, 1.0))
        fisher_cap = float(max(0.0, self.VALIDATED_FISHER_MAX))

        with torch.no_grad():
            for name, param in self.adapter.named_parameters():
                if not param.requires_grad:
                    continue

                p = param.detach().clone().to(self.device)

                fisher = self._ewc_fisher.get(name)
                if fisher is None:
                    fisher = torch.zeros_like(p)

                if self._fixed_anchor_active and name in self._fixed_anchor:
                    old_anchor = self._fixed_anchor[name]
                    abs_diff = (p - old_anchor).abs().reshape(-1)
                    if abs_diff.numel() > 0:
                        thr = torch.quantile(abs_diff, q)
                        stable_mask = (p - old_anchor).abs() <= thr
                    else:
                        stable_mask = torch.ones_like(p, dtype=torch.bool)
                else:
                    stable_mask = torch.ones_like(p, dtype=torch.bool)

                fisher = fisher + stable_mask.to(fisher.dtype) * boost
                if fisher_cap > 0.0:
                    fisher.clamp_(max=fisher_cap)

                self._ewc_fisher[name] = fisher
                self._ewc_theta_star[name] = p

    def _blend_with_historical_gradients(self, blend):
        if not self.HISTORICAL_GRADIENT_ENABLED:
            return

        blend = float(np.clip(blend, 0.0, 1.0))
        if blend <= 0.0 or not self._historical_grad_ema:
            return

        for name, param in self.adapter.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            hist = self._historical_grad_ema.get(name)
            if hist is None:
                continue

            param.grad.mul_(1.0 - blend).add_(hist, alpha=blend)

    def _update_historical_gradients(self):
        if not self.HISTORICAL_GRADIENT_ENABLED:
            return

        momentum = float(np.clip(self.HISTORICAL_GRADIENT_MOMENTUM, 0.0, 0.9999))

        for name, param in self.adapter.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad.detach().clone()
            hist = self._historical_grad_ema.get(name)
            if hist is None:
                self._historical_grad_ema[name] = grad
            else:
                hist.mul_(momentum).add_(grad, alpha=(1.0 - momentum))

    def _ewc_penalty(self):
        if self.adapter is None:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.adapter.named_parameters():
            if not param.requires_grad:
                continue

            fisher = self._ewc_fisher.get(name)
            if self._fixed_anchor_active and name in self._fixed_anchor:
                theta_star = self._fixed_anchor[name]
            else:
                theta_star = self._ewc_theta_star.get(name)
            if fisher is None or theta_star is None:
                continue

            penalty = penalty + torch.sum(fisher * (param - theta_star) ** 2)

        return penalty

    def _refresh_fixed_anchor(self):
        self._fixed_anchor = {}
        self._fixed_anchor_active = False

        if self.adapter is None:
            return

        with torch.no_grad():
            for name, param in self.adapter.named_parameters():
                if not param.requires_grad:
                    continue
                self._fixed_anchor[name] = param.detach().clone().to(self.device)

        self._fixed_anchor_active = bool(self._fixed_anchor)

    def _fixed_anchor_penalty(self):
        if self.adapter is None or not self._fixed_anchor_active:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.adapter.named_parameters():
            if not param.requires_grad:
                continue

            anchor = self._fixed_anchor.get(name)
            if anchor is None:
                continue

            penalty = penalty + torch.sum((param - anchor) ** 2)

        return penalty

    def _ewc_update_after_backward(self):
        """Online-EWC Fisher/anchor update from current gradients."""
        if self.adapter is None:
            return

        decay = float(np.clip(self.EWC_FISHER_DECAY, 0.0, 0.9999))

        with torch.no_grad():
            for name, param in self.adapter.named_parameters():
                if not param.requires_grad:
                    continue

                g = param.grad
                if g is None:
                    continue

                g2 = g.detach() ** 2

                if name not in self._ewc_fisher:
                    self._ewc_fisher[name] = torch.zeros_like(g2)
                if name not in self._ewc_theta_star:
                    self._ewc_theta_star[name] = param.detach().clone()

                self._ewc_fisher[name].mul_(decay).add_(g2, alpha=(1.0 - decay))
