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
    INTENT_STOP_THRESHOLD = 0.5
    INTENT_CENTER_MARGIN_NORM = 0.12
    DEEP_ADAPTIVE_CLASS_WEIGHTING_ENABLED = False
    DEEP_CLASS_STATS_MOMENTUM = 0.97
    DEEP_CLASS_FREQ_GAIN = 0.55
    DEEP_CLASS_HARDNESS_GAIN = 0.75
    DEEP_CLASS_WEIGHT_MIN = 0.65
    DEEP_CLASS_WEIGHT_MAX = 2.6

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

        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = self._determine_checkpoint_dir(checkpoint_path)
        self.learning_rate = learning_rate

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
            from fca.learning.live_policy_head import LivePolicyHead

            self.feature_extractor = FeatureExtractor(
                feature_model_path,
                use_tpu=use_tpu,
                num_threads=num_threads,
            )

            # Trainable policy head on CPU (or selected torch device).
            self.adapter = LivePolicyHead(
                feature_dim=512,
                hidden1=256,
                hidden2=128,
                num_angle_classes=self.NUM_ANGLE_CLASSES,
                num_experts=4,
            ).to(device)

            self.optimizer = torch.optim.AdamW(
                self.adapter.parameters(),
                lr=learning_rate,
                weight_decay=1e-4,
            )

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

        self.weights_lock = threading.Lock()
        self._validated_save_count = 0

        checkpoint_loaded = False
        if self.adapter is not None and checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint_loaded = self._load_checkpoint(checkpoint_path)

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

            load_kind = self._load_adapter_state(state)
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

        with self.weights_lock:
            load_kind = self._load_adapter_state(state)
            self.adapter.eval()
            self.checkpoint_path = checkpoint_path
            self.checkpoint_dir = self._determine_checkpoint_dir(checkpoint_path)
            self._validated_save_count = validated_save_count
            self._manual_checkpoint_locked = False
            self._last_manual_checkpoint_saved_at = None
            self._smoothed_angle_car = None
            self._last_mode_for_smoothing = None
            self._reset_ewc_state()
            self._reset_historical_grad_state()
            if self.FIXED_ANCHOR_ENABLED:
                self._refresh_fixed_anchor()
            else:
                self._fixed_anchor = {}
                self._fixed_anchor_active = False

        print(f"[controller] switched policy head -> {checkpoint_path} ({load_kind})")
        return checkpoint_path

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
            "final_angle_car": float(final_angle_car),
            "final_speed_car": int(final_speed_car),
            "feature_ms": float(feature_ms),
            "adapter_ms": float(adapter_ms),
            "feature_gate": 1.0,
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
            "final_angle_car": float(final_angle_car),
            "final_speed_car": int(final_speed_car),
            "feature_ms": 0.0,
            "adapter_ms": 0.0,
            "feature_gate": 0.0,
        }

    def _predict_deep_policy(self, image, mode):
        from fca.learning.live_policy_head import angle_expected_value

        t0 = time.time()
        deep_features = self.feature_extractor.extract(image)
        t_feature = time.time()

        x = torch.tensor(deep_features, dtype=torch.float32, device=self.device).view(1, -1)

        with self.weights_lock:
            self.adapter.eval()
            with torch.no_grad():
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
            "final_angle_car": float(final_angle_car),
            "final_speed_car": int(final_speed_car),
            "feature_ms": float(feature_ms),
            "adapter_ms": float(adapter_ms),
            "feature_gate": 1.0,
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
    ):
        if self.adapter is None:
            return 0.0

        with self.weights_lock:
            self.adapter.train()

            batch_features = batch_features.to(self.device)
            batch_target_deltas = batch_target_deltas.to(self.device)
            batch_target_speeds = batch_target_speeds.to(self.device)

            if batch_target_deltas.ndim == 1:
                batch_target_deltas = batch_target_deltas.unsqueeze(1)
            if batch_target_speeds.ndim == 1:
                batch_target_speeds = batch_target_speeds.unsqueeze(1)

            if self.adapter_type == "deep":
                gate_probs = None
                intent_logits = None
                if hasattr(self.adapter, "forward_with_gate"):
                    (
                        angle_logits,
                        speed_logit,
                        gate_probs,
                        intent_logits,
                        _intent_probs,
                    ) = self.adapter.forward_with_gate(batch_features)
                else:
                    angle_logits, speed_logit = self.adapter(batch_features)

                angle_targets = torch.clamp(batch_target_deltas, 0.0, 1.0)
                angle_class = torch.round(angle_targets * (self.NUM_ANGLE_CLASSES - 1)).long().squeeze(1)
                angle_ce = F.cross_entropy(angle_logits, angle_class, reduction="none")

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

                if self.INTENT_ROUTING_ENABLED and intent_logits is not None:
                    intent_targets = self._derive_intent_targets(angle_targets, batch_target_speeds)
                    intent_loss = F.cross_entropy(intent_logits, intent_targets)
                    loss = loss + float(self.INTENT_LOSS_WEIGHT) * intent_loss

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

                    loss = (
                        loss
                        + float(self.MOE_LOAD_BALANCE_WEIGHT) * load_balance_loss
                        + float(self.MOE_GATE_ENTROPY_WEIGHT) * entropy_deficit
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
