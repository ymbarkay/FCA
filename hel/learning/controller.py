"""
hel/learning/controller.py

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

from hel.perception.base_model import BaseModel
from hel.learning.adapter_scalar import ScalarAdapter, ANGLE_DELTA_BOUND


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
    EWC_THETA_MOMENTUM = 0.998
    EWC_WARMUP_STEPS = 8

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

        self.checkpoint_path = checkpoint_path
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

            from hel.perception.feature_extractor import FeatureExtractor
            from hel.learning.live_policy_head import LivePolicyHead

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

        if self.adapter is not None and checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        self._ewc_steps = 0
        self._ewc_fisher = {}
        self._ewc_theta_star = {}

        if self.adapter is not None:
            self._reset_ewc_state()

        self._smoothed_angle_car = None
        self._last_mode_for_smoothing = None

    def _load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                state = checkpoint["model_state"]
            else:
                state = checkpoint

            self.adapter.load_state_dict(state)
            print(f"[controller] loaded adapter from {checkpoint_path}")

        except Exception as e:
            print(f"[controller] WARN — could not load adapter: {e}")

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

    def _predict_deep_policy(self, image, mode):
        from hel.learning.live_policy_head import angle_expected_value

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

        if self.adapter_type == "deep":
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
                angle_logits, speed_logit = self.adapter(batch_features)

                angle_targets = torch.clamp(batch_target_deltas, 0.0, 1.0)
                angle_class = torch.round(angle_targets * (self.NUM_ANGLE_CLASSES - 1)).long().squeeze(1)

                angle_loss = F.cross_entropy(angle_logits, angle_class)

                if train_speed:
                    speed_loss = F.binary_cross_entropy_with_logits(speed_logit, batch_target_speeds)
                else:
                    speed_loss = torch.tensor(0.0, device=self.device)

                loss = 2.0 * angle_loss + speed_loss

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
                loss = loss + self.EWC_LAMBDA * ewc_penalty

            self.optimizer.zero_grad()
            loss.backward()
            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), clip_grad_norm)
            if self.EWC_ENABLED:
                self._ewc_update_after_backward()
            self.optimizer.step()
            self.adapter.eval()
            self._ewc_steps += 1

        return float(loss.item())

    def save_checkpoint(self):
        if self.adapter is None or self.checkpoint_path is None:
            return

        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)

        with self.weights_lock:
            if self.adapter_type == "deep":
                torch.save(
                    {
                        "model_state": self.adapter.state_dict(),
                        "feature_dim": 512,
                        "num_angle_classes": self.NUM_ANGLE_CLASSES,
                        "architecture": "LivePolicyHead-512-256-128",
                    },
                    self.checkpoint_path,
                )
            else:
                torch.save(self.adapter.state_dict(), self.checkpoint_path)

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

                    self.adapter.load_state_dict(state)
                    self.adapter.eval()

                checkpoint_loaded = True
                print(f"[controller] reset adapter from checkpoint: {self.checkpoint_path}")

            except Exception as e:
                print(f"[controller] WARN — reset from checkpoint failed: {e}")

        if checkpoint_loaded:
            self._reset_ewc_state()
            return

        # Fallback: if no checkpoint is available, reinitialize parameters.
        with self.weights_lock:
            for _name, module in self.adapter.named_modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            self.adapter.eval()

        self._reset_ewc_state()

        print("[controller] reset adapter by reinitializing parameters (no checkpoint)")

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

    def _ewc_penalty(self):
        if self.adapter is None:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.adapter.named_parameters():
            if not param.requires_grad:
                continue

            fisher = self._ewc_fisher.get(name)
            theta_star = self._ewc_theta_star.get(name)
            if fisher is None or theta_star is None:
                continue

            penalty = penalty + torch.sum(fisher * (param - theta_star) ** 2)

        return penalty

    def _ewc_update_after_backward(self):
        """Online-EWC Fisher/anchor update from current gradients."""
        if self.adapter is None:
            return

        decay = float(np.clip(self.EWC_FISHER_DECAY, 0.0, 0.9999))
        theta_m = float(np.clip(self.EWC_THETA_MOMENTUM, 0.0, 0.9999))

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

                # Slowly move the reference anchor so adaptation is not frozen.
                self._ewc_theta_star[name].mul_(theta_m).add_(
                    param.detach(),
                    alpha=(1.0 - theta_m),
                )
