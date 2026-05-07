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

    def __init__(
        self,
        base_model_path,
        adapter_type="scalar",
        checkpoint_path=None,
        learning_rate=1e-3,
        max_speed=,
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

            self.adapter.load_state_dict(state)
            print(f"[controller] loaded adapter from {checkpoint_path}")
            return True

        except Exception as e:
            print(f"[controller] WARN — could not load adapter: {e}")
            return False

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
                    "architecture": "LivePolicyHead-512-256-128",
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
        with self.weights_lock:
            return {
                "checkpoint_path": self.checkpoint_path or "",
                "checkpoint_locked": self._manual_checkpoint_locked,
                "last_manual_checkpoint_saved_at": self._last_manual_checkpoint_saved_at,
                "validated_save_count": int(max(0, self._validated_save_count)),
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

                    self.adapter.load_state_dict(state)
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
