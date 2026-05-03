"""
fca/learning/replay_buffer.py — FIFO buffer of teaching samples.

Stores (input_features, target_delta_angle, target_speed_norm) tuples.
The trainer thread samples small batches from this for gradient steps.
"""
import random
import threading
from collections import deque

import torch


class ReplayBuffer:
    """Thread-safe replay buffer for online learning."""

    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def add(self, input_features, target_delta_angle, target_speed_norm, sample_kind="correction"):
        """input_features: tensor or numpy array — stored on CPU."""
        if isinstance(input_features, torch.Tensor):
            features = input_features.detach().cpu().clone()
        else:
            features = torch.tensor(input_features, dtype=torch.float32)

        with self.lock:
            self.buffer.append((
                features,
                float(target_delta_angle),
                float(target_speed_norm),
                str(sample_kind),
            ))

    def sample(self, batch_size):
        """Return a tuple (features, target_deltas, target_speeds) or None."""
        with self.lock:
            if len(self.buffer) == 0:
                return None

            n = min(batch_size, len(self.buffer))

            correction_samples = []
            anchor_samples = []
            for item in self.buffer:
                kind = item[3] if len(item) > 3 else "correction"
                if kind == "anchor":
                    anchor_samples.append(item)
                else:
                    correction_samples.append(item)

            if correction_samples and anchor_samples:
                target_corr = n // 2
                target_anchor = n - target_corr

                n_corr = min(target_corr, len(correction_samples))
                n_anchor = min(target_anchor, len(anchor_samples))

                remaining = n - (n_corr + n_anchor)
                if remaining > 0:
                    extra_corr = min(remaining, len(correction_samples) - n_corr)
                    n_corr += extra_corr
                    remaining -= extra_corr

                if remaining > 0:
                    extra_anchor = min(remaining, len(anchor_samples) - n_anchor)
                    n_anchor += extra_anchor

                chosen_corr = random.sample(
                    correction_samples,
                    n_corr,
                )
                chosen_anchor = random.sample(
                    anchor_samples,
                    n_anchor,
                )

                batch = chosen_corr + chosen_anchor
            else:
                batch = random.sample(self.buffer, n)

        features_list = [b[0] for b in batch]
        deltas = torch.tensor([b[1] for b in batch], dtype=torch.float32).unsqueeze(-1)
        speeds = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(-1)

        # Stack features — handle variable input dim by re-shaping
        features = torch.stack([
            f.squeeze(0) if f.dim() > 1 else f
            for f in features_list
        ], dim=0)

        return features, deltas, speeds

    def __len__(self):
        with self.lock:
            return len(self.buffer)

    def correction_count(self):
        """Count correction samples (non-anchor) currently in buffer."""
        with self.lock:
            c = 0
            for item in self.buffer:
                kind = item[3] if len(item) > 3 else "correction"
                if kind != "anchor":
                    c += 1
            return c

    def clear(self):
        with self.lock:
            self.buffer.clear()
