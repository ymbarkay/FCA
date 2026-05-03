"""
fca/learning/adapter_scalar.py — Stage A: scalar-input adapter.

Takes the base model's scalar outputs as input. Limited adaptation capability
(can't truly distinguish visual environments) but lets us validate the entire
pipeline (state machine, replay buffer, GUI, training loop) without doing the
model split first.

INPUT:  [base_angle_norm (1), base_speed_prob (1)]            shape (2,)
OUTPUT: [delta_angle_norm (1), delta_speed_logit (1)]         shape (2,)

The delta is bounded to ±0.20 normalised (≈ ±16 degrees) via tanh.
"""
import torch
import torch.nn as nn


# Hardcoded bound on how much the adapter can correct.
# 0.20 in normalised units = 16 degrees in car units (since range is 80°)
# Limits catastrophic policy overwrites.
ANGLE_DELTA_BOUND = 0.20


class ScalarAdapter(nn.Module):
    """Small MLP. Input dim 2, output dim 2."""

    INPUT_DIM = 2
    HIDDEN_DIM = 32
    OUTPUT_DIM = 2

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(self.INPUT_DIM, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.ReLU(),
        )
        self.delta_angle_head = nn.Linear(self.HIDDEN_DIM, 1)
        self.delta_speed_head = nn.Linear(self.HIDDEN_DIM, 1)

        # Initialise output heads to zero — adapter starts as identity
        nn.init.zeros_(self.delta_angle_head.weight)
        nn.init.zeros_(self.delta_angle_head.bias)
        nn.init.zeros_(self.delta_speed_head.weight)
        nn.init.zeros_(self.delta_speed_head.bias)

    def forward(self, x):
        """
        x: tensor of shape (batch, 2) = [base_angle_norm, base_speed_prob]
        returns: (delta_angle_norm, delta_speed_logit) each shape (batch, 1)
        """
        h = self.trunk(x)
        delta_angle = ANGLE_DELTA_BOUND * torch.tanh(self.delta_angle_head(h))
        delta_speed_logit = self.delta_speed_head(h)
        return delta_angle, delta_speed_logit

    def get_input_features(self, base_angle_norm, base_speed_prob,
                            image=None, deep_features=None):
        """Build input tensor from base model outputs.

        Signature matches DeepAdapter for interchangeability — extra kwargs ignored.
        """
        return torch.tensor(
            [[float(base_angle_norm), float(base_speed_prob)]],
            dtype=torch.float32,
        )
