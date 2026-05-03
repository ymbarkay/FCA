"""
hel/learning/live_policy_head.py

PyTorch policy head trained on 512-d visual features.
Produces:
  - angle class logits (17 classes)
  - speed logit (binary)
"""
import torch
import torch.nn as nn


class LivePolicyHead(nn.Module):
    def __init__(self, feature_dim=512, hidden1=256, hidden2=128, num_angle_classes=17):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_angle_classes = num_angle_classes

        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
        )

        self.angle_head = nn.Linear(hidden2, num_angle_classes)
        self.speed_head = nn.Linear(hidden2, 1)

    def forward(self, x):
        h = self.net(x)
        angle_logits = self.angle_head(h)
        speed_logit = self.speed_head(h)
        return angle_logits, speed_logit


def angle_expected_value(angle_logits):
    probs = torch.softmax(angle_logits, dim=-1)
    values = torch.linspace(0.0, 1.0, angle_logits.shape[-1], device=angle_logits.device)
    return probs @ values
