"""
fca/learning/adapter_deep.py

Stage B adapter.

Input:
    deep feature vector, e.g. 512-d from trained MobileNetV2 dense_2
    + base_angle_norm
    + base_speed_prob

Default input dimension:
    512 + 2 = 514

Architecture:
    514 → 64 → 32 → 16 → heads

Parameter count:
    ~35k

Outputs:
    delta_angle_norm bounded to ±0.20
    delta_speed_logit unbounded
"""
import torch
import torch.nn as nn


ANGLE_DELTA_BOUND = 0.08


class DeepAdapter(nn.Module):
    def __init__(
        self,
        feature_dim=512,
        hidden1=64,
        hidden2=32,
        hidden3=16,
        angle_delta_bound=ANGLE_DELTA_BOUND,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.input_dim = feature_dim + 2
        self.angle_delta_bound = angle_delta_bound

        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden1),
            nn.ReLU(inplace=True),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(inplace=True),
        )

        self.angle_head = nn.Linear(hidden3, 1)
        self.speed_head = nn.Linear(hidden3, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset hidden layers normally, but zero the output heads so the adapter
        starts with exactly zero behaviour change.
        """
        for module in self.net:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        nn.init.zeros_(self.angle_head.weight)
        nn.init.zeros_(self.angle_head.bias)
        nn.init.zeros_(self.speed_head.weight)
        nn.init.zeros_(self.speed_head.bias)

    def forward(self, x):
        h = self.net(x)

        delta_angle = self.angle_delta_bound * torch.tanh(self.angle_head(h))
        delta_speed_logit = self.speed_head(h)

        return delta_angle, delta_speed_logit

    def get_input_features(
        self,
        base_angle_norm,
        base_speed_prob,
        image=None,
        deep_features=None,
    ):
        """
        Build adapter input tensor.

        For deep adapter, deep_features must be provided.

        Returns:
            tensor shape: (1, feature_dim + 2)
        """
        if deep_features is None:
            raise ValueError(
                "DeepAdapter requires deep_features. "
                "Controller must call FeatureExtractor.extract(image)."
            )

        if isinstance(deep_features, torch.Tensor):
            feat = deep_features.detach().float().view(-1)
        else:
            feat = torch.tensor(deep_features, dtype=torch.float32).view(-1)

        if feat.numel() != self.feature_dim:
            raise ValueError(
                f"Expected deep feature dim {self.feature_dim}, got {feat.numel()}"
            )

        scalars = torch.tensor(
            [float(base_angle_norm), float(base_speed_prob)],
            dtype=torch.float32,
        )

        x = torch.cat([feat, scalars], dim=0).unsqueeze(0)

        return x