"""
fca/learning/live_policy_head.py

PyTorch policy head trained on 512-d visual features.
Produces:
  - angle class logits (17 classes)
  - speed logit (binary)

The current head is a lightweight mixture-of-experts (MoE):
  - shared stem learns common features
  - several experts specialize on subsets of situations
  - a learned gate mixes expert outputs per sample

Legacy single-head checkpoints are still supported and are expanded into the
MoE experts on load, preserving existing trained behavior.
"""
import torch
import torch.nn as nn


class _PolicyExpert(nn.Module):
    def __init__(self, hidden1, hidden2, num_angle_classes):
        super().__init__()
        self.hidden = nn.Linear(hidden1, hidden2)
        self.activation = nn.ReLU(inplace=True)
        self.angle_head = nn.Linear(hidden2, num_angle_classes)
        self.speed_head = nn.Linear(hidden2, 1)

    def forward(self, x):
        hidden = self.activation(self.hidden(x))
        angle_logits = self.angle_head(hidden)
        speed_logit = self.speed_head(hidden)
        return angle_logits, speed_logit


class LivePolicyHead(nn.Module):
    def __init__(
        self,
        feature_dim=512,
        hidden1=256,
        hidden2=128,
        num_angle_classes=17,
        num_experts=4,
        num_intents=4,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_angle_classes = num_angle_classes
        self.num_experts = max(1, int(num_experts))
        self.num_intents = max(2, int(num_intents))
        self.architecture_name = (
            f"LivePolicyIntentMoEHead-{feature_dim}-{hidden1}-{hidden2}x{self.num_experts}"
        )

        self.stem = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.gate = nn.Linear(hidden1, self.num_experts)
        self.intent_head = nn.Linear(hidden1, self.num_intents)
        self.intent_to_gate = nn.Linear(self.num_intents, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                _PolicyExpert(
                    hidden1=hidden1,
                    hidden2=hidden2,
                    num_angle_classes=num_angle_classes,
                )
                for _ in range(self.num_experts)
            ]
        )

        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.intent_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.intent_head.bias)
        nn.init.zeros_(self.intent_to_gate.weight)

    def forward_with_gate(self, x):
        shared = self.stem(x)
        intent_logits = self.intent_head(shared)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        gate_logits = self.gate(shared) + self.intent_to_gate(intent_probs)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        angle_logits = []
        speed_logits = []
        for expert in self.experts:
            expert_angle, expert_speed = expert(shared)
            angle_logits.append(expert_angle)
            speed_logits.append(expert_speed)

        angle_stack = torch.stack(angle_logits, dim=1)
        speed_stack = torch.stack(speed_logits, dim=1)
        mix = gate_probs.unsqueeze(-1)

        angle_logits = torch.sum(angle_stack * mix, dim=1)
        speed_logit = torch.sum(speed_stack * mix, dim=1)
        return angle_logits, speed_logit, gate_probs, intent_logits, intent_probs

    def forward(self, x):
        angle_logits, speed_logit, _gate_probs, _intent_logits, _intent_probs = self.forward_with_gate(x)
        return angle_logits, speed_logit

    def load_compatible_state_dict(self, state_dict):
        """Load native MoE checkpoints or expand legacy single-head checkpoints."""
        native_keys = {"stem.0.weight", "gate.weight", "experts.0.hidden.weight"}
        if native_keys.issubset(state_dict.keys()):
            self.load_state_dict(state_dict, strict=False)
            if "intent_head.weight" in state_dict:
                return "moe-intent"
            return "moe"

        legacy_keys = {
            "net.0.weight",
            "net.0.bias",
            "net.1.weight",
            "net.1.bias",
            "net.4.weight",
            "net.4.bias",
            "angle_head.weight",
            "angle_head.bias",
            "speed_head.weight",
            "speed_head.bias",
        }
        if not legacy_keys.issubset(state_dict.keys()):
            self.load_state_dict(state_dict)
            return "moe"

        with torch.no_grad():
            self.stem[0].weight.copy_(state_dict["net.0.weight"])
            self.stem[0].bias.copy_(state_dict["net.0.bias"])
            self.stem[1].weight.copy_(state_dict["net.1.weight"])
            self.stem[1].bias.copy_(state_dict["net.1.bias"])

            for expert in self.experts:
                expert.hidden.weight.copy_(state_dict["net.4.weight"])
                expert.hidden.bias.copy_(state_dict["net.4.bias"])
                expert.angle_head.weight.copy_(state_dict["angle_head.weight"])
                expert.angle_head.bias.copy_(state_dict["angle_head.bias"])
                expert.speed_head.weight.copy_(state_dict["speed_head.weight"])
                expert.speed_head.bias.copy_(state_dict["speed_head.bias"])

        return "legacy-single-head"


def angle_expected_value(angle_logits):
    probs = torch.softmax(angle_logits, dim=-1)
    values = torch.linspace(0.0, 1.0, angle_logits.shape[-1], device=angle_logits.device)
    return probs @ values
