import torch
import torch.nn as nn


def angle_expected_value(angle_logits):
    probs = torch.softmax(angle_logits, dim=-1)
    values = torch.linspace(0.0, 1.0, angle_logits.shape[-1], device=angle_logits.device)
    return probs @ values


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


class DensePolicyHead(nn.Module):
    def __init__(
        self,
        feature_dim=512,
        hidden1=256,
        hidden2=128,
        num_angle_classes=17,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_angle_classes = num_angle_classes
        self.architecture_name = f"LivePolicyHead-{feature_dim}-{hidden1}-{hidden2}"

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
        hidden = self.net(x)
        angle_logits = self.angle_head(hidden)
        speed_logit = self.speed_head(hidden)
        return angle_logits, speed_logit

    @staticmethod
    def _average_tensors(state_dict, keys):
        tensors = [state_dict[key] for key in keys if key in state_dict]
        if not tensors:
            raise KeyError(f"No tensors found for keys: {keys}")
        return torch.stack(tensors, dim=0).mean(dim=0)

    def load_compatible_state_dict(self, state_dict):
        dense_keys = {
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
        if dense_keys.issubset(state_dict.keys()):
            self.load_state_dict(state_dict, strict=False)
            return "dense"

        moe_keys = {"stem.0.weight", "gate.weight", "experts.0.hidden.weight"}
        if moe_keys.issubset(state_dict.keys()):
            expert_hidden_keys = [
                key for key in state_dict.keys() if key.startswith("experts.") and key.endswith("hidden.weight")
            ]
            expert_hidden_bias_keys = [
                key for key in state_dict.keys() if key.startswith("experts.") and key.endswith("hidden.bias")
            ]
            expert_angle_weight_keys = [
                key for key in state_dict.keys() if key.startswith("experts.") and key.endswith("angle_head.weight")
            ]
            expert_angle_bias_keys = [
                key for key in state_dict.keys() if key.startswith("experts.") and key.endswith("angle_head.bias")
            ]
            expert_speed_weight_keys = [
                key for key in state_dict.keys() if key.startswith("experts.") and key.endswith("speed_head.weight")
            ]
            expert_speed_bias_keys = [
                key for key in state_dict.keys() if key.startswith("experts.") and key.endswith("speed_head.bias")
            ]

            with torch.no_grad():
                self.net[0].weight.copy_(state_dict["stem.0.weight"])
                self.net[0].bias.copy_(state_dict["stem.0.bias"])
                self.net[1].weight.copy_(state_dict["stem.1.weight"])
                self.net[1].bias.copy_(state_dict["stem.1.bias"])
                self.net[4].weight.copy_(self._average_tensors(state_dict, expert_hidden_keys))
                self.net[4].bias.copy_(self._average_tensors(state_dict, expert_hidden_bias_keys))
                self.angle_head.weight.copy_(self._average_tensors(state_dict, expert_angle_weight_keys))
                self.angle_head.bias.copy_(self._average_tensors(state_dict, expert_angle_bias_keys))
                self.speed_head.weight.copy_(self._average_tensors(state_dict, expert_speed_weight_keys))
                self.speed_head.bias.copy_(self._average_tensors(state_dict, expert_speed_bias_keys))
            return "collapsed-moe"

        self.load_state_dict(state_dict)
        return "dense"


class BaselineMoEPolicyHead(nn.Module):
    def __init__(
        self,
        feature_dim=512,
        hidden1=256,
        hidden2=128,
        num_angle_classes=17,
        num_experts=4,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_angle_classes = num_angle_classes
        self.num_experts = max(1, int(num_experts))
        self.architecture_name = f"LivePolicyMoEHead-{feature_dim}-{hidden1}-{hidden2}x{self.num_experts}"

        self.stem = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.gate = nn.Linear(hidden1, self.num_experts)
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

    def _gate_logits(self, shared):
        return self.gate(shared)

    @staticmethod
    def _gate_probs_from_logits(gate_logits, gate_temperature=1.0):
        temp = float(gate_temperature) if gate_temperature is not None else 1.0
        temp = max(0.25, temp)
        return torch.softmax(gate_logits / temp, dim=-1)

    def _stack_expert_outputs(self, shared):
        angle_logits = []
        speed_logits = []
        for expert in self.experts:
            expert_angle, expert_speed = expert(shared)
            angle_logits.append(expert_angle)
            speed_logits.append(expert_speed)
        return torch.stack(angle_logits, dim=1), torch.stack(speed_logits, dim=1)

    @staticmethod
    def _mix_expert_outputs(angle_stack, speed_stack, gate_probs):
        mix = gate_probs.unsqueeze(-1)
        angle_logits = torch.sum(angle_stack * mix, dim=1)
        speed_logit = torch.sum(speed_stack * mix, dim=1)
        return angle_logits, speed_logit

    def forward_with_gate(self, x, gate_temperature=1.0):
        shared = self.stem(x)
        gate_logits = self._gate_logits(shared)
        gate_probs = self._gate_probs_from_logits(gate_logits, gate_temperature=gate_temperature)

        angle_stack, speed_stack = self._stack_expert_outputs(shared)
        angle_logits, speed_logit = self._mix_expert_outputs(angle_stack, speed_stack, gate_probs)
        return angle_logits, speed_logit, gate_probs

    def forward(self, x):
        angle_logits, speed_logit, _gate_probs = self.forward_with_gate(x)
        return angle_logits, speed_logit

    def _load_legacy_dense_state(self, state_dict):
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

    def load_compatible_state_dict(self, state_dict):
        native_keys = {"stem.0.weight", "gate.weight", "experts.0.hidden.weight"}
        if native_keys.issubset(state_dict.keys()):
            self.load_state_dict(state_dict, strict=False)
            if "task_head.weight" in state_dict:
                return "moe-context-task"
            if "intent_head.weight" in state_dict:
                return "moe-intent"
            return "moe"

        dense_keys = {
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
        if dense_keys.issubset(state_dict.keys()):
            self._load_legacy_dense_state(state_dict)
            return "legacy-single-head"

        self.load_state_dict(state_dict)
        return "moe"


class IntentMoEPolicyHead(BaselineMoEPolicyHead):
    def __init__(
        self,
        feature_dim=512,
        hidden1=256,
        hidden2=128,
        num_angle_classes=17,
        num_experts=4,
        num_intents=4,
    ):
        self.num_intents = max(2, int(num_intents))
        super().__init__(
            feature_dim=feature_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            num_angle_classes=num_angle_classes,
            num_experts=num_experts,
        )
        self.architecture_name = (
            f"LivePolicyIntentMoEHead-{feature_dim}-{hidden1}-{hidden2}x{self.num_experts}"
        )
        self.intent_head = nn.Linear(hidden1, self.num_intents)
        self.intent_to_gate = nn.Linear(self.num_intents, self.num_experts, bias=False)

        nn.init.normal_(self.intent_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.intent_head.bias)
        nn.init.zeros_(self.intent_to_gate.weight)

    def _gate_logits(self, shared):
        intent_logits = self.intent_head(shared)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        gate_logits = self.gate(shared) + self.intent_to_gate(intent_probs)
        return gate_logits, intent_logits, intent_probs

    def forward_with_gate(self, x, gate_temperature=1.0):
        shared = self.stem(x)
        gate_logits, intent_logits, intent_probs = self._gate_logits(shared)
        gate_probs = self._gate_probs_from_logits(gate_logits, gate_temperature=gate_temperature)

        angle_stack, speed_stack = self._stack_expert_outputs(shared)
        angle_logits, speed_logit = self._mix_expert_outputs(angle_stack, speed_stack, gate_probs)
        return angle_logits, speed_logit, gate_probs, intent_logits, intent_probs

    def forward(self, x):
        angle_logits, speed_logit, *_rest = self.forward_with_gate(x)
        return angle_logits, speed_logit


class IntentSupervisedMoEPolicyHead(IntentMoEPolicyHead):
    def __init__(
        self,
        feature_dim=512,
        hidden1=256,
        hidden2=128,
        num_angle_classes=17,
        num_experts=4,
        num_intents=4,
    ):
        super().__init__(
            feature_dim=feature_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            num_angle_classes=num_angle_classes,
            num_experts=num_experts,
            num_intents=num_intents,
        )
        self.architecture_name = (
            f"LivePolicyIntentSupervisedMoEHead-{feature_dim}-{hidden1}-{hidden2}x{self.num_experts}"
        )
        self.intent_gate_scale = nn.Parameter(torch.tensor(1.0))

        with torch.no_grad():
            self.intent_to_gate.weight.zero_()
            for index in range(min(self.num_intents, self.num_experts)):
                self.intent_to_gate.weight[index, index] = 1.0

    def _gate_logits(self, shared):
        intent_logits = self.intent_head(shared)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        gate_logits = self.gate(shared) + self.intent_gate_scale * self.intent_to_gate(intent_probs)
        return gate_logits, intent_logits, intent_probs

    def forward_with_gate(self, x, gate_temperature=1.0):
        shared = self.stem(x)
        gate_logits, intent_logits, intent_probs = self._gate_logits(shared)
        gate_probs = self._gate_probs_from_logits(gate_logits, gate_temperature=gate_temperature)
        angle_stack, speed_stack = self._stack_expert_outputs(shared)
        angle_logits, speed_logit = self._mix_expert_outputs(angle_stack, speed_stack, gate_probs)
        return (
            angle_logits,
            speed_logit,
            gate_probs,
            intent_logits,
            intent_probs,
            None,
            None,
            angle_stack,
            speed_stack,
            gate_logits,
        )


class ContextTaskMoEPolicyHead(IntentMoEPolicyHead):
    def __init__(
        self,
        feature_dim=512,
        hidden1=256,
        hidden2=128,
        num_angle_classes=17,
        num_experts=4,
        num_intents=4,
    ):
        super().__init__(
            feature_dim=feature_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            num_angle_classes=num_angle_classes,
            num_experts=num_experts,
            num_intents=num_intents,
        )
        self.architecture_name = (
            f"LivePolicyContextTaskMoEHead-{feature_dim}-{hidden1}-{hidden2}x{self.num_experts}"
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden2),
        )
        self.task_head = nn.Linear(hidden2, self.num_intents)
        self.context_to_gate = nn.Linear(hidden2, self.num_experts, bias=False)
        self.task_to_gate = nn.Linear(self.num_intents, self.num_experts, bias=False)

        nn.init.normal_(self.task_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.task_head.bias)
        nn.init.zeros_(self.context_to_gate.weight)
        nn.init.zeros_(self.task_to_gate.weight)

    def forward_with_gate(self, x, gate_temperature=1.0):
        shared = self.stem(x)
        intent_logits = self.intent_head(shared)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        context_embedding = self.context_encoder(shared)
        task_logits = self.task_head(context_embedding)
        task_probs = torch.softmax(task_logits, dim=-1)
        gate_logits = (
            self.gate(shared)
            + self.intent_to_gate(intent_probs)
            + self.context_to_gate(context_embedding)
            + self.task_to_gate(task_probs)
        )
        gate_probs = self._gate_probs_from_logits(gate_logits, gate_temperature=gate_temperature)

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
        return (
            angle_logits,
            speed_logit,
            gate_probs,
            intent_logits,
            intent_probs,
            task_logits,
            task_probs,
            context_embedding,
        )

    def forward(self, x):
        (
            angle_logits,
            speed_logit,
            _gate_probs,
            _intent_logits,
            _intent_probs,
            _task_logits,
            _task_probs,
            _context_embedding,
        ) = self.forward_with_gate(x)
        return angle_logits, speed_logit

    def load_compatible_state_dict(self, state_dict):
        native_keys = {"stem.0.weight", "gate.weight", "experts.0.hidden.weight"}
        if native_keys.issubset(state_dict.keys()):
            self.load_state_dict(state_dict, strict=False)
            if "task_to_gate.weight" not in state_dict and "intent_to_gate.weight" in state_dict:
                with torch.no_grad():
                    self.task_to_gate.weight.copy_(state_dict["intent_to_gate.weight"])
            if "task_head.weight" in state_dict:
                return "moe-context-task"
            if "intent_head.weight" in state_dict:
                return "moe-intent"
            return "moe"

        dense_keys = {
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
        if dense_keys.issubset(state_dict.keys()):
            self._load_legacy_dense_state(state_dict)
            return "legacy-single-head"

        self.load_state_dict(state_dict)
        return "moe-context-task"