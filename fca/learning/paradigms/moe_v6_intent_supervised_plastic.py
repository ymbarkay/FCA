from fca.learning.paradigms.common import IntentSupervisedMoEPolicyHead
from fca.learning.paradigms.registry import LearningParadigmSpec


class LivePolicyHead(IntentSupervisedMoEPolicyHead):
    pass


def build_adapter(**kwargs):
    return LivePolicyHead(**kwargs)


PARADIGM = LearningParadigmSpec(
    paradigm_id="moe_v6_intent_supervised_plastic",
    label="MoE v6 intent-supervised plastic experts",
    description="Intent-routed MoE with direct selected-expert action loss and gate supervision.",
    family="moe",
    build_adapter=build_adapter,
    controller_overrides={
        "MOE_BALANCING_ENABLED": True,
        "MOE_LOAD_BALANCE_WEIGHT": 0.01,
        "MOE_GATE_ENTROPY_WEIGHT": 0.0015,
        "INTENT_ROUTING_ENABLED": True,
        "INTENT_EXPERT_SUPERVISION_ENABLED": True,
        "INTENT_EXPERT_DIRECT_LOSS_WEIGHT": 1.0,
        "INTENT_EXPERT_GATE_LOSS_WEIGHT": 0.35,
        "DEEP_ADAPTIVE_CLASS_WEIGHTING_ENABLED": False,
        "CONTEXT_TASK_ROUTING_ENABLED": False,
    },
)