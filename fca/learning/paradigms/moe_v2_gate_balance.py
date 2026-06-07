from fca.learning.paradigms.common import BaselineMoEPolicyHead
from fca.learning.paradigms.registry import LearningParadigmSpec


class LivePolicyHead(BaselineMoEPolicyHead):
    pass


def build_adapter(**kwargs):
    return LivePolicyHead(**kwargs)


PARADIGM = LearningParadigmSpec(
    paradigm_id="moe_v2_gate_balance",
    label="MoE v2 gate balance",
    description="MoE baseline plus batch load-balancing and gate entropy regularization.",
    family="moe",
    build_adapter=build_adapter,
    controller_overrides={
        "MOE_BALANCING_ENABLED": True,
        "INTENT_ROUTING_ENABLED": False,
        "DEEP_ADAPTIVE_CLASS_WEIGHTING_ENABLED": False,
        "CONTEXT_TASK_ROUTING_ENABLED": False,
    },
)