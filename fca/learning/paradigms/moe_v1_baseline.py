from fca.learning.paradigms.common import BaselineMoEPolicyHead
from fca.learning.paradigms.registry import LearningParadigmSpec


class LivePolicyHead(BaselineMoEPolicyHead):
    pass


def build_adapter(**kwargs):
    return LivePolicyHead(**kwargs)


PARADIGM = LearningParadigmSpec(
    paradigm_id="moe_v1_baseline",
    label="MoE v1 baseline",
    description="Plain soft-gated MoE with shared stem and no extra routing losses.",
    family="moe",
    build_adapter=build_adapter,
    controller_overrides={
        "MOE_BALANCING_ENABLED": False,
        "INTENT_ROUTING_ENABLED": False,
        "DEEP_ADAPTIVE_CLASS_WEIGHTING_ENABLED": False,
        "CONTEXT_TASK_ROUTING_ENABLED": False,
    },
)