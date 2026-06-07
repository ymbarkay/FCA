from fca.learning.paradigms.common import BaselineMoEPolicyHead
from fca.learning.paradigms.registry import LearningParadigmSpec


class LivePolicyHead(BaselineMoEPolicyHead):
    pass


def build_adapter(**kwargs):
    return LivePolicyHead(**kwargs)


PARADIGM = LearningParadigmSpec(
    paradigm_id="moe_v3_adaptive_class_weight",
    label="MoE v3 adaptive class weight",
    description="Gate-balanced MoE with adaptive angle-class reweighting for rare or hard targets.",
    family="moe",
    build_adapter=build_adapter,
    controller_overrides={
        "MOE_BALANCING_ENABLED": True,
        "INTENT_ROUTING_ENABLED": False,
        "DEEP_ADAPTIVE_CLASS_WEIGHTING_ENABLED": True,
        "CONTEXT_TASK_ROUTING_ENABLED": False,
    },
)