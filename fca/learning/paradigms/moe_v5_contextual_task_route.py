from fca.learning.paradigms.common import ContextTaskMoEPolicyHead
from fca.learning.paradigms.registry import LearningParadigmSpec


class LivePolicyHead(ContextTaskMoEPolicyHead):
    pass


def build_adapter(**kwargs):
    return LivePolicyHead(**kwargs)


PARADIGM = LearningParadigmSpec(
    paradigm_id="moe_v5_contextual_task_route",
    label="MoE v5 contextual task route",
    description="Intent-routed MoE extended with contextual task embeddings and extra routing losses.",
    family="moe",
    build_adapter=build_adapter,
    controller_overrides={
        "MOE_BALANCING_ENABLED": True,
        "INTENT_ROUTING_ENABLED": True,
        "DEEP_ADAPTIVE_CLASS_WEIGHTING_ENABLED": False,
        "CONTEXT_TASK_ROUTING_ENABLED": True,
    },
)