from fca.learning.paradigms.common import DensePolicyHead
from fca.learning.paradigms.registry import LearningParadigmSpec


class LivePolicyHead(DensePolicyHead):
    pass


def build_adapter(**kwargs):
    return LivePolicyHead(**kwargs)


PARADIGM = LearningParadigmSpec(
    paradigm_id="dense_single_head",
    label="Dense single head",
    description="Single shared dense policy head without expert routing.",
    family="dense",
    build_adapter=build_adapter,
    controller_overrides={
        "MOE_BALANCING_ENABLED": False,
        "INTENT_ROUTING_ENABLED": False,
        "DEEP_ADAPTIVE_CLASS_WEIGHTING_ENABLED": False,
        "CONTEXT_TASK_ROUTING_ENABLED": False,
    },
)