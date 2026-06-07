from fca.learning.paradigms.common import IntentSupervisedMoEPolicyHead
from fca.learning.paradigms.registry import LearningParadigmSpec


class LivePolicyHead(IntentSupervisedMoEPolicyHead):
    pass


def build_adapter(**kwargs):
    return LivePolicyHead(**kwargs)


PARADIGM = LearningParadigmSpec(
    paradigm_id="moe_v6_1_stabilized_ownership",
    label="MoE v6.1 stabilized ownership",
    description="TEACH-only intent-owned expert supervision with softer collaborative routing and stronger rehearsal.",
    family="moe",
    build_adapter=build_adapter,
    controller_overrides={
        "MOE_BALANCING_ENABLED": True,
        "MOE_LOAD_BALANCE_WEIGHT": 0.012,
        "MOE_GATE_ENTROPY_WEIGHT": 0.0020,
        "INTENT_ROUTING_ENABLED": True,
        "INTENT_EXPERT_SUPERVISION_ENABLED": True,
        "INTENT_EXPERT_SUPERVISION_TEACH_ONLY": True,
        "INTENT_EXPERT_DIRECT_LOSS_WEIGHT": 1.0,
        "INTENT_EXPERT_GATE_LOSS_WEIGHT": 0.28,
        "INFERENCE_GATE_TEMPERATURE": 1.10,
        "TRAIN_GATE_TEMPERATURE": 1.0,
        "TEACH_GATE_TEMPERATURE": 1.12,
        "REHEARSAL_GATE_TEMPERATURE": 1.0,
        "TEACH_LOAD_BALANCE_WEIGHT_MULT": 0.25,
        "TEACH_GATE_ENTROPY_WEIGHT_MULT": 0.15,
        "REHEARSAL_LOAD_BALANCE_WEIGHT_MULT": 1.20,
        "REHEARSAL_GATE_ENTROPY_WEIGHT_MULT": 1.60,
        "GATE_LR_SCALE": 0.50,
        "FOCUSED_REHEARSAL_BATCH_SCALE": 1.25,
        "REHEARSAL_BATCH_SIZE_SCALE": 1.15,
        "REHEARSAL_STEPS_SCALE": 1.25,
        "DEEP_ADAPTIVE_CLASS_WEIGHTING_ENABLED": False,
        "CONTEXT_TASK_ROUTING_ENABLED": False,
    },
)