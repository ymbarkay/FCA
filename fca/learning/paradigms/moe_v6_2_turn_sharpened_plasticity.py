from fca.learning.paradigms.common import IntentSupervisedMoEPolicyHead
from fca.learning.paradigms.registry import LearningParadigmSpec


class LivePolicyHead(IntentSupervisedMoEPolicyHead):
    pass


def build_adapter(**kwargs):
    return LivePolicyHead(**kwargs)


PARADIGM = LearningParadigmSpec(
    paradigm_id="moe_v6_2_turn_sharpened_plasticity",
    label="MoE v6.2 turn-sharpened plasticity",
    description="v6.1 ownership with turn-specific focused TEACH sharpening for left/right acquisition.",
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
        "INTENT_CENTER_MARGIN_NORM": 0.05,
        "INTENT_EXPERT_DIRECT_LOSS_WEIGHT_BY_INTENT": {
            "left": 1.35,
            "right": 1.35,
        },
        "INTENT_EXPERT_GATE_LOSS_WEIGHT_BY_INTENT": {
            "left": 0.42,
            "right": 0.42,
        },
        "INFERENCE_GATE_TEMPERATURE": 1.10,
        "TRAIN_GATE_TEMPERATURE": 1.0,
        "TEACH_GATE_TEMPERATURE": 1.12,
        "REHEARSAL_GATE_TEMPERATURE": 1.0,
        "TEACH_GATE_TEMPERATURE_BY_INTENT": {
            "left": 0.92,
            "right": 0.92,
        },
        "TEACH_LOAD_BALANCE_WEIGHT_MULT": 0.25,
        "TEACH_GATE_ENTROPY_WEIGHT_MULT": 0.15,
        "REHEARSAL_LOAD_BALANCE_WEIGHT_MULT": 1.20,
        "REHEARSAL_GATE_ENTROPY_WEIGHT_MULT": 1.60,
        "TEACH_LOAD_BALANCE_WEIGHT_MULT_BY_INTENT": {
            "left": 0.10,
            "right": 0.10,
        },
        "TEACH_GATE_ENTROPY_WEIGHT_MULT_BY_INTENT": {
            "left": 0.05,
            "right": 0.05,
        },
        "GATE_LR_SCALE": 0.50,
        "FOCUSED_REHEARSAL_BATCH_SCALE": 1.25,
        "FOCUSED_REHEARSAL_BATCH_SCALE_BY_INTENT": {
            "left": 0.50,
            "right": 0.50,
            "straight": 0.50,
        },
        "FOCUSED_TARGET_REPEAT_SCALE_BY_INTENT": {
            "left": 2.50,
            "right": 2.50,
            "straight": 2.0,
        },
        "TEACH_FOCUSED_STEP_SCALE_BY_INTENT": {
            "left": 1.45,
            "right": 1.45,
            "straight": 1.25,
        },
        "TEACH_FOCUSED_LR_SCALE_BY_INTENT": {
            "left": 2.00,
            "right": 2.00,
            "straight": 1.65,
        },
        "TEACH_FOCUSED_MAX_LR_MULTIPLIER_BY_INTENT": {
            "left": 8.50,
            "right": 8.50,
            "straight": 7.00,
        },
        "REHEARSAL_BATCH_SIZE_SCALE": 1.15,
        "REHEARSAL_STEPS_SCALE": 1.25,
        "DEEP_ADAPTIVE_CLASS_WEIGHTING_ENABLED": False,
        "CONTEXT_TASK_ROUTING_ENABLED": False,
    },
)