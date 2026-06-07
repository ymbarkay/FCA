"""Compatibility wrapper for the default online learning paradigm (MoE v4 intent routing)."""

from fca.learning.paradigms import angle_expected_value
from fca.learning.paradigms.moe_v4_intent_routing import LivePolicyHead

__all__ = ["LivePolicyHead", "angle_expected_value"]
