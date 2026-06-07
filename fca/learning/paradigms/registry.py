from dataclasses import dataclass, field
import importlib
import pkgutil
from typing import Any, Callable


@dataclass(frozen=True)
class LearningParadigmSpec:
    paradigm_id: str
    label: str
    description: str
    family: str
    build_adapter: Callable[..., Any]
    controller_overrides: dict[str, Any] = field(default_factory=dict)

    def to_snapshot(self):
        return {
            "id": self.paradigm_id,
            "label": self.label,
            "description": self.description,
            "family": self.family,
        }


_DISCOVERED_PARADIGMS = None


def _discover_paradigms():
    global _DISCOVERED_PARADIGMS
    if _DISCOVERED_PARADIGMS is not None:
        return _DISCOVERED_PARADIGMS

    package = importlib.import_module("fca.learning.paradigms")
    registry = {}

    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name in {"common", "registry"}:
            continue

        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        paradigm = getattr(module, "PARADIGM", None)
        if paradigm is None:
            continue

        registry[paradigm.paradigm_id] = paradigm

    _DISCOVERED_PARADIGMS = registry
    return _DISCOVERED_PARADIGMS


def get_learning_paradigm(paradigm_id):
    paradigms = _discover_paradigms()
    if paradigm_id not in paradigms:
        available = ", ".join(sorted(paradigms.keys())) or "none"
        raise ValueError(f"Unknown learning paradigm '{paradigm_id}'. Available: {available}")
    return paradigms[paradigm_id]


def list_learning_paradigms():
    paradigms = _discover_paradigms()
    return [paradigms[key] for key in sorted(paradigms.keys())]


def list_learning_paradigm_snapshots():
    return [spec.to_snapshot() for spec in list_learning_paradigms()]