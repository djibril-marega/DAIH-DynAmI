"""Metrics used by the offline structural controller."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math


@dataclass
class StateMetrics:
    state: str
    free_energy: float
    # Legacy L2 norm kept for diagnostics only; it no longer drives phases.
    gradient_norm: float
    # Component-wise dF/dmu, retained without norm reduction for structural phases.
    gradient_raw: list[float]
    # L1 sum of absolute dF/dmu components, used by continuous phases 1-4.
    gradient_sum: float
    gradient_finite: bool
    precision: float
    complexity: float
    accuracy: float
    # Strict-mode aggregate sensitivity of local generative outputs to Q_i.
    local_sensitivity: float = 0.0
    # Frobenius norm of Hessian d²F_i/dQ_i² (strict paper mode).
    hessian_norm: float = 0.0
    local_sensitivity_finite: bool = False
    hessian_finite: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, RuntimeError):
        return default
    return result if math.isfinite(result) else default
