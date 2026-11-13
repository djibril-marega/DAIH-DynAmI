"""Numerical diagnostics used by the adaptive offline controller."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math


@dataclass
class StateMetrics:
    state: str
    free_energy: float
    gradient_norm: float
    gradient_raw: list[float]
    gradient_sum: float
    gradient_finite: bool
    precision: float
    complexity: float
    accuracy: float
    local_sensitivity: float = 0.0
    hessian_norm: float = 0.0
    local_sensitivity_finite: bool = False
    hessian_finite: bool = False
    shock: float = 0.0
    shock_finite: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, RuntimeError):
        return default
    return result if math.isfinite(result) else default
