"""
Offline learning controller for Agent_DAIH.

One call to OfflineTrainer.step() is one global offline iteration:
1. evaluate the supplied historical observations;
2. optimize trainable torch modules through variational free energy;
3. keep the current posterior beliefs Q unchanged;
4. evaluate the test observations;
5. update the per-state seven-phase controller and persist its plan.

Structural mutations are applied automatically through the declarative model
specification. Per-state compatibility constraints are supplied in OfflineConfig.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
import json
import math
import time

import torch
from torch import nn

from .. import inference, preprocessing
from .offline_metrics import StateMetrics, finite_float
from .offline_architecture import OfflineArchitecture


@dataclass
class OfflineConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    belief_gradient_clip: float = 100.0
    dt: float = 0.1
    patience: int = 4
    relative_tolerance: float = 1e-3
    decorrelation_tolerance: float = 1e-3
    # Controller mode: "current" preserves v3 behavior; "strict" follows
    # the paper's temporal derivative-pair formalization.
    detection_mode: str = "current"
    derivative_window: int = 8
    derivative_min_samples: int = 4
    derivative_pair_mode: str = "both"  # both | local_precision | global_precision
    local_saturation_tolerance: float = 1e-3
    precision_saturation_tolerance: float = 1e-3
    derivative_correlation_threshold: float = 0.8
    derivative_movement_tolerance: float = 1e-6
    # Phase-1 unstructured pruning. Applied to shared nn.Modules, not to a
    # fictitious state-specific parameter subset.
    prune_fraction: float = 0.05
    prune_min_remaining_per_tensor: int = 1
    # Phase-2 hyperparameter shrinkage toward explicit stable targets.
    hyperparameter_regularization_strength: float = 0.15
    hyperparameter_targets: dict[str, float] = field(default_factory=lambda: {
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "lr_motors": 1.0,
        "lr_policies": 1.0,
        "lr_obs": 1.0,
    })
    min_precision: float = 1e-8
    max_precision: float = 1e8
    lr_min: float = 1e-4
    lr_max: float = 2.0
    adapt_learning_rates: bool = True
    neuron_step: int = 16
    max_total_states: int | None = 64
    state_constraints: dict[str, dict[str, Any]] = field(default_factory=dict)
    checkpoint_dir: str | None = "offline_artifacts"
    plan_filename: str = "offline_plan.json"
    checkpoint_filename: str = "offline_before.pt"
    final_checkpoint_filename: str = "offline_after.pt"
    best_checkpoint_filename: str = "offline_best_agent.pt"


@dataclass
class StateProgress:
    phase: int = 1
    history: deque = field(default_factory=lambda: deque(maxlen=8))
    stable: bool = False
    pending_mutation: bool = False


class OfflineTrainer:
    """Stateful offline trainer; progress survives successive method calls."""

    PHASE_NAMES = {
        1: "generative_weights",
        2: "hyperparameters",
        3: "neurons",
        4: "intermediate_layers",
        5: "generalized_coordinates",
        6: "hidden_states",
        7: "hierarchical_layers",
    }

    def __init__(self, agent, config: OfflineConfig | dict | None = None):
        self.agent = agent
        if config is None:
            self.config = OfflineConfig()
        elif isinstance(config, OfflineConfig):
            self.config = config
        else:
            self.config = OfflineConfig(**config)

        self._validate_config()
        self.progress: dict[str, StateProgress] = {}
        self.iteration = 0
        self.pending_plan: dict[str, Any] | None = None
        self.last_metrics: dict[str, StateMetrics] = {}
        self._optimizer_signature: tuple[tuple[int, ...], ...] = ()
        self._optimizer = None
        self._pending_optimizer_state = None
        self.best_free_energy: float | None = None
        self.best_iteration: int | None = None
        self.best_checkpoint_path: str | None = None

    def _validate_config(self) -> None:
        if self.config.detection_mode not in {"current", "strict"}:
            raise ValueError("detection_mode must be 'current' or 'strict'.")
        if self.config.derivative_pair_mode not in {"both", "local_precision", "global_precision"}:
            raise ValueError("derivative_pair_mode must be both, local_precision or global_precision.")
        if (self.config.derivative_window < 3 or self.config.derivative_min_samples < 3
                or self.config.derivative_min_samples > self.config.derivative_window):
            raise ValueError("Require 3 <= derivative_min_samples <= derivative_window.")
        if not 0.0 <= self.config.derivative_correlation_threshold <= 1.0:
            raise ValueError("derivative_correlation_threshold must be in [0, 1].")
        if not 0.0 <= self.config.prune_fraction < 1.0:
            raise ValueError("prune_fraction must be in [0, 1).")
        if int(self.config.prune_min_remaining_per_tensor) < 1:
            raise ValueError("prune_min_remaining_per_tensor must be >= 1.")
        if not 0.0 <= self.config.hyperparameter_regularization_strength <= 1.0:
            raise ValueError("hyperparameter_regularization_strength must be in [0, 1].")
        if self.config.patience < 2:
            raise ValueError("patience must be >= 2 in current detection mode.")

    def invalidate_optimizer(self):
        """Drop Adam state after a structural/module parameter replacement."""
        self._optimizer = None
        self._optimizer_signature = ()
        self._pending_optimizer_state = None

    def state_dict(self) -> dict:
        """Serializable offline-controller state for exact training resumption."""
        return {
            "config": asdict(self.config),
            "iteration": self.iteration,
            "progress": {
                name: {
                    "phase": item.phase,
                    "history": [metric.to_dict() for metric in item.history],
                    "stable": item.stable,
                    "pending_mutation": item.pending_mutation,
                }
                for name, item in self.progress.items()
            },
            "pending_plan": deepcopy(self.pending_plan),
            "best_free_energy": self.best_free_energy,
            "best_iteration": self.best_iteration,
            "best_checkpoint_path": self.best_checkpoint_path,
            "optimizer_state": (
                deepcopy(self._optimizer.state_dict())
                if self._optimizer is not None else deepcopy(self._pending_optimizer_state)
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore controller history; Adam state is loaded lazily on first step."""
        if state.get("config"):
            self.config = OfflineConfig(**state["config"])
        self.iteration = int(state.get("iteration", 0))
        self.progress = {}
        for name, item in state.get("progress", {}).items():
            progress = StateProgress(
                phase=int(item.get("phase", 1)),
                history=deque(maxlen=self.config.derivative_window),
                stable=bool(item.get("stable", False)),
                pending_mutation=bool(item.get("pending_mutation", False)),
            )
            for metric in item.get("history", []):
                progress.history.append(StateMetrics(**metric))
            self.progress[name] = progress
        self.pending_plan = deepcopy(state.get("pending_plan"))
        self.best_free_energy = state.get("best_free_energy")
        self.best_iteration = state.get("best_iteration")
        self.best_checkpoint_path = state.get("best_checkpoint_path")
        self._pending_optimizer_state = deepcopy(state.get("optimizer_state"))
        self._optimizer = None
        self._optimizer_signature = ()

    # ------------------------------------------------------------------
    # Evaluation and optimization
    # ------------------------------------------------------------------

    def _prepared_state(self, state_name: str, observations: dict):
        return preprocessing.preprocessing_layer_for_free_energy(
            state_name,
            self.agent.navigator,
            self.agent.functions_map,
            self.agent.inputs_map,
            self.agent.Q_,
            self.agent.pi_priors,
            self.agent.pi_obs,
            observations,
            self.agent.states_list,
        )

    def _state_objective(self, state_name: str, observations: dict):
        prepared = self._prepared_state(state_name, observations)
        posterior = prepared["state_posterior"].detach().clone().requires_grad_(True)
        pi_prior = prepared["pi_prior"].detach().clone().requires_grad_(True)
        prediction = prepared["prediction"]
        if not prediction.requires_grad:
            prediction = prediction.detach().requires_grad_(True)

        pi_obs = prepared["pi_obs"].clamp_min(self.config.min_precision)
        gamma_obs = torch.log(pi_obs).detach().requires_grad_(True)

        F_value = self.agent.free_energy(
            prepared["state_prior"],
            posterior,
            prepared["f_transition"],
            pi_prior,
            prepared["actual"],
            pi_obs,
            gamma_obs,
            prediction,
        )
        return prepared, posterior, F_value

    def _local_sensitivity(self, state_name: str, posterior: torch.Tensor) -> tuple[float, bool]:
        """Frobenius norm of local f/g/p/V output Jacobians with respect to Q_i.

        This is evaluated only in strict mode. The posterior is injected into
        the local input map before re-evaluating the generative functions, so
        the derivative is not taken through a detached preprocessing result.
        """
        from . import model_components

        base_inputs = dict(self.agent.inputs_map)
        state_value = posterior.detach().clone().requires_grad_(True)
        base_inputs[state_name] = state_value

        def outputs(value):
            local_inputs = dict(base_inputs)
            local_inputs[state_name] = value
            parts = []
            transition = model_components.get_transition_prediction(
                self.agent.navigator, self.agent.functions_map, local_inputs,
                state_name, self.agent.states_list, self.agent.Q_
            )
            if transition is not None:
                parts.append(transition.reshape(-1))
            prediction = model_components.get_state_prediction(
                self.agent.navigator, self.agent.functions_map, local_inputs,
                state_name, self.agent.states_list, self.agent.Q_
            )
            if prediction is not None:
                parts.append(prediction.reshape(-1))
            for function_type in ("p", "V"):
                if self.agent.navigator.function(state_name, function_type) is not None:
                    local_output = preprocessing.execute_flat_function(
                        self.agent.navigator, self.agent.functions_map,
                        local_inputs, state_name, function_type,
                        self.agent.states_list, function_type, self.agent.Q_
                    )
                    if local_output is not None:
                        parts.append(local_output.reshape(-1))
            if not parts:
                return value.new_zeros((1,))
            return torch.cat(parts)

        try:
            jacobian = torch.autograd.functional.jacobian(
                outputs, state_value, create_graph=False, strict=False
            )
            if not torch.isfinite(jacobian).all():
                return 0.0, False
            return finite_float(torch.linalg.vector_norm(jacobian).item()), True
        except (RuntimeError, ValueError, TypeError, KeyError):
            return 0.0, False

    def _measure_state(self, state_name: str, observations: dict) -> tuple[StateMetrics, torch.Tensor]:
        prepared, posterior, F_value = self._state_objective(state_name, observations)

        gradient = torch.autograd.grad(
            F_value, posterior, retain_graph=True, create_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            gradient_tensor = torch.zeros_like(posterior.detach()).reshape(-1)
            gradient_finite = True
        else:
            gradient_tensor = gradient.detach().reshape(-1)
            gradient_finite = bool(torch.isfinite(gradient_tensor).all().item())

        # Keep the actual components: reducing them to a norm loses the
        # coordinate-wise signal needed by structural decisions.
        gradient_raw = (
            [float(value) for value in gradient_tensor.cpu().tolist()]
            if gradient_finite else []
        )
        gradient_sum = (
            finite_float(gradient_tensor.abs().sum().item())
            if gradient_finite else 0.0
        )
        # Retained only as a backward-compatible diagnostic.
        gradient_norm = (
            finite_float(torch.linalg.vector_norm(gradient_tensor).item())
            if gradient_finite else 0.0
        )

        hessian_norm = 0.0
        hessian_finite = False
        try:
            hessian = inference.second_derivative_F(posterior, F_value).detach()
            hessian_norm = finite_float(torch.linalg.vector_norm(hessian).item())
            hessian_finite = bool(torch.isfinite(hessian).all().item())
            # Precision is the non-negative mean curvature, not the raw
            # signed trace (which could classify a saddle as high precision).
            eigenvalues = torch.linalg.eigvalsh(0.5 * (hessian + hessian.transpose(-1, -2)))
            precision = finite_float(eigenvalues.clamp_min(0).mean().item())
        except (RuntimeError, ValueError):
            precision = 0.0

        complexity = finite_float(
            (0.5 * prepared["pi_prior"].detach() *
             (posterior.detach() + prepared["f_transition"].detach()
              - prepared["state_prior"].detach()).square()).sum().item()
        )
        accuracy = finite_float(
            (-0.5 * prepared["pi_obs"].detach() *
             (prepared["actual"].detach() - prepared["prediction"].detach()).square()).sum().item()
        )
        if self.config.detection_mode == "strict":
            local_sensitivity, local_sensitivity_finite = self._local_sensitivity(
                state_name, prepared["state_posterior_init_size"]
            )
        else:
            local_sensitivity, local_sensitivity_finite = 0.0, False
        metric = StateMetrics(
            state=state_name,
            free_energy=finite_float(F_value.detach().item()),
            gradient_norm=gradient_norm,
            gradient_raw=gradient_raw,
            gradient_sum=gradient_sum,
            gradient_finite=gradient_finite,
            precision=precision,
            complexity=complexity,
            accuracy=accuracy,
            local_sensitivity=local_sensitivity,
            hessian_norm=hessian_norm,
            local_sensitivity_finite=local_sensitivity_finite,
            hessian_finite=hessian_finite,
        )
        return metric, F_value

    def _unique_modules(self) -> list[nn.Module]:
        modules = []
        seen = set()
        for value in (self.agent.functions_map or {}).values():
            if isinstance(value, nn.Module) and id(value) not in seen:
                modules.append(value)
                seen.add(id(value))
        return modules

    def _parameter_groups(self):
        """Separate registered local policy modules from other world-model weights."""
        policy_parameter_ids = set()
        policy_parameters = []
        other_parameters = []
        seen = set()

        for module in self._unique_modules():
            local_modules = getattr(module, "_local_modules", None)
            if local_modules is not None:
                for key, local_module in local_modules.items():
                    if key.endswith(":p"):
                        for parameter in local_module.parameters():
                            if parameter.requires_grad and id(parameter) not in seen:
                                policy_parameters.append(parameter)
                                policy_parameter_ids.add(id(parameter))
                                seen.add(id(parameter))

        for module in self._unique_modules():
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    other_parameters.append(parameter)
                    seen.add(id(parameter))

        groups = []
        if other_parameters:
            groups.append({"params": other_parameters, "lr": self.config.learning_rate})
        if policy_parameters:
            groups.append({"params": policy_parameters, "lr": self.config.learning_rate * self.agent.lr_policies})
        return groups

    def _optimize_world_model(self, observations: dict):
        groups = self._parameter_groups()
        if not groups:
            return {"updated": False, "loss": None, "parameters": 0}

        signature = tuple(
            tuple(id(parameter) for parameter in group["params"])
            for group in groups
        )
        if self._optimizer is None or signature != self._optimizer_signature:
            self._optimizer = torch.optim.AdamW(
                groups, weight_decay=self.config.weight_decay
            )
            self._optimizer_signature = signature
            if self._pending_optimizer_state is not None:
                # A checkpoint can restore Adam moments after the model and
                # its modules have been reconstructed.
                self._optimizer.load_state_dict(self._pending_optimizer_state)
                self._pending_optimizer_state = None
        else:
            # Keep Adam's moments, while honoring live learning-rate changes.
            for optimizer_group, parameter_group in zip(self._optimizer.param_groups, groups):
                optimizer_group["lr"] = parameter_group["lr"]
                optimizer_group["weight_decay"] = self.config.weight_decay
        optimizer = self._optimizer
        optimizer.zero_grad(set_to_none=True)

        losses = []
        state_names = [
            name for name, node in self.agent.navigator.graph.items()
            if node.get("type") == "state"
        ]
        for state_name in state_names:
            _, _, F_value = self._state_objective(state_name, observations)
            if F_value.requires_grad and torch.isfinite(F_value):
                losses.append(F_value)

        if not losses:
            return {"updated": False, "loss": None, "parameters": 0}

        total_loss = torch.stack([loss.reshape(()) for loss in losses]).sum()
        if not total_loss.requires_grad:
            return {"updated": False, "loss": finite_float(total_loss.item()), "parameters": 0}

        total_loss.backward()
        parameters = [
            parameter
            for group in groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
            optimizer.step()

        return {
            "updated": bool(parameters),
            "loss": finite_float(total_loss.detach().item()),
            "parameters": sum(parameter.numel() for parameter in parameters),
        }

    def _evaluate(self, observations: dict) -> tuple[float, dict[str, StateMetrics]]:
        metrics = {}
        total = 0.0
        for state_name, node in self.agent.navigator.graph.items():
            if node.get("type") != "state":
                continue
            metric, _ = self._measure_state(state_name, observations)
            metrics[state_name] = metric
            total += metric.free_energy
        return finite_float(total), metrics

    # ------------------------------------------------------------------
    # Seven-phase per-state decision system
    # ------------------------------------------------------------------

    @staticmethod
    def _relative_slope(values: list[float]) -> float:
        """Least-squares slope normalized by the series' absolute mean."""
        n = len(values)
        if n < 2:
            return 0.0
        x = torch.arange(n, dtype=torch.float64)
        y = torch.tensor(values, dtype=torch.float64)
        x = x - x.mean()
        y_centered = y - y.mean()
        slope = (x * y_centered).sum() / x.square().sum().clamp_min(1e-12)
        scale = y.abs().mean().clamp_min(1e-8)
        return finite_float((slope / scale).item())

    @staticmethod
    def _pearson_increments(x_values: list[float], y_values: list[float]) -> tuple[float, float, float]:
        dx = torch.tensor(x_values[1:], dtype=torch.float64) - torch.tensor(x_values[:-1], dtype=torch.float64)
        dy = torch.tensor(y_values[1:], dtype=torch.float64) - torch.tensor(y_values[:-1], dtype=torch.float64)
        if dx.numel() < 2:
            return 0.0, 0.0, 0.0
        sx, sy = dx.std(unbiased=False), dy.std(unbiased=False)
        if sx <= 1e-12 or sy <= 1e-12:
            return 0.0, finite_float(sx.item()), finite_float(sy.item())
        corr = torch.corrcoef(torch.stack([dx, dy]))[0, 1]
        return finite_float(corr.item()), finite_float(sx.item()), finite_float(sy.item())

    def _strict_pairs(self, history: deque):
        items = list(history)[-self.config.derivative_window:]
        if len(items) < self.config.derivative_min_samples:
            return []
        pairs = []
        if (self.config.derivative_pair_mode in ("both", "local_precision")
                and all(m.local_sensitivity_finite and m.hessian_finite for m in items)):
            pairs.append(("L", [m.local_sensitivity for m in items]))
        if (self.config.derivative_pair_mode in ("both", "global_precision")
                and all(m.gradient_finite and m.hessian_finite for m in items)):
            pairs.append(("G", [m.gradient_norm for m in items]))
        return [(name, values, [m.hessian_norm for m in items]) for name, values in pairs]

    def _strict_double_saturated(self, history: deque) -> bool:
        for _, x_values, h_values in self._strict_pairs(history):
            slope_x = self._relative_slope(x_values)
            slope_h = self._relative_slope(h_values)
            if (slope_x >= -self.config.local_saturation_tolerance and
                    slope_h <= self.config.precision_saturation_tolerance):
                return True
        return False

    def _strict_decorrelated(self, history: deque) -> bool:
        for _, x_values, h_values in self._strict_pairs(history):
            corr, movement_x, movement_h = self._pearson_increments(x_values, h_values)
            scale_x = max(sum(abs(v) for v in x_values) / len(x_values), 1e-8)
            scale_h = max(sum(abs(v) for v in h_values) / len(h_values), 1e-8)
            if (corr >= self.config.derivative_correlation_threshold and
                    movement_x / scale_x >= self.config.derivative_movement_tolerance and
                    movement_h / scale_h >= self.config.derivative_movement_tolerance):
                return True
        return False

    def _is_decorrelated(self, history: deque, phase: int) -> bool:
        if self.config.detection_mode == "strict":
            return self._strict_decorrelated(history)
        if len(history) < 2:
            return False
        previous, current = history[-2], history[-1]
        if not previous.gradient_finite or not current.gradient_finite:
            return False

        if phase <= 4:
            old = abs(previous.gradient_sum)
            new = abs(current.gradient_sum)
            gradient_rise = new > old * (1.0 + self.config.decorrelation_tolerance)
        else:
            old = previous.gradient_raw
            new = current.gradient_raw
            if not old or len(old) != len(new):
                return False
            eps = 1e-8
            gradient_rise = any(
                abs(new_i) > max(abs(old_i), eps) * (1.0 + self.config.decorrelation_tolerance)
                for old_i, new_i in zip(old, new)
            )

        precision_fall = current.precision < previous.precision * (
            1.0 - self.config.decorrelation_tolerance
        )
        return gradient_rise and precision_fall

    def _is_double_saturated(self, history: deque, phase: int) -> bool:
        if self.config.detection_mode == "strict":
            return self._strict_double_saturated(history)
        if len(history) < self.config.patience:
            return False
        first, last = history[-self.config.patience], history[-1]
        if not first.gradient_finite or not last.gradient_finite:
            return False

        if phase <= 4:
            old = abs(first.gradient_sum)
            new = abs(last.gradient_sum)
            gradient_improvement = (old - new) / max(old, 1e-8)
            gradient_saturated = gradient_improvement <= self.config.relative_tolerance
        else:
            old = first.gradient_raw
            new = last.gradient_raw
            if not old or len(old) != len(new):
                return False
            # Test each component independently. A coordinate with a strong
            # residual gradient cannot be hidden by small gradients elsewhere.
            component_improvements = [
                (abs(old_i) - abs(new_i)) / max(abs(old_i), 1e-8)
                for old_i, new_i in zip(old, new)
            ]
            gradient_saturated = all(
                value <= self.config.relative_tolerance
                for value in component_improvements
            )

        precision_improvement = (
            last.precision - first.precision
        ) / max(abs(first.precision), 1e-8)
        return (
            gradient_saturated
            and precision_improvement <= self.config.relative_tolerance
        )

    @staticmethod
    def _generic_name(prefix: str, existing: set[str]) -> str:
        index = 1
        while f"{prefix}_{index}" in existing:
            index += 1
        return f"{prefix}_{index}"


    def _mutation_for(self, state_name: str, phase: int, decorrelated: bool):
        """Create a generic mutation command; constraints live in OfflineConfig."""
        navigator = self.agent.navigator
        layer_values = navigator.layers()
        layer = navigator.get_state_layer(state_name)
        if decorrelated:
            operations = {
                1: {"action": "prune_obsolete_weights", "target": state_name},
                2: {"action": "regularize_hyperparameters", "target": state_name},
                3: {"action": "prune_neurons", "target": state_name},
                4: {"action": "prune_intermediate_layer", "target": state_name},
                5: {"action": "remove_low_utility_coordinate", "target": state_name},
                6: {"action": "remove_redundant_state", "target": state_name},
                7: {"action": "remove_redundant_hierarchy_layer", "target": state_name},
            }
            return operations.get(phase, {"action": "none", "target": state_name})

        operations = {
            3: {"action": "add_neurons", "target": state_name},
            4: {"action": "add_intermediate_layer", "target": state_name},
            5: {"action": "add_generalized_coordinate", "target": state_name},
            6: {"action": "add_state", "target": state_name},
            7: {"action": "add_hierarchy_layer", "target": state_name},
        }
        return operations.get(phase, {"action": "none", "target": state_name})

    def _update_progress(self, metrics: dict[str, StateMetrics]):
        decisions = {}
        proposed = []
        for state_name, metric in metrics.items():
            progress = self.progress.get(state_name)
            if progress is None:
                progress = StateProgress(history=deque(maxlen=self.config.derivative_window))
                self.progress[state_name] = progress
            elif progress.history.maxlen != self.config.derivative_window:
                progress.history = deque(progress.history, maxlen=self.config.derivative_window)
            progress.history.append(metric)

            decorrelated = self._is_decorrelated(progress.history, progress.phase)
            saturated = self._is_double_saturated(progress.history, progress.phase)
            phase_before = progress.phase
            operation = None

            if progress.stable:
                decisions[state_name] = {
                    "phase_before": phase_before,
                    "phase": progress.phase,
                    "phase_name": self.PHASE_NAMES[progress.phase],
                    "gradient_norm": metric.gradient_norm,
                    "gradient_sum": metric.gradient_sum,
                    "gradient_raw": metric.gradient_raw,
                    "gradient_measure": (
                        "temporal_derivative_pairs_LH_GH" if self.config.detection_mode == "strict"
                        else "sum_abs" if phase_before <= 4 else "raw_components"
                    ),
                    "gradient_finite": metric.gradient_finite,
                    "precision": metric.precision,
                    "local_sensitivity": metric.local_sensitivity,
                    "hessian_norm": metric.hessian_norm,
                    "detection_mode": self.config.detection_mode,
                    "double_saturation": saturated,
                    "decorrelation": decorrelated,
                    "stable": True,
                    "mutation": None,
                }
                continue

            # Do not stack further structural mutations on top of a plan
            # that has not yet been rebuilt and evaluated.
            if progress.pending_mutation:
                decisions[state_name] = {
                    "phase_before": phase_before,
                    "phase": progress.phase,
                    "phase_name": self.PHASE_NAMES[progress.phase],
                    "gradient_norm": metric.gradient_norm,
                    "gradient_sum": metric.gradient_sum,
                    "gradient_raw": metric.gradient_raw,
                    "gradient_measure": (
                        "temporal_derivative_pairs_LH_GH" if self.config.detection_mode == "strict"
                        else "sum_abs" if phase_before <= 4 else "raw_components"
                    ),
                    "gradient_finite": metric.gradient_finite,
                    "precision": metric.precision,
                    "local_sensitivity": metric.local_sensitivity,
                    "hessian_norm": metric.hessian_norm,
                    "detection_mode": self.config.detection_mode,
                    "double_saturation": saturated,
                    "decorrelation": decorrelated,
                    "stable": progress.stable,
                    "pending_mutation": True,
                    "mutation": None,
                }
                continue

            if decorrelated:
                operation = self._mutation_for(state_name, progress.phase, True)
                proposed.append(operation)
                progress.pending_mutation = True
            elif saturated:
                if progress.phase < 7:
                    # Phase 1 and 2 are continuous stages. A state only
                    # reaches phase 3 after both have saturated in sequence.
                    progress.phase += 1
                    if progress.phase >= 3:
                        operation = self._mutation_for(state_name, progress.phase, False)
                        proposed.append(operation)
                        progress.pending_mutation = True
                else:
                    progress.stable = True

            decisions[state_name] = {
                "phase_before": phase_before,
                "phase": progress.phase,
                "phase_name": self.PHASE_NAMES[progress.phase],
                "gradient_norm": metric.gradient_norm,
                "gradient_sum": metric.gradient_sum,
                "gradient_raw": metric.gradient_raw,
                "gradient_measure": (
                    "temporal_derivative_pairs_LH_GH" if self.config.detection_mode == "strict"
                    else "sum_abs" if phase_before <= 4 else "raw_components"
                ),
                "gradient_finite": metric.gradient_finite,
                "precision": metric.precision,
                "local_sensitivity": metric.local_sensitivity,
                "hessian_norm": metric.hessian_norm,
                "detection_mode": self.config.detection_mode,
                "double_saturation": saturated,
                "decorrelation": decorrelated,
                "stable": progress.stable,
                "mutation": operation,
            }
        return decisions, proposed

    # ------------------------------------------------------------------
    # Persistence and structural rebuild contract
    # ------------------------------------------------------------------

    def _consider_best_agent(self, free_energy: float) -> bool:
        """Save a complete deployable checkpoint when test free energy improves.

        Free energy is minimized. The first evaluated model is retained as the
        initial best, so a training run never loses a better starting point.
        """
        score = finite_float(free_energy)
        if self.best_free_energy is not None and score >= self.best_free_energy:
            return False

        self.best_free_energy = score
        self.best_iteration = self.iteration

        if not self.config.checkpoint_dir:
            self.best_checkpoint_path = None
            return True

        from .persistence import save_agent

        directory = Path(self.config.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.config.best_checkpoint_filename
        # Save the complete deployable agent, not only diagnostic module weights.
        self.best_checkpoint_path = str(save_agent(self.agent, path))
        return True

    def _checkpoint_before(self, F_before: float):
        if not self.config.checkpoint_dir:
            return None
        directory = Path(self.config.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.config.checkpoint_filename
        state = {
            "iteration": self.iteration,
            "F_global_test_before": F_before,
            "world_model": {
                key: module.state_dict()
                for key, module in enumerate(self._unique_modules())
            },
            "D": deepcopy(self.agent.D),
        }
        torch.save(state, path)
        return str(path)

    def _checkpoint_after(self, F_after: float):
        if not self.config.checkpoint_dir:
            return None
        directory = Path(self.config.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.config.final_checkpoint_filename
        state = {
            "iteration": self.iteration,
            "F_global_test_after": F_after,
            "world_model": {
                str(index): module.state_dict()
                for index, module in enumerate(self._unique_modules())
            },
            "D": deepcopy(self.agent.D),
            "generative_functions": deepcopy(self.agent.generative_functions),
            "pi_states": deepcopy(self.agent.pi_states),
            "pi_obs": deepcopy(self.agent.pi_obs),
            "progress": {
                name: {
                    "phase": item.phase,
                    "stable": item.stable,
                    "pending_mutation": item.pending_mutation,
                    "history": [metric.to_dict() for metric in item.history],
                }
                for name, item in self.progress.items()
            },
        }
        torch.save(state, path)
        return str(path)

    def _write_plan(self, plan: dict[str, Any]):
        if not self.config.checkpoint_dir:
            return None
        directory = Path(self.config.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.config.plan_filename
        path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def _apply_rebuild_callback(self, callback: Callable | None, plan: dict):
        if callback is None:
            return False
        updated = callback(
            plan=deepcopy(plan),
            D=deepcopy(self.agent.D),
            generative_functions=deepcopy(self.agent.generative_functions),
            observations=deepcopy(self.agent.observations),
            actions=deepcopy(self.agent.actions),
            functions_map=self.agent.functions_map,
        )
        if updated is None:
            return False
        required = {"D", "generative_functions"}
        missing = required - set(updated)
        if missing:
            raise ValueError(
                "The offline rebuild callback must return at least "
                f"{sorted(required)}; missing {sorted(missing)}."
            )

        # Reset graph/distribution-derived structures before calling the
        # same public model initializer used by the application. Keep existing
        # posterior beliefs across the rebuild, just like the internal path.
        old_Q = deepcopy(self.agent.Q_)
        self.agent.D = updated["D"]
        self.agent.graph = None
        self.agent.navigator = None
        self.agent.Q_ = None
        self.agent.pi_states = None
        self.agent.pi_priors = None
        self.agent.inputs_map = None
        self.agent.pi_p = None

        self.agent.model(
            generative_functions=updated["generative_functions"],
            functions_map=updated.get("functions_map", self.agent.functions_map),
            observations=updated.get("observations", self.agent.observations),
            actions=updated.get("actions", self.agent.actions),
            pi_p=updated.get("pi_p"),
        )
        OfflineArchitecture._restore_existing_beliefs(self.agent, old_Q)
        if getattr(self.agent, "_local_function_factory", None) is not None:
            self.agent.install_local_functions()
        self.invalidate_optimizer()
        self.progress = {
            name: StateProgress()
            for name, node in self.agent.navigator.graph.items()
            if node.get("type") == "state"
        }
        self.pending_plan = None
        return True

    # ------------------------------------------------------------------
    # Public iteration
    # ------------------------------------------------------------------

    def step(
        self,
        observations: dict,
        *,
        test_observations: dict | None = None,
        rebuild_callback: Callable | None = None,
    ) -> dict[str, Any]:
        self._validate_config()
        if self.agent.navigator is None:
            raise RuntimeError("Call Agent_DAIH.model() before training_offline().")
        if not isinstance(observations, dict):
            raise TypeError("observations must be a dictionary.")
        if test_observations is None:
            test_observations = observations

        self.iteration += 1

        # Test baseline and save the pre-training model before any update.
        test_copy = deepcopy(test_observations)
        test_copy, _, _ = inference.handle_missing_observations(
            test_copy, deepcopy(self.agent.pi_obs), deepcopy(self.agent.pi_obs_substitued)
        )
        F_before, _ = self._evaluate(test_copy)
        checkpoint_path = self._checkpoint_before(F_before)
        # Keep the starting model if subsequent optimization makes it worse.
        best_agent_updated = self._consider_best_agent(F_before)

        # Historical observations are accepted as supplied. Missing-value
        # bookkeeping is local to this offline step: it must not mutate the
        # agent's observation precisions or posterior beliefs.
        training_copy = deepcopy(observations)
        training_copy, _, _ = inference.handle_missing_observations(
            training_copy,
            deepcopy(self.agent.pi_obs),
            deepcopy(self.agent.pi_obs_substitued),
        )

        # Offline training updates generative modules only. In particular,
        # beliefs_update() is deliberately NOT called here: Q, pi_obs,
        # pi_states and the inference thresholds belong to online inference.
        optimization = self._optimize_world_model(training_copy)
        belief_updates = []

        # Test after training using the caller's held-out observations.
        test_copy_after = deepcopy(test_observations)
        test_copy_after, _, _ = inference.handle_missing_observations(
            test_copy_after, deepcopy(self.agent.pi_obs), deepcopy(self.agent.pi_obs_substitued)
        )
        F_after, metrics_after = self._evaluate(test_copy_after)
        self.last_metrics = metrics_after

        decisions, proposed = self._update_progress(metrics_after)
        plan = {
            "version": "4.0",
            "detection_mode": self.config.detection_mode,
            "detection_parameters": {
                "derivative_window": self.config.derivative_window,
                "derivative_min_samples": self.config.derivative_min_samples,
                "derivative_pair_mode": self.config.derivative_pair_mode,
                "local_saturation_tolerance": self.config.local_saturation_tolerance,
                "precision_saturation_tolerance": self.config.precision_saturation_tolerance,
                "derivative_correlation_threshold": self.config.derivative_correlation_threshold,
                "derivative_movement_tolerance": self.config.derivative_movement_tolerance,
                "patience": self.config.patience,
                "relative_tolerance": self.config.relative_tolerance,
                "decorrelation_tolerance": self.config.decorrelation_tolerance,
            },
            "iteration": self.iteration,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "F_global_test_before": F_before,
            "F_global_test_after": F_after,
            "delta_F_global_test": F_after - F_before,
            "phases": decisions,
            "mutations": proposed,
            "status": "ready_to_rebuild" if proposed else "optimizing",
            "note": (
                "Offline updates generative modules and architecture only; "
                "posterior beliefs Q and online observation precisions are not updated. "
                "Structural mutations rebuild the declarative model."
            ),
        }

        # Mutations are now applied internally. The optional callback remains
        # only as a compatibility override for applications with custom rules.
        plan_to_apply = plan if proposed else None
        plan_to_persist = plan
        plan_path = self._write_plan(plan_to_persist)

        rebuild_result = {"applied": [], "skipped": []}
        rebuilt = False
        if plan_to_apply is not None:
            if rebuild_callback is not None:
                rebuilt = self._apply_rebuild_callback(rebuild_callback, plan_to_apply)
                rebuild_result = {"applied": proposed if rebuilt else [], "skipped": [] if rebuilt else proposed}
            else:
                rebuild_result = OfflineArchitecture(self.agent, self.config).apply(proposed)
                rebuilt = bool(rebuild_result.get("rebuilt", False))
            plan_to_persist["status"] = (
                "rebuilt" if rebuilt else
                "updated_in_place" if rebuild_result.get("applied") else
                "no_mutation_applied"
            )
            plan_to_persist["rebuilt"] = rebuilt
            plan_to_persist["mutation_result"] = rebuild_result
            for skipped_mutation in rebuild_result.get("skipped", []):
                target = skipped_mutation.get("target")
                if target in self.progress:
                    progress = self.progress[target]
                    progress.pending_mutation = False
                    if skipped_mutation.get("action") in {
                        "prune_obsolete_weights", "regularize_hyperparameters"
                    }:
                        # A phase-1/2 operation with no safe candidate is a
                        # completed no-op; continue the staged process rather
                        # than retrying the same in-place mutation forever.
                        if progress.phase < 7:
                            progress.phase += 1
                        else:
                            progress.stable = True
                    else:
                        progress.stable = True
            if not rebuilt:
                # Phase-1/2 operations are in-place (masking / scalar
                # regularization), not architecture rebuilds.
                for applied_mutation in rebuild_result.get("applied", []):
                    target = applied_mutation.get("target")
                    if target in self.progress:
                        self.progress[target].pending_mutation = False
            self.pending_plan = None
            plan_path = self._write_plan(plan_to_persist)
            if rebuilt or rebuild_result.get("applied"):
                # Evaluate the updated/rebuilt declaration on the same test batch.
                test_copy_rebuilt = deepcopy(test_observations)
                test_copy_rebuilt, _, _ = inference.handle_missing_observations(
                    test_copy_rebuilt, deepcopy(self.agent.pi_obs), deepcopy(self.agent.pi_obs_substitued)
                )
                F_after, metrics_after = self._evaluate(test_copy_rebuilt)
                self.last_metrics = metrics_after
                plan_to_persist["F_global_test_after"] = F_after
                plan_to_persist["delta_F_global_test"] = F_after - F_before
                plan_to_persist["post_mutation_states"] = {
                    name: metric.to_dict() for name, metric in metrics_after.items()
                }
                plan_path = self._write_plan(plan_to_persist)

        if self.config.adapt_learning_rates:
            # Smooth bounded adaptation. Rates change during the transition
            # from structural phase 4 toward phase 5 as requested, and remain
            # responsive to the observed global objective afterwards.
            phases = [item.phase for item in self.progress.values()]
            if phases and max(phases) >= 5:
                ratio = 0.95 if F_after <= F_before else 1.05
                self.agent.lr_motors = min(
                    self.config.lr_max,
                    max(self.config.lr_min, self.agent.lr_motors * ratio),
                )
                self.agent.lr_policies = min(
                    self.config.lr_max,
                    max(self.config.lr_min, self.agent.lr_policies * ratio),
                )
                self.agent.lr_obs = min(
                    self.config.lr_max,
                    max(self.config.lr_min, self.agent.lr_obs * ratio),
                )

        # Compare the fully updated agent, including any applied mutation.
        best_agent_updated = (
            self._consider_best_agent(F_after) or best_agent_updated
        )
        final_checkpoint_path = self._checkpoint_after(F_after)

        return {
            "iteration": self.iteration,
            "optimization": optimization,
            "belief_updates": belief_updates,  # always empty: offline never updates Q
            "q_updated_offline": False,
            "F_global_test_before": F_before,
            "F_global_test_after": F_after,
            "delta_F_global_test": F_after - F_before,
            "states": {name: metric.to_dict() for name, metric in metrics_after.items()},
            "phases": decisions,
            "mutations": (
                plan_to_persist.get("mutations", [])
                if plan_to_persist is not None else proposed
            ),
            "rebuilt": rebuilt,
            "mutation_result": rebuild_result,
            "checkpoint_path": checkpoint_path,
            "final_checkpoint_path": final_checkpoint_path,
            "best_agent_updated": best_agent_updated,
            "best_free_energy": self.best_free_energy,
            "best_iteration": self.best_iteration,
            "best_agent_checkpoint_path": self.best_checkpoint_path,
            "plan_path": plan_path,
            "learning_rates": {
                "motors": self.agent.lr_motors,
                "policies": self.agent.lr_policies,
                "observations": self.agent.lr_obs,
            },
        }
