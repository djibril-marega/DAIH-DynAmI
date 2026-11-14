"""Adaptive offline optimization for Agent_DAIH (v7).

The shared WorldModel is optimized against the sum of all state free energies.
Policy generators are optimized separately with expected-free-energy policy
 gradients and a G_prior/G_posterior alignment loss. The controller has one
shared progression for WorldModel phases 1-3 and independent per-state
progression for agent-architecture phases 4-7.

Offline learning performs temporary per-observation inference, then restores persistent posterior beliefs Q and online observation precisions.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from contextlib import contextmanager
import json
import math
import time

import torch
from torch import nn

from .. import inference, preprocessing, model_components
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
    # Reference detector retained by the project.
    detection_mode: str = "current"
    derivative_window: int = 8
    derivative_min_samples: int = 4
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

    # Adaptive controller: the shock is a temporal proxy of the third-order effect
    # (variation of local curvature), not an explicitly materialized 3rd tensor.
    shock_threshold: float = 0.15
    shock_window: int = 4
    phase_transition_mode: str = "forward_only"  # forward_only | bidirectional
    backtrack_tolerance: float = 1e-3
    phase_min_iterations: int = 3

    # Dataset semantics: temporal carries beliefs through a trajectory; independent resets beliefs per sample.
    observation_mode: str = "temporal"  # temporal | independent

    # Training scopes. The WorldModel objective remains the sum of all F_i.
    train_world_model: bool = True
    train_policy: bool = True
    # The motor function is user-owned. Its training is opt-in and delegated
    # to the user-provided motor_train_fn callback.
    train_motor: bool = False
    weights_only: bool = False  # disables all architecture/phase mutations
    policy_learning_rate: float | None = None
    policy_efe_weight: float = 1.0
    policy_alignment_weight: float = 1.0
    policy_baseline_momentum: float = 0.9

    neuron_step: int = 16
    world_model_min_neurons: int = 64
    world_model_max_neurons: int = 512
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
    phase_iterations: int = 0
    phase_entry_free_energy: float | None = None
    phase_best_free_energy: float | None = None


class OfflineTrainer:
    """Stateful offline trainer; progress survives successive method calls."""

    PHASE_NAMES = {
        1: "shared_world_model_weights",
        2: "generative_hyperparameters",
        3: "shared_world_model_capacity",
        4: "state_local_intermediate_layers",
        5: "state_local_generalized_coordinates",
        6: "state_local_under_states",
        7: "agent_states",
        8: "agent_hierarchy",
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
        # One shared controller governs WorldModel phases 1-3. Per-state
        # controllers are created only when the local agent phases begin.
        self.global_progress = StateProgress(history=deque(maxlen=self.config.derivative_window))
        self.progress: dict[str, StateProgress] = {}
        self.iteration = 0
        self.pending_plan: dict[str, Any] | None = None
        self.last_metrics: dict[str, StateMetrics] = {}
        self._optimizer_signature: tuple[tuple[int, ...], ...] = ()
        self._optimizer = None
        self._pending_optimizer_state = None
        self._policy_optimizer = None
        self._policy_optimizer_signature: tuple[int, ...] = ()
        self._pending_policy_optimizer_state = None
        self._policy_baselines: dict[str, float] = {}
        self.best_free_energy: float | None = None
        self.best_iteration: int | None = None
        self.best_checkpoint_path: str | None = None

    def _validate_config(self) -> None:
        if self.config.detection_mode != "current":
            raise ValueError("detection_mode must be 'current'.")
        if self.config.observation_mode not in {"temporal", "independent"}:
            raise ValueError("observation_mode must be 'temporal' or 'independent'.")
        if (self.config.derivative_window < 3 or self.config.derivative_min_samples < 3
                or self.config.derivative_min_samples > self.config.derivative_window):
            raise ValueError("Require 3 <= derivative_min_samples <= derivative_window.")
        if not 0.0 <= self.config.prune_fraction < 1.0:
            raise ValueError("prune_fraction must be in [0, 1).")
        if int(self.config.prune_min_remaining_per_tensor) < 1:
            raise ValueError("prune_min_remaining_per_tensor must be >= 1.")
        if not 0.0 <= self.config.hyperparameter_regularization_strength <= 1.0:
            raise ValueError("hyperparameter_regularization_strength must be in [0, 1].")
        if self.config.patience < 2:
            raise ValueError("patience must be >= 2 in current detection mode.")
        if self.config.phase_transition_mode not in {"forward_only", "bidirectional"}:
            raise ValueError("phase_transition_mode must be 'forward_only' or 'bidirectional'.")
        if self.config.shock_threshold < 0 or self.config.shock_window < 2:
            raise ValueError("shock_threshold must be >= 0 and shock_window >= 2.")
        if self.config.phase_min_iterations < 1:
            raise ValueError("phase_min_iterations must be >= 1.")
        if self.config.world_model_min_neurons < 16:
            raise ValueError("world_model_min_neurons must be >= 16.")
        if self.config.world_model_max_neurons < self.config.world_model_min_neurons:
            raise ValueError("world_model_max_neurons must be >= world_model_min_neurons.")
        for state_name, constraints in self.config.state_constraints.items():
            min_coordinates = int(constraints.get("min_coordinates", 1))
            max_coordinates = constraints.get("max_coordinates")
            min_under_state = int(constraints.get(
                "min_under_state", constraints.get("min_size", 1)
            ))
            max_under_state = constraints.get(
                "max_under_state", constraints.get("max_size")
            )
            min_states = int(constraints.get("min_states", 1))
            max_states = constraints.get("max_states")
            if min_coordinates < 1 or (max_coordinates is not None
                    and int(max_coordinates) < min_coordinates):
                raise ValueError(
                    f"Invalid coordinate bounds for state {state_name!r}: "
                    "require 1 <= min_coordinates <= max_coordinates (or None)."
                )
            if min_under_state < 1 or (max_under_state is not None
                    and int(max_under_state) < min_under_state):
                raise ValueError(
                    f"Invalid under_state bounds for state {state_name!r}: "
                    "require 1 <= min_under_state <= max_under_state (or None)."
                )
            if min_states < 1 or (max_states is not None
                    and int(max_states) < min_states):
                raise ValueError(
                    f"Invalid state-count bounds for {state_name!r}: "
                    "require 1 <= min_states <= max_states (or None)."
                )
        if not 0.0 <= self.config.policy_baseline_momentum < 1.0:
            raise ValueError("policy_baseline_momentum must be in [0, 1).")
        if self.config.policy_learning_rate is not None and self.config.policy_learning_rate <= 0:
            raise ValueError("policy_learning_rate must be positive when provided.")
        if self.config.policy_efe_weight < 0 or self.config.policy_alignment_weight < 0:
            raise ValueError("Policy loss weights must be non-negative.")

    def invalidate_optimizer(self):
        """Drop Adam state after a structural/module parameter replacement."""
        self._optimizer = None
        self._optimizer_signature = ()
        self._pending_optimizer_state = None
        self._policy_optimizer = None
        self._policy_optimizer_signature = ()
        self._pending_policy_optimizer_state = None

    def state_dict(self) -> dict:
        """Serializable offline-controller state for exact training resumption."""
        return {
            "config": asdict(self.config),
            "iteration": self.iteration,
            "global_progress": {
                "phase": self.global_progress.phase,
                "history": [metric.to_dict() for metric in self.global_progress.history],
                "stable": self.global_progress.stable,
                "pending_mutation": self.global_progress.pending_mutation,
                "phase_iterations": self.global_progress.phase_iterations,
                "phase_entry_free_energy": self.global_progress.phase_entry_free_energy,
                "phase_best_free_energy": self.global_progress.phase_best_free_energy,
            },
            "progress": {
                name: {
                    "phase": item.phase,
                    "history": [metric.to_dict() for metric in item.history],
                    "stable": item.stable,
                    "pending_mutation": item.pending_mutation,
                    "phase_iterations": item.phase_iterations,
                    "phase_entry_free_energy": item.phase_entry_free_energy,
                    "phase_best_free_energy": item.phase_best_free_energy,
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
            "policy_optimizer_state": (
                deepcopy(self._policy_optimizer.state_dict())
                if self._policy_optimizer is not None else deepcopy(self._pending_policy_optimizer_state)
            ),
            "policy_baselines": deepcopy(self._policy_baselines),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore controller history; Adam state is loaded lazily on first step."""
        if state.get("config"):
            self.config = OfflineConfig(**state["config"])
        self.iteration = int(state.get("iteration", 0))
        legacy_controller_state = "global_progress" not in state
        global_item = state.get("global_progress", {})
        self.global_progress = StateProgress(
            phase=int(global_item.get("phase", 1)),
            history=deque(maxlen=self.config.derivative_window),
            stable=bool(global_item.get("stable", False)),
            pending_mutation=bool(global_item.get("pending_mutation", False)),
            phase_iterations=int(global_item.get("phase_iterations", 0)),
            phase_entry_free_energy=global_item.get("phase_entry_free_energy"),
            phase_best_free_energy=global_item.get("phase_best_free_energy"),
        )
        for metric in global_item.get("history", []):
            self.global_progress.history.append(StateMetrics(**metric))
        self.progress = {}
        # Older controller versions had seven local phases: their phase 6
        # (states) and phase 7 (hierarchy) shift to 7 and 8 after inserting the
        # independent under_state phase at position 6.
        legacy_progress = {} if legacy_controller_state else state.get("progress", {})
        for name, item in legacy_progress.items():
            saved_phase = int(item.get("phase", 1))
            migrated_phase = saved_phase + 1 if saved_phase >= 6 else saved_phase
            progress = StateProgress(
                phase=migrated_phase,
                history=deque(maxlen=self.config.derivative_window),
                stable=bool(item.get("stable", False)),
                pending_mutation=bool(item.get("pending_mutation", False)),
                phase_iterations=int(item.get("phase_iterations", 0)),
                phase_entry_free_energy=item.get("phase_entry_free_energy"),
                phase_best_free_energy=item.get("phase_best_free_energy"),
            )
            for metric in item.get("history", []):
                progress.history.append(StateMetrics(**metric))
            self.progress[name] = progress
        self.pending_plan = deepcopy(state.get("pending_plan"))
        self.best_free_energy = state.get("best_free_energy")
        self.best_iteration = state.get("best_iteration")
        self.best_checkpoint_path = state.get("best_checkpoint_path")
        self._pending_optimizer_state = deepcopy(state.get("optimizer_state"))
        self._pending_policy_optimizer_state = deepcopy(state.get("policy_optimizer_state"))
        self._policy_baselines = {k: float(v) for k, v in state.get("policy_baselines", {}).items()}
        self._optimizer = None
        self._optimizer_signature = ()
        self._policy_optimizer = None
        self._policy_optimizer_signature = ()

    # ------------------------------------------------------------------
    # Evaluation and optimization
    # ------------------------------------------------------------------

    _BELIEF_FIELDS = (
        "Q_", "pi_obs", "pi_obs_substitued", "pi_priors", "pi_states",
        "inputs_map", "pi_threshold",
    )

    def _snapshot_beliefs(self) -> dict[str, Any]:
        return {name: deepcopy(getattr(self.agent, name)) for name in self._BELIEF_FIELDS}

    def _restore_beliefs(self, snapshot: dict[str, Any]) -> None:
        for name, value in snapshot.items():
            setattr(self.agent, name, deepcopy(value))

    @contextmanager
    def _using_beliefs(self, snapshot: dict[str, Any]):
        current = self._snapshot_beliefs()
        self._restore_beliefs(snapshot)
        try:
            yield
        finally:
            self._restore_beliefs(current)

    @staticmethod
    def _normalize_observations(observations: Any) -> list[dict]:
        if isinstance(observations, dict):
            return [deepcopy(observations)]
        if isinstance(observations, (list, tuple)) and observations:
            if not all(isinstance(item, dict) for item in observations):
                raise TypeError("A sequence dataset must contain dictionaries, one per observation.")
            return [deepcopy(item) for item in observations]
        raise TypeError("observations must be a dictionary or a non-empty sequence of dictionaries.")

    @staticmethod
    def _reset_motor(motor_fn: Callable | None, initial_observation: dict) -> None:
        """Reset optional user-owned motor-function memory before a replay."""
        if motor_fn is None:
            return
        reset = getattr(motor_fn, "reset", None)
        if callable(reset):
            reset(deepcopy(initial_observation))

    @staticmethod
    def _call_motor(
        motor_fn: Callable,
        observation: dict,
        motors: dict[str, float] | None,
    ) -> dict:
        """Let the user's motor function materialize an action in-place.

        The callback receives the observation schema defined by the caller and
        the flat motor-output dictionary (or None). It may mutate the supplied
        observation and return it, or return None after an in-place mutation.
        """
        if not callable(motor_fn):
            raise TypeError("motor_fn must be callable.")
        result = motor_fn(observation, deepcopy(motors) if motors else None)
        if result is None:
            return observation
        if not isinstance(result, dict):
            raise TypeError("motor_fn must return an observation dictionary or None.")
        return result

    def _build_belief_contexts(
        self,
        observations: list[dict],
        initial: dict[str, Any],
        *,
        motor_fn: Callable | None = None,
        motor_trace: list[dict[str, Any]] | None = None,
    ):
        """Run E-steps in order and apply each motor output to its observation.

        Historical input dictionaries are copied. The user callback receives
        that same working dictionary and can modify only fields already
        present in the user-defined observation schema.
        """
        if not hasattr(self.agent, "beliefs_update"):
            raise RuntimeError("Temporal/independent offline inference requires Agent_DAIH.beliefs_update().")
        if motor_fn is not None and self.config.observation_mode != "temporal":
            raise ValueError("motor_fn requires observation_mode='temporal'.")

        contexts = []
        if motor_fn is not None:
            self._reset_motor(motor_fn, observations[0])

        old_offline = getattr(self.agent, "offline", False)
        old_dt = getattr(self.agent, "_offline_dt", None)
        old_clip = getattr(self.agent, "_offline_belief_gradient_clip", None)
        try:
            self.agent.offline = True
            self.agent._offline_dt = self.config.dt
            self.agent._offline_belief_gradient_clip = self.config.belief_gradient_clip
            for index, observation in enumerate(observations):
                if self.config.observation_mode == "independent":
                    self._restore_beliefs(initial)
                prepared = deepcopy(observation)
                prepared, pi_obs, pi_substituted = preprocessing.handle_missing_observations(
                    prepared, deepcopy(self.agent.pi_obs), deepcopy(self.agent.pi_obs_substitued)
                )
                self.agent.pi_obs = pi_obs
                self.agent.pi_obs_substitued = pi_substituted

                # Flatten motor outputs: state names are not part of the
                # environment's observation/action contract.
                motors: dict[str, float] = {}
                state_free_energies: dict[str, float | None] = {}
                for item in self.agent.beliefs_update(prepared):
                    if not isinstance(item, tuple) or len(item) != 3:
                        continue
                    state_motors, free_energy, state_name = item
                    state_free_energies[state_name] = (
                        float(free_energy.detach().cpu().item())
                        if torch.is_tensor(free_energy) and torch.isfinite(free_energy).all()
                        else None
                    )
                    if isinstance(state_motors, dict):
                        motors.update(deepcopy(state_motors))

                if motor_fn is not None:
                    prepared = self._call_motor(motor_fn, prepared, motors or None)

                prepared, self.agent.pi_obs, self.agent.pi_obs_substitued = (
                    preprocessing.handle_missing_observations(
                        prepared,
                        self.agent.pi_obs,
                        self.agent.pi_obs_substitued,
                    )
                )

                
                if motor_trace is not None:
                    motor_trace.append({
                        "observation_index": index,
                        "motors": deepcopy(motors) if motors else None,
                        "observation_after_motor": deepcopy(prepared),
                        "free_energy_by_state": state_free_energies,
                    })

                contexts.append((prepared, self._snapshot_beliefs()))
        finally:
            self._restore_beliefs(initial)
            self.agent.offline = old_offline
            if old_dt is None:
                if hasattr(self.agent, "_offline_dt"):
                    delattr(self.agent, "_offline_dt")
            else:
                self.agent._offline_dt = old_dt
            if old_clip is None:
                if hasattr(self.agent, "_offline_belief_gradient_clip"):
                    delattr(self.agent, "_offline_belief_gradient_clip")
            else:
                self.agent._offline_belief_gradient_clip = old_clip
        return contexts

    @staticmethod
    def _mean_policy_results(results: list[dict[str, Any]]) -> dict[str, Any]:
        active = [r for r in results if r.get("loss") is not None]
        return {
            "updated": any(r.get("updated", False) for r in results),
            "loss": finite_float(sum(r["loss"] for r in active) / len(active)) if active else None,
            "parameters": max((r.get("parameters", 0) for r in results), default=0),
            "candidates": sum(r.get("candidates", 0) for r in results),
            "observations_processed": len(results),
            "per_observation": results,
        }

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

    def _policy_parameters(self) -> list[torch.nn.Parameter]:
        """Parameters belonging to global or state-local PolicyGenerators."""
        parameters, seen = [], set()
        for module in self._unique_modules():
            generator = getattr(module, "policy_generator", None)
            if isinstance(generator, nn.Module):
                for parameter in generator.parameters():
                    if parameter.requires_grad and id(parameter) not in seen:
                        parameters.append(parameter)
                        seen.add(id(parameter))
            local_modules = getattr(module, "_local_modules", None)
            if local_modules is not None:
                for key, local_module in local_modules.items():
                    if key.endswith(":p"):
                        for parameter in local_module.parameters():
                            if parameter.requires_grad and id(parameter) not in seen:
                                parameters.append(parameter)
                                seen.add(id(parameter))
        return parameters

    def _parameter_groups(self):
        """World-model parameters only; policy generators have their own loss."""
        policy_ids = {id(parameter) for parameter in self._policy_parameters()}
        parameters, seen = [], set()
        for module in self._unique_modules():
            for parameter in module.parameters():
                if (parameter.requires_grad and id(parameter) not in seen
                        and id(parameter) not in policy_ids):
                    parameters.append(parameter)
                    seen.add(id(parameter))
        if not parameters:
            return []
        return [{"params": parameters, "lr": self.config.learning_rate}]

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
        batches = observations if (isinstance(observations, list) and observations and isinstance(observations[0], tuple)) else [(observations, self._snapshot_beliefs())]
        for frame, belief_context in batches:
            with self._using_beliefs(belief_context):
                for state_name in state_names:
                    _, _, F_value = self._state_objective(state_name, frame)
                    if F_value.requires_grad and torch.isfinite(F_value):
                        losses.append(F_value / max(1, len(batches)))

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

    def _policy_candidate_posterior(self, policy_output, base_inputs=None):
        """Evaluate the candidate sequence with the agent's existing EFE path.

        Action decoding is discrete in the current agent API. Consequently the
        EFE term is used as a REINFORCE cost; G_prior remains directly regressed
        toward the detached G_posterior.
        """
        local_q = deepcopy(self.agent.Q_)
        source_inputs = self.agent.inputs_map if base_inputs is None else base_inputs
        local_inputs = {
            key: (value.detach().clone() if torch.is_tensor(value) else deepcopy(value))
            for key, value in source_inputs.items()
        }
        posterior_values = []
        sequence = policy_output.policy[0]
        for action_t in sequence:
            if not torch.isfinite(action_t).all():
                break
            if abs(float(action_t.detach().sum().item()) - 1.0) > 1e-4:
                break
            state_parents, action_name = preprocessing.get_one_hot_action_parents(
                action_t, self.agent.actions_id, self.agent.navigator
            )
            for parent_state in state_parents:
                actions_for_state = self.agent.navigator.child_nodes(
                    parent_state, node_type="action"
                )
                if action_name not in actions_for_state:
                    continue
                local_inputs[parent_state] = local_inputs[parent_state].detach().clone().requires_grad_(True)
                prepared = preprocessing.preprocessing_for_excepted_free_energy(
                    actions_for_state,
                    action_name,
                    parent_state,
                    local_inputs,
                    self.agent.pi_states,
                    self.agent.navigator,
                    self.agent.functions_map,
                    local_q,
                    self.agent.states_list,
                )
                g_posterior = self.agent.execepted_free_energy(
                    prepared["pi_priors"],
                    prepared["co_variances"],
                    prepared["transition_cost"],
                )
                posterior_values.append(g_posterior)
                new_precision = inference.seconde_derivative_from_output(
                    prepared["temporal_pred"], local_inputs[parent_state]
                )
                local_q[parent_state] = inference.update_distributions(
                    new_precision.detach(),
                    prepared["temporal_pred"].detach(),
                    local_q[parent_state],
                    eps=1e-6,
                )
        if not posterior_values:
            return None
        return torch.stack([value.reshape(()) for value in posterior_values]).mean()

    def _policy_inputs_from_observations(self, observations: dict) -> dict:
        """Overlay observed factors on a copy of the current agent input map."""
        inputs = {
            key: (value.detach().clone() if torch.is_tensor(value) else deepcopy(value))
            for key, value in self.agent.inputs_map.items()
        }
        for group_values in observations.values():
            if not isinstance(group_values, dict):
                continue
            for factor_name, raw_value in group_values.items():
                reference = inputs.get(factor_name)
                if torch.is_tensor(reference):
                    value = torch.as_tensor(
                        raw_value, device=reference.device, dtype=reference.dtype
                    )
                    if value.numel() == reference.numel():
                        value = value.reshape_as(reference)
                else:
                    value = torch.as_tensor(raw_value, dtype=torch.float32)
                inputs[factor_name] = value
        return inputs

    def _optimize_policy(self, observations: dict):
        parameters = self._policy_parameters()
        if not parameters:
            return {"updated": False, "loss": None, "parameters": 0, "candidates": 0}

        signature = tuple(id(parameter) for parameter in parameters)
        lr = (
            float(self.config.policy_learning_rate)
            if self.config.policy_learning_rate is not None
            else float(self.config.learning_rate) * float(self.agent.lr_policies)
        )
        if self._policy_optimizer is None or signature != self._policy_optimizer_signature:
            self._policy_optimizer = torch.optim.AdamW(
                parameters, lr=lr, weight_decay=self.config.weight_decay
            )
            self._policy_optimizer_signature = signature
            if self._pending_policy_optimizer_state is not None:
                self._policy_optimizer.load_state_dict(self._pending_policy_optimizer_state)
                self._pending_policy_optimizer_state = None
        else:
            for group in self._policy_optimizer.param_groups:
                group["lr"] = lr
                group["weight_decay"] = self.config.weight_decay

        self._policy_optimizer.zero_grad(set_to_none=True)
        policy_inputs = self._policy_inputs_from_observations(observations)
        try:
            candidates = model_components.get_policy_candidates(
                self.agent.navigator,
                self.agent.functions_map,
                policy_inputs,
                self.agent.states_list,
                self.agent.Q_,
            )
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
            return {"updated": False, "loss": None, "parameters": 0,
                    "candidates": 0, "reason": f"candidate_generation_failed: {exc}"}

        losses = []
        diagnostics = {}
        for state_name, candidate in candidates.items():
            policy_output = getattr(candidate, "policy", None)
            if policy_output is None or policy_output.logits is None:
                continue
            try:
                g_posterior = self._policy_candidate_posterior(policy_output, policy_inputs)
                if g_posterior is None or not torch.isfinite(g_posterior):
                    continue
                g_posterior = g_posterior.detach()
                g_prior = policy_output.G.reshape(-1).mean()

                logits = policy_output.logits
                actions = policy_output.policy
                mask = policy_output.temporal_mask.to(dtype=logits.dtype)
                log_probabilities = torch.log_softmax(logits, dim=-1)[..., :actions.shape[-1]]
                selected_log_probability = (actions * log_probabilities).sum(dim=-1)
                sequence_log_probability = (selected_log_probability * mask).sum()

                baseline = self._policy_baselines.get(state_name, 0.0)
                advantage = g_posterior - float(baseline)
                efe_loss = advantage * (-sequence_log_probability)
                alignment_loss = (g_prior - g_posterior).square()
                total = (
                    float(self.config.policy_efe_weight) * efe_loss
                    + float(self.config.policy_alignment_weight) * alignment_loss
                )
                if torch.isfinite(total) and total.requires_grad:
                    losses.append(total.reshape(()))
                self._policy_baselines[state_name] = (
                    float(self.config.policy_baseline_momentum) * float(baseline)
                    + (1.0 - float(self.config.policy_baseline_momentum)) * float(g_posterior.item())
                )
                diagnostics[state_name] = {
                    "G_prior": finite_float(g_prior.detach().item()),
                    "G_posterior": finite_float(g_posterior.item()),
                    "efe_loss": finite_float(efe_loss.detach().item()),
                    "alignment_loss": finite_float(alignment_loss.detach().item()),
                }
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, UnboundLocalError) as exc:
                diagnostics[state_name] = {"skipped": str(exc)}

        if not losses:
            return {"updated": False, "loss": None, "parameters": 0,
                    "candidates": len(candidates), "states": diagnostics}

        total_loss = torch.stack(losses).sum()
        total_loss.backward()
        active_parameters = [p for p in parameters if p.grad is not None]
        if active_parameters:
            torch.nn.utils.clip_grad_norm_(active_parameters, self.config.gradient_clip)
            self._policy_optimizer.step()
        return {
            "updated": bool(active_parameters),
            "loss": finite_float(total_loss.detach().item()),
            "parameters": sum(p.numel() for p in active_parameters),
            "candidates": len(candidates),
            "states": diagnostics,
        }

    def _evaluate(self, observations) -> tuple[float, dict[str, StateMetrics]]:
        batches = observations if (isinstance(observations, list) and observations and isinstance(observations[0], tuple)) else [(observations, self._snapshot_beliefs())]
        collected: dict[str, list[StateMetrics]] = defaultdict(list)
        for frame, belief_context in batches:
            with self._using_beliefs(belief_context):
                for state_name, node in self.agent.navigator.graph.items():
                    if node.get("type") != "state":
                        continue
                    metric, _ = self._measure_state(state_name, frame)
                    collected[state_name].append(metric)
        metrics = {}
        for state_name, items in collected.items():
            n = len(items)
            metrics[state_name] = StateMetrics(
                state=state_name,
                free_energy=sum(x.free_energy for x in items) / n,
                gradient_norm=sum(x.gradient_norm for x in items) / n,
                gradient_raw=[v for x in items for v in x.gradient_raw],
                gradient_sum=sum(x.gradient_sum for x in items) / n,
                gradient_finite=all(x.gradient_finite for x in items),
                precision=sum(x.precision for x in items) / n,
                complexity=sum(x.complexity for x in items) / n,
                accuracy=sum(x.accuracy for x in items) / n,
                local_sensitivity=sum(x.local_sensitivity for x in items) / n,
                hessian_norm=sum(x.hessian_norm for x in items) / n,
                local_sensitivity_finite=all(x.local_sensitivity_finite for x in items),
                hessian_finite=all(x.hessian_finite for x in items),
            )
        total = sum(metric.free_energy for metric in metrics.values())
        return finite_float(total), metrics

    # ------------------------------------------------------------------
    # Eight-phase per-state decision system
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

    def _is_decorrelated(self, history: deque, phase: int) -> bool:
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
        if decorrelated:
            operations = {
                1: {"action": "prune_obsolete_weights", "target": state_name},
                2: {"action": "regularize_hyperparameters", "target": state_name},
                3: {"action": "prune_neurons", "target": state_name},
                4: {"action": "prune_intermediate_layer", "target": state_name},
                5: {"action": "remove_low_utility_coordinate", "target": state_name},
                6: {"action": "remove_under_state", "target": state_name},
                7: {"action": "remove_redundant_state", "target": state_name},
                8: {"action": "remove_redundant_hierarchy_layer", "target": state_name},
            }
            return operations.get(phase, {"action": "none", "target": state_name})

        operations = {
            3: {"action": "add_neurons", "target": state_name},
            4: {"action": "add_intermediate_layer", "target": state_name},
            5: {"action": "add_generalized_coordinate", "target": state_name},
            6: {"action": "add_under_state", "target": state_name},
            7: {"action": "add_state", "target": state_name},
            8: {"action": "add_hierarchy_layer", "target": state_name},
        }
        return operations.get(phase, {"action": "none", "target": state_name})

    def _shock(self, history: deque) -> float:
        """Relative temporal change of curvature (a practical third-order proxy)."""
        items = list(history)[-self.config.shock_window:]
        if len(items) < 2:
            return 0.0
        values = []
        for previous, current in zip(items[:-1], items[1:]):
            if not previous.hessian_finite or not current.hessian_finite:
                continue
            curvature_change = abs(current.hessian_norm - previous.hessian_norm) / (
                abs(previous.hessian_norm) + 1e-8
            )
            gradient_change = abs(current.gradient_norm - previous.gradient_norm) / (
                abs(previous.gradient_norm) + 1e-8
            )
            values.append(0.5 * (curvature_change + gradient_change))
        return finite_float(sum(values) / len(values)) if values else 0.0

    def _shock_exceeds_local_precision(self, shock: float, precision: float) -> bool:
        # Normalize curvature to [0, 1] before comparing it with relative shock.
        local_precision = max(0.0, precision) / (1.0 + max(0.0, precision))
        return shock > max(float(self.config.shock_threshold), local_precision)

    @staticmethod
    def _aggregate_metrics(metrics: dict[str, StateMetrics]) -> StateMetrics:
        items = list(metrics.values())
        if not items:
            raise ValueError("Cannot aggregate an empty metric collection.")
        return StateMetrics(
            state="__world_model__",
            free_energy=finite_float(sum(item.free_energy for item in items)),
            gradient_norm=finite_float(math.sqrt(sum(item.gradient_norm ** 2 for item in items))),
            gradient_raw=[value for item in items for value in item.gradient_raw],
            gradient_sum=finite_float(sum(item.gradient_sum for item in items)),
            gradient_finite=all(item.gradient_finite for item in items),
            precision=finite_float(sum(item.precision for item in items) / len(items)),
            complexity=finite_float(sum(item.complexity for item in items)),
            accuracy=finite_float(sum(item.accuracy for item in items)),
            local_sensitivity=finite_float(sum(item.local_sensitivity for item in items)),
            hessian_norm=finite_float(math.sqrt(sum(item.hessian_norm ** 2 for item in items))),
            local_sensitivity_finite=all(item.local_sensitivity_finite for item in items),
            hessian_finite=all(item.hessian_finite for item in items),
        )

    def _initialize_local_phases(self, metrics: dict[str, StateMetrics]) -> None:
        for state_name, metric in metrics.items():
            if state_name not in self.progress:
                self.progress[state_name] = StateProgress(
                    phase=4,
                    history=deque(maxlen=self.config.derivative_window),
                    phase_entry_free_energy=metric.free_energy,
                    phase_best_free_energy=metric.free_energy,
                )

    def _update_progress(self, metrics: dict[str, StateMetrics]):
        decisions = {}
        proposed = []
        if self.config.weights_only:
            return ({"world_model": {
                "phase": self.global_progress.phase,
                "phase_name": self.PHASE_NAMES[min(self.global_progress.phase, 8)],
                "weights_only": True,
                "mutation": None,
            }}, proposed)

        # --------------------------------------------------------------
        # Shared WorldModel: phases 1-3 are controlled globally from all F_i.
        # --------------------------------------------------------------
        aggregate = self._aggregate_metrics(metrics)
        global_state = self.global_progress
        if global_state.history.maxlen != self.config.derivative_window:
            global_state.history = deque(global_state.history, maxlen=self.config.derivative_window)
        global_state.history.append(aggregate)
        global_state.phase_iterations += 1
        if global_state.phase_best_free_energy is None:
            global_state.phase_best_free_energy = aggregate.free_energy
        else:
            global_state.phase_best_free_energy = min(
                global_state.phase_best_free_energy, aggregate.free_energy
            )
        global_shock = self._shock(global_state.history)
        aggregate.shock = global_shock
        aggregate.shock_finite = len(global_state.history) >= 2
        global_phase_before = global_state.phase
        global_operation = None

        # A bidirectional controller may return to shared-model optimization
        # if the first local level degrades the global objective.
        if (self.config.phase_transition_mode == "bidirectional"
                and global_state.phase >= 4
                and global_state.phase_iterations >= self.config.patience
                and global_state.phase_entry_free_energy is not None):
            tolerance = float(self.config.backtrack_tolerance) * max(
                abs(global_state.phase_entry_free_energy), 1e-8
            )
            if (aggregate.free_energy > global_state.phase_entry_free_energy + tolerance
                    and global_state.phase_best_free_energy is not None
                    and global_state.phase_best_free_energy >= global_state.phase_entry_free_energy - tolerance):
                global_state.phase = 3
                global_state.phase_iterations = 0
                global_state.phase_entry_free_energy = aggregate.free_energy
                global_state.phase_best_free_energy = aggregate.free_energy
                global_state.history.clear()
                self.progress.clear()
                decisions["world_model"] = {
                    "phase_before": global_phase_before,
                    "phase": 3,
                    "phase_name": self.PHASE_NAMES[3],
                    "free_energy": aggregate.free_energy,
                    "shock": global_shock,
                    "transition": "backtrack_to_shared_world_model",
                    "mutation": None,
                }
                return decisions, proposed

        if global_state.phase < 4:
            if global_state.pending_mutation:
                decisions["world_model"] = {
                    "phase_before": global_phase_before,
                    "phase": global_state.phase,
                    "phase_name": self.PHASE_NAMES[global_state.phase],
                    "free_energy": aggregate.free_energy,
                    "shock": global_shock,
                    "pending_mutation": True,
                    "mutation": None,
                }
                return decisions, proposed

            global_decorrelated = self._is_decorrelated(global_state.history, global_state.phase)
            global_transition = (
                global_state.phase_iterations >= self.config.phase_min_iterations
                and len(global_state.history) >= self.config.derivative_min_samples
                and self._shock_exceeds_local_precision(global_shock, aggregate.precision)
            )
            if global_decorrelated:
                target = next((
                    name for name in metrics
                    if global_state.phase != 3 or self.config.state_constraints.get(name, {}).get("allow_neurons", True)
                ), next(iter(metrics)))
                global_operation = self._mutation_for(target, global_state.phase, True)
                if global_operation.get("action") != "none":
                    global_operation["scope"] = "world_model"
                    proposed.append(global_operation)
                    global_state.pending_mutation = True
            elif global_transition:
                if global_state.phase < 3:
                    global_state.phase += 1
                    global_state.phase_iterations = 0
                    global_state.phase_entry_free_energy = aggregate.free_energy
                    global_state.phase_best_free_energy = aggregate.free_energy
                    global_state.history.clear()
                    global_state.history.append(aggregate)
                    if global_state.phase == 3:
                        target = next((
                            name for name in metrics
                            if self.config.state_constraints.get(name, {}).get("allow_neurons", True)
                        ), next(iter(metrics)))
                        global_operation = self._mutation_for(target, 3, False)
                        global_operation["scope"] = "world_model"
                        proposed.append(global_operation)
                        global_state.pending_mutation = True
                else:
                    # Shared representation is considered sufficiently explored.
                    # Subsequent architecture decisions are local to each state.
                    global_state.phase = 4
                    global_state.phase_iterations = 0
                    global_state.phase_entry_free_energy = aggregate.free_energy
                    global_state.phase_best_free_energy = aggregate.free_energy
                    global_state.history.clear()
                    global_state.history.append(aggregate)
                    self._initialize_local_phases(metrics)

            decisions["world_model"] = {
                "phase_before": global_phase_before,
                "phase": global_state.phase,
                "phase_name": self.PHASE_NAMES[global_state.phase],
                "free_energy": aggregate.free_energy,
                "gradient_norm": aggregate.gradient_norm,
                "gradient_sum": aggregate.gradient_sum,
                "precision": aggregate.precision,
                "hessian_norm": aggregate.hessian_norm,
                "shock": global_shock,
                "shock_threshold": self.config.shock_threshold,
                "shock_triggered": global_transition,
                "decorrelation": global_decorrelated,
                "mutation": global_operation,
                "scope": "all_states_shared_world_model",
            }
            if global_state.phase < 4:
                return decisions, proposed

        # --------------------------------------------------------------
        # Agent architecture: phases 4-7 are controlled independently per state.
        # --------------------------------------------------------------
        self._initialize_local_phases(metrics)
        for state_name, metric in metrics.items():
            progress = self.progress[state_name]
            if progress.history.maxlen != self.config.derivative_window:
                progress.history = deque(progress.history, maxlen=self.config.derivative_window)
            progress.history.append(metric)
            progress.phase_iterations += 1
            if progress.phase_best_free_energy is None:
                progress.phase_best_free_energy = metric.free_energy
            else:
                progress.phase_best_free_energy = min(
                    progress.phase_best_free_energy, metric.free_energy
                )
            shock = self._shock(progress.history)
            metric.shock = shock
            metric.shock_finite = len(progress.history) >= 2
            shock_trigger = (
                progress.phase_iterations >= self.config.phase_min_iterations
                and len(progress.history) >= self.config.derivative_min_samples
                and self._shock_exceeds_local_precision(shock, metric.precision)
            )
            decorrelated = self._is_decorrelated(progress.history, progress.phase)
            phase_before = progress.phase
            operation = None

            if (self.config.phase_transition_mode == "bidirectional"
                    and progress.phase > 4
                    and progress.phase_iterations >= self.config.patience
                    and progress.phase_entry_free_energy is not None):
                tolerance = float(self.config.backtrack_tolerance) * max(
                    abs(progress.phase_entry_free_energy), 1e-8
                )
                if (metric.free_energy > progress.phase_entry_free_energy + tolerance
                        and progress.phase_best_free_energy is not None
                        and progress.phase_best_free_energy >= progress.phase_entry_free_energy - tolerance):
                    progress.phase -= 1
                    progress.phase_iterations = 0
                    progress.phase_entry_free_energy = metric.free_energy
                    progress.phase_best_free_energy = metric.free_energy
                    progress.history.clear()
                    decisions[state_name] = {
                        "phase_before": phase_before,
                        "phase": progress.phase,
                        "phase_name": self.PHASE_NAMES[progress.phase],
                        "free_energy": metric.free_energy,
                        "shock": shock,
                        "transition": "backtrack",
                        "mutation": None,
                    }
                    continue

            if progress.stable:
                decisions[state_name] = {
                    "phase_before": phase_before,
                    "phase": progress.phase,
                    "phase_name": self.PHASE_NAMES[progress.phase],
                    "free_energy": metric.free_energy,
                    "gradient_norm": metric.gradient_norm,
                    "precision": metric.precision,
                    "hessian_norm": metric.hessian_norm,
                    "shock": shock,
                    "stable": True,
                    "mutation": None,
                }
                continue

            if progress.pending_mutation:
                decisions[state_name] = {
                    "phase_before": phase_before,
                    "phase": progress.phase,
                    "phase_name": self.PHASE_NAMES[progress.phase],
                    "free_energy": metric.free_energy,
                    "shock": shock,
                    "pending_mutation": True,
                    "mutation": None,
                }
                continue

            if decorrelated:
                operation = self._mutation_for(state_name, progress.phase, True)
                if operation.get("action") != "none":
                    operation["scope"] = "state_local"
                    proposed.append(operation)
                    progress.pending_mutation = True
            elif shock_trigger:
                if progress.phase < 8:
                    progress.phase += 1
                    progress.phase_iterations = 0
                    progress.phase_entry_free_energy = metric.free_energy
                    progress.phase_best_free_energy = metric.free_energy
                    progress.history.clear()
                    progress.history.append(metric)
                    operation = self._mutation_for(state_name, progress.phase, False)
                    if operation.get("action") != "none":
                        operation["scope"] = "state_local"
                        proposed.append(operation)
                        progress.pending_mutation = True

            if (progress.phase == 8 and not shock_trigger and not progress.pending_mutation
                    and len(progress.history) >= self.config.patience):
                progress.stable = True

            decisions[state_name] = {
                "phase_before": phase_before,
                "phase": progress.phase,
                "phase_name": self.PHASE_NAMES[progress.phase],
                "free_energy": metric.free_energy,
                "gradient_norm": metric.gradient_norm,
                "gradient_sum": metric.gradient_sum,
                "gradient_raw": metric.gradient_raw,
                "gradient_finite": metric.gradient_finite,
                "precision": metric.precision,
                "local_sensitivity": metric.local_sensitivity,
                "hessian_norm": metric.hessian_norm,
                "shock": shock,
                "shock_threshold": self.config.shock_threshold,
                "shock_triggered": shock_trigger,
                "decorrelation": decorrelated,
                "stable": progress.stable,
                "mutation": operation,
                "scope": "state_local",
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
            "global_progress": {
                "phase": self.global_progress.phase,
                "stable": self.global_progress.stable,
                "pending_mutation": self.global_progress.pending_mutation,
                "phase_iterations": self.global_progress.phase_iterations,
                "phase_entry_free_energy": self.global_progress.phase_entry_free_energy,
                "phase_best_free_energy": self.global_progress.phase_best_free_energy,
                "history": [metric.to_dict() for metric in self.global_progress.history],
            },
            "progress": {
                name: {
                    "phase": item.phase,
                    "stable": item.stable,
                    "pending_mutation": item.pending_mutation,
                    "phase_iterations": item.phase_iterations,
                    "phase_entry_free_energy": item.phase_entry_free_energy,
                    "phase_best_free_energy": item.phase_best_free_energy,
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
        local_phase = 4 if self.global_progress.phase >= 4 else 1
        self.progress = {
            name: StateProgress(phase=local_phase)
            for name, node in self.agent.navigator.graph.items()
            if node.get("type") == "state"
        } if local_phase >= 4 else {}
        self.pending_plan = None
        return True

    # ------------------------------------------------------------------
    # Public iteration
    # ------------------------------------------------------------------

    def step(
        self,
        observations: dict | list[dict] | tuple[dict, ...],
        *,
        test_observations: dict | list[dict] | tuple[dict, ...] | None = None,
        rebuild_callback: Callable | None = None,
        motor_fn: Callable | None = None,
        motor_train_fn: Callable | None = None,
    ) -> dict[str, Any]:
        self._validate_config()
        if self.agent.navigator is None:
            raise RuntimeError("Call Agent_DAIH.model() before training_offline().")
        training_frames = self._normalize_observations(observations)
        test_frames = self._normalize_observations(
            test_observations if test_observations is not None else observations
        )
        if self.config.train_motor and motor_fn is None:
            raise ValueError("train_motor=True requires a user-supplied motor_fn.")
        if self.config.train_motor and not callable(motor_train_fn):
            raise ValueError("train_motor=True requires motor_train_fn(records, observations).")
        if motor_fn is not None and self.config.observation_mode != "temporal":
            raise ValueError("A motor_fn requires observation_mode='temporal'.")
        self.iteration += 1

        # Beliefs are ephemeral during offline training. Build one posterior
        # context per frame, carrying it forward only in temporal mode.
        initial_beliefs = self._snapshot_beliefs()
        test_contexts = self._build_belief_contexts(
            test_frames, initial_beliefs, motor_fn=motor_fn
        )
        F_before, _ = self._evaluate(test_contexts)
        checkpoint_path = self._checkpoint_before(F_before)
        # Keep the starting model if subsequent optimization makes it worse.
        best_agent_updated = self._consider_best_agent(F_before)

        motor_trace: list[dict[str, Any]] = []
        training_contexts = self._build_belief_contexts(
            training_frames,
            initial_beliefs,
            motor_fn=motor_fn,
            motor_trace=motor_trace,
        )
        training_batches = [(frame, context) for frame, context in training_contexts]

        # The E-step updates temporary Q_t contexts; the M-step optimizes the
        # generative model against the mean sequence objective. Persistent agent
        # beliefs and online observation precisions are restored afterwards.
        optimization = (
            self._optimize_world_model(training_batches)
            if self.config.train_world_model
            else {"updated": False, "loss": None, "parameters": 0, "disabled": True}
        )
        if self.config.train_policy:
            policy_results = []
            for frame, context in training_batches:
                with self._using_beliefs(context):
                    policy_results.append(self._optimize_policy(frame))
            policy_optimization = self._mean_policy_results(policy_results)
        else:
            policy_optimization = {"updated": False, "loss": None, "parameters": 0, "disabled": True}

        motor_optimization = {"updated": False, "disabled": True}
        if self.config.train_motor:
            motor_optimization = motor_train_fn(
                deepcopy(motor_trace), deepcopy(training_frames)
            )
            if motor_optimization is None:
                motor_optimization = {"updated": True, "result": None}

        belief_updates = [
            {
                "observation_index": item["observation_index"],
                "inference_updated": True,
                "motors": deepcopy(item["motors"]),
                "observation_after_motor": deepcopy(item["observation_after_motor"]),
            }
            for item in motor_trace
        ]

        # Re-infer test beliefs with the updated generative model and motor callback.
        test_contexts_after = self._build_belief_contexts(
            test_frames, initial_beliefs, motor_fn=motor_fn
        )
        F_after, metrics_after = self._evaluate(test_contexts_after)
        self.last_metrics = metrics_after

        decisions, proposed = self._update_progress(metrics_after)
        plan = {
            "version": "7.1",
            "observation_mode": self.config.observation_mode,
            "observations_processed": len(training_frames),
            "phase_transition_mode": self.config.phase_transition_mode,
            "shock_threshold": self.config.shock_threshold,
            "detection_mode": self.config.detection_mode,
            "detection_parameters": {
                "derivative_window": self.config.derivative_window,
                "derivative_min_samples": self.config.derivative_min_samples,
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
            "optimization": optimization,
            "policy_optimization": policy_optimization,
            "note": (
                "Shared WorldModel parameters are optimized against the sum of all state free energies. "
                "Policy generators use expected-free-energy REINFORCE and G_prior/G_posterior alignment. "
                "Phase-4 context blocks are state/layer-local; later coordinate/state/hierarchy mutations "
                "are local to their target declaration. Per-observation beliefs are temporary; "
                "persistent Q and online precision state are restored after offline inference. "
                "The user motor callback receives flat motor outputs and can modify existing "
                "fields of the working observation dictionary."
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
                if skipped_mutation.get("scope") == "world_model":
                    self.global_progress.pending_mutation = False
                    if self.global_progress.phase < 4:
                        self.global_progress.phase += 1
                        self.global_progress.phase_iterations = 0
                        self.global_progress.history.clear()
                        if self.global_progress.phase >= 4:
                            self._initialize_local_phases(metrics_after)
                elif target in self.progress:
                    local_progress = self.progress[target]
                    local_progress.pending_mutation = False
                    if local_progress.phase < 8:
                        # A disabled, capped or inapplicable dimension is
                        # skipped; it must not freeze the state before later
                        # structural levels are considered.
                        local_progress.phase += 1
                        local_progress.phase_iterations = 0
                        local_progress.phase_entry_free_energy = (
                            metrics_after[target].free_energy if target in metrics_after else None
                        )
                        local_progress.phase_best_free_energy = local_progress.phase_entry_free_energy
                        local_progress.history.clear()
                    else:
                        local_progress.stable = True
            if not rebuilt:
                # Phase-1/2 operations are in-place (masking / scalar
                # regularization), not architecture rebuilds.
                for applied_mutation in rebuild_result.get("applied", []):
                    target = applied_mutation.get("target")
                    if applied_mutation.get("scope") == "world_model":
                        self.global_progress.pending_mutation = False
                        self.global_progress.phase_iterations = 0
                        self.global_progress.history.clear()
                    elif target in self.progress:
                        self.progress[target].pending_mutation = False
                        self.progress[target].phase_iterations = 0
                        self.progress[target].history.clear()
            self.pending_plan = None
            plan_path = self._write_plan(plan_to_persist)
            if rebuilt or rebuild_result.get("applied"):
                # Evaluate the updated/rebuilt declaration on the same test batch.
                if rebuilt:
                    # A structural rebuild may change posterior dimensions.
                    initial_beliefs = self._snapshot_beliefs()
                test_contexts_rebuilt = self._build_belief_contexts(
                    test_frames, initial_beliefs, motor_fn=motor_fn
                )
                F_after, metrics_after = self._evaluate(test_contexts_rebuilt)
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
            phases = [self.global_progress.phase] + [item.phase for item in self.progress.values()]
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
            "policy_optimization": policy_optimization,
            "motor_optimization": motor_optimization,
            "motor_outputs": [deepcopy(item["motors"]) for item in motor_trace],
            "motor_trace": motor_trace,
            "belief_updates": belief_updates,  # ephemeral per-observation inference; persistent Q is restored
            "q_updated_offline": False,
            "observation_mode": self.config.observation_mode,
            "observations_processed": len(training_frames),
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
