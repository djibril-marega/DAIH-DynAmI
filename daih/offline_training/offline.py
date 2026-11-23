"""Adaptive offline optimization for Agent_DAIH (v12).

The shared WorldGenerativeModel is optimized against the sum of all state free energies.
Policy generators are optimized separately with expected-free-energy policy
 gradients and a G_prior/G_posterior alignment loss. The controller has one
shared progression for WorldGenerativeModel phases 1-3 and independent per-state
progression for agent-architecture phases 4-8.

Offline learning performs temporary per-observation inference, then restores persistent posterior beliefs Q and online observation precisions.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from contextlib import contextmanager
import json
import math
import time
import os
import tempfile

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
    # Maximum number of historical observations passed through f/g in one
    # native model batch. Gradients accumulate across chunks before one step.
    model_batch_size: int = 64
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

    # Training scopes. The WorldGenerativeModel objective remains the sum of all F_i.
    train_world_generative_model: bool = True
    train_policy: bool = True
    # The motor function is user-owned. Its training is opt-in and delegated
    # to the user-provided motor_train_fn callback.
    train_motor: bool = False
    weights_only: bool = False  # disables all architecture/phase mutations
    max_phase: int = 8  # cumulative phase ceiling: 3 means phases 1, 2 and 3
    policy_learning_rate: float | None = None
    policy_efe_weight: float = 1.0
    policy_alignment_weight: float = 1.0
    policy_baseline_momentum: float = 0.9

    neuron_step: int = 1
    world_generative_model_min_neurons: int = 64
    world_generative_model_max_neurons: int = 512
    max_total_states: int | None = 64  # Legacy global safety cap; graph_constraints.max_states is preferred.
    graph_constraints: dict[str, Any] = field(default_factory=dict)
    state_constraints: dict[str, dict[str, Any]] = field(default_factory=dict)
    checkpoint_dir: str | None = "offline_artifacts"
    checkpoint_every: int = 50  # Full "after" checkpoint cadence, in offline steps.
    save_phase_checkpoints: bool = True  # Mandatory end-of-phase full-agent snapshots by default.
    phase_checkpoint_prefix: str = "offline_phase"
    plan_filename: str = "offline_plan.json"
    checkpoint_filename: str = "offline_before.pt"
    final_checkpoint_filename: str = "offline_after.pt"
    best_checkpoint_filename: str = "offline_best_agent.pt"

    # DAIH offline paradigm v8: three macro-phases.
    global_precision: float = 1e-3
    macro_patience: int = 5
    macro_relative_tolerance: float = 1e-3
    divergence_tolerance: float = 0.05
    initial_top_k: int = 1
    top_k_step: int = 1
    temporal_horizon: int = 8  # Legacy compatibility only; Dreamer H comes from policy sequence length.
    temporal_loss_weight: float = 1.0
    policy_precision_min: float = 1e-9
    structural_evaluation_limit: int = 1
    enable_structural_phase: bool = True

    # Persistent observations collected during online inference.
    inference_memory_path: str = "offline_artifacts/observation_memory.pt"
    transition_memory_path: str = "offline_artifacts/transition_memory.pt"
    inference_memory_capacity: int = 256
    transition_memory_capacity: int = 256
    inference_memory_old_fraction: float = 0.25
    include_inference_memory: bool = True
    include_transition_memory: bool = True
    temporal_precision: float | None = None


@dataclass
class StateProgress:
    phase: int = 1
    history: deque = field(default_factory=lambda: deque(maxlen=8))
    stable: bool = False
    pending_mutation: bool = False
    phase_iterations: int = 0
    phase_entry_free_energy: float | None = None
    phase_best_free_energy: float | None = None


class _LegacyOfflineTrainer:
    """Stateful offline trainer; progress survives successive method calls."""

    PHASE_NAMES = {
        1: "shared_world_generative_model_weights",
        2: "generative_hyperparameters",
        3: "shared_world_generative_model_capacity",
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
        # One shared controller governs WorldGenerativeModel phases 1-3. Per-state
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
        if int(self.config.model_batch_size) < 1:
            raise ValueError("model_batch_size must be >= 1.")
        if self.config.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1.")
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
        if not 1 <= int(self.config.max_phase) <= 8:
            raise ValueError("max_phase must be an integer between 1 and 8.")
        if self.config.world_generative_model_min_neurons < 16:
            raise ValueError("world_generative_model_min_neurons must be >= 16.")
        if self.config.world_generative_model_max_neurons < self.config.world_generative_model_min_neurons:
            raise ValueError("world_generative_model_max_neurons must be >= world_generative_model_min_neurons.")
        graph_constraints = self.config.graph_constraints
        min_states = int(graph_constraints.get("min_states", 1))
        max_states = graph_constraints.get("max_states")
        if min_states < 1 or (max_states is not None and int(max_states) < min_states):
            raise ValueError(
                "Invalid graph state-count bounds: "
                "require 1 <= graph_constraints['min_states'] <= "
                "graph_constraints['max_states'] (or None)."
            )
        if (self.config.max_total_states is not None
                and int(self.config.max_total_states) < min_states):
            raise ValueError("max_total_states must be >= graph_constraints.min_states.")

        graph_level_keys = {
            "min_states", "max_states", "allow_state_addition", "allow_state_removal"
        }
        for state_name, constraints in self.config.state_constraints.items():
            for flag in (
                "allow_mutations", "allow_removal", "allow_neurons",
                "allow_intermediate_layers", "allow_coordinates",
                "allow_under_state", "allow_hierarchy_addition",
                "allow_hierarchy_removal",
            ):
                if flag in constraints and not isinstance(constraints[flag], bool):
                    raise ValueError(
                        f"state_constraints[{state_name!r}][{flag!r}] must be a boolean."
                    )
            misplaced = graph_level_keys.intersection(constraints)
            if misplaced:
                names = ", ".join(sorted(misplaced))
                raise ValueError(
                    f"Graph-level constraints ({names}) must be declared in "
                    "graph_constraints, not state_constraints."
                )
            min_coordinates = int(constraints.get("min_coordinates", 1))
            max_coordinates = constraints.get("max_coordinates")
            min_under_state = int(constraints.get(
                "min_under_state", constraints.get("min_size", 1)
            ))
            max_under_state = constraints.get(
                "max_under_state", constraints.get("max_size")
            )
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
        """Materialize one batch of observations from a dict or iterable of dicts.

        Dataset loaders may provide generators to avoid building an additional
        intermediate list. A training step needs to replay the same batch for
        evaluation and optimization, so the iterable is materialized exactly
        once here.
        """
        if isinstance(observations, dict):
            return [deepcopy(observations)]
        if isinstance(observations, (str, bytes)) or not isinstance(observations, Iterable):
            raise TypeError(
                "observations must be a dictionary or a non-empty iterable of dictionaries."
            )
        frames = list(observations)
        if not frames:
            raise TypeError("observations must be a dictionary or a non-empty iterable of dictionaries.")
        if not all(isinstance(item, dict) for item in frames):
            raise TypeError("An observations iterable must contain dictionaries, one per observation.")
        return [deepcopy(item) for item in frames]

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
                allowed_states = self._observation_compatibility.get(id(observation))
                prepared = deepcopy(observation)
                if allowed_states is not None:
                    self._observation_compatibility[id(prepared)] = set(allowed_states)
                prepared, pi_obs, pi_substituted = preprocessing.handle_missing_observations(
                    prepared,
                    deepcopy(self.agent.pi_obs),
                    deepcopy(self.agent.pi_obs_substitued),
                    navigator=self.agent.navigator,
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
                        navigator=self.agent.navigator,
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
            state_min=self.agent.min_state_value,
            state_max=self.agent.max_state_value,
            observation_bounds=self.agent.observation_bounds,
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

    def _supports_native_batch(self, state_name: str) -> bool:
        """Return whether this state's f/g path is backed by batch-aware WorldGenerativeModels.

        User-registered local callables are deliberately excluded unless they
        explicitly advertise ``supports_batch = True``. This avoids silently
        changing the semantics of legacy scalar callables.
        """
        from ..world_generative_model import WorldGenerativeModel

        relevant = [state_name]
        relevant.extend(self.agent.navigator.parent_nodes(state_name, node_type="state"))
        for name in relevant:
            for kind in ("f", "g"):
                ref = self.agent.navigator.function(name, kind)
                if ref is None:
                    continue
                module = (self.agent.functions_map or {}).get(ref)
                if isinstance(module, WorldGenerativeModel):
                    state_id = self.agent.states_list.index(name) + 1
                    layer_id = self.agent.navigator.get_state_layer(name)
                    if module.has_local_function(state_id, layer_id, kind):
                        local_key = (state_id, layer_id, kind)
                        local_fn = module._local_functions.get(local_key)
                        if local_fn is None:
                            module_key = module._local_module_keys.get(local_key)
                            local_fn = module._local_modules.get(module_key)
                        if not bool(getattr(local_fn, "supports_batch", False)):
                            return False
                elif not bool(getattr(module, "supports_batch", False)):
                    return False
        return True

    def _prepare_state_objectives_batch(self, state_name: str, entries):
        """Prepare a time batch with native batched f/g model calls.

        Each entry contains its own post-inference belief snapshot. The snapshots
        were produced in chronological order by _build_belief_contexts; batching
        here only combines independent numerical model evaluations after those
        sequential belief updates have completed.
        """
        frames = [frame for _, frame, _ in entries]
        contexts = [context for _, _, context in entries]
        batch_size = len(entries)
        batched_inputs = preprocessing.stack_inputs_maps(
            [context["inputs_map"] for context in contexts]
        )
        if batched_inputs is None:
            raise ValueError("Observation contexts do not have stack-compatible model inputs.")

        first_Q = contexts[0]["Q_"]
        state_prior = model_components.get_parent_g_predictions_batch(
            self.agent.navigator, self.agent.functions_map, batched_inputs,
            state_name, self.agent.states_list, first_Q, batch_size,
        )
        f_transition = model_components.get_transition_prediction_batch(
            self.agent.navigator, self.agent.functions_map, batched_inputs,
            state_name, self.agent.states_list, first_Q,
            state_min=self.agent.min_state_value,
            state_max=self.agent.max_state_value, batch_size=batch_size,
        ).unsqueeze(1)
        prediction = model_components.get_state_prediction_batch(
            self.agent.navigator, self.agent.functions_map, batched_inputs,
            state_name, self.agent.states_list, first_Q,
            state_min=self.agent.min_state_value,
            state_max=self.agent.max_state_value,
            observation_bounds=self.agent.observation_bounds,
            batch_size=batch_size,
        )

        state_posteriors = []
        pi_priors = []
        actuals = []
        pi_observations = []
        for frame, context in zip(frames, contexts):
            state_posteriors.append(
                context["inputs_map"][state_name].reshape(1, -1)
            )
            pi_priors.append(model_components.get_parent_g_precisions(
                self.agent.navigator, state_name, context["pi_priors"], context["Q_"]
            ))
            actuals.append(model_components.get_actual(
                self.agent.navigator, context["Q_"], state_name, frame
            ))
            pi_observations.append(model_components.get_actual_precision(
                self.agent.navigator, context["pi_obs"], context["Q_"], state_name
            ))

        state_posterior = torch.stack(state_posteriors, dim=0)
        pi_prior = torch.stack(pi_priors, dim=0)
        actual = torch.stack(actuals, dim=0)
        pi_obs = torch.stack(pi_observations, dim=0)

        if state_prior.shape[1] == 0:
            state_prior = torch.zeros_like(state_posterior)
            pi_prior = torch.full_like(state_posterior, fill_value=1e-7)

        # Match the original scalar objective's precision floor and autograd
        # boundaries exactly, while preserving each frame as a separate item.
        posterior = state_posterior.detach().clone().requires_grad_(True)
        pi_prior = pi_prior.detach().clone().requires_grad_(True)
        pi_obs = pi_obs.clamp_min(self.config.min_precision)
        gamma_obs = torch.log(pi_obs).detach().requires_grad_(True)
        if not prediction.requires_grad:
            prediction = prediction.detach().requires_grad_(True)

        objective_inputs = []
        for index in range(batch_size):
            objective_inputs.append((
                state_prior[index], posterior[index], f_transition[index],
                pi_prior[index], actual[index], pi_obs[index],
                gamma_obs[index], prediction[index],
            ))
        return objective_inputs

    def _measure_objective_inputs(
        self, state_name: str, objective_inputs: tuple[torch.Tensor, ...],
        metric_pi_obs: torch.Tensor | None = None,
    ) -> tuple[StateMetrics, torch.Tensor]:
        (state_prior, posterior, f_transition, pi_prior, actual,
         pi_obs, gamma_obs, prediction) = objective_inputs
        F_value = self.agent.free_energy(
            state_prior, posterior, f_transition, pi_prior,
            actual, pi_obs, gamma_obs, prediction,
        )

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

        gradient_raw = (
            [float(value) for value in gradient_tensor.cpu().tolist()]
            if gradient_finite else []
        )
        gradient_sum = (
            finite_float(gradient_tensor.abs().sum().item())
            if gradient_finite else 0.0
        )
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
            eigenvalues = torch.linalg.eigvalsh(
                0.5 * (hessian + hessian.transpose(-1, -2))
            )
            precision = finite_float(eigenvalues.clamp_min(0).mean().item())
        except (RuntimeError, ValueError):
            precision = 0.0

        complexity = finite_float(
            (0.5 * pi_prior.detach() *
             (posterior.detach() + f_transition.detach()
              - state_prior.detach()).square()).sum().item()
        )
        accuracy_precision = pi_obs if metric_pi_obs is None else metric_pi_obs
        accuracy = finite_float(
            (-0.5 * accuracy_precision.detach() *
             (actual.detach() - prediction.detach()).square()).sum().item()
        )
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
            local_sensitivity=0.0,
            hessian_norm=hessian_norm,
            local_sensitivity_finite=False,
            hessian_finite=hessian_finite,
        )
        return metric, F_value

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

        gradient_raw = (
            [float(value) for value in gradient_tensor.cpu().tolist()]
            if gradient_finite else []
        )
        gradient_sum = (
            finite_float(gradient_tensor.abs().sum().item())
            if gradient_finite else 0.0
        )
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
            eigenvalues = torch.linalg.eigvalsh(
                0.5 * (hessian + hessian.transpose(-1, -2))
            )
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
            local_sensitivity=0.0,
            hessian_norm=hessian_norm,
            local_sensitivity_finite=False,
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

    def _allowed_states_for_frame(self, frame):
        """None means all current states; a set is a per-observation compatibility mask."""
        return self._observation_compatibility.get(id(frame))

    def _objective_inputs_from_prepared(self, prepared: dict) -> tuple[torch.Tensor, ...]:
        """Build the exact tensor arguments used by ``_state_objective``.

        Keeping this preparation outside ``vmap`` is deliberate: graph lookup,
        observation decoding, and per-frame belief contexts are Python/stateful
        operations. The resulting numerical free-energy calculation is pure and
        can safely be mapped over a batch.
        """
        posterior = prepared["state_posterior"].detach().clone().requires_grad_(True)
        pi_prior = prepared["pi_prior"].detach().clone().requires_grad_(True)
        prediction = prepared["prediction"]
        if not prediction.requires_grad:
            prediction = prediction.detach().requires_grad_(True)

        pi_obs = prepared["pi_obs"].clamp_min(self.config.min_precision)
        gamma_obs = torch.log(pi_obs).detach().requires_grad_(True)
        return (
            prepared["state_prior"],
            posterior,
            prepared["f_transition"],
            pi_prior,
            prepared["actual"],
            pi_obs,
            gamma_obs,
            prediction,
        )

    def _free_energy_batch(self, objective_inputs: list[tuple[torch.Tensor, ...]]):
        """Evaluate same-shaped per-frame objectives with ``torch.vmap``.

        A fallback keeps support for unusual custom functions whose outputs
        change shape from one frame to another. In the common fixed-shape case,
        vmap removes the Python loop around the per-frame free-energy algebra
        while retaining autograd paths into each model prediction.
        """
        if not objective_inputs:
            return []
        # For very small batches, Python dispatch costs less than setting up
        # the vmap transform; keep the scalar path in that case.
        if len(objective_inputs) < 4:
            return [self.agent.free_energy(*items) for items in objective_inputs]

        same_shapes = all(
            all(tuple(items[i].shape) == tuple(objective_inputs[0][i].shape)
                for i in range(len(objective_inputs[0])))
            for items in objective_inputs[1:]
        )
        if not same_shapes:
            return [self.agent.free_energy(*items) for items in objective_inputs]

        stacked = tuple(
            torch.stack([items[index] for items in objective_inputs], dim=0)
            for index in range(len(objective_inputs[0]))
        )
        # Agent_DAIH.free_energy is a pure tensor expression; mapping its
        # leading batch dimension is safe and preserves its exact formula.
        try:
            mapped = torch.vmap(self.agent.free_energy)(*stacked)
            return list(mapped.unbind(0))
        except (RuntimeError, TypeError, NotImplementedError):
            # Preserve support for applications that subclass/override the
            # free-energy function with operations lacking vmap rules.
            return [self.agent.free_energy(*items) for items in objective_inputs]

    def _optimize_world_generative_model(self, observations: dict):
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

        state_names = [
            name for name, node in self.agent.navigator.graph.items()
            if node.get("type") == "state"
        ]
        batches = observations if (isinstance(observations, list) and observations and isinstance(observations[0], tuple)) else [(observations, self._snapshot_beliefs())]

        # Preserve chronological belief inference, then batch compatible
        # model evaluations. Chunks limit activation memory; backward is
        # accumulated across chunks and the optimizer still steps exactly once.
        vectorized_state_count = 0
        vectorized_frame_count = 0
        total_loss_value = 0.0
        has_gradient_loss = False
        chunk_size = max(1, int(self.config.model_batch_size))

        for state_name in state_names:
            entries = []
            for frame_index, (frame, belief_context) in enumerate(batches):
                allowed_states = self._allowed_states_for_frame(frame)
                if allowed_states is None or state_name in allowed_states:
                    entries.append((frame_index, frame, belief_context))
            if not entries:
                continue

            for start in range(0, len(entries), chunk_size):
                chunk = entries[start:start + chunk_size]
                objective_inputs = None
                if len(chunk) >= 2 and self._supports_native_batch(state_name):
                    try:
                        objective_inputs = self._prepare_state_objectives_batch(
                            state_name, chunk
                        )
                        vectorized_state_count += 1
                        vectorized_frame_count += len(chunk)
                    except (RuntimeError, TypeError, ValueError, KeyError, IndexError,
                            AttributeError, NotImplementedError):
                        # A legacy local callable or heterogeneous input shape
                        # falls back to the scalar path for this chunk only.
                        objective_inputs = None

                if objective_inputs is None:
                    objective_inputs = []
                    for _, frame, belief_context in chunk:
                        with self._using_beliefs(belief_context):
                            prepared = self._prepared_state(state_name, frame)
                        objective_inputs.append(
                            self._objective_inputs_from_prepared(prepared)
                        )

                values = self._free_energy_batch(objective_inputs)
                finite_losses = [
                    value.reshape(()) / max(1, len(batches))
                    for value in values
                    if value.requires_grad and bool(torch.isfinite(value).detach().item())
                ]
                if finite_losses:
                    chunk_loss = torch.stack(finite_losses).sum()
                    total_loss_value += float(chunk_loss.detach().item())
                    if chunk_loss.requires_grad:
                        chunk_loss.backward()
                        has_gradient_loss = True

        if not has_gradient_loss:
            return {
                "updated": False,
                "loss": finite_float(total_loss_value) if total_loss_value else None,
                "parameters": 0,
                "vectorized_states": vectorized_state_count,
                "vectorized_frames": vectorized_frame_count,
            }

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
            "loss": finite_float(total_loss_value),
            "parameters": sum(parameter.numel() for parameter in parameters),
            "vectorized_states": vectorized_state_count,
            "vectorized_frames": vectorized_frame_count,
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
                    state_min=self.agent.min_state_value,
                    state_max=self.agent.max_state_value,
                    observation_bounds=self.agent.observation_bounds,
                    reward_min=self.agent.min_reward_value,
                    reward_max=self.agent.max_reward_value,
                )
                g_posterior = self.agent.execepted_free_energy(
                    prepared["pi_priors"],
                    prepared["co_variances"],
                    prepared["transition_cost"],
                )
                posterior_values.append(g_posterior)
                predicted = prepared["temporal_pred"]
                if isinstance(predicted, (tuple, list)) and predicted:
                    predicted = predicted[0]
                new_precision = inference.seconde_derivative_from_output(
                    predicted,
                    local_inputs[parent_state],
                    min_precision=self.agent.min_precision,
                    max_precision=self.agent.max_precision,
                    eps=self.agent.numerical_eps,
                )
                local_q[parent_state] = inference.update_distributions(
                    new_precision.detach(),
                    predicted.detach(),
                    local_q[parent_state],
                    min_state_value=self.agent.min_state_value,
                    max_state_value=self.agent.max_state_value,
                    min_precision=self.agent.min_precision,
                    max_precision=self.agent.max_precision,
                    eps=self.agent.numerical_eps,
                )
                # The next policy step conditions on the preceding imagined state.
                local_inputs[parent_state] = predicted
        if not posterior_values:
            return None
        return torch.stack([value.reshape(()) for value in posterior_values]).mean()

    def _policy_inputs_from_observations(self, observations: dict) -> dict:
        """Overlay observed factors on a copy of the current agent input map."""
        inputs = {
            key: (value.detach().clone() if torch.is_tensor(value) else deepcopy(value))
            for key, value in self.agent.inputs_map.items()
        }
        for group_name, group_values in observations.items():
            if isinstance(group_values, dict):
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
                continue

            # A vector observation is supplied directly under its group name.
            reference = inputs.get(group_name)
            if torch.is_tensor(reference):
                value = torch.as_tensor(
                    group_values, device=reference.device, dtype=reference.dtype
                )
                if value.numel() == reference.numel():
                    value = value.reshape_as(reference)
            else:
                value = torch.as_tensor(group_values, dtype=torch.float32)
            inputs[group_name] = value
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
                alignment_loss = (g_prior - g_posterior).square() * self._policy_precision(state_name)
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
        state_names = [
            name for name, node in self.agent.navigator.graph.items()
            if node.get("type") == "state"
        ]

        # As in optimization, all belief snapshots are already computed in
        # chronological order. Evaluation batches only the pure f/g forwards;
        # gradients and curvature diagnostics remain per observation.
        for state_name in state_names:
            entries = []
            for frame, belief_context in batches:
                allowed_states = self._allowed_states_for_frame(frame)
                if allowed_states is None or state_name in allowed_states:
                    entries.append((frame, belief_context))
            if not entries:
                continue

            chunk_size = max(1, int(self.config.model_batch_size))
            for start in range(0, len(entries), chunk_size):
                chunk = entries[start:start + chunk_size]
                objective_inputs = None
                if len(chunk) >= 2 and self._supports_native_batch(state_name):
                    try:
                        indexed_entries = [
                            (index, frame, context)
                            for index, (frame, context) in enumerate(chunk)
                        ]
                        objective_inputs = self._prepare_state_objectives_batch(
                            state_name, indexed_entries
                        )
                    except (RuntimeError, TypeError, ValueError, KeyError, IndexError,
                            AttributeError, NotImplementedError):
                        objective_inputs = None

                if objective_inputs is not None:
                    for items, (_, context) in zip(objective_inputs, chunk):
                        raw_pi_obs = model_components.get_actual_precision(
                            self.agent.navigator, context["pi_obs"],
                            context["Q_"], state_name,
                        )
                        metric, _ = self._measure_objective_inputs(
                            state_name, items, metric_pi_obs=raw_pi_obs
                        )
                        collected[state_name].append(metric)
                else:
                    for frame, belief_context in chunk:
                        with self._using_beliefs(belief_context):
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
    # Eight-phase adaptive decision system
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
            state="__world_generative_model__",
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
        max_phase = int(self.config.max_phase)
        # A resumed controller may have been saved at a later phase. Respect
        # the current CLI ceiling without resetting learned model parameters.
        self.global_progress.phase = min(self.global_progress.phase, max_phase if max_phase < 4 else 4)
        for item in self.progress.values():
            item.phase = min(item.phase, max_phase)
        if self.config.weights_only:
            return ({"world_generative_model": {
                "phase": self.global_progress.phase,
                "phase_name": self.PHASE_NAMES[min(self.global_progress.phase, 8)],
                "weights_only": True,
                "mutation": None,
            }}, proposed)

        # --------------------------------------------------------------
        # Shared WorldGenerativeModel: phases 1-3 are controlled globally from all F_i.
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
                decisions["world_generative_model"] = {
                    "phase_before": global_phase_before,
                    "phase": 3,
                    "phase_name": self.PHASE_NAMES[3],
                    "free_energy": aggregate.free_energy,
                    "shock": global_shock,
                    "transition": "backtrack_to_shared_world_generative_model",
                    "mutation": None,
                }
                return decisions, proposed

        if global_state.phase < 4:
            if global_state.pending_mutation:
                decisions["world_generative_model"] = {
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
                    if global_state.phase != 3 or (
                        self.config.state_constraints.get(name, {}).get("allow_mutations", True)
                        and self.config.state_constraints.get(name, {}).get("allow_neurons", True)
                    )
                ), next(iter(metrics)))
                global_operation = self._mutation_for(target, global_state.phase, True)
                if global_operation.get("action") != "none":
                    global_operation["scope"] = "world_generative_model"
                    proposed.append(global_operation)
                    global_state.pending_mutation = True
            elif global_transition:
                if global_state.phase < min(3, max_phase):
                    global_state.phase += 1
                    global_state.phase_iterations = 0
                    global_state.phase_entry_free_energy = aggregate.free_energy
                    global_state.phase_best_free_energy = aggregate.free_energy
                    global_state.history.clear()
                    global_state.history.append(aggregate)
                    if global_state.phase == 3:
                        target = next((
                            name for name in metrics
                            if (
                                self.config.state_constraints.get(name, {}).get("allow_mutations", True)
                                and self.config.state_constraints.get(name, {}).get("allow_neurons", True)
                            )
                        ), next(iter(metrics)))
                        global_operation = self._mutation_for(target, 3, False)
                        global_operation["scope"] = "world_generative_model"
                        proposed.append(global_operation)
                        global_state.pending_mutation = True
                elif global_state.phase == 3 and max_phase >= 4:
                    # Shared representation is sufficiently explored; local
                    # architecture phases are enabled only when requested.
                    global_state.phase = 4
                    global_state.phase_iterations = 0
                    global_state.phase_entry_free_energy = aggregate.free_energy
                    global_state.phase_best_free_energy = aggregate.free_energy
                    global_state.history.clear()
                    global_state.history.append(aggregate)
                    self._initialize_local_phases(metrics)
                # At the selected ceiling, remain in that phase. Optimization
                # and same-phase maintenance can continue, but later phases cannot start.

            decisions["world_generative_model"] = {
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
                "scope": "all_states_shared_world_generative_model",
            }
            if global_state.phase < 4:
                return decisions, proposed

        # --------------------------------------------------------------
        # Agent architecture: phases 4-8 are controlled independently per state.
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
                if progress.phase < min(8, max_phase):
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

            if (progress.phase == min(8, max_phase) and not shock_trigger and not progress.pending_mutation
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
        """Save the complete initial state once for this run."""
        if not self.config.checkpoint_dir:
            return None
        from .persistence import save_agent

        directory = Path(self.config.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.config.checkpoint_filename
        # Preserve the original starting point if a run is resumed in-place.
        if path.exists():
            return str(path)
        return str(save_agent(self.agent, path))

    def _checkpoint_after(self, F_after: float, *, force: bool = False):
        """Periodically save a complete resumable state, including trainer/Adam state."""
        if not self.config.checkpoint_dir:
            return None
        if not force and self.iteration % self.config.checkpoint_every != 0:
            return None
        from .persistence import save_agent

        directory = Path(self.config.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.config.final_checkpoint_filename
        return str(save_agent(self.agent, path))

    def _checkpoint_phase_end(self, phase: int, phase_name: str):
        """Persist a complete agent at each executed macro-phase boundary."""
        if not bool(getattr(self.config, "save_phase_checkpoints", True)):
            return None
        from .persistence import save_agent
        directory = Path(self.config.checkpoint_dir or "offline_artifacts")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.config.phase_checkpoint_prefix}_{int(phase)}_{phase_name}_iter_{self.iteration:08d}.pt"
        return str(save_agent(self.agent, path))

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
        if rebuild_callback is not None:
            raise ValueError(
                "Paradigm v8 applies structural operations through OfflineArchitecture; "
                "rebuild_callback is not used."
            )
        if self.agent.navigator is None:
            raise RuntimeError("Call Agent_DAIH.model() before training_offline().")
        training_frames = self._normalize_observations(observations)
        # When no explicit test set is supplied, reuse the already materialized
        # batch. This is essential for one-shot iterators/generators, which
        # cannot be consumed a second time.
        test_frames = (
            training_frames
            if test_observations is None
            else self._normalize_observations(test_observations)
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
            self._optimize_world_generative_model(training_batches)
            if self.config.train_world_generative_model
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
            "version": "7.2",
            "max_phase": int(self.config.max_phase),
            "weights_only": bool(self.config.weights_only),
            "train_world_generative_model": bool(self.config.train_world_generative_model),
            "train_policy": bool(self.config.train_policy),
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
                "Shared WorldGenerativeModel parameters are optimized against the sum of all state free energies. "
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
                if skipped_mutation.get("scope") == "world_generative_model":
                    self.global_progress.pending_mutation = False
                    if self.global_progress.phase < min(3, int(self.config.max_phase)):
                        self.global_progress.phase += 1
                        self.global_progress.phase_iterations = 0
                        self.global_progress.history.clear()
                    elif self.global_progress.phase == 3 and int(self.config.max_phase) >= 4:
                        self.global_progress.phase = 4
                        self.global_progress.phase_iterations = 0
                        self.global_progress.history.clear()
                        self._initialize_local_phases(metrics_after)
                elif target in self.progress:
                    local_progress = self.progress[target]
                    local_progress.pending_mutation = False
                    if local_progress.phase < min(8, int(self.config.max_phase)):
                        # A disabled or inapplicable dimension is skipped, but
                        # never advance beyond the phase ceiling selected by CLI.
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
                    if applied_mutation.get("scope") == "world_generative_model":
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


# =============================================================================
# Offline paradigm v8: hierarchical prediction / temporal prediction / BMR
# =============================================================================
class OfflineTrainer(_LegacyOfflineTrainer):
    """
    Three-macro-phase offline trainer.

    Macro-phases I and II alternate on every step until both objectives have
    converged or clearly diverged. Macro-phase III evaluates one structural
    deletion per step; accepted deletion restarts learning at macro-phase I.
    If all tested structures are retained, one capacity addition from a
    retained structural family is evaluated.
    """

    MACRO_NAMES = {
        1: "hierarchical_predictions",
        2: "temporal_predictions",
        3: "architectural_restructuring",
    }

    def __init__(self, agent, config: OfflineConfig | dict | None = None):
        super().__init__(agent, config)
        self.agent._offline_trainer = self
        # The online collector and offline reader must share one file and one cap.
        self.agent.inference_memory_path = self.config.inference_memory_path
        self.agent.transition_memory_path = self.config.transition_memory_path
        self.agent.inference_memory_capacity = int(self.config.inference_memory_capacity)
        self.agent.transition_memory_capacity = int(self.config.transition_memory_capacity)
        self.agent.inference_memory_old_fraction = float(self.config.inference_memory_old_fraction)
        self.macro_phase = 1
        self.top_k = max(1, int(self.config.initial_top_k))
        self._hierarchical_history = deque(maxlen=max(3, self.config.macro_patience))
        self._temporal_history = deque(maxlen=max(3, self.config.macro_patience))
        self._hierarchical_done = False
        self._temporal_done = False
        self._structural_candidates = []
        self._structural_cursor = 0
        self._retained_candidates = []
        self._structural_log = []
        self._creation_cursor = 0
        self._macro_cycles = 0
        self._last_temporal_loss = None
        self._last_replay_info = {"loaded": 0, "new": 0, "total": 0}
        self._last_observation_state_scores = {}
        self._temporal_surprise_trace = []
        self._observation_compatibility = {}

    def _validate_config(self):
        super()._validate_config()
        if self.config.global_precision <= 0:
            raise ValueError("global_precision must be strictly positive.")
        if self.config.macro_patience < 3:
            raise ValueError("macro_patience must be >= 3.")
        if self.config.macro_relative_tolerance < 0 or self.config.divergence_tolerance <= 0:
            raise ValueError("Macro-phase tolerances must be non-negative and divergence_tolerance positive.")
        if self.config.initial_top_k < 1 or self.config.top_k_step < 1:
            raise ValueError("initial_top_k and top_k_step must be >= 1.")
        if self.config.temporal_loss_weight < 0:
            raise ValueError("temporal_loss_weight must be non-negative.")
        if int(self.config.inference_memory_capacity) < 1:
            raise ValueError("inference_memory_capacity must be >= 1.")
        if int(self.config.transition_memory_capacity) < 1:
            raise ValueError("transition_memory_capacity must be >= 1.")
        if not 0.0 <= float(self.config.inference_memory_old_fraction) < 1.0:
            raise ValueError("inference_memory_old_fraction must be in [0, 1).")
        if self.config.temporal_precision is not None and float(self.config.temporal_precision) <= 0:
            raise ValueError("temporal_precision must be strictly positive when provided.")

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["paradigm"] = "daih_offline_v12"
        state["macro_controller"] = {
            "macro_phase": self.macro_phase,
            "top_k": self.top_k,
            "hierarchical_history": list(self._hierarchical_history),
            "temporal_history": list(self._temporal_history),
            "hierarchical_done": self._hierarchical_done,
            "temporal_done": self._temporal_done,
            "structural_candidates": deepcopy(self._structural_candidates),
            "structural_cursor": self._structural_cursor,
            "retained_candidates": deepcopy(self._retained_candidates),
            "structural_log": deepcopy(self._structural_log),
            "creation_cursor": self._creation_cursor,
            "macro_cycles": self._macro_cycles,
            "last_temporal_loss": self._last_temporal_loss,
            "last_replay_info": deepcopy(self._last_replay_info),
        }
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        macro = state.get("macro_controller", {})
        self.macro_phase = int(macro.get("macro_phase", 1))
        self.top_k = max(1, int(macro.get("top_k", self.config.initial_top_k)))
        self._hierarchical_history = deque(
            macro.get("hierarchical_history", []),
            maxlen=max(3, self.config.macro_patience),
        )
        self._temporal_history = deque(
            macro.get("temporal_history", []),
            maxlen=max(3, self.config.macro_patience),
        )
        self._hierarchical_done = bool(macro.get("hierarchical_done", False))
        self._temporal_done = bool(macro.get("temporal_done", False))
        self._structural_candidates = deepcopy(macro.get("structural_candidates", []))
        self._structural_cursor = int(macro.get("structural_cursor", 0))
        self._retained_candidates = deepcopy(macro.get("retained_candidates", []))
        self._structural_log = deepcopy(macro.get("structural_log", []))
        self._creation_cursor = int(macro.get("creation_cursor", 0))
        self._macro_cycles = int(macro.get("macro_cycles", 0))
        self._last_temporal_loss = macro.get("last_temporal_loss")
        self._last_replay_info = deepcopy(macro.get("last_replay_info", {"loaded": 0, "new": 0, "total": 0}))

    @staticmethod
    def _scalar(value, default=0.0) -> float:
        try:
            if torch.is_tensor(value):
                value = value.detach().float().mean().cpu().item()
            result = float(value)
            return result if math.isfinite(result) else float(default)
        except (TypeError, ValueError, RuntimeError):
            return float(default)

    def _global_precision(self) -> float:
        return max(float(self.config.global_precision), float(self.config.min_precision))

    def _state_precision(self, state_name: str) -> float:
        values = getattr(self.agent, "pi_states", {}) or {}
        return max(self._scalar(values.get(state_name), self._global_precision()),
                   float(self.config.min_precision))

    def _policy_precision(self, state_name: str) -> float:
        values = getattr(self.agent, "pi_p", {}) or {}
        return max(self._scalar(values.get(state_name), 1.0),
                   float(self.config.policy_precision_min))

    @staticmethod
    def _distribution_mean(distribution):
        if hasattr(distribution, "loc"):
            return distribution.loc
        if hasattr(distribution, "mean"):
            return distribution.mean
        return torch.as_tensor(distribution)

    def _observation_scores(self, contexts):
        scores = []
        self._last_observation_state_scores = {}
        for index, (frame, belief_context) in enumerate(contexts):
            with self._using_beliefs(belief_context):
                value, metrics = self._evaluate([(frame, belief_context)])
            scores.append((index, float(value)))
            self._last_observation_state_scores[index] = {
                name: self._scalar(metric.free_energy)
                for name, metric in metrics.items()
            }
        return scores

    @staticmethod
    def _causal_reference_name(reference):
        """Return a hashable state/node name from a graph parent reference.

        Navigator implementations may expose parents either as plain state
        names or as structured dictionaries (for example
        {"name": "state_a"} / {"state_name": "state_a"}).  Never use the
        dictionary itself as a graph key or set member.
        """
        if isinstance(reference, str):
            return reference
        if isinstance(reference, (int, float)):
            return str(reference)
        if isinstance(reference, dict):
            for key in ("state_name", "name", "node_name", "state", "node", "id", "key"):
                if key in reference:
                    value = OfflineTrainer._causal_reference_name(reference[key])
                    if value is not None:
                        return value
            return None
        if isinstance(reference, (tuple, list)):
            for item in reference:
                value = OfflineTrainer._causal_reference_name(item)
                if value is not None:
                    return value
            return None
        for attribute in ("state_name", "name", "node_name", "id"):
            value = getattr(reference, attribute, None)
            if value is not None and value is not reference:
                normalized = OfflineTrainer._causal_reference_name(value)
                if normalized is not None:
                    return normalized
        return None

    def _causal_ancestors(self, state_names):
        """Expand surprising states to graph ancestors, accepting structured parents."""
        navigator = getattr(self.agent, "navigator", None)
        graph = getattr(navigator, "graph", {}) or {}
        if not isinstance(graph, dict):
            return sorted({name for item in state_names
                           if (name := self._causal_reference_name(item)) is not None})

        visited = {
            name for item in state_names
            if (name := self._causal_reference_name(item)) is not None
        }
        queue = list(visited)
        while queue:
            name = queue.pop(0)
            node = graph.get(name, {})
            if not isinstance(node, dict):
                continue
            parents = node.get("parents", []) or []
            # A single parent can itself be a mapping, rather than a list.
            if isinstance(parents, dict):
                parents = [parents]
            elif isinstance(parents, (str, int, float)):
                parents = [parents]
            for parent_ref in parents:
                parent = self._causal_reference_name(parent_ref)
                if parent is not None and parent in graph and parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
        return sorted(visited)

    def _select_surprising_contexts(self, contexts):
        scores = self._observation_scores(contexts)
        ranked = sorted(scores, key=lambda item: item[1], reverse=True)
        count = min(max(1, int(self.top_k)), int(self.config.inference_memory_capacity), len(ranked))
        selected_ids = {index for index, _ in ranked[:count]}
        selected = [item for index, item in enumerate(contexts) if index in selected_ids]
        state_surprises = []
        for index in selected_ids:
            for state_name, state_f in self._last_observation_state_scores.get(index, {}).items():
                state_surprises.append((state_name, state_f, index))
        state_surprises.sort(key=lambda item: item[1], reverse=True)
        surprising_states = []
        for state_name, _, _ in state_surprises:
            if state_name not in surprising_states:
                surprising_states.append(state_name)
        causal_factors = self._causal_ancestors(surprising_states)
        return selected, {
            "top_k_requested": int(self.top_k),
            "top_k_effective": count,
            "top_k_capacity": int(self.config.inference_memory_capacity),
            "observations_total": len(contexts),
            "observations_selected": count,
            "ranked_indices": [index for index, _ in ranked],
            "surprising_states": surprising_states,
            "causal_explanatory_factors": causal_factors,
            "state_surprise_ranking": [
                {"state": name, "free_energy": value, "observation_index": index}
                for name, value, index in state_surprises
            ],
        }

    def _objective_status(self, history, precision=None):
        """Absolute convergence test for one minimized objective."""
        if len(history) < self.config.macro_patience:
            return "active"
        values = list(history)
        tolerance = float(precision if precision is not None else self._global_precision())
        variation = abs(values[-1] - values[0])
        return "converged" if variation <= tolerance else "active"

    def _joint_macro_status(self):
        """Stop phases I/II only on joint convergence or opposing progress.

        Both objectives are minimized. A clear disagreement in their trends
        (one improves while the other worsens beyond its precision) is treated
        as divergence of the coupled learning cycle.
        """
        required = int(self.config.macro_patience)
        if len(self._hierarchical_history) < required or len(self._temporal_history) < required:
            return "active", {"reason": "insufficient_history"}
        h = list(self._hierarchical_history)
        t = list(self._temporal_history)
        dh = h[-1] - h[0]
        dt = t[-1] - t[0]
        hp = self._global_precision()
        tp = float(self.config.temporal_precision or self._global_precision())
        h_stable = abs(dh) <= hp
        t_stable = abs(dt) <= tp
        if h_stable and t_stable:
            return "converged", {"delta_hierarchical": dh, "delta_temporal": dt,
                                 "precision_hierarchical": hp, "precision_temporal": tp}
        opposing = (dh < -hp and dt > tp) or (dh > hp and dt < -tp)
        if opposing:
            return "diverged", {"delta_hierarchical": dh, "delta_temporal": dt,
                                "precision_hierarchical": hp, "precision_temporal": tp}
        return "active", {"delta_hierarchical": dh, "delta_temporal": dt,
                           "precision_hierarchical": hp, "precision_temporal": tp}

    def _load_inference_memory(self):
        """Return observation records with their recording-time architecture metadata."""
        if not self.config.include_inference_memory:
            return []
        path = Path(self.config.inference_memory_path)
        if not path.exists():
            return []
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
        except Exception:
            return []
        if isinstance(payload, dict):
            items = payload.get("observations", payload.get("frames", []))
        elif isinstance(payload, list):
            items = payload
        else:
            items = []
        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("observation"), dict):
                records.append({
                    "observation": deepcopy(item["observation"]),
                    "architecture": deepcopy(item.get("architecture", {})),
                    "surprise": item.get("surprise"),
                })
            else:
                # Legacy raw-observation records have no architecture metadata.
                records.append({"observation": deepcopy(item), "architecture": None,
                                "surprise": None})
        return records

    def _compatible_observation_states(self, stored_architecture):
        """Keep only states whose recorded dimensions still match the live model."""
        if not isinstance(stored_architecture, dict) or not stored_architecture:
            # Legacy memories lack enough metadata to prove a mismatch.
            return None
        current = self.agent._offline_memory_architecture()
        compatible = set()
        for state_name, old in stored_architecture.items():
            now = current.get(state_name)
            if not isinstance(old, dict) or not isinstance(now, dict):
                continue
            try:
                old_count = int(old.get("under_states", -1))
                current_count = int(now.get("under_states", -2))
            except (TypeError, ValueError):
                continue
            if old_count != current_count:
                continue
            old_shapes = old.get("distribution_shapes")
            new_shapes = now.get("distribution_shapes")
            if old_shapes == new_shapes:
                compatible.add(state_name)
        return compatible

    def _load_transition_memory(self):
        if not self.config.include_transition_memory:
            return []
        path = Path(self.config.transition_memory_path)
        if not path.exists():
            return []
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
        except Exception:
            return []
        if isinstance(payload, dict):
            items = payload.get("transitions", [])
        elif isinstance(payload, list):
            items = payload
        else:
            items = []
        return [deepcopy(item) for item in items if isinstance(item, dict)
                and isinstance(item.get("initial_state"), dict)
                and isinstance(item.get("next_state"), dict)]

    def _merge_inference_memory(self, new_frames):
        """Phase I reader: observation memory only, with component-wise compatibility masks."""
        records = self._load_inference_memory()
        old_frames = []
        self._observation_compatibility = {}
        for record in records:
            frame = deepcopy(record.get("observation", {}))
            if not isinstance(frame, dict):
                continue
            allowed = self._compatible_observation_states(record.get("architecture"))
            if allowed is not None:
                if not allowed:
                    continue
                self._observation_compatibility[id(frame)] = allowed
            old_frames.append(frame)
        merged = old_frames + new_frames
        self._last_replay_info = {
            "observation_memory_loaded": len(old_frames),
            "new_observations": len(new_frames),
            "observation_replay_total": len(merged),
            "top_k_capacity": int(self.config.inference_memory_capacity),
            "compatibility_masked_records": sum(
                1 for frame in old_frames if self._observation_compatibility.get(id(frame)) is not None
            ),
        }
        return merged

    def _read_online_free_energy_reference(self):
        """Read persistent sum/count; this statistic is not bounded by sample retention."""
        path = Path(self.config.inference_memory_path)
        if not path.exists():
            return {"sum": 0.0, "count": 0, "mean": None}
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            stats = payload.get("online_free_energy_stats", {}) if isinstance(payload, dict) else {}
            total = float(stats.get("sum", 0.0))
            count = int(stats.get("count", 0))
            return {"sum": total, "count": count, "mean": total / count if count > 0 else None}
        except Exception:
            return {"sum": 0.0, "count": 0, "mean": None}

    def _reset_online_free_energy_reference(self):
        """Consume the online interval only after a successful offline step."""
        path = Path(self.config.inference_memory_path)
        if not path.exists():
            return
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict):
                payload = {"observations": payload if isinstance(payload, list) else []}
            payload["online_free_energy_stats"] = {"sum": 0.0, "count": 0, "mean": None}
            fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
            os.close(fd)
            try:
                torch.save(payload, temp_name)
                os.replace(temp_name, path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
        except Exception as exc:
            raise RuntimeError(f"Unable to reset online free-energy reference: {exc}") from exc

    @staticmethod
    def _distribution_shape(distribution):
        mean = getattr(distribution, "mean", None)
        if mean is None:
            mean = getattr(distribution, "loc", None)
        if mean is None:
            return None
        try:
            return tuple(mean.shape)
        except Exception:
            return None

    def _state_snapshot_compatible(self, stored_snapshot, state_name):
        """Check one state's posterior structure without rejecting its siblings."""
        stored_q = stored_snapshot.get("Q_", {}) if isinstance(stored_snapshot, dict) else {}
        current_q = getattr(self.agent, "Q_", {}) or {}
        if state_name not in stored_q or state_name not in current_q:
            return False
        old_distributions = stored_q.get(state_name) or []
        new_distributions = current_q.get(state_name) or []
        if len(old_distributions) != len(new_distributions):
            return False
        old_shapes = [self._distribution_shape(value) for value in old_distributions]
        new_shapes = [self._distribution_shape(value) for value in new_distributions]
        return old_shapes == new_shapes

    def _partial_transition_beliefs(self, stored_snapshot):
        """Overlay only compatible stored states onto current beliefs.

        The current architecture remains the default. Deleted states and states
        whose under-state count or posterior tensor shapes changed are omitted.
        """
        current = self._snapshot_beliefs()
        compatible = []
        stored_q = stored_snapshot.get("Q_", {}) if isinstance(stored_snapshot, dict) else {}
        current_q = current.get("Q_", {}) or {}
        for state_name in list(current_q):
            if not self._state_snapshot_compatible(stored_snapshot, state_name):
                continue
            compatible.append(state_name)
            for field in ("Q_", "inputs_map", "pi_states", "pi_priors",
                          "pi_threshold"):
                old_map = stored_snapshot.get(field)
                new_map = current.get(field)
                if isinstance(old_map, dict) and isinstance(new_map, dict):
                    if state_name in old_map and state_name in new_map:
                        new_map[state_name] = deepcopy(old_map[state_name])
        # Keep only the states that can be replayed; callers iterate this list.
        return current, compatible

    def _snapshots_are_consecutive(self, previous_next, following_initial, atol=1e-6, rtol=1e-5):
        """Return true when two stored snapshots describe the same awake instant."""
        left = previous_next.get("Q_", {}) if isinstance(previous_next, dict) else {}
        right = following_initial.get("Q_", {}) if isinstance(following_initial, dict) else {}
        common = [name for name in left if name in right]
        if not common:
            return False
        compared = 0
        for name in common:
            if len(left[name] or []) != len(right[name] or []):
                return False
            for a, b in zip(left[name] or [], right[name] or []):
                av = self._distribution_mean(a).detach()
                bv = self._distribution_mean(b).detach()
                # A changed/deleted component does not invalidate compatible
                # sibling states in the same historical snapshot.
                if av.shape != bv.shape or not torch.allclose(av, bv, atol=atol, rtol=rtol):
                    continue
                compared += 1
        return compared > 0

    def _transition_chains(self, transitions, horizon):
        """Return the longest contiguous prefix available, capped at policy horizon H.

        H is a maximum rollout length, not a minimum required history length.
        A shorter non-empty chain is valid when memory contains fewer transitions.
        """
        horizon = max(1, int(horizon))
        if not transitions:
            return []
        candidate = [transitions[0]]
        limit = min(len(transitions), horizon)
        for offset in range(1, limit):
            if not self._snapshots_are_consecutive(
                candidate[-1]["next_state"], transitions[offset]["initial_state"]
            ):
                break
            candidate.append(transitions[offset])
        return [(0, candidate)]

    def _awake_belief_for_transition(self, transition):
        """s_t^y: actual posterior belief captured after the awake inference at t."""
        belief = transition.get("awake_belief")
        if isinstance(belief, dict) and belief:
            return belief
        return transition.get("next_state", {}).get("Q_", {})

    def _dream_trajectory_loss(self, transition, chain, policy_output, policy_state):
        """Roll out the available contiguous steps, up to policy horizon H.

        The recorded transition action may be None: it is not used to discard or
        truncate the awake target. Dream actions are generated by the current policy.
        """
        policy_tensor = policy_output.policy
        sequence = policy_tensor[0] if policy_tensor.ndim >= 3 else policy_tensor
        policy_horizon = int(sequence.shape[0])
        rollout_horizon = len(chain)
        if policy_horizon < 1 or rollout_horizon < 1 or rollout_horizon > policy_horizon:
            return [], 0
        # Use every available contiguous transition up to the policy's maximum horizon.
        sequence = sequence[:rollout_horizon]
        initial = transition["initial_state"]
        initial_context, compatible_initial = self._partial_transition_beliefs(initial)
        if not compatible_initial:
            return [], 0
        losses = []
        completed_steps = 0
        with self._using_beliefs(initial_context):
            local_q = deepcopy(self.agent.Q_)
            local_inputs = {
                key: (value.detach().clone() if torch.is_tensor(value) else deepcopy(value))
                for key, value in self.agent.inputs_map.items()
            }
            for t, action_t in enumerate(sequence):
                if not torch.isfinite(action_t).all():
                    return [], 0
                if abs(float(action_t.detach().sum().item()) - 1.0) > 1e-4:
                    return [], 0
                parent_states, action_name = preprocessing.get_one_hot_action_parents(
                    action_t, self.agent.actions_id, self.agent.navigator
                )
                awake_q = self._awake_belief_for_transition(chain[t])
                step_losses = []
                for parent_state in parent_states:
                    if parent_state not in local_inputs or parent_state not in local_q:
                        continue
                    actions_for_state = self.agent.navigator.child_nodes(
                        parent_state, node_type="action"
                    )
                    if action_name not in actions_for_state:
                        continue
                    source = local_inputs[parent_state]
                    if torch.is_tensor(source) and not source.requires_grad:
                        source = source.detach().clone().requires_grad_(True)
                        local_inputs[parent_state] = source
                    prepared = preprocessing.preprocessing_for_excepted_free_energy(
                        actions_for_state, action_name, parent_state, local_inputs,
                        self.agent.pi_states, self.agent.navigator,
                        self.agent.functions_map, local_q, self.agent.states_list,
                        state_min=self.agent.min_state_value,
                        state_max=self.agent.max_state_value,
                        observation_bounds=self.agent.observation_bounds,
                        reward_min=self.agent.min_reward_value,
                        reward_max=self.agent.max_reward_value,
                    )
                    predicted = prepared.get("temporal_pred")
                    # Generative APIs may return (prediction, mask); the mask
                    # is metadata and is not part of the predicted state.
                    if isinstance(predicted, (tuple, list)) and predicted:
                        predicted = predicted[0]
                    target_distributions = awake_q.get(parent_state)
                    if predicted is None or not target_distributions:
                        continue
                    target = torch.stack([
                        self._distribution_mean(dist).detach()
                        for dist in target_distributions
                    ])
                    predicted = predicted if torch.is_tensor(predicted) else torch.as_tensor(predicted)
                    target = target.to(device=predicted.device, dtype=predicted.dtype)
                    if predicted.shape != target.shape:
                        if predicted.numel() != target.numel():
                            continue
                        predicted = predicted.reshape_as(target)
                    # Exact specified WorldGenerativeModel objective: sum_t ||s_x(t)-s_y(t)||^2.
                    step_loss = (predicted - target).square().sum()
                    if torch.isfinite(step_loss):
                        losses.append(step_loss)
                        step_losses.append(self._scalar(step_loss.detach()))
                        self._temporal_surprise_trace.append({
                            "memory": "awake_sleep_trajectory",
                            "policy_state": str(policy_state),
                            "transition_state": str(parent_state),
                            "horizon_step": int(t + 1),
                            "recorded_action": chain[t].get("action"),
                            "recorded_action_missing": chain[t].get("action") is None,
                            "awake_belief_target": "next_state.Q_",
                            "surprise": self._scalar(step_loss.detach()),
                        })
                    # The imagined belief is propagated, never replaced by the awake target.
                    new_precision = inference.seconde_derivative_from_output(
                        predicted,
                        source,
                        min_precision=self.agent.min_precision,
                        max_precision=self.agent.max_precision,
                        eps=self.agent.numerical_eps,
                    )
                    local_q[parent_state] = inference.update_distributions(
                        new_precision.detach(),
                        predicted.detach(),
                        local_q[parent_state],
                        min_state_value=self.agent.min_state_value,
                        max_state_value=self.agent.max_state_value,
                        min_precision=self.agent.min_precision,
                        max_precision=self.agent.max_precision,
                        eps=self.agent.numerical_eps,
                    )
                    local_inputs[parent_state] = predicted
                if not step_losses:
                    # A missing compatible state at a time step invalidates this rollout;
                    # do not silently shorten H or change the stated objective.
                    return [], 0
                completed_steps += 1
        return losses, completed_steps

    def _temporal_loss(self):
        """Phase II trains on the longest available contiguous awake/dream chain,
        up to the maximum horizon defined by the current policy sequence.
        """
        transitions = self._load_transition_memory()
        if not transitions:
            return None, 0, {"transitions_loaded": 0, "trajectories_selected": 0}
        # Policy horizon is model-derived. Generate candidates from each compatible
        # initial snapshot; the policy sequence itself defines H.
        candidates = []
        score_by_index = {}
        for index, transition in enumerate(transitions):
            score = transition.get("surprise")
            if score is None or not math.isfinite(self._scalar(score, float("nan"))):
                score = 0.0
            score_by_index[index] = float(score)
            initial = transition.get("initial_state", {})
            try:
                context, compatible = self._partial_transition_beliefs(initial)
                if not compatible:
                    continue
                with self._using_beliefs(context):
                    policy_inputs = {
                        key: (value.detach().clone() if torch.is_tensor(value) else deepcopy(value))
                        for key, value in self.agent.inputs_map.items()
                    }
                    policy_candidates = model_components.get_policy_candidates(
                        self.agent.navigator, self.agent.functions_map, policy_inputs,
                        self.agent.states_list, self.agent.Q_,
                    )
                has_usable_chain = False
                for policy_candidate in policy_candidates.values():
                    policy_output = getattr(policy_candidate, "policy", None)
                    if policy_output is None or getattr(policy_output, "policy", None) is None:
                        continue
                    policy_tensor = policy_output.policy
                    sequence = policy_tensor[0] if policy_tensor.ndim >= 3 else policy_tensor
                    horizon = int(sequence.shape[0])
                    chains = self._transition_chains(transitions[index:], horizon)
                    if chains and chains[0][0] == 0 and chains[0][1]:
                        has_usable_chain = True
                        break
                if has_usable_chain:
                    candidates.append((index, transition, policy_candidates))
            except Exception:
                continue
        capacity = int(self.config.transition_memory_capacity)
        effective_k = min(max(1, int(self.top_k)), capacity, len(candidates))
        candidates.sort(key=lambda item: score_by_index.get(item[0], 0.0), reverse=True)
        selected = candidates[:effective_k]
        losses = []
        trajectory_count = 0
        used_starts = []
        used_horizons = []
        for index, transition, policy_candidates in selected:
            for policy_state, candidate in policy_candidates.items():
                policy_output = getattr(candidate, "policy", None)
                if policy_output is None or getattr(policy_output, "policy", None) is None:
                    continue
                policy_tensor = policy_output.policy
                sequence = policy_tensor[0] if policy_tensor.ndim >= 3 else policy_tensor
                horizon = int(sequence.shape[0])
                chains = self._transition_chains(transitions[index:], horizon)
                if not chains or chains[0][0] != 0 or not chains[0][1]:
                    continue
                chain = chains[0][1]
                rollout_losses, steps = self._dream_trajectory_loss(
                    transition, chain, policy_output, policy_state
                )
                if rollout_losses and steps == len(chain):
                    losses.extend(rollout_losses)
                    trajectory_count += 1
                    used_starts.append(index)
                    used_horizons.append({"requested": horizon, "used": steps})
        if not losses:
            return None, 0, {
                "transitions_loaded": len(transitions), "trajectories_selected": 0,
                "top_k_effective": effective_k,
                "reason": "no_usable_compatible_awake_sleep_chain",
            }
        # Sum over time, then average only over selected trajectories to keep scale
        # independent of the number of Top-K samples while preserving sum_t within each.
        temporal = torch.stack(losses).sum() / max(1, trajectory_count)
        return temporal, trajectory_count, {
            "transitions_loaded": len(transitions), "trajectories_selected": trajectory_count,
            "top_k_effective": effective_k, "trajectory_start_indices": used_starts,
            "horizon_source": "policy_sequence_length_maximum",
            "horizons_used": used_horizons,
        }

    def _optimize_temporal_world_generative_model(self):
        groups = self._parameter_groups()
        if not groups:
            return {"updated": False, "loss": None, "pairs": 0, "parameters": 0}
        signature = tuple(tuple(id(p) for p in g["params"]) for g in groups)
        if self._optimizer is None or signature != self._optimizer_signature:
            self._optimizer = torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)
            self._optimizer_signature = signature
            if self._pending_optimizer_state is not None:
                self._optimizer.load_state_dict(self._pending_optimizer_state)
                self._pending_optimizer_state = None
        self._optimizer.zero_grad(set_to_none=True)
        temporal, trajectories, info = self._temporal_loss()
        if temporal is None or not temporal.requires_grad or not torch.isfinite(temporal):
            return {"updated": False, "loss": None, "pairs": 0,
                    "trajectories": trajectories, "parameters": 0, **info}
        loss = float(self.config.temporal_loss_weight) * temporal
        loss.backward()
        parameters = [p for g in groups for p in g["params"] if p.grad is not None]
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
            self._optimizer.step()
        return {
            "updated": bool(parameters), "loss": self._scalar(temporal),
            "weighted_loss": self._scalar(loss), "pairs": trajectories,
            "trajectories": trajectories,
            "parameters": sum(p.numel() for p in parameters), **info,
        }

    def _candidate_structures(self):
        from .offline_architecture import OfflineArchitecture
        architecture = OfflineArchitecture(self.agent, self.config)
        candidates = []
        states = [
            name for name, node in self.agent.navigator.graph.items()
            if node.get("type") == "state"
        ]
        for state in states:
            constraint = architecture._constraint(state)
            if constraint["allow_coordinates"]:
                count = architecture._coordinate_count(self.agent.D[state]["mu"])
                if count > int(constraint["min_coordinates"]):
                    candidates.append({"action": "remove_low_utility_coordinate", "target": state,
                                       "kind": "coordinate"})
            if constraint["allow_under_state"]:
                count = architecture._under_state_count(self.agent.D[state]["mu"])
                if count > int(constraint["min_under_state"]):
                    candidates.append({"action": "remove_under_state", "target": state,
                                       "kind": "under_state"})
            if constraint["allow_removal"] and len(states) > 1:
                candidates.append({"action": "remove_redundant_state", "target": state,
                                   "kind": "state"})
            if constraint["allow_hierarchy_removal"] and len(states) > 1:
                candidates.append({"action": "remove_redundant_hierarchy_layer", "target": state,
                                   "kind": "hierarchy"})
            if constraint["allow_intermediate_layers"]:
                state_id = self.agent.states_list.index(state) + 1
                layer_id = int(self.agent.generative_functions[state].get("layer", 0))
                modules = {id(v): v for v in (self.agent.functions_map or {}).values()
                           if isinstance(v, nn.Module) and hasattr(v, "local_context_layer_count")}
                layer_count = max(
                    (m.local_context_layer_count(state_id, layer_id) for m in modules.values()),
                    default=0,
                )
                if layer_count > int(constraint["min_intermediate_layers"]):
                    candidates.append({"action": "prune_intermediate_layer", "target": state,
                                       "kind": "layer"})
        if states:
            candidates.insert(0, {"action": "prune_neurons", "target": states[0], "kind": "neuron"})
        return candidates

    def _snapshot_agent_for_trial(self):
        # Keep the trainer identity out of the deepcopy to avoid copying the
        # optimizer/controller cycle; deepcopy preserves shared module aliases.
        state = {k: v for k, v in self.agent.__dict__.items() if k != "_offline_trainer"}
        return deepcopy(state)

    def _restore_agent_trial(self, snapshot):
        trainer = self
        self.agent.__dict__.clear()
        self.agent.__dict__.update(snapshot)
        self.agent._offline_trainer = trainer
        self.invalidate_optimizer()

    def _evaluate_current_agent(self, frames):
        initial = self._snapshot_beliefs()
        contexts = self._build_belief_contexts(frames, initial)
        value, metrics = self._evaluate(contexts)
        return value, metrics

    def _apply_one_structure(self, candidate):
        from .offline_architecture import OfflineArchitecture
        return OfflineArchitecture(self.agent, self.config).apply([candidate])

    def _precision_for_candidate(self, candidate):
        target = candidate.get("target")
        if candidate.get("kind") in {"coordinate", "under_state", "state", "hierarchy", "layer"}:
            return self._state_precision(target)
        return self._global_precision()

    @staticmethod
    def _candidate_free_energy(candidate, metrics, global_value):
        kind = candidate.get("kind")
        target = candidate.get("target")
        if kind in {"coordinate", "under_state", "layer"} and target in metrics:
            return float(metrics[target].free_energy)
        return float(global_value)

    def _structure_signature(self):
        """Return a compact, comparable description of the current architecture."""
        graph = getattr(getattr(self.agent, "navigator", None), "graph", {}) or {}
        state_names = tuple(sorted(
            str(name) for name, node in graph.items()
            if isinstance(node, dict) and node.get("type") == "state"
        ))
        graph_nodes = tuple(sorted(str(name) for name in graph.keys()))
        distributions = getattr(self.agent, "D", {}) or {}
        distribution_shapes = []
        for name, value in distributions.items():
            mu = value.get("mu") if isinstance(value, dict) else getattr(value, "mu", None)
            distribution_shapes.append((str(name), tuple(getattr(mu, "shape", ()))))
        distribution_shapes = tuple(sorted(distribution_shapes))
        module_shapes = []
        for name, module in (getattr(self.agent, "functions_map", {}) or {}).items():
            if isinstance(module, nn.Module):
                module_shapes.append((str(name), tuple(
                    (str(parameter_name), tuple(parameter.shape))
                    for parameter_name, parameter in module.named_parameters()
                )))
        return {
            "states": state_names,
            "graph_nodes": graph_nodes,
            "distribution_shapes": distribution_shapes,
            "module_shapes": tuple(sorted(module_shapes)),
        }

    @staticmethod
    def _structure_diff(before, after):
        before_states = set(before.get("states", ()))
        after_states = set(after.get("states", ()))
        before_nodes = set(before.get("graph_nodes", ()))
        after_nodes = set(after.get("graph_nodes", ()))
        return {
            "states_added": sorted(after_states - before_states),
            "states_removed": sorted(before_states - after_states),
            "graph_nodes_added": sorted(after_nodes - before_nodes),
            "graph_nodes_removed": sorted(before_nodes - after_nodes),
            "before": before,
            "after": after,
        }

    def _restart_learning_cycle(self):
        self.macro_phase = 1
        self._hierarchical_done = False
        self._temporal_done = False
        self._hierarchical_history.clear()
        self._temporal_history.clear()
        self._structural_candidates = []
        self._structural_cursor = 0
        self._retained_candidates = []
        self._creation_cursor = 0
        self._macro_cycles += 1
        self.invalidate_optimizer()

    def _structural_step(self, test_frames):
        if not self._structural_candidates:
            self._structural_candidates = self._candidate_structures()
            self._structural_cursor = 0
            self._retained_candidates = []
            self._creation_cursor = 0

        if self._structural_cursor < len(self._structural_candidates):
            candidate = deepcopy(self._structural_candidates[self._structural_cursor])
            self._structural_cursor += 1
            before_signature = self._structure_signature()
            snapshot = self._snapshot_agent_for_trial()
            before_global, before_metrics = self._evaluate_current_agent(test_frames)
            before = self._candidate_free_energy(candidate, before_metrics, before_global)
            result = self._apply_one_structure(candidate)
            if not result.get("applied"):
                after_signature = self._structure_signature()
                diff = self._structure_diff(before_signature, after_signature)
                record = {
                    "candidate": candidate, "action": candidate.get("action"),
                    "target": candidate.get("target"), "accepted": False,
                    "element_added": None, "element_removed": None,
                    "structure_changed": before_signature != after_signature,
                    "structure_diff": diff, "reason": result.get("skipped", []),
                    "F_before": before, "F_after": before,
                    "architecture_result": result,
                    "outcome": "not_applicable",
                }
                self._structural_log.append(record)
                return record

            after_global, after_metrics = self._evaluate_current_agent(test_frames)
            after = self._candidate_free_energy(candidate, after_metrics, after_global)
            delta = after - before
            threshold = 1.0 / self._precision_for_candidate(candidate)
            useful = delta > threshold
            if useful:
                self._restore_agent_trial(snapshot)
                self._retained_candidates.append(candidate)
                outcome = "retained_component"
                accepted = False
            else:
                self.invalidate_optimizer()
                outcome = "component_removed_restart_learning"
                accepted = True
                self._restart_learning_cycle()

            after_signature = self._structure_signature()
            diff = self._structure_diff(before_signature, after_signature)
            changed = before_signature != after_signature
            removed = ({"kind": candidate.get("kind"), "target": candidate.get("target")}
                       if accepted and changed else None)
            record = {
                "candidate": candidate, "action": candidate.get("action"),
                "target": candidate.get("target"), "accepted": accepted,
                "element_added": None, "element_removed": removed,
                "structure_changed": changed, "structure_diff": diff,
                "outcome": outcome, "F_before": before,
                "F_after": after if not useful else before,
                "delta_F": delta, "precision": self._precision_for_candidate(candidate),
                "threshold": threshold, "architecture_result": result,
            }
            self._structural_log.append(record)
            return record

        # Once all deletion candidates have been evaluated, try one addition.
        if self._retained_candidates and self._creation_cursor < len(self._retained_candidates):
            source = self._retained_candidates[self._creation_cursor]
            self._creation_cursor += 1
            add_actions = {
                "neuron": "add_neurons", "coordinate": "add_generalized_coordinate",
                "under_state": "add_under_state", "state": "add_state",
                "hierarchy": "add_hierarchy_layer", "layer": "add_intermediate_layer",
            }
            creation = {"action": add_actions[source["kind"]],
                        "target": source["target"], "kind": source["kind"]}
            before_signature = self._structure_signature()
            snapshot = self._snapshot_agent_for_trial()
            before, _ = self._evaluate_current_agent(test_frames)
            result = self._apply_one_structure(creation)
            if result.get("applied"):
                after, _ = self._evaluate_current_agent(test_frames)
                gain = before - after
                success = gain > self._global_precision()
            else:
                after, gain, success = before, 0.0, False
            if success:
                self.invalidate_optimizer()
                outcome = "capacity_added_restart_learning"
                self._restart_learning_cycle()
            else:
                self._restore_agent_trial(snapshot)
                outcome = "capacity_addition_rejected"
            after_signature = self._structure_signature()
            diff = self._structure_diff(before_signature, after_signature)
            changed = before_signature != after_signature
            added = ({"kind": creation.get("kind"), "target": creation.get("target")}
                     if success and changed else None)
            record = {
                "candidate": creation, "action": creation.get("action"),
                "target": creation.get("target"), "accepted": success,
                "element_added": added, "element_removed": None,
                "structure_changed": changed, "structure_diff": diff,
                "outcome": outcome, "F_before": before,
                "F_after": after if success else before,
                "delta_F": before - after, "global_precision": self._global_precision(),
                "architecture_result": result,
            }
            self._structural_log.append(record)
            return record

        record = {
            "candidate": None, "action": None, "target": None,
            "accepted": False, "element_added": None, "element_removed": None,
            "structure_changed": False,
            "outcome": ("all_capacity_additions_rejected" if self._retained_candidates
                        else "no_structural_candidate_available"),
        }
        self._structural_log.append(record)
        return record

    def step(
        self,
        observations: dict | list[dict] | tuple[dict, ...],
        *,
        test_observations: dict | list[dict] | tuple[dict, ...] | None = None,
        rebuild_callback: Callable | None = None,
        motor_fn: Callable | None = None,
        motor_train_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Run one offline cycle: phase I and II use their own persistent memories.

        Observation frames are consumed only by the hierarchical objective.
        Temporal Dreamer loss and policy optimization read transition memory only.
        """
        self._validate_config()
        self.agent.inference_memory_path = self.config.inference_memory_path
        self.agent.transition_memory_path = self.config.transition_memory_path
        self.agent.inference_memory_capacity = int(self.config.inference_memory_capacity)
        self.agent.transition_memory_capacity = int(self.config.transition_memory_capacity)
        self.agent.inference_memory_old_fraction = float(self.config.inference_memory_old_fraction)
        if self.agent.navigator is None:
            raise RuntimeError("Call Agent_DAIH.model() before training_offline().")
        if self.config.train_motor and motor_fn is None:
            raise ValueError("train_motor=True requires a user-supplied motor_fn.")
        if self.config.train_motor and not callable(motor_train_fn):
            raise ValueError("train_motor=True requires motor_train_fn(records, observations).")
        if motor_fn is not None and self.config.observation_mode != "temporal":
            raise ValueError("A motor_fn requires observation_mode='temporal'.")

        online_reference = self._read_online_free_energy_reference()
        new_frames = self._normalize_observations(observations)
        frames = self._merge_inference_memory(new_frames)
        # Evaluation defaults to the caller's current batch, not the bounded
        # observation replay memory. This keeps phase-II training isolated.
        test_frames = new_frames if test_observations is None else self._normalize_observations(test_observations)
        self.iteration += 1
        initial_beliefs = self._snapshot_beliefs()
        motor_trace = []
        contexts = self._build_belief_contexts(
            frames, initial_beliefs, motor_fn=motor_fn, motor_trace=motor_trace
        )
        test_contexts = self._build_belief_contexts(
            test_frames, initial_beliefs, motor_fn=motor_fn
        )
        F_before, _ = self._evaluate(test_contexts)
        checkpoint_path = self._checkpoint_before(F_before)
        best_updated = self._consider_best_agent(F_before)

        structural = None
        hierarchical = {"updated": False, "loss": None}
        temporal = {"updated": False, "loss": None}
        policy = {"updated": False, "loss": None}
        top_k_info = {"top_k_requested": int(self.top_k), "top_k_effective": 0}
        temporal_surprises = []
        phase_checkpoints = {}

        if self.macro_phase in (1, 2):
            # PHASE I — observation memory only.
            selected, top_k_info = self._select_surprising_contexts(contexts)
            hierarchical = self._optimize_world_generative_model(selected)
            F_hierarchical, _ = self._evaluate(selected)
            self._hierarchical_history.append(float(F_hierarchical))
            status_h = self._objective_status(
                self._hierarchical_history, self._global_precision()
            )
            phase_checkpoints["phase_1"] = self._checkpoint_phase_end(
                1, self.MACRO_NAMES[1]
            )

            # PHASE II — transition memory only. No observation contexts or raw
            # observation frames are passed into either temporal or policy loss.
            self._temporal_surprise_trace = []
            temporal = self._optimize_temporal_world_generative_model()
            temporal_surprises = sorted(
                self._temporal_surprise_trace,
                key=lambda item: item["surprise"], reverse=True,
            )
            start_indices = temporal.get("trajectory_start_indices", [])
            stored_transitions = self._load_transition_memory()
            policy_results = []
            for transition_index in sorted(set(start_indices)):
                if transition_index < 0 or transition_index >= len(stored_transitions):
                    continue
                initial_snapshot = stored_transitions[transition_index].get("initial_state", {})
                context, compatible = self._partial_transition_beliefs(initial_snapshot)
                if not compatible:
                    continue
                with self._using_beliefs(context):
                    # Empty observation overlay: policy inputs come from the
                    # compatible stored transition initial state only.
                    policy_results.append(self._optimize_policy({}))
            policy = self._mean_policy_results(policy_results)
            temporal_value = temporal.get("loss")
            if temporal_value is None:
                status_t = "active"
            else:
                self._temporal_history.append(float(temporal_value))
                temporal_precision = float(
                    self.config.temporal_precision or self._global_precision()
                )
                status_t = self._objective_status(
                    self._temporal_history, temporal_precision
                )
            self._last_temporal_loss = temporal_value
            joint_status, joint_details = self._joint_macro_status()
            self._hierarchical_done = joint_status in {"converged", "diverged"}
            self._temporal_done = self._hierarchical_done
            F_after, metrics_after = self._evaluate(test_contexts)
            if self._hierarchical_done and self.config.enable_structural_phase:
                self.macro_phase = 3
            phase_info = {
                "macro_phase": self.macro_phase,
                "macro_phase_name": self.MACRO_NAMES[self.macro_phase],
                "hierarchical": {
                    "status": status_h,
                    "objective": F_hierarchical,
                    "top_k": top_k_info,
                    "done": self._hierarchical_done,
                },
                "temporal": {
                    "status": status_t,
                    "objective": temporal_value,
                    "done": self._temporal_done,
                    "trajectories": temporal.get("trajectories", 0),
                    "details": {k: v for k, v in temporal.items()
                                if k not in {"loss", "weighted_loss"}},
                },
                "joint_status": joint_status,
                "joint_details": joint_details,
                "most_surprising_transitions": temporal_surprises[:max(1, self.top_k)],
            }
        else:
            # PHASE III — perform one structural operation only. The structural
            # phase is a single intervention between two learning cycles, not
            # a phase that may consume the remaining offline steps.
            F_after, metrics_after = self._evaluate(test_contexts)
            structural_phase = self.macro_phase
            structural = self._structural_step(test_frames)
            if structural.get("accepted"):
                F_after, metrics_after = self._evaluate_current_agent(test_frames)

            # Phase III is a single structural intervention per trigger.
            # Whether the candidate is accepted, rejected, or not applicable,
            # always return to phases I and II on the next offline step.
            # Accepted mutations already restart the cycle inside
            # _structural_step; avoid restarting/incrementing the cycle twice.
            if self.macro_phase == 3:
                self._restart_learning_cycle()

            phase_info = {
                "macro_phase": structural_phase,
                "macro_phase_name": self.MACRO_NAMES[structural_phase],
                "structural": structural,
                "next_macro_phase": self.macro_phase,
                "next_macro_phase_name": self.MACRO_NAMES[self.macro_phase],
            }

        # Top-K progression is referenced to all online inferences since the
        # previous offline run, not to the mean of the current replay batch.
        online_mean = online_reference.get("mean")
        top_k_progression = {
            "online_inferences": int(online_reference.get("count", 0)),
            "online_mean_free_energy": online_mean,
            "offline_final_free_energy": float(F_after),
            "global_precision": self._global_precision(),
            "advanced": False,
        }
        if online_mean is not None:
            residual = float(F_after) - float(online_mean)
            advanced = residual > self._global_precision()
            top_k_progression["residual_offline_minus_online"] = residual
            top_k_progression["advanced"] = advanced
            top_k_progression["top_k_before"] = int(self.top_k)
            if advanced:
                max_k = max(
                    int(self.config.inference_memory_capacity),
                    int(self.config.transition_memory_capacity),
                )
                self.top_k = min(max_k, int(self.top_k) + int(self.config.top_k_step))
            top_k_progression["top_k_after"] = int(self.top_k)
        top_k_info["progression_reference"] = top_k_progression

        # Save the complete agent after the corresponding phase has finished.
        # Phase I is saved immediately after its own update above; phase II/III
        # are saved here so their checkpoints include the final Top-K controller state.
        if "phase_1" in phase_checkpoints:
            phase_checkpoints["phase_2"] = self._checkpoint_phase_end(
                2, self.MACRO_NAMES[2]
            )
        elif structural is not None:
            phase_checkpoints["phase_3"] = self._checkpoint_phase_end(
                3, self.MACRO_NAMES[3]
            )

        motor_optimization = {"updated": False, "disabled": True}
        if self.config.train_motor:
            motor_optimization = motor_train_fn(
                deepcopy(motor_trace), deepcopy(frames)
            )
            if motor_optimization is None:
                motor_optimization = {"updated": True, "result": None}

        self.last_metrics = metrics_after
        best_updated = self._consider_best_agent(F_after) or best_updated
        final_checkpoint_path = self._checkpoint_after(F_after)
        plan = {
            "version": "12.0",
            "paradigm": "awake_belief_vs_dream_trajectory_with_separate_memories",
            "iteration": self.iteration,
            "macro_cycles": self._macro_cycles,
            "macro_phase": self.macro_phase,
            "macro_phase_name": self.MACRO_NAMES[self.macro_phase],
            "top_k": self.top_k,
            "top_k_progression": deepcopy(top_k_progression),
            "inference_memory": deepcopy(self._last_replay_info),
            "online_free_energy_reference": deepcopy(online_reference),
            "temporal_surprise_trace": deepcopy(self._temporal_surprise_trace),
            "F_global_test_before": F_before,
            "F_global_test_after": F_after,
            "delta_F_global_test": F_after - F_before,
            "hierarchical_optimization": hierarchical,
            "temporal_optimization": temporal,
            "policy_optimization": policy,
            "motor_optimization": motor_optimization,
            "structural": structural,
            "phase_status": phase_info,
            "phase_checkpoints": phase_checkpoints,
            "q_updated_offline": False,
        }
        plan_path = self._write_plan(plan)
        # Reset only after the whole offline call and its checkpoints succeeded.
        self._reset_online_free_energy_reference()
        return {
            "iteration": self.iteration,
            "paradigm": "daih_offline_v12",
            "macro_phase": self.macro_phase,
            "macro_phase_name": self.MACRO_NAMES[self.macro_phase],
            "macro_cycles": self._macro_cycles,
            "hierarchical_optimization": hierarchical,
            "temporal_optimization": temporal,
            "policy_optimization": policy,
            "motor_optimization": motor_optimization,
            "motor_trace": deepcopy(motor_trace),
            "structural": structural,
            "phases": phase_info,
            "states": {name: metric.to_dict() for name, metric in metrics_after.items()},
            "F_global_test_before": F_before,
            "F_global_test_after": F_after,
            "delta_F_global_test": F_after - F_before,
            "top_k": self.top_k,
            "top_k_progression": top_k_progression,
            "online_free_energy_reference": online_reference,
            "inference_memory": deepcopy(self._last_replay_info),
            "temporal_surprise_trace": deepcopy(self._temporal_surprise_trace),
            "q_updated_offline": False,
            "checkpoint_path": checkpoint_path,
            "final_checkpoint_path": final_checkpoint_path,
            "phase_checkpoints": phase_checkpoints,
            "best_agent_updated": best_updated,
            "best_free_energy": self.best_free_energy,
            "best_iteration": self.best_iteration,
            "best_agent_checkpoint_path": self.best_checkpoint_path,
            "plan_path": plan_path,
            "structural_log": deepcopy(self._structural_log[-20:]),
        }

