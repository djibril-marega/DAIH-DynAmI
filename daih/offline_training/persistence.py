"""Checkpoint persistence for Agent_DAIH.

Only declarative architecture, tensors, distributions and nn.Module state_dicts
are serialized. User Python callables are restored through explicit factories.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn.utils import prune


FORMAT_VERSION = 1


def _cpu_tree(value: Any):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return deepcopy(value)


def _serialize_distribution(distribution: Any) -> dict:
    if isinstance(distribution, torch.distributions.MultivariateNormal):
        return {
            "type": "multivariate_normal",
            "loc": _cpu_tree(distribution.loc),
            "covariance_matrix": _cpu_tree(distribution.covariance_matrix),
        }
    if isinstance(distribution, torch.distributions.Normal):
        return {
            "type": "normal",
            "loc": _cpu_tree(distribution.loc),
            "scale": _cpu_tree(distribution.scale),
        }
    raise TypeError(
        "Unsupported Q distribution for checkpoint: "
        f"{type(distribution).__module__}.{type(distribution).__name__}. "
        "Add an explicit serializer before saving this distribution."
    )


def _deserialize_distribution(data: dict, device: torch.device | str):
    kind = data["type"]
    if kind == "multivariate_normal":
        return torch.distributions.MultivariateNormal(
            loc=data["loc"].to(device),
            covariance_matrix=data["covariance_matrix"].to(device),
        )
    if kind == "normal":
        return torch.distributions.Normal(
            loc=data["loc"].to(device),
            scale=data["scale"].to(device),
        )
    raise ValueError(f"Unknown serialized distribution type: {kind!r}")


def _snapshot_q(q: dict | None):
    if q is None:
        return None
    return {
        state: [_serialize_distribution(d) for d in distributions]
        for state, distributions in q.items()
    }


def _restore_cached_beliefs(agent) -> None:
    """Synchronize cached moments after restoring the authoritative Q_ object."""
    agent.mu = {}
    agent.variance = {}
    agent.covariances = {}
    agent.state_priors = {}
    for state, distributions in agent.Q_.items():
        means, variances, covariances = [], [], []
        for distribution in distributions:
            means.append(distribution.loc.detach())
            if isinstance(distribution, torch.distributions.MultivariateNormal):
                covariance = distribution.covariance_matrix.detach()
                covariances.append(covariance)
                variances.append(torch.diagonal(covariance, dim1=-2, dim2=-1))
            else:
                variance = distribution.scale.detach().square()
                variances.append(variance)
                covariances.append(torch.diag_embed(variance) if variance.ndim else None)
        agent.mu[state] = torch.stack(means)
        agent.variance[state] = torch.stack(variances)
        if all(c is not None for c in covariances):
            agent.covariances[state] = torch.stack(covariances)
        agent.state_priors[state] = agent.mu[state]


def _restore_pruning_reparameterizations(module: nn.Module, state: dict) -> None:
    """Recreate PyTorch pruning wrappers before strict state_dict loading.

    A pruned module stores ``weight_orig`` and ``weight_mask`` instead of a
    plain ``weight`` parameter. Reapply the mask parametrization on the freshly
    constructed module so checkpoints containing v4 phase-1 pruning reload
    with strict=True and retain their effective sparsity.
    """
    suffix = ".weight_orig"
    for key, original in state.items():
        if key == "weight_orig":
            prefix = ""
        elif key.endswith(suffix):
            prefix = key[:-len(suffix)]
        else:
            continue
        mask_key = f"{prefix}.weight_mask" if prefix else "weight_mask"
        if mask_key not in state:
            raise RuntimeError(f"Pruned parameter {key!r} has no matching mask {mask_key!r}.")
        target = module.get_submodule(prefix) if prefix else module
        if hasattr(target, "weight_orig"):
            continue
        parameter = getattr(target, "weight")
        mask = state[mask_key].to(device=parameter.device, dtype=parameter.dtype)
        prune.custom_from_mask(target, "weight", mask)


def save_agent(agent, path: str | Path) -> Path:
    """Save a deployable agent checkpoint (callables are represented by a manifest)."""
    if agent.D is None or agent.generative_functions is None or agent.functions_map is None:
        raise RuntimeError("The agent must be built with model() before it can be saved.")

    module_states = {}
    module_types = {}
    for key, value in agent.functions_map.items():
        if isinstance(value, nn.Module):
            module_states[key] = _cpu_tree(value.state_dict())
            module_types[key] = f"{type(value).__module__}.{type(value).__qualname__}"

    local_manifest = []
    seen_modules = set()
    for key, value in agent.functions_map.items():
        if not isinstance(value, nn.Module) or id(value) in seen_modules:
            continue
        seen_modules.add(id(value))
        for state_id, layer_id, kind in getattr(value, "_local_functions", {}):
            local_manifest.append({
                "module_key": key, "state_id": state_id,
                "layer_id": layer_id, "function_type": kind,
                "callable_kind": "python",
            })
        for state_id, layer_id, kind in getattr(value, "_local_module_keys", {}):
            local_manifest.append({
                "module_key": key, "state_id": state_id,
                "layer_id": layer_id, "function_type": kind,
                "callable_kind": "torch_module",
            })

    checkpoint = {
        "format_version": FORMAT_VERSION,
        "package_version": "3.0.0",
        "architecture": {
            "D": _cpu_tree(agent.D),
            "generative_functions": deepcopy(agent.generative_functions),
            "observations": deepcopy(agent.observations or {}),
            "actions": deepcopy(agent.actions or {}),
        },
        "module_states": module_states,
        "module_types": module_types,
        "local_function_manifest": local_manifest,
        "beliefs": _snapshot_q(agent.Q_),
        "precisions": {
            "pi_states": _cpu_tree(agent.pi_states),
            "pi_priors": _cpu_tree(agent.pi_priors),
            "pi_obs": _cpu_tree(agent.pi_obs),
            "pi_p": _cpu_tree(getattr(agent, "pi_p", None)),
            "pi_threshold": _cpu_tree(agent.pi_threshold),
            "pi_obs_substitued": _cpu_tree(agent.pi_obs_substitued),
        },
        "offline_trainer": (
            agent._offline_trainer.state_dict()
            if agent._offline_trainer is not None else None
        ),
        "runtime": {
            "lr_motors": agent.lr_motors,
            "lr_policies": agent.lr_policies,
            "lr_obs": agent.lr_obs,
            "max_policies_iter": agent.max_policies_iter,
        },
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(destination)
    return destination


def load_agent(
    agent_class,
    path: str | Path,
    *,
    functions_map_factory: Callable[[dict], dict],
    local_functions_factory: Callable | None = None,
    map_location: str | torch.device = "cpu",
):
    """Rebuild an agent and restore its parameters and posterior beliefs.

    functions_map_factory(checkpoint) must return the application's symbolic
    function map, including the expected WorldModel modules.
    local_functions_factory(agent, checkpoint) must re-register user callables.
    """
    if not callable(functions_map_factory):
        raise TypeError("functions_map_factory must be callable.")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {checkpoint.get('format_version')!r}; "
            f"expected {FORMAT_VERSION}."
        )

    architecture = checkpoint["architecture"]
    functions_map = functions_map_factory(checkpoint)
    if not isinstance(functions_map, dict):
        raise TypeError("functions_map_factory must return a dict.")

    agent = agent_class()
    agent.D = _cpu_tree(architecture["D"])
    agent.model(
        generative_functions=deepcopy(architecture["generative_functions"]),
        functions_map=functions_map,
        observations=deepcopy(architecture["observations"]),
        actions=deepcopy(architecture["actions"]),
        pi_p=checkpoint["precisions"].get("pi_p"),
    )

    local_manifest = checkpoint.get("local_function_manifest", [])
    if local_manifest and local_functions_factory is None:
        raise RuntimeError(
            "This checkpoint contains state-specific local functions. "
            "Pass local_functions_factory so the application can recreate them."
        )
    if local_functions_factory is not None:
        if not callable(local_functions_factory):
            raise TypeError("local_functions_factory must be callable or None.")
        local_functions_factory(agent, checkpoint)
        if getattr(agent, "_local_function_factory", None) is not None:
            agent.install_local_functions()

    # A manifest mismatch means the checkpoint could not be restored as an
    # operational agent; fail rather than silently falling back to global nets.
    for entry in local_manifest:
        module = functions_map.get(entry["module_key"])
        key = (entry["state_id"], entry["layer_id"], entry["function_type"])
        if not isinstance(module, nn.Module) or not getattr(module, "has_local_function", lambda *a: False)(*key):
            raise RuntimeError(
                "Local function was not restored: "
                f"module={entry['module_key']!r}, key={key!r}. "
                "Check local_functions_factory."
            )

    # Restore module weights after factories have registered any local nn.Modules.
    missing_modules = []
    for key, state in checkpoint.get("module_states", {}).items():
        module = functions_map.get(key)
        if not isinstance(module, nn.Module):
            missing_modules.append(key)
            continue
        _restore_pruning_reparameterizations(module, state)
        module.load_state_dict(state, strict=True)
    if missing_modules:
        raise RuntimeError(
            "functions_map_factory did not recreate these nn.Module entries: "
            + ", ".join(missing_modules)
        )

    beliefs = checkpoint.get("beliefs")
    if beliefs is not None:
        agent.Q_ = {
            state: [
                _deserialize_distribution(item, map_location)
                for item in distributions
            ]
            for state, distributions in beliefs.items()
        }
        if set(agent.Q_) != set(agent.generative_functions):
            raise RuntimeError(
                "Checkpoint beliefs and architecture disagree: "
                f"Q={sorted(agent.Q_)}, architecture={sorted(agent.generative_functions)}"
            )
        _restore_cached_beliefs(agent)
        from . import preprocessing
        agent.inputs_map = preprocessing.sync_state_inputs(agent.D, agent.Q_, agent.inputs_map)

    precisions = checkpoint.get("precisions", {})
    for name in ("pi_states", "pi_priors", "pi_obs", "pi_p", "pi_threshold", "pi_obs_substitued"):
        if name in precisions and precisions[name] is not None:
            setattr(agent, name, _cpu_tree(precisions[name]))

    # model() initialized these derived input aliases before the saved precision
    # dictionaries were restored. Refresh them so the first inference step sees
    # the checkpoint values rather than the D-derived defaults.
    for state_name, precision in (agent.pi_states or {}).items():
        key = f"pi_{state_name}"
        if key in agent.inputs_map:
            agent.inputs_map[key] = precision
    for state_name, precision in (agent.pi_p or {}).items():
        for key in (f"pi_p_{state_name}", f"error_{state_name}"):
            if key in agent.inputs_map:
                agent.inputs_map[key] = precision

    runtime = checkpoint.get("runtime", {})
    for name in ("lr_motors", "lr_policies", "lr_obs", "max_policies_iter"):
        if name in runtime:
            setattr(agent, name, runtime[name])

    agent._local_function_factory = getattr(agent, "_local_function_factory", None)
    trainer_state = checkpoint.get("offline_trainer")
    if trainer_state is not None:
        from .offline import OfflineTrainer
        agent._offline_trainer = OfflineTrainer(agent, config=trainer_state.get("config"))
        agent._offline_trainer.load_state_dict(trainer_state)
    else:
        agent._offline_trainer = None
    return agent
