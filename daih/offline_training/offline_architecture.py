"""Automatic, declarative architecture mutations for offline DAIH training."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn.utils import prune

from ..world_model import WorldModel


class OfflineArchitecture:
    """Applies generic mutations and rebuilds Agent_DAIH through agent.model().

    Local callables are deliberately not inspected for shape compatibility.
    Their contracts are the user's responsibility and are expressed through
    OfflineConfig.state_constraints.
    """

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config

    def _constraint(self, state_name: str) -> dict:
        defaults = {
            "allow_neurons": True,
            "allow_intermediate_layers": True,
            "allow_under_state": True,
            "allow_state_addition": True,
            "allow_state_removal": True,
            "allow_hierarchy_addition": True,
            "allow_hierarchy_removal": True,
            "min_size": 1,
            "max_size": None,
            "min_neurons": 16,
            "max_neurons": 1024,
            "min_intermediate_layers": 0,
            "max_intermediate_layers": 8,
            "min_states": 1,
            "max_states": None,
        }
        defaults.update(self.config.state_constraints.get(state_name, {}))
        return defaults

    @staticmethod
    def _dimension_limit(value: Any, key: str, default):
        if value is None:
            return default
        if isinstance(value, dict):
            return value.get(key, default)
        if isinstance(value, (tuple, list)):
            # A list is interpreted as a shape template, matching the
            # corresponding mu layout: [0] -> one coordinate and
            # [[0, 0], [0, 0]] -> two under_state per order.
            if not value:
                return default
            last_axis = value[-1]
            return len(last_axis) if isinstance(last_axis, (tuple, list)) else len(value)
        return int(value)

    @staticmethod
    def _current_state_parameters(agent) -> dict:
        """Use current posterior means/variances as the next D initialization.""" # under_state
        result = deepcopy(agent.D)
        for name, distributions in agent.Q_.items():
            means, variances = [], []
            for q in distributions:
                means.append(torch.as_tensor(q.loc).detach().cpu())
                if isinstance(q, torch.distributions.MultivariateNormal):
                    covariance = q.covariance_matrix.detach().cpu()
                    variances.append(torch.diagonal(covariance, dim1=-2, dim2=-1))
                else:
                    variances.append(torch.as_tensor(q.scale).detach().cpu().square())
            mu = torch.stack(means)
            variance = torch.stack(variances)
            result[name]["mu"] = mu.tolist()
            result[name]["variance"] = variance.tolist()
            result[name].pop("covariance", None)
        return result

    @staticmethod
    def _resize_tensor_rows(values, add: bool, fill_value):
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
            squeeze = True
        else:
            squeeze = False
        if add:
            extra = torch.full((tensor.shape[0], 1), float(fill_value), dtype=tensor.dtype)
            tensor = torch.cat([tensor, extra], dim=1)
        else:
            tensor = tensor[:, :-1]
        return tensor.squeeze(-1).tolist() if squeeze else tensor.tolist()

    def _mutate_under_state(self, D, state_name: str, add: bool):
        c = self._constraint(state_name)
        mu = torch.as_tensor(D[state_name]["mu"], dtype=torch.float32)
        if mu.ndim == 1:
            under_state_count = 1
        else:
            under_state_count = mu.shape[-1]
        min_size = self._dimension_limit(c.get("min_size"), "under_state", 1)
        max_size = self._dimension_limit(c.get("max_size"), "under_state", None)
        if add and max_size is not None and under_state_count >= int(max_size):
            return False
        if not add and under_state_count <= int(min_size):
            return False

        variance = D[state_name].get("variance")
        if variance is None:
            variance = torch.ones_like(mu).tolist()
        if add:
            D[state_name]["mu"] = self._resize_tensor_rows(D[state_name]["mu"], True, 0.0)
            D[state_name]["variance"] = self._resize_tensor_rows(variance, True, 1.0)
        else:
            D[state_name]["mu"] = self._resize_tensor_rows(D[state_name]["mu"], False, 0.0)
            D[state_name]["variance"] = self._resize_tensor_rows(variance, False, 1.0)
        D[state_name].pop("covariance", None)
        return True

    @staticmethod
    def _generic_name(prefix: str, names: set[str]) -> str:
        index = 1
        while f"{prefix}_{index}" in names:
            index += 1
        return f"{prefix}_{index}"

    def _clone_state_declaration(
        self, source: str, new_name: str, parents: list[str], layer: int,
        input_f: list[str] | None = None,
    ):
        source_spec = self.agent.generative_functions[source]
        spec = {
            "layer": layer,
            "parents": list(parents),
            "f": source_spec["f"],
            "input_f": list(input_f or [new_name]),
            "output_f": [f"{new_name}_transition"],
            "g": source_spec["g"],
            "input_g": [new_name],
            "output_g": deepcopy(source_spec.get("output_g", source_spec.get("outpu_g", []))),
        }
        # Ancestors of action-conditioned states must expose p. The new
        # hierarchy node receives a generic p contract when the template has it.
        if source_spec.get("p") is not None:
            spec.update({
                "p": source_spec["p"],
                "input_p": [new_name],
                "output_p": [f"{new_name}_pred_policy"],
            })
        return spec

    def _add_state(self, D, gf, target: str, hierarchy: bool):
        self._last_add_state_reason = None
        c = self._constraint(target)
        if c.get("max_states") is not None and len(gf) >= int(c["max_states"]):
            return None
        if self.config.max_total_states is not None and len(gf) >= int(self.config.max_total_states):
            return None
        names = set(gf)
        prefix = "offline_hierarchy_state" if hierarchy else "offline_state"
        new_name = self._generic_name(prefix, names)
        layers = [int(v.get("layer", 0)) for v in gf.values()]
        next_layer = max(layers, default=0) + 1
        old_parents = list(gf[target].get("parents") or [])

        if hierarchy:
            # Insert a generic parent between the previous parents and target.
            # Carry the motor/action factors required by target.output_g so
            # graph validation and the motor routing contract remain valid.
            motor_groups = set(self.agent.actions or {}) & set(self.agent.observations or {})
            required_factors = []
            for output_name in (gf[target].get("output_g") or gf[target].get("outpu_g") or []):
                for group, names_in_group in (self.agent.observations or {}).items():
                    if output_name in names_in_group and group in motor_groups and group not in required_factors:
                        required_factors.append(group)
            if len(required_factors) > 1:
                return None
            generic_inputs = [new_name] + required_factors
            # The new state is action-dependent when it carries motor
            # factors. Keep only ancestors that satisfy the framework's
            # required-p rule; other previous parents are no longer direct
            # parents of the inserted generic layer.
            ancestor_candidates = []
            visited = set()
            stack = list(old_parents)
            while stack:
                candidate = stack.pop(0)
                if candidate in visited or candidate not in gf:
                    continue
                visited.add(candidate)
                ancestor_candidates.append(candidate)
                stack.extend(gf[candidate].get("parents") or [])
            new_parents = [
                parent for parent in ancestor_candidates
                if not required_factors or gf[parent].get("p") is not None
            ]
            gf[new_name] = self._clone_state_declaration(
                target, new_name, new_parents, next_layer, input_f=generic_inputs
            )
            action_names = {
                action for group in (self.agent.actions or {}).values() for action in group
            }
            action_groups = set(self.agent.actions or {})
            target_is_action_dependent = any(
                item in action_names or item in action_groups
                for item in (gf[target].get("input_f") or [])
            )
            if target_is_action_dependent and gf[new_name].get("p") is None:
                p_source = next((gf[parent] for parent in ancestor_candidates if gf[parent].get("p") is not None), None)
                if p_source is None:
                    del gf[new_name]
                    return None
                gf[new_name].update({
                    "p": p_source["p"],
                    "input_p": [new_name],
                    "output_p": [f"{new_name}_pred_policy"],
                })
            if required_factors:
                # The inserted parent predicts non-motor observations. The
                # target remains responsible for motor outputs, and its new
                # parent exposes the motor factor in input_f.
                motor_observations = {
                    observation_name
                    for group in required_factors
                    for observation_name in (self.agent.observations or {}).get(group, [])
                }
                gf[new_name]["output_g"] = [
                    output for output in (gf[target].get("output_g") or gf[target].get("outpu_g") or [])
                    if output not in motor_observations and output not in required_factors
                ]
                if not gf[new_name]["output_g"]:
                    # A generic state can model itself as its output when no
                    # non-motor observation remains available.
                    gf[new_name]["output_g"] = [new_name]
            if required_factors:
                v_template = next(
                    (gf[parent] for parent in old_parents
                     if gf[parent].get("V") is not None
                     and any(factor in (gf[parent].get("input_f") or []) for factor in required_factors)),
                    None,
                )
                if v_template is None:
                    del gf[new_name]
                    return None
                gf[new_name].update({
                    "V": v_template["V"],
                    "input_V": [f"{new_name}_transition"],
                    "output_V": [f"{new_name}_reward"],
                })
            if target not in gf[new_name]["output_g"]:
                gf[new_name]["output_g"].append(target)
            gf[target]["parents"] = [new_name]
        else:
            # Add a generic child under target. The parent's g contract must
            # explicitly predict the child state; a parents edge alone is not
            # enough for relation priors or parent prediction extraction.
            target_spec = gf[target]
            target_g_name = target_spec.get("g")
            target_g_module = self.agent.functions_map.get(target_g_name)
            target_id = self.agent.states_list.index(target) + 1
            target_layer = int(target_spec.get("layer", 0))
            has_local_g = (
                isinstance(target_g_module, WorldModel)
                and target_g_module.has_local_function(target_id, target_layer, "g")
            )
            if has_local_g and getattr(self.agent, "_local_function_factory", None) is None:
                self._last_add_state_reason = "local_parent_g_requires_factory"
                return None
            gf[new_name] = self._clone_state_declaration(target, new_name, [target], next_layer)
            outputs = list(target_spec.get("output_g") or target_spec.get("outpu_g") or [])
            if new_name not in outputs:
                outputs.append(new_name)
            target_spec["output_g"] = outputs

        D[new_name] = deepcopy(D[target])
        return new_name

    @staticmethod
    def _remove_state(D, gf, state_name: str, min_states: int = 1):
        if state_name not in gf or len(gf) <= int(min_states):
            return False
        # Do not silently break a declared function contract. A state with
        # explicit IO references must be rewired by the application first.
        for other_name, spec in gf.items():
            if other_name == state_name:
                continue
            for key in ("input_f", "input_g", "input_p", "input_V", "output_f", "output_g", "output_p", "output_V"):
                if state_name in (spec.get(key) or []):
                    return False
        parents = list(gf[state_name].get("parents") or [])
        for other_name, spec in gf.items():
            if other_name == state_name:
                continue
            if state_name in (spec.get("parents") or []):
                spec["parents"] = [p for p in spec.get("parents", []) if p != state_name]
                for parent in parents:
                    if parent not in spec["parents"]:
                        spec["parents"].append(parent)
        del gf[state_name]
        D.pop(state_name, None)
        return True

    @staticmethod
    def _resize_world_model(old: WorldModel, add_neurons: int = 0, add_layer: bool = False):
        old_hidden = int(old.hidden_dim)
        old_policy_hidden = int(old.policy_generator.initial_hidden.out_features)
        new_hidden = max(16, old_hidden + add_neurons)
        # Policy capacity is optimized by its own objective, not by the
        # shared generative WorldModel neuron phase.
        new_policy_hidden = old_policy_hidden
        new = WorldModel(
            hidden_dim=new_hidden,
            id_embedding_dim=old.state_embedding.embedding_dim,
            id_buckets=old.id_buckets,
            policy_hidden_dim=new_policy_hidden,
            vocab_size=old.policy_generator.vocab_size,
        )
        try:
            device = next(old.parameters()).device
            new = new.to(device)
        except StopIteration:
            pass

        # Recreate state/layer-local blocks before copying their tensors.
        for key, blocks in getattr(old, "local_context_blocks", {}).items():
            state_id, layer_id = key[1:].split("_l", 1)
            for _ in blocks:
                new.add_local_context_layer(int(state_id), int(layer_id))

        old_state = old.state_dict()
        new_state = new.state_dict()
        for key, target in new_state.items():
            source = old_state.get(key)
            if source is None or source.ndim != target.ndim:
                continue
            slices = tuple(slice(0, min(a, b)) for a, b in zip(source.shape, target.shape))
            target[slices] = source[slices].to(device=target.device, dtype=target.dtype)
        new.load_state_dict(new_state)

        # Preserve extra generic intermediate blocks.
        extra = int(getattr(old, "_offline_extra_layers", 0))
        if add_layer:
            block = nn.Sequential(nn.Linear(new_hidden, new_hidden), nn.GELU())
            block = block.to(next(new.parameters()).device)
            new.context_encoder = nn.Sequential(*list(new.context_encoder.children()), block)
            extra += 1
        elif extra > 0:
            children = list(new.context_encoder.children())
            new.context_encoder = nn.Sequential(*children[:-1])
            extra -= 1
        new._offline_extra_layers = extra

        # Transfer registered local functions; IDs are remapped after rebuild.
        for key, function in old._local_functions.items():
            new.register_local_function(key[0], key[1], key[2], function)
        for key, module_key in old._local_module_keys.items():
            new.register_local_function(key[0], key[1], key[2], old._local_modules[module_key])
        return new

    @staticmethod
    def _linear_saliency(weight: torch.Tensor) -> torch.Tensor:
        gradient = getattr(weight, "grad", None)
        if gradient is None:
            return weight.detach().abs()
        return weight.detach().abs() * gradient.detach().abs()

    @classmethod
    def _reorder_world_model_channels(cls, model: WorldModel) -> list[int]:
        """Move least useful shared hidden channels to the tail before pruning.

        Importance is a first-order |weight * gradient| proxy aggregated over
        the shared encoder/decoders and local context adapters. This is safer
        than blindly deleting the final channels, but is not a full ablation
        estimate of the free-energy change.
        """
        width = int(model.hidden_dim)
        scores = torch.zeros(width, device=next(model.parameters()).device)
        input_prefix_layers = []
        output_layers = []

        def collect(module, input_width=None, output_width=None):
            for child in module.modules():
                if not isinstance(child, nn.Linear):
                    continue
                if child.out_features == width:
                    output_layers.append(child)
                if input_width is not None and child.in_features == input_width:
                    input_prefix_layers.append((child, width))
                elif child.in_features == width:
                    input_prefix_layers.append((child, width))

        collect(model.context_encoder)
        collect(model.f_decoder)
        collect(model.gv_decoder)
        collect(model.local_fusion)
        collect(model.local_context_blocks)

        # The first decoder/fusion layers have metadata appended after the
        # hidden context, so their hidden channels occupy the input prefix.
        for layer in input_prefix_layers:
            linear, _ = layer
            saliency = cls._linear_saliency(linear.weight)
            scores += saliency[:, :width].sum(dim=0).to(scores)
        for linear in output_layers:
            saliency = cls._linear_saliency(linear.weight)
            scores += saliency.sum(dim=1).to(scores)
            if linear.bias is not None:
                bias_grad = linear.bias.grad
                bias_score = (linear.bias.detach().abs() if bias_grad is None
                              else linear.bias.detach().abs() * bias_grad.detach().abs())
                scores += bias_score.to(scores)

        # Keep the most useful channels first; resize then drops the tail.
        order = torch.argsort(scores, descending=True)
        for module in (model.context_encoder, model.f_decoder, model.gv_decoder,
                       model.local_fusion, model.local_context_blocks):
            for linear in module.modules():
                if not isinstance(linear, nn.Linear):
                    continue
                with torch.no_grad():
                    if linear.out_features == width:
                        linear.weight.copy_(linear.weight.index_select(0, order))
                        if linear.bias is not None:
                            linear.bias.copy_(linear.bias.index_select(0, order))
                    if linear.in_features == width:
                        linear.weight.copy_(linear.weight.index_select(1, order))
                    elif linear.in_features > width and module in (
                            model.f_decoder, model.gv_decoder, model.local_fusion):
                        # For context-plus-coordinate inputs, reorder only the
                        # leading context columns; leave coordinate features intact.
                        linear.weight[:, :width].copy_(linear.weight[:, :width].index_select(1, order))
        return order.detach().cpu().tolist()

    def _replace_world_model(self, delta: int = 0, add_layer: bool = False):
        old_modules = []
        seen = set()
        for value in self.agent.functions_map.values():
            if isinstance(value, WorldModel) and id(value) not in seen:
                seen.add(id(value))
                old_modules.append(value)
        replacements = {id(old): self._resize_world_model(old, delta, add_layer) for old in old_modules}
        for key, value in list(self.agent.functions_map.items()):
            if id(value) in replacements:
                self.agent.functions_map[key] = replacements[id(value)]

    def _regularize_hyperparameters(self) -> dict:
        """Shrink selected scalar training hyperparameters toward configured targets.

        Positive scale parameters are interpolated geometrically to preserve
        positivity across orders of magnitude. The regularization is explicit
        and deterministic; it does not silently change observation precisions.
        """
        strength = float(self.config.hyperparameter_regularization_strength)
        targets = dict(self.config.hyperparameter_targets or {})
        allowed = {"learning_rate", "weight_decay", "lr_motors", "lr_policies", "lr_obs"}
        changed = {}
        for name, target in targets.items():
            if name not in allowed or not hasattr(self.agent if name.startswith("lr_") else self.config, name):
                continue
            owner = self.agent if name.startswith("lr_") else self.config
            current = float(getattr(owner, name))
            target = float(target)
            if current <= 0 or target <= 0:
                value = current + strength * (target - current)
            else:
                value = current * ((target / current) ** strength)
            setattr(owner, name, value)
            changed[name] = {"before": current, "target": target, "after": value}
        return changed

    def _prune_obsolete_weights(self) -> dict:
        """Permanently mask low-utility weights in shared trainable modules.

        Utility is approximated by |w|*|grad(w)| from the immediately
        preceding optimization step. The lowest configured fraction is
        removed globally, while preserving a minimum number of active entries
        in every tensor. Since WorldModel instances may be shared by several
        states, pruning is intentionally reported as shared-module pruning.
        """
        fraction = float(self.config.prune_fraction)
        if fraction <= 0:
            return {"pruned": 0, "candidates": 0, "scope": "shared_modules"}

        modules, seen = [], set()
        for root in (self.agent.functions_map or {}).values():
            if isinstance(root, nn.Module) and id(root) not in seen:
                modules.append(root)
                seen.add(id(root))

        tensors = []
        candidates = []
        for root in modules:
            for module_name, module in root.named_modules():
                path_parts = module_name.split(".") if module_name else []
                if ("policy_generator" in path_parts
                        or any(part.endswith(":p") for part in path_parts)):
                    continue
                for registered_name, parameter in module.named_parameters(recurse=False):
                    if registered_name not in ("weight", "weight_orig"):
                        continue
                    parameter_name = "weight"
                    if parameter.ndim < 2 or not parameter.requires_grad:
                        continue
                    if parameter.grad is None:
                        continue
                    current_mask = getattr(module, "weight_mask", None)
                    if current_mask is None:
                        active = torch.ones_like(parameter, dtype=torch.bool)
                        effective = parameter.detach()
                    else:
                        active = current_mask.detach().bool()
                        effective = parameter.detach() * current_mask.detach()
                    gradient = parameter.grad.detach()
                    if not torch.isfinite(effective).all() or not torch.isfinite(gradient).all():
                        continue
                    key = (id(root), module_name, parameter_name)
                    tensors.append((module, parameter_name, key, active.clone()))
                    utility = (effective.abs() * gradient.abs()).reshape(-1)
                    active_flat = active.reshape(-1)
                    for flat_index in torch.nonzero(active_flat, as_tuple=False).reshape(-1).tolist():
                        candidates.append((float(utility[flat_index].item()), key, flat_index))

        if not candidates:
            return {"pruned": 0, "candidates": 0, "scope": "shared_modules"}

        scores = sorted(item[0] for item in candidates)
        target_count = min(len(candidates), max(1, int(len(candidates) * fraction)))
        cutoff = scores[target_count - 1]
        max_score = scores[-1]
        # If the utility distribution is flat (or entirely zero), there is no
        # evidence that a subset is obsolete; do not prune arbitrarily.
        if max_score <= 1e-12 or cutoff >= max_score * (1.0 - 1e-6):
            return {
                "pruned": 0, "candidates": len(candidates),
                "scope": "shared_modules", "reason": "no_separable_low_utility_tail",
            }
        min_remaining = max(1, int(self.config.prune_min_remaining_per_tensor))
        remaining = {key: int(mask.sum().item()) for _, _, key, mask in tensors}
        row_remaining = {}
        row_widths = {}
        for module, parameter_name, key, mask in tensors:
            shape = getattr(module, f"{parameter_name}_orig", getattr(module, parameter_name)).shape
            row_width = int(torch.tensor(shape[1:]).prod().item())
            row_widths[key] = row_width
            flat_mask = mask.reshape(-1)
            for row in range(shape[0]):
                row_remaining[(key, row)] = int(flat_mask[row * row_width:(row + 1) * row_width].sum().item())
        chosen = {key: [] for _, _, key, _ in tensors}
        pruned = 0
        for _, key, flat_index in sorted(candidates, key=lambda item: item[0]):
            if pruned >= target_count:
                break
            row = flat_index // row_widths[key]
            if remaining[key] <= min_remaining or row_remaining[(key, row)] <= min_remaining:
                continue
            chosen[key].append(flat_index)
            remaining[key] -= 1
            row_remaining[(key, row)] -= 1
            pruned += 1

        for module, parameter_name, key, active in tensors:
            indices = chosen[key]
            if not indices:
                continue
            mask = active.to(dtype=getattr(module, parameter_name).dtype)
            mask_flat = mask.reshape(-1)
            mask_flat[torch.tensor(indices, device=mask.device, dtype=torch.long)] = 0
            if hasattr(module, f"{parameter_name}_orig"):
                existing = getattr(module, f"{parameter_name}_mask")
                with torch.no_grad():
                    existing.mul_(mask)
            else:
                prune.custom_from_mask(module, parameter_name, mask)

        return {
            "pruned": pruned,
            "candidates": len(candidates),
            "fraction_requested": fraction,
            "scope": "shared_modules",
            "criterion": "global_lowest_abs_weight_times_abs_gradient",
            "utility_cutoff": cutoff,
            "utility_max": max_score,
        }

    def apply(self, mutations: list[dict]) -> dict:
        if not mutations:
            return {"applied": [], "skipped": []}

        # Preserve the actual posterior, not the original arbitrary D values.
        D = self._current_state_parameters(self.agent)
        gf = deepcopy(self.agent.generative_functions)
        applied, skipped = [], []
        world_neuron_delta = 0
        local_layer_changes = []
        structural_applied = False
        global_inplace_results = {}

        for mutation in mutations:
            state = mutation.get("target")
            action = mutation.get("action")
            if state not in gf:
                skipped.append({**mutation, "reason": "target_state_missing"})
                continue
            c = self._constraint(state)
            ok = False
            if action == "prune_obsolete_weights":
                result = global_inplace_results.get(action)
                if result is None:
                    result = self._prune_obsolete_weights()
                    global_inplace_results[action] = result
                if result["pruned"]:
                    applied.append({**mutation, "result": result})
                else:
                    skipped.append({**mutation, "reason": "no_eligible_weights", "result": result})
                continue
            elif action == "regularize_hyperparameters":
                changed = global_inplace_results.get(action)
                if changed is None:
                    changed = self._regularize_hyperparameters()
                    global_inplace_results[action] = changed
                if changed:
                    applied.append({**mutation, "result": {"changed": changed}})
                else:
                    skipped.append({**mutation, "reason": "no_configured_hyperparameters"})
                continue
            elif action == "add_neurons" and c["allow_neurons"]:
                world_neuron_delta += int(self.config.neuron_step)
                ok = True
            elif action == "prune_neurons" and c["allow_neurons"]:
                world_neuron_delta -= int(self.config.neuron_step)
                ok = True
            elif action in {"add_intermediate_layer", "prune_intermediate_layer"} and c["allow_intermediate_layers"]:
                # Phase 4 is local: it changes only the context path for the
                # state/layer that triggered the mutation, never the shared encoder.
                state_id = self.agent.states_list.index(state) + 1
                layer_id = int(gf[state].get("layer", 0))
                modules, seen = [], set()
                for value in self.agent.functions_map.values():
                    if isinstance(value, WorldModel) and id(value) not in seen:
                        modules.append(value)
                        seen.add(id(value))
                counts = [m.local_context_layer_count(state_id, layer_id) for m in modules]
                current_layers = max(counts, default=0)
                if action == "add_intermediate_layer":
                    ok = current_layers < int(c["max_intermediate_layers"])
                    if ok:
                        for module in modules:
                            module.add_local_context_layer(state_id, layer_id)
                else:
                    ok = current_layers > int(c["min_intermediate_layers"])
                    if ok:
                        for module in modules:
                            module.remove_local_context_layer(state_id, layer_id)
                if ok:
                    local_layer_changes.append({
                        "state_id": state_id,
                        "layer_id": layer_id,
                        "layers": current_layers + (1 if action == "add_intermediate_layer" else -1),
                    })
            elif action == "add_generalized_coordinate" and c["allow_under_state"]:
                ok = self._mutate_under_state(D, state, True)
            elif action == "remove_low_utility_coordinate" and c["allow_under_state"]:
                ok = self._mutate_under_state(D, state, False)
            elif action == "add_state" and c["allow_state_addition"]:
                new_name = self._add_state(D, gf, state, hierarchy=False)
                if new_name is None:
                    skipped.append({
                        **mutation,
                        "reason": getattr(self, "_last_add_state_reason", "max_states_reached"),
                    })
                else:
                    applied.append({**mutation, "created_state": new_name})
                    structural_applied = True
                continue
            elif action == "remove_redundant_state" and c["allow_state_removal"]:
                ok = self._remove_state(D, gf, state, min_states=c["min_states"])
            elif action == "add_hierarchy_layer" and c["allow_hierarchy_addition"]:
                new_name = self._add_state(D, gf, state, hierarchy=True)
                if new_name is None:
                    skipped.append({
                        **mutation,
                        "reason": getattr(self, "_last_add_state_reason", "max_states_reached"),
                    })
                else:
                    applied.append({**mutation, "created_state": new_name})
                    structural_applied = True
                continue
            elif action == "remove_redundant_hierarchy_layer" and c["allow_hierarchy_removal"]:
                ok = self._remove_state(D, gf, state, min_states=c["min_states"])
            else:
                skipped.append({**mutation, "reason": "disabled_or_unsupported"})
                continue

            (applied if ok else skipped).append(
                {**mutation, "reason": None if ok else "constraint_limit_or_noop"}
            )
            if action not in {
                "add_intermediate_layer", "prune_intermediate_layer",
                "add_neurons", "prune_neurons",
            }:
                structural_applied = structural_applied or bool(ok)

        if structural_applied:
            # Validate the complete declarative result before touching the
            # live agent or replacing any registered module.
            try:
                from . import model, graph
                model.validate_model(D, gf, self.agent.actions)
                graph.build_graph(gf, self.agent.observations, self.agent.actions)
                model.init_distributions(D, list(gf))
            except Exception as exc:
                skipped.extend({**item, "reason": f"declaration_validation_failed: {exc}"} for item in applied)
                return {"applied": [], "skipped": skipped}

        if world_neuron_delta:
            # Do not allow the shared representation to collapse below its
            # configured minimum; the most restrictive target limit applies.
            limits = [
                self._constraint(m.get("target"))
                for m in mutations
                if m.get("action") in ("add_neurons", "prune_neurons")
                and m.get("target") in gf
                and self._constraint(m.get("target"))["allow_neurons"]
            ]
            if limits:
                old_modules = [v for v in self.agent.functions_map.values() if isinstance(v, WorldModel)]
                if old_modules:
                    old_dim = old_modules[0].hidden_dim
                    min_dim = max(
                        int(self.config.world_model_min_neurons),
                        max(int(x["min_neurons"]) for x in limits),
                    )
                    max_dim = min(
                        int(self.config.world_model_max_neurons),
                        min(int(x["max_neurons"]) for x in limits),
                    )
                    delta = max(min_dim - old_dim, min(max_dim - old_dim, world_neuron_delta))
                    if delta:
                        if delta < 0:
                            seen_models = set()
                            for module in old_modules:
                                if id(module) not in seen_models:
                                    seen_models.add(id(module))
                                    self._reorder_world_model_channels(module)
                        self._replace_world_model(delta=delta)
                        trainer = getattr(self.agent, "_offline_trainer", None)
                        if trainer is not None:
                            trainer.invalidate_optimizer()
        rebuilt = False
        if structural_applied:
            self._rebuild(D, gf)
            rebuilt = True
        elif local_layer_changes:
            trainer = getattr(self.agent, "_offline_trainer", None)
            if trainer is not None:
                trainer.invalidate_optimizer()
        return {
            "applied": applied,
            "skipped": skipped,
            "rebuilt": rebuilt,
            "local_layer_changes": local_layer_changes,
        }

    @staticmethod
    def _restore_existing_beliefs(agent, old_Q):
        """Restore existing posterior values after model() rebuilds the graph.

        Equal-sized distributions are copied exactly (including full MVN
        covariance). If a state's coordinate count changed, the overlapping
        means/covariance block is retained and only newly added under_state
        keep their D-derived initialization.
        """
        if not old_Q:
            return
        for state_name, old_distributions in old_Q.items():
            if state_name not in agent.Q_:
                continue
            fresh_distributions = agent.Q_[state_name]
            restored = []
            for index, fresh in enumerate(fresh_distributions):
                if index >= len(old_distributions):
                    restored.append(fresh)
                    continue
                old = old_distributions[index]
                if isinstance(old, torch.distributions.MultivariateNormal) and isinstance(
                    fresh, torch.distributions.MultivariateNormal
                ):
                    old_loc = old.loc.detach().clone().to(fresh.loc)
                    fresh_loc = fresh.loc.detach().clone()
                    old_cov = old.covariance_matrix.detach().clone().to(fresh.covariance_matrix)
                    fresh_cov = fresh.covariance_matrix.detach().clone()
                    n = min(old_loc.numel(), fresh_loc.numel())
                    fresh_loc[:n] = old_loc[:n]
                    fresh_cov[:n, :n] = old_cov[:n, :n]
                    restored.append(torch.distributions.MultivariateNormal(
                        loc=fresh_loc, covariance_matrix=fresh_cov
                    ))
                elif isinstance(old, torch.distributions.Normal) and isinstance(
                    fresh, torch.distributions.Normal
                ):
                    restored.append(deepcopy(old))
                else:
                    restored.append(fresh)
            agent.Q_[state_name] = restored

        # Keep the agent's public cached moments synchronized with restored Q.
        for state_name, distributions in agent.Q_.items():
            means = torch.stack([q.loc.detach() for q in distributions])
            variances = []
            covariances = []
            for q in distributions:
                if isinstance(q, torch.distributions.MultivariateNormal):
                    covariance = q.covariance_matrix.detach()
                    covariances.append(covariance)
                    variances.append(torch.diagonal(covariance, dim1=-2, dim2=-1))
                else:
                    variance = q.scale.detach().square()
                    variances.append(variance)
                    covariances.append(torch.diag_embed(variance) if variance.ndim else None)
            agent.mu[state_name] = means
            agent.variance[state_name] = torch.stack(variances)
            if all(value is not None for value in covariances):
                agent.covariances[state_name] = torch.stack(covariances)
            agent.state_priors[state_name] = means

    def _rebuild(self, D, gf):
        agent = self.agent
        old_states = list(agent.states_list)
        old_gf = deepcopy(agent.generative_functions or {})
        old_pi_states = deepcopy(agent.pi_states)
        # Q_ is the authoritative posterior. Snapshot it BEFORE model state is
        # cleared below; otherwise deepcopy(None) silently loses every belief.
        old_Q = deepcopy(agent.Q_)
        old_pi_p_all = deepcopy(getattr(agent, "pi_p", None)) or {}
        old_pi_p = {name: value for name, value in old_pi_p_all.items() if name in D}
        old_functions_map = agent.functions_map

        # Remap local-function registrations if state IDs shifted after deletion.
        new_states = list(D)
        for module in {id(v): v for v in old_functions_map.values() if isinstance(v, WorldModel)}.values():
            registrations = []
            for key, fn in module._local_functions.items():
                old_id, layer_id, kind = key
                if 0 < old_id <= len(old_states):
                    state_name = old_states[old_id - 1]
                    if state_name in new_states:
                        registrations.append((new_states.index(state_name) + 1, layer_id, kind, fn))
            for key, module_key in module._local_module_keys.items():
                old_id, layer_id, kind = key
                if 0 < old_id <= len(old_states):
                    state_name = old_states[old_id - 1]
                    if state_name in new_states:
                        registrations.append((new_states.index(state_name) + 1, layer_id, kind, module._local_modules[module_key]))
            module._local_functions.clear()
            module._local_modules = nn.ModuleDict()
            module._local_module_keys.clear()
            for state_id, layer_id, kind, fn in registrations:
                module.register_local_function(state_id, layer_id, kind, fn)

        agent.D = D
        agent.graph = None
        agent.navigator = None
        agent.Q_ = None
        agent.inputs_map = None
        agent.pi_priors = None
        agent.state_priors = None
        agent.covariances = None
        agent.pi_states = None
        agent.pi_p = None

        # Preserve precision for unchanged dimensions and initialize only new
        # entries from the updated variance.
        preserved_pi = {}
        for name, spec in D.items():
            variance = torch.as_tensor(spec.get("variance", 1.0), dtype=torch.float32)
            fallback = torch.reciprocal(variance.clamp_min(1e-8))
            old = old_pi_states.get(name)
            if old is not None:
                old = torch.as_tensor(old, dtype=torch.float32)
                flat_old, flat_new = old.reshape(-1), fallback.reshape(-1)
                n = min(flat_old.numel(), flat_new.numel())
                flat_new[:n] = flat_old[:n].clamp(1e-8, 1e8)
                preserved_pi[name] = flat_new.reshape(fallback.shape)
        agent.pi_states = preserved_pi

        # Existing Q entries survive the rebuild; only newly introduced
        # under_state/states use D-based initialization.
        agent.model(
            generative_functions=gf,
            functions_map=old_functions_map,
            observations=agent.observations,
            actions=agent.actions,
            pi_p=old_pi_p,
        )
        self._restore_existing_beliefs(agent, old_Q)
        new_state_names = [name for name in D if name not in old_states]
        changed_output_states = [
            name for name in D
            if name in old_gf
            and list(old_gf[name].get("output_g") or old_gf[name].get("outpu_g") or [])
                != list(gf[name].get("output_g") or gf[name].get("outpu_g") or [])
        ]
        factory_states = list(dict.fromkeys(new_state_names + changed_output_states))
        if factory_states and getattr(agent, "_local_function_factory", None) is not None:
            # If a changed parent's old local g no longer matches output_g,
            # remove it before asking the application factory for the new contract.
            required_local_g = []
            for state_name in changed_output_states:
                old_spec = old_gf[state_name]
                module = agent.functions_map.get(old_spec.get("g"))
                new_id = agent.states_list.index(state_name) + 1
                new_layer = int(gf[state_name].get("layer", 0))
                if isinstance(module, WorldModel) and module.has_local_function(new_id, new_layer, "g"):
                    module.unregister_local_function(new_id, new_layer, "g")
                    required_local_g.append((state_name, old_spec.get("g")))
            installed = agent.install_local_functions(factory_states)
            installed_set = set(installed)
            missing = [
                (state_name, "g", function_name)
                for state_name, function_name in required_local_g
                if (state_name, "g", function_name) not in installed_set
            ]
            if missing:
                raise RuntimeError(
                    "The local-function factory did not provide an updated g callable "
                    "for a parent whose output_g contract changed: " + repr(missing)
                )
        # New graph now owns the updated declarations. Fail loudly if a
        # requested state did not make it through the public model rebuild.
        agent.generative_functions = deepcopy(gf)
        declared_states = set(agent.generative_functions)
        parameter_states = set(agent.D)
        graph_states = {
            name for name, node in agent.navigator.graph.items()
            if node.get("type") == "state"
        }
        if declared_states != parameter_states or declared_states != graph_states:
            raise RuntimeError(
                "Offline rebuild produced an inconsistent model: "
                f"declarations={sorted(declared_states)}, "
                f"D={sorted(parameter_states)}, graph={sorted(graph_states)}"
            )
        # Keep the phase reached by existing states. A structural rebuild
        # invalidates their recent derivative history, not their progress.
        from .offline import StateProgress
        trainer = agent._offline_trainer
        previous_progress = trainer.progress
        next_progress = {}
        for name, node in agent.navigator.graph.items():
            if node.get("type") != "state":
                continue
            default_phase = 4 if getattr(trainer, "global_progress", None) is not None and trainer.global_progress.phase >= 4 else 1
            progress = previous_progress.get(name, StateProgress(phase=default_phase))
            progress.history.clear()
            progress.pending_mutation = False
            progress.stable = False
            progress.phase_iterations = 0
            progress.phase_entry_free_energy = None
            progress.phase_best_free_energy = None
            next_progress[name] = progress
        trainer.progress = next_progress
        if hasattr(trainer, "invalidate_optimizer"):
            trainer.invalidate_optimizer()
        trainer.pending_plan = None
