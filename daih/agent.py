import torch
from . import utils, model, preprocessing, inference, model_components
import math
import copy
from pathlib import Path
import os
import tempfile


# Deep Active Inference Hybride Framework with Dynamic Planning Horizon
# and Amortized Active Inference (DAIH-DynAmI)
class Agent_DAIH:
    def __init__(
        self,
        *,
        max_grad_norm=5.0,
        min_state_value=-1e+3,
        max_state_value=1e+3,
        min_observation_value=-1e+3,
        max_observation_value=1e+3,
        observation_bounds=None,
        min_reward_value=-1e+2,
        max_reward_value=1e+2,
        min_precision=1e-4,
        max_precision=1e+4,
        min_obs_precision=1e-4,
        max_obs_precision=1e+4,
        min_policy_precision=1e-4,
        max_policy_precision=1e+4,
        max_policy_error=10,
        numerical_eps=1e-6,
    ):
        self.D = None

        # Numerical limits: configurable when the agent is instantiated.
        self.max_grad_norm = max_grad_norm
        self.min_state_value = min_state_value
        self.max_state_value = max_state_value
        self.min_observation_value = min_observation_value
        self.max_observation_value = max_observation_value
        self._observation_bounds_config = (
            {} if observation_bounds is None else copy.deepcopy(observation_bounds)
        )
        self.observation_bounds = {}
        self.min_reward_value = min_reward_value
        self.max_reward_value = max_reward_value
        self.min_precision = min_precision
        self.max_precision = max_precision
        self.min_obs_precision = min_obs_precision
        self.max_obs_precision = max_obs_precision
        self.min_policy_precision = min_policy_precision
        self.max_policy_precision = max_policy_precision
        self.max_policy_error = max_policy_error
        self.numerical_eps = numerical_eps

        model.validate_agent_parameters(
            max_grad_norm=self.max_grad_norm,
            min_state_value=self.min_state_value,
            max_state_value=self.max_state_value,
            min_observation_value=self.min_observation_value,
            max_observation_value=self.max_observation_value,
            min_reward_value=self.min_reward_value,
            max_reward_value=self.max_reward_value,
            min_precision=self.min_precision,
            max_precision=self.max_precision,
            min_obs_precision=self.min_obs_precision,
            max_obs_precision=self.max_obs_precision,
            min_policy_precision=self.min_policy_precision,
            max_policy_precision=self.max_policy_precision,
            max_policy_error=self.max_policy_error,
            numerical_eps=self.numerical_eps,
        )

        self.pi_obs = {}
        self.covariances = None
        self.state_priors = None
        self.pi_states = None
        self.pi_priors = None

        self.Q_ = None

        self.graph = None
        self.navigator = None
        self.functions_map = None
        self.inputs_map = None

        self.pi_threshold = {}
        self.actions_id = {}
        self.pi_obs_substitued = {}

        self.max_policies_iter = 100

        self.lr_motors = 1
        self.lr_policies = 1
        self.lr_obs = 1

        self.inference_memory_path = "offline_artifacts/observation_memory.pt"
        self.transition_memory_path = "offline_artifacts/transition_memory.pt"
        self.inference_memory_capacity = 256
        self.transition_memory_capacity = 256
        self.inference_memory_old_fraction = 0.25
        self.initial_top_k = 1
        self._pending_transition = None
        self._last_inference_free_energy = {}

    def model(self, generative_functions=None, functions_map=None,
              observations=None, actions=None, pi_p=None):
        if generative_functions is not None:
            self.generative_functions = generative_functions
        elif not hasattr(self, "generative_functions"):
            self.generative_functions = {}

        if observations is not None:
            self.observations = observations
        elif not hasattr(self, "observations"):
            self.observations = {}

        if actions is not None:
            self.actions = actions
        elif not hasattr(self, "actions"):
            self.actions = {}

        generative_functions = self.generative_functions
        observations = self.observations
        actions = self.actions

        self.observation_bounds = model.init_observation_bounds(
            observations,
            self._observation_bounds_config,
            self.min_observation_value,
            self.max_observation_value,
        )

        states = model.validate_model(self.D, generative_functions, actions)
        _, idx_to_actions = model.actions_index(actions)
        self.actions_id = idx_to_actions
        self.states_list = utils.dict_keys_to_list(self.D)

        (
            self.mu,
            self.variance,
            self.covariances,
            self.state_priors,
            self.Q_,
            self.pi_states,
        ) = model.init_distributions(self.D, states, self.pi_states)

        self.graph, navigator, self.functions_map = model.init_graph(
            self.graph, generative_functions, functions_map, observations, actions
        )
        if navigator is not None:
            self.navigator = navigator

        self.pi_priors = model.init_relation_priors(self.navigator, self.pi_states)
        self.inputs_map = preprocessing.sync_state_inputs(
            self.D, self.Q_, self.inputs_map
        )
        utils.init_actions(actions, self.inputs_map)

        self.pi_p, self.inputs_map = model.init_p(
            generative_functions, self.pi_states, pi_p, self.inputs_map
        )
        self.pi_threshold = model.sum_priors(self.pi_states, self.navigator)

    @staticmethod
    def _atomic_torch_save(payload, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=path.name, suffix=".tmp", dir=str(path.parent)
        )
        os.close(fd)
        try:
            torch.save(payload, temp_name)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    @staticmethod
    def _load_memory_items(path, key):
        path = Path(path)
        if not path.exists():
            return []
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            items = payload.get(key, []) if isinstance(payload, dict) else payload
            return items if isinstance(items, list) else []
        except Exception:
            return []

    @staticmethod
    def _load_memory_payload(path):
        path = Path(path)
        if not path.exists():
            return {}
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, list):
                return {"_legacy_items": payload}
            return {}
        except Exception:
            return {}

    def _retain_memory_item(self, path, key, item, capacity):
        """Keep a bounded sample while preserving independent memory metadata."""
        payload = self._load_memory_payload(path)
        items = payload.get(key, payload.get("_legacy_items", []))
        items = items if isinstance(items, list) else []
        payload.pop("_legacy_items", None)
        capacity = max(1, int(capacity))
        old_fraction = min(
            .95, max(0., float(self.inference_memory_old_fraction))
        )
        old_quota = min(len(items), int(round(capacity * old_fraction)))
        recent_quota = max(0, capacity - old_quota - 1)
        older_pool = items[:-recent_quota] if recent_quota else items

        if old_quota and older_pool:
            if old_quota == 1:
                retained_old = [older_pool[0]]
            else:
                indices = [
                    round(i * (len(older_pool) - 1) / (old_quota - 1))
                    for i in range(old_quota)
                ]
                retained_old = [older_pool[i] for i in indices]
        else:
            retained_old = []

        recent = items[-recent_quota:] if recent_quota else []
        merged = (retained_old + recent + [copy.deepcopy(item)])[-capacity:]
        payload[key] = merged
        payload["version"] = max(3, int(payload.get("version", 0) or 0))
        payload["capacity"] = capacity
        self._atomic_torch_save(payload, path)

    def _accumulate_online_free_energy(self, free_energy):
        """Persist the mean-reference accumulator independently of bounded retention."""
        path = Path(self.inference_memory_path)
        payload = self._load_memory_payload(path)
        if "_legacy_items" in payload and "observations" not in payload:
            payload["observations"] = payload.pop("_legacy_items")

        stats = payload.get("online_free_energy_stats", {})
        if not isinstance(stats, dict):
            stats = {}
        if free_energy is None:
            return

        value = float(free_energy)
        if not math.isfinite(value):
            return

        stats["sum"] = float(stats.get("sum", 0.0)) + value
        stats["count"] = int(stats.get("count", 0)) + 1
        stats["mean"] = stats["sum"] / max(1, stats["count"])
        stats["last"] = value
        payload["online_free_energy_stats"] = stats
        payload.setdefault("observations", [])
        payload["version"] = max(3, int(payload.get("version", 0) or 0))
        payload["capacity"] = int(self.inference_memory_capacity)
        self._atomic_torch_save(payload, path)

    def _offline_memory_architecture(self, snapshot=None):
        """Describe the state-level structure present when a memory is recorded."""
        result = {}
        source_q = (snapshot or {}).get("Q_", getattr(self, "Q_", {})) or {}
        state_names = list(source_q) or list(getattr(self, "states_list", []) or [])

        for state_name in state_names:
            distributions = source_q.get(state_name, [])
            dims = []
            for distribution in distributions or []:
                mean = getattr(distribution, "mean", None)
                if mean is None:
                    mean = getattr(distribution, "loc", None)
                if mean is None:
                    dims.append(None)
                else:
                    try:
                        dims.append(tuple(mean.shape))
                    except Exception:
                        dims.append(None)

            result[str(state_name)] = {
                "under_states": len(distributions or []),
                "distribution_shapes": dims,
            }
        return result

    def _record_inference_observation(self, observations, surprise=None):
        if not isinstance(observations, dict):
            return
        item = {
            "observation": copy.deepcopy(observations),
            "architecture": self._offline_memory_architecture(),
            "surprise": float(surprise or 0.0),
        }
        self._retain_memory_item(
            self.inference_memory_path, "observations", item,
            self.inference_memory_capacity
        )

    def _record_transition(self, initial_state, action, next_state, surprise=None):
        item = {
            "initial_state": copy.deepcopy(initial_state),
            "action": copy.deepcopy(action),
            "next_state": copy.deepcopy(next_state),
            "architecture": {
                "initial": self._offline_memory_architecture(initial_state),
                "next": self._offline_memory_architecture(next_state),
            },
            "surprise": (
                float(surprise)
                if surprise is not None and math.isfinite(float(surprise))
                else None
            ),
        }
        self._retain_memory_item(
            self.transition_memory_path,
            "transitions",
            item,
            getattr(self, "transition_memory_capacity", self.inference_memory_capacity),
        )

    def inference(self, observations):
        (
            observations,
            self.pi_obs,
            self.pi_obs_substitued,
        ) = preprocessing.handle_missing_observations(
            observations, self.pi_obs, self.pi_obs_substitued
        )

        for motors in self.beliefs_update(observations):
            yield f"action motor: {motors}"

        current_state = {
            "Q_": copy.deepcopy(self.Q_),
            "inputs_map": copy.deepcopy(self.inputs_map),
            "pi_states": copy.deepcopy(self.pi_states),
            "pi_priors": copy.deepcopy(self.pi_priors),
            "pi_obs": copy.deepcopy(self.pi_obs),
            "pi_obs_substitued": copy.deepcopy(self.pi_obs_substitued),
            "pi_threshold": copy.deepcopy(self.pi_threshold),
        }

        surprise_values = [
            value for value in self._last_inference_free_energy.values()
            if isinstance(value, (int, float)) and math.isfinite(value)
        ]
        observation_surprise = sum(surprise_values) if surprise_values else None

        if self._pending_transition is not None:
            self._record_transition(
                self._pending_transition["initial_state"],
                self._pending_transition.get("action"),
                current_state,
                surprise=observation_surprise,
            )

        self._accumulate_online_free_energy(observation_surprise)
        self._record_inference_observation(observations, observation_surprise)

        policy_selected = self.policy_selection()
        action = policy_selected.get("action") if isinstance(policy_selected, dict) else None
        self._pending_transition = {"initial_state": current_state, "action": action}
        yield f"policy selected: {policy_selected}"

    def free_energy(
        self,
        state_prior,
        state_posterior,
        D_poresterior,
        pi_prior,
        actual,
        pi_obs,
        gamma_obs,
        prediction,
    ):
        def complexity(state_posterior, state_prior, gamma_obs, pi_prior):
            return (
                torch.sum(
                    0.5 * pi_prior
                    * ((state_posterior + D_poresterior - state_prior) ** 2)
                )
                - torch.sum(0.5 * gamma_obs)
                - torch.sum(0.5 * torch.log(pi_prior))
            )

        def accuracy(actual, pi_obs, prediction):
            return torch.sum(-0.5 * pi_obs * ((actual - prediction) ** 2))

        return complexity(
            state_posterior, state_prior, gamma_obs, pi_prior
        ) - accuracy(actual, pi_obs, prediction)

    def beliefs_update(self, observations):
        self._last_inference_free_energy = {}
        layers_list = self.navigator.layers()

        for layer in layers_list:
            states_of_layer = self.navigator.states_by_layer(layer)
            F_layer_list = []
            pi_priors_list = []

            for state_name in states_of_layer:
                prepared = preprocessing.preprocessing_layer_for_free_energy(
                    state_name,
                    self.navigator,
                    self.functions_map,
                    self.inputs_map,
                    self.Q_,
                    self.pi_priors,
                    self.pi_obs,
                    observations,
                    self.states_list,
                    state_min=self.min_state_value,
                    state_max=self.max_state_value,
                    observation_bounds=self.observation_bounds
                )

                gamma_obs = torch.log(prepared["pi_obs"]).requires_grad_(True)
                F = self.free_energy(
                    prepared["state_prior"],
                    prepared["state_posterior"].requires_grad_(True),
                    prepared["f_transition"],
                    prepared["pi_prior"].requires_grad_(True),
                    prepared["actual"],
                    prepared["pi_obs"],
                    gamma_obs,
                    prepared["prediction"].requires_grad_(True)
                )

                try:
                    self._last_inference_free_energy[state_name] = float(
                        F.detach().cpu().item()
                    )
                except Exception:
                    self._last_inference_free_energy[state_name] = float("inf")

                motors = self.motor_command(state_name, F, prepared["prediction"])

                new_posterior = inference.euler_update(
                    prepared["state_posterior"],
                    0.1,
                    F,
                    max_grad_norm=self.max_grad_norm,
                )

                self.pi_obs = inference.update_pi_obs(
                    self.navigator,
                    self.pi_obs,
                    gamma_obs,
                    F,
                    self.Q_,
                    state_name,
                    min_precision=self.min_obs_precision,
                    max_precision=self.max_obs_precision,
                    dt=0.1,
                    lr_obs=self.lr_obs,
                )

                new_posterior_init_size = new_posterior.reshape(
                    prepared["state_posterior_init_size"].shape
                )
                state_posterior_init_size = prepared["state_posterior"].reshape(
                    prepared["state_posterior_init_size"].shape
                )

                new_precision_hessian = inference.second_derivative_F(
                    state_posterior_init_size, F
                )

                Q = inference.update_distributions(
                    new_precision_hessian.detach(),
                    new_posterior_init_size.detach(),
                    self.Q_[state_name],
                    min_state_value=self.min_state_value,
                    max_state_value=self.max_state_value,
                    min_precision=self.min_precision,
                    max_precision=self.max_precision,
                    eps=self.numerical_eps,
                )
                self.Q_[state_name] = Q

                inference.update_pi_prior(
                    self.pi_states,
                    self.pi_priors,
                    new_precision_hessian,
                    state_name,
                    min_precision=self.min_precision,
                    max_precision=self.max_precision,
                    eps=self.numerical_eps,
                )

                if f"pi_{state_name}" in self.inputs_map:
                    self.inputs_map[f"pi_{state_name}"] = self.pi_states[state_name]

                F_layer_list.append(F)
                pi_priors_list.append(prepared["pi_prior"])

                if motors:
                    yield motors

            F_layer = torch.stack(F_layer_list).sum()
            threshold = (1 / F_layer) < self.pi_threshold[layer]

            new_pi_threshold = inference.update_threshold(
                pi_priors_list, F_layer_list,
                self.pi_threshold[layer], dt=0.1, k=1.0
            )
            self.pi_threshold[layer] = new_pi_threshold.detach()

            if not threshold:
                break

        self.inputs_map = preprocessing.sync_state_inputs(
            self.D, self.Q_, self.inputs_map
        )

    def execepted_free_energy(self, pi_priors, variances, transition_cost):
        def excepted_ambiguity(pi_priors):
            gaussian_entropy = 0.5 * math.log(2 * math.e * math.pi)
            log_prior_precision = 0.5 * torch.log(pi_priors)
            return torch.sum(gaussian_entropy - log_prior_precision)

        def risk(variances, transition_cost):
            log_det = torch.logdet(2 * torch.pi * torch.e * variances)
            return torch.sum(0.5 * log_det - transition_cost)

        return excepted_ambiguity(pi_priors) + risk(variances, transition_cost)

    def policy_selection(self):
        i = 0
        while True:
            i += 1
            if i >= self.max_policies_iter:
                raise RuntimeError(
                    f"Policy selection exceeded the limit of "
                    f"{self.max_policies_iter} iterations. "
                    "Interrupted to prevent an infinite loop."
                )

            policies_candidats = model_components.get_policy_candidates(
                self.navigator,
                self.functions_map,
                self.inputs_map,
                self.states_list,
                self.Q_,
            )
            admited_candidats = []
            is_policy_finded = False

            for state_name, policy_values in policies_candidats.items():
                local_Q_ = copy.deepcopy(self.Q_)
                local_inputs_map = copy.deepcopy(self.inputs_map)

                for action_t in policy_values.policy.policy[0]:
                    if torch.sum(action_t) != 1:
                        break

                    state_parents, action_name = (
                        preprocessing.get_one_hot_action_parents(
                            action_t, self.actions_id, self.navigator
                        )
                    )

                    for parent_of_action in state_parents:
                        all_action_of_state = self.navigator.child_nodes(
                            parent_of_action, node_type="action"
                        )

                        if action_name not in all_action_of_state:
                            continue

                        local_inputs_map[parent_of_action].requires_grad_(True)
                        prepared = preprocessing.preprocessing_for_excepted_free_energy(
                            all_action_of_state,
                            action_name,
                            parent_of_action,
                            local_inputs_map,
                            self.pi_states,
                            self.navigator,
                            self.functions_map,
                            local_Q_,
                            self.states_list,
                            state_min=self.min_state_value,
                            state_max=self.max_state_value,
                            observation_bounds=self.observation_bounds,
                            reward_min=self.min_reward_value,
                            reward_max=self.max_reward_value,
                        )

                        G_posterior = self.execepted_free_energy(
                            prepared["pi_priors"],
                            prepared["co_variances"],
                            prepared["transition_cost"],
                        )

                        new_precision = inference.seconde_derivative_from_output(
                            prepared["temporal_pred"],
                            local_inputs_map[parent_of_action],
                            min_precision=self.min_precision,
                            max_precision=self.max_precision,
                            eps=self.numerical_eps,
                        )

                        Q = inference.update_distributions(
                            new_precision.detach(),
                            prepared["temporal_pred"].detach(),
                            local_Q_[parent_of_action],
                            min_state_value=self.min_state_value,
                            max_state_value=self.max_state_value,
                            min_precision=self.min_precision,
                            max_precision=self.max_precision,
                            eps=self.numerical_eps,
                        )
                        local_Q_[parent_of_action] = Q

                G_prior = policy_values.policy.G

                try:
                    pi_p = self.pi_p[state_name].requires_grad_(True)
                    error_eps = inference.policy_error(
                        G_prior,
                        G_posterior,
                        pi_p,
                        max_policy=self.max_policy_error,
                    )

                    if error_eps <= 1:
                        first_action = policy_values.policy.policy[0][0]
                        admited_candidats.append([first_action, G_posterior])
                    else:
                        print('error_eps : ', error_eps)
                        self.inputs_map[f"error_{state_name}"] = error_eps.detach()
                except UnboundLocalError:
                    pass

            # Update policy precision using the last evaluated policy,
            # as in the original implementation.
            try:
                next_pi_p = utils.update_wih_grad_desc(
                    pi_p, error_eps, dt=0.1, k=self.lr_policies
                )
                self.pi_p[state_name] = torch.exp(next_pi_p.detach())
            except UnboundLocalError:
                break

            if admited_candidats:
                best_candidat = min(admited_candidats, key=lambda item: item[1])
                is_policy_finded = True

            if is_policy_finded:
                action_group, new_action_oh = inference.update_current_action(
                    best_candidat, self.actions_id, self.navigator
                )
                self.inputs_map[action_group[0]] = new_action_oh
                self.inputs_map[f"error_{state_name}"] = (
                    self.pi_p[state_name].detach().item()
                )

                first_action = policy_values.policy.policy[0][0]
                _, first_action_name = preprocessing.get_one_hot_action_parents(
                    first_action, self.actions_id, self.navigator
                )

                return {
                    "action": first_action_name,
                    "G_prior": G_prior,
                    "G_posterior": G_posterior.detach().item(),
                }

    def motor_command(self, current_state, F, g_output):
        states = self.navigator.states_with_motors()
        if current_state not in states:
            return {}

        motors_name = self.navigator.motors_by_state(current_state)
        outputs_g_name = self.navigator.output_g(current_state)
        motors_indexes = [
            (motor, outputs_g_name.index(motor))
            for motor in motors_name
            if motor in outputs_g_name
        ]

        if not motors_indexes:
            return {}

        new_y = utils.update_wih_grad_desc(
            g_output, F, dt=0.1, k=self.lr_motors
        )
        return {
            name: new_y[index].detach().item()
            for name, index in motors_indexes
        }

    def training_offline(
        self,
        observations,
        config=None,
        *,
        test_observations=None,
        rebuild_callback=None,
        motor_fn=None,
        motor_train_fn=None,
    ):
        """Run one iteration of the three-macro-phase offline trainer."""
        from .offline_training.offline import OfflineTrainer

        trainer = getattr(self, "_offline_trainer", None)
        if trainer is None:
            trainer = OfflineTrainer(self, config=config)
            self._offline_trainer = trainer
        elif config is not None:
            from .offline_training.offline import OfflineConfig
            trainer.config = (
                config
                if isinstance(config, OfflineConfig)
                else OfflineConfig(**config)
            )
            trainer._validate_config()

        return trainer.step(
            observations,
            test_observations=test_observations,
            rebuild_callback=rebuild_callback,
            motor_fn=motor_fn,
            motor_train_fn=motor_train_fn,
        )

    def offline(self, observations, config=None, **kwargs):
        """Backward-compatible short alias for training_offline()."""
        return self.training_offline(observations, config=config, **kwargs)
