import torch
from . import utils, model, preprocessing, inference, model_components
from .offline_training.offline import OfflineTrainer, OfflineConfig
import math, copy
from .offline_training import persistence

# Deep Active Inference Hybride Framework with Dynamic Planning Horizon and Amortized Active Inference (DAIH-DynAmI)
class Agent_DAIH():
    def __init__(self):
        self.D = None
    
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

        self.offline = False

        # Original declarative model specification is retained so the
        # offline structural controller can prepare a safe rebuild.
        self.generative_functions = None
        self.observations = None
        self.actions = None
        self._offline_trainer = None
        # User-supplied factory for state/layer-specific local callables.
        # It is deliberately not serialized; load() receives it explicitly.
        self._local_function_factory = None
    
    
    def model(self, generative_functions=None, functions_map=None, observations=None, actions=None, pi_p=None):
        self.generative_functions = copy.deepcopy(generative_functions)
        self.observations = copy.deepcopy(observations or {})
        self.actions = copy.deepcopy(actions or {})
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

        self.graph, navigator, self.functions_map = model.init_graph(self.graph, generative_functions, functions_map, observations, actions)

        if navigator is not None:
            self.navigator = navigator

        self.pi_priors = model.init_relation_priors(self.navigator, self.pi_states)

        self.inputs_map = preprocessing.sync_state_inputs(self.D, self.Q_, self.inputs_map)

        utils.init_actions(actions, self.inputs_map)

        self.pi_p, self.inputs_map = model.init_p(generative_functions, self.pi_states, pi_p, self.inputs_map)

        self.pi_threshold = model.sum_priors(self.pi_states, self.navigator)





    def register_local_function_factory(self, factory):
        """Register the application callback used for local functions.

        Signature:
            factory(agent, state_name, state_id, layer_id, function_type,
                    function_name, declaration) -> callable | None

        The returned callable is registered on functions_map[function_name]
        when that entry is a WorldModel. Return None to use its global model.
        """
        if factory is not None and not callable(factory):
            raise TypeError("factory must be callable or None.")
        self._local_function_factory = factory
        return self

    def install_local_functions(self, state_names=None):
        """Install factory-provided local functions for selected states.

        Returns the installed (state_name, function_type, function_name) keys.
        """
        if self._local_function_factory is None:
            return []
        from .world_model import WorldModel
        names = list(state_names) if state_names is not None else list(self.generative_functions or {})
        installed = []
        function_types = ("f", "g", "p", "V")
        for state_name in names:
            declaration = self.generative_functions[state_name]
            state_id = self.states_list.index(state_name) + 1
            layer_id = int(declaration.get("layer", 0))
            for function_type in function_types:
                function_name = declaration.get(function_type)
                if not function_name or function_name not in self.functions_map:
                    continue
                module = self.functions_map[function_name]
                if not isinstance(module, WorldModel):
                    continue
                local_function = self._local_function_factory(
                    self, state_name, state_id, layer_id, function_type,
                    function_name, declaration
                )
                if local_function is None:
                    continue
                module.register_local_function(
                    state_id, layer_id, function_type, local_function
                )
                installed.append((state_name, function_type, function_name))
        return installed

    def save(self, path):
        """Save architecture, module weights, beliefs and precision tensors."""
        return persistence.save_agent(self, path)

    @classmethod
    def load(
        cls, path, *, functions_map_factory, local_functions_factory=None,
        map_location="cpu"
    ):
        """Load an agent using application-provided function factories.

        Only load checkpoints created by a trusted source.
        """
        return persistence.load_agent(
            cls, path, functions_map_factory=functions_map_factory,
            local_functions_factory=local_functions_factory,
            map_location=map_location
        )

    def inference(self, observations):
        #print("inside pi obs: ", self.pi_obs)
        (
            observations, 
            self.pi_obs, 
            self.pi_obs_substitued
        ) = inference.handle_missing_observations(observations, self.pi_obs, self.pi_obs_substitued)

        for motors in self.beliefs_update(observations):
            yield f"action motor: {motors}"

        policy_selected = self.policy_selection()
        yield f"policy selected: {policy_selected}"


    # Continuous variational free energy
    def free_energy(self, state_prior, state_posterior, D_poresterior, pi_prior, actual, pi_obs, gamma_obs, prediction):
        def complexity(state_posterior, state_prior, gamma_obs, pi_prior):
            return torch.sum(0.5*pi_prior*((state_posterior+D_poresterior-state_prior) ** 2)) - \
            torch.sum(0.5*gamma_obs) - torch.sum(0.5*torch.log(pi_prior))
        
        def accuracy(actual, pi_obs, prediction):
            return torch.sum(-0.5*pi_obs*((actual-prediction) ** 2))

        return  complexity(state_posterior, state_prior, gamma_obs, pi_prior) - accuracy(actual, pi_obs, prediction)

    def beliefs_update(self, observations):
        # Retrieve states level by level to start with level 1
        layers_list = self.navigator.layers()
        for layer in layers_list:
            states_of_layer = self.navigator.states_by_layer(layer)
            F_layer_list = []
            pi_priors_list = []
            for state_name in states_of_layer:
                # Data prprocessing
                prepared = preprocessing.preprocessing_layer_for_free_energy(state_name, 
                                                            self.navigator, self.functions_map, self.inputs_map, 
                                                            self.Q_, self.pi_priors, 
                                                            self.pi_obs, observations, self.states_list
                                                            )
                safe_pi_obs = torch.nan_to_num(
                    prepared["pi_obs"], nan=1.0, posinf=1e8, neginf=1e-9
                ).clamp(min=1e-9, max=1e8)
                gamma_obs = torch.log(safe_pi_obs).requires_grad_(True)
                
                # Free energy calcul
                F = self.free_energy(
                        prepared["state_prior"],
                        prepared["state_posterior"].requires_grad_(True),
                        prepared["f_transition"],
                        prepared["pi_prior"].requires_grad_(True),
                        prepared["actual"],
                        safe_pi_obs,
                        gamma_obs,
                        prepared["prediction"].requires_grad_(True),
                    )

                # Reject a numerically invalid state update without allowing a
                # NaN to contaminate Q_, precisions, or subsequent iterations.
                if not torch.isfinite(F).all():
                    if self.offline:
                        yield None, torch.zeros((), dtype=torch.float32), state_name
                    continue

                # command motors
                motors = self.motor_command(state_name, F, prepared["prediction"])

                # mean update
                offline_dt = getattr(self, "_offline_dt", 0.1) if self.offline else 0.1
                belief_clip = getattr(self, "_offline_belief_gradient_clip", 100.0)
                new_posterior = inference.euler_update(
                    prepared["state_posterior"], offline_dt, F,
                    max_grad_norm=belief_clip if self.offline else 100.0,
                )

                # Π_obs update
                self.pi_obs = inference.update_pi_obs(self.navigator, self.pi_obs, gamma_obs, F, self.Q_, state_name, 
                                                      dt=offline_dt, lr_obs=self.lr_obs
                                                    )
                # re-init original size
                new_posterior_init_size = new_posterior.reshape(prepared["state_posterior_init_size"].shape)
                state_posterior_init_size = prepared["state_posterior"].reshape(prepared["state_posterior_init_size"].shape)

                # covariance update
                new_precision_hessian = inference.second_derivative_F(state_posterior_init_size, F)

                # update beliefs
                Q = inference.update_distributions(new_precision_hessian.detach(), new_posterior_init_size.detach(), self.Q_[state_name], eps=1e-6)
                self.Q_[state_name] = Q

                # Π_prior update
                inference.update_pi_prior(self.pi_states, self.pi_priors, new_precision_hessian, state_name, eps=1e-6)
                if f"pi_{state_name}" in self.inputs_map:
                    self.inputs_map[f"pi_{state_name}"] = self.pi_states[state_name]

                # F layer
                F_layer_list.append(F)
                pi_priors_list.append(prepared["pi_prior"])

                if not self.offline:
                    if motors:
                        yield motors
                else:
                    if motors:
                        yield motors, F, state_name
                    else:
                        yield None, F, state_name

            # Π_threshold : a layer can be skipped if every state objective
            # was rejected because of non-finite numerical values.
            if not F_layer_list:
                continue
            F_layer = torch.stack(F_layer_list).sum()
            threshold = (1/F_layer) < self.pi_threshold[layer]

            new_pi_threshold = inference.update_threshold(pi_priors_list, F_layer_list, self.pi_threshold[layer], dt=offline_dt if self.offline else 0.1, k=1.0)
            self.pi_threshold[layer] = new_pi_threshold.detach()

            if not threshold:
                break

        # update for generative function inputs
        self.inputs_map = preprocessing.sync_state_inputs(self.D, self.Q_, self.inputs_map)

    
    def execepted_free_energy(self, pi_priors, variances, transition_cost):
        def excepted_ambiguity(pi_priors):
            gaussian_entropy = (0.5*math.log(2*math.e*math.pi))
            log_prior_precision = (0.5*torch.log(pi_priors))
            return torch.sum(gaussian_entropy-log_prior_precision)
        
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
                    f"Policy selection exceeded the limit of {self.max_policies_iter} "
                    f"iterations. Interrupted to prevent an infinite loop."
                )
            #print("self.inputs_map : ", self.inputs_map)
            policies_candidats = model_components.get_policy_candidates(self.navigator, self.functions_map, self.inputs_map, self.states_list, self.Q_)
            admited_candidats = []
            #print("policies_candidats: ", policies_candidats["life_expectancy"])
            is_policy_finded = False
            for state_name, policy_values in policies_candidats.items():
                local_Q_ = copy.deepcopy(self.Q_)
                local_inputs_map = copy.deepcopy(self.inputs_map)

                for action_t in policy_values.policy.policy[0]:
                    #print("action_t : ", action_t)
                    if torch.sum(action_t) != 1:
                        break
                    # get all state that depends on this one-hot action "action_t"
                    state_parents, action_name = preprocessing.get_one_hot_action_parents(action_t, self.actions_id, self.navigator)

                    for parent_of_action in state_parents:

                        # get all action which "parent_of_action" state depends on
                        all_action_of_state = self.navigator.child_nodes(parent_of_action, node_type="action")

                        # pass state that not depends on "action_name" 
                        if action_name not in all_action_of_state:
                            continue

                        local_inputs_map[parent_of_action].requires_grad_(True)
                        # get all actions vectors for "parent_of_action" function f inputs
                        prepared = preprocessing.preprocessing_for_excepted_free_energy(all_action_of_state, action_name, parent_of_action, 
                                                                                        local_inputs_map, self.pi_states, self.navigator, 
                                                                                        self.functions_map, local_Q_, self.states_list
                                                                            )


                        G_posterior = self.execepted_free_energy(prepared["pi_priors"], prepared["co_variances"], prepared["transition_cost"])

                        # belief projected update
                        new_precision = inference.seconde_derivative_from_output(prepared["temporal_pred"], local_inputs_map[parent_of_action])
                        Q = inference.update_distributions(new_precision.detach(), prepared["temporal_pred"].detach(), local_Q_[parent_of_action], eps=1e-6)
                        local_Q_[parent_of_action] = Q


                # compare G_prior to G_posterior
                G_prior = policy_values.policy.G
                #print("G_prior: ", G_prior)
                #print("G_posterior: ", G_posterior)
                try:
                    pi_p = self.pi_p[state_name].requires_grad_(True)
                    #print("pi_p: ", pi_p)
                    error_eps = inference.policy_error(G_prior, G_posterior, pi_p)
                    #print("error_eps: ", error_eps)

                    if error_eps <= 1:
                        fist_action = policy_values.policy.policy[0][0]
                        admited_candidats.append([fist_action, G_posterior])
                    else: 
                        self.inputs_map[f"error_{state_name}"] = torch.log(torch.tensor(1+1e-7+error_eps.detach().item()))
                except UnboundLocalError:
                    pass

            # Update Π_P
            try:
                next_pi_p = utils.update_wih_grad_desc(pi_p, error_eps, dt=0.1, k=self.lr_policies)
                self.pi_p[state_name] = torch.clamp(next_pi_p, min=1e-9, max=1e+6).detach()
            except UnboundLocalError:
                print("None policies")
                break

            # Sort the best policy candidat or update error_eps
            if admited_candidats:
                best_candidat = min(admited_candidats, key=lambda x: x[1])
                is_policy_finded = True 

            # Update the new action and init error for next time step
            if is_policy_finded:
                action_group, new_action_oh = inference.update_current_action(best_candidat, self.actions_id, self.navigator)
                self.inputs_map[action_group[0]] = new_action_oh
                self.inputs_map[f"error_{state_name}"] = self.pi_p[state_name].detach().item()

                # get first action name
                a_one_hot = policy_values.policy.policy[0][0] # fist action one-hot
                #print("a_one_hot: ", a_one_hot)
                #print("self.actions_id: ", self.actions_id)
                _, first_action_name = preprocessing.get_one_hot_action_parents(a_one_hot, self.actions_id, self.navigator)

                return  {"action": first_action_name, # fist action name
                            "G_prior": G_prior,
                            "G_posterior": G_posterior.detach().item()
                        }

    
    def motor_command(self, current_state, F, g_output):
        states = self.navigator.states_with_motors()

        if current_state not in states:
            return {}

        motors_name = self.navigator.motors_by_state(current_state)
        outputs_g_name = self.navigator.output_g(current_state)

        # Map each motor to its output index
        motors_indexes = [
            (m, outputs_g_name.index(m))
            for m in motors_name
            if m in outputs_g_name
        ]

        if not motors_indexes:
            return {}

        # grad descente on complet output_g
        if not torch.isfinite(F).all() or not torch.isfinite(g_output).all():
            return {}
        motor_dt = getattr(self, "_offline_dt", 0.1) if self.offline else 0.1
        new_y = utils.update_wih_grad_desc(g_output, F, dt=motor_dt, k=self.lr_motors)
        if not torch.isfinite(new_y).all():
            return {}

        # Keep only the envolved motors 
        return {
            name: new_y[index].detach().item()
            for name, index in motors_indexes
        }

    def training_offline(
        self,
        observations,
        *,
        test_observations=None,
        config=None,
        rebuild_callback=None,
    ):
        """
        Run exactly one global offline-learning iteration.

        Parameters
        ----------
        observations:
            Historical observations used for this training iteration.
        test_observations:
            Optional held-out observations used to compare global free
            energy before and after the iteration. Defaults to observations.
        config:
            OfflineConfig instance or a dictionary of OfflineConfig values.
        rebuild_callback:
            Optional callback that applies the proposed structural plan and
            returns a new model specification. See README_OFFLINE.md.

        Returns
        -------
        dict
            Metrics, per-state phase decisions, proposed mutations and paths.
        """
        if self._offline_trainer is None:
            self._offline_trainer = OfflineTrainer(self, config=config)
        elif config is not None:
            # Reusing the same config on every call must NOT reset phase
            # histories or the global iteration counter.

            self._offline_trainer.config = (
                config if isinstance(config, OfflineConfig)
                else OfflineConfig(**config)
            )

        return self._offline_trainer.step(
            observations,
            test_observations=test_observations,
            rebuild_callback=rebuild_callback,
        )
