import torch
import utils, model, preprocessing, inference, model_components
import math, copy

# Deep Active Inference Hybride Framework with Dynamic Planning Horizon and Amortized Active Inference (DAIH-DynAmI)
class Agent_DAIH():
    def __init__(self):
        #super().__init__()
        self.D = None
    
        self.pi_obs = None

        self.covariances = None

        self.state_priors = None

        self.pi_priors = None

        self.Q_ = None

        self.graph = None
        self.navigator = None
        self.functions_map = None
        self.inputs_map = None

        self.learning_rate = 0.1
        self.num_iterations = 100
        self.tolerance_F = 1e-5
        self.tolerance_Q = 1e-5

        self.gamma = 1
        self.max_motor_iters = 100
        self.lr_motor = 0.1
        self.tolerance_F_motor = 1e-5
        self.tolerance_action = 1e-5

        self.pi_threshold = {}
        self.actions_id = {}
    
    
    def model(self, generative_functions=None, functions_map=None, observations=None, pi_obs=None, actions=None, pi_p=None):
        states = model.validate_model(self.D, generative_functions, actions)
        _, idx_to_actions = model.actions_index(actions)
        self.actions_id = idx_to_actions

        (
            self.mu,
            self.variance,
            self.covariances,
            self.state_priors,
            self.Q_,
            self.pi_priors,
        ) = model.init_distributions(self.D, states, self.pi_priors)

        self.graph, navigator, self.functions_map = model.init_graph(self.graph, generative_functions, functions_map, observations, actions)

        if navigator is not None:
            self.navigator = navigator

        self.inputs_map = preprocessing.sync_state_inputs(self.D, self.Q_, self.inputs_map)

        utils.init_actions(actions, self.inputs_map)

        self.pi_p, self.inputs_map = model.init_p(generative_functions, self.pi_priors,  pi_p, self.inputs_map)

        self.pi_obs = model.init_observations(states, pi_obs)

        self.pi_threshold = model.sum_priors(self.pi_priors, self.navigator)




    def inference(self, observations):
        for motors in self.beliefs_update(observations):
            yield f"action motor: {motors}"

        policy_selected = self.policy_selection()
        yield f"policy selected: {policy_selected}"


    # Continuous variational free energy
    def free_energy(self, state_prior, state_posterior, D_poresterior, pi_prior, actual, pi_obs, gamma_obs, prediction):
        def complexity(state_posterior, state_prior, gamma_obs, pi_prior):
            return torch.sum(0.5*pi_prior*((state_posterior-D_poresterior-state_prior) ** 2)) - \
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
                                                            self.graph, self.Q_, self.pi_priors, 
                                                            self.pi_obs, observations
                                                            )
                gamma_obs = torch.log(prepared["pi_obs"]).requires_grad_(True)
                
                # Free energy calcul
                F = self.free_energy(
                        prepared["state_prior"],
                        prepared["state_posterior"].requires_grad_(True),
                        prepared["D_posterior"],
                        prepared["pi_prior"].requires_grad_(True),
                        prepared["actual"],
                        prepared["pi_obs"],
                        gamma_obs,
                        prepared["prediction"].requires_grad_(True),
                    )

                # command motors
                motors = self.motor_command(state_name, F, prepared["prediction"])

                # mean update
                new_posterior = inference.euler_update(prepared["state_posterior"], 0.1, F)

                # Π_obs update
                new_gamma_obs = utils.update_wih_grad_desc(gamma_obs, F, dt=0.1, k=1.0)
                self.pi_obs[state_name] = torch.clamp(torch.exp(new_gamma_obs), min=1e-9).detach()

                # covariance update
                new_precision_hessian = inference.second_derivative_F(prepared["state_posterior"], F)

                # update beliefs
                Q = inference.update_distributions(new_precision_hessian.detach(), new_posterior.detach(), self.Q_[state_name], eps=1e-6)
                self.Q_[state_name] = Q

                # Π_prior update
                inference.update_pi_prior(self.pi_priors, new_precision_hessian, state_name, eps=1e-6)

                # F layer
                F_layer_list.append(F)
                pi_priors_list.append(prepared["pi_prior"])

                if motors:
                    yield motors

            # Π_threshold : 
            F_layer = torch.stack(F_layer_list).sum()
            threshold = (1/F_layer) < self.pi_threshold[layer]

            new_pi_threshold = inference.update_threshold(pi_priors_list, F_layer_list, self.pi_threshold[layer], dt=0.1, k=1.0)
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
            entropy_gaussian_excepted = 0.5*torch.log(2*torch.pi*torch.e*(variances ** 2))
            return torch.sum(entropy_gaussian_excepted - transition_cost)

        return excepted_ambiguity(pi_priors) + risk(variances, transition_cost)
    
    
    def policy_selection(self):
        i = 0
        while True:
            i += 1
            policies_candidats = model_components.get_policy_candidates(self.navigator, self.functions_map, self.inputs_map)
            admited_candidats = []
            is_policy_finded = False
            for state_name, policy in policies_candidats.items():
                local_Q_ = copy.deepcopy(self.Q_)
                local_inputs_map = copy.deepcopy(self.inputs_map)

                for action_t in policy[0]:
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
                                                                                        local_inputs_map,
                                                                            self.pi_priors, self.navigator, self.functions_map, local_Q_)


                        G_posterior = self.execepted_free_energy(prepared["pi_priors"], prepared["co_variances"], prepared["transition_cost"])

                        # belief projected update
                        new_precision = inference.jacobian_from_output(prepared["temporal_pred"], local_inputs_map[parent_of_action])
                        Q = inference.update_distributions(new_precision.detach(), prepared["temporal_pred"].detach(), local_Q_[parent_of_action], eps=1e-6)
                        local_Q_[parent_of_action] = Q

                # compare G_prior to G_posterior
                G_prior = policy[1]
                try:
                    pi_p = self.pi_p[state_name].requires_grad_(True)
                    error_eps = inference.policy_error(G_prior, G_posterior, pi_p)

                    if error_eps <= 1:
                        fist_action = policy[0][0]
                        admited_candidats.append([fist_action, G_posterior])
                    else: 
                        self.inputs_map[f"error_{state_name}"] = error_eps.detach()
                except UnboundLocalError:
                    pass

            # Update Π_P
            try:
                next_pi_p = utils.update_wih_grad_desc(pi_p, error_eps, dt=0.1, k=100)
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
                self.inputs_map[f"error_{state_name}"] = self.pi_p[state_name].detach()
                
                return  {"action": action_name,
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

        # Descente de gradient sur le vecteur complet grad descente on complet output_g
        new_y = utils.update_wih_grad_desc(g_output, F, dt=0.1, k=1.0)

        # Keep only the envolved motors
        return {
            name: new_y[index].detach().item()
            for name, index in motors_indexes
        }


    