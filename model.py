import torch
import torch.distributions as dist
import graph 


def sum_priors(pi_priors, navigator):
    dict_sum_per_layer = {}

    for layer in navigator.layers():
        states_of_layer = navigator.states_by_layer(layer)

        total = sum(
            pi_priors[state].sum()
            for state in states_of_layer
            if state in pi_priors
        )

        dict_sum_per_layer[layer] = total

    return dict_sum_per_layer




def validate_model(D, generative_functions, actions=None):
    """Validates the functions and hierarchical dependencies."""

    if D is None:
        raise ValueError("D must be defined.")

    if generative_functions is None:
        raise ValueError("generative_functions must be defined.")

    actions = actions or {}
    states = list(generative_functions.keys())

    if set(D.keys()) != set(states):
        missing = set(states) - set(D.keys())
        extra = set(D.keys()) - set(states)

        raise ValueError(
            "D and generative_functions must contain "
            "the same states. "
            f"Missing states in D: {missing}. "
            f"Extra states in D: {extra}."
        )

    # Index individual actions and their groups
    action_names = {
        action_name
        for action_list in actions.values()
        for action_name in action_list
    }
    action_groups = set(actions.keys())

    # Validate f, g, and p for each state
    for state_name, state in generative_functions.items():

        # f is mandatory
        for key in ("f", "input_f", "output_f"):
            if state.get(key) is None:
                raise ValueError(
                    f"State '{state_name}' must define '{key}'."
                )

        # g is mandatory for each state
        for key in ("g", "input_g"):
            if state.get(key) is None:
                raise ValueError(
                    f"State '{state_name}' must define '{key}'."
                )

        output_g = state.get("output_g", state.get("outpu_g"))

        if output_g is None:
            raise ValueError(
                f"State '{state_name}' must define 'output_g'."
            )

        # V is generally optional, but its metadata is
        # mandatory whenever the function is declared.
        if state.get("V") is not None:
            for key in ("input_V", "output_V"):
                if state.get(key) is None:
                    raise ValueError(
                        f"State '{state_name}' defines V, "
                        f"but '{key}' is missing."
                    )

        # p is optional, but its metadata is mandatory
        # whenever the function is declared.
        if state.get("p") is not None:
            for key in ("input_p", "output_p"):
                if state.get(key) is None:
                    raise ValueError(
                        f"State '{state_name}' defines p, "
                        f"but '{key}' is missing."
                    )

        # Check hierarchical references
        for parent_name in state.get("parents") or []:
            if parent_name not in generative_functions:
                raise KeyError(
                    f"Parent '{parent_name}' of '{state_name}' "
                    "does not exist."
                )

    # Identify states that have actions as children.
    # In the graph, these children are declared via input_f.
    action_child_states = set()

    for state_name, state in generative_functions.items():
        if any(
            input_name in action_names or input_name in action_groups
            for input_name in (state.get("input_f") or [])
        ):
            action_child_states.add(state_name)

    # V is mandatory for every state that has action children.
    for state_name in action_child_states:
        state = generative_functions[state_name]
        if state.get("V") is None:
            raise ValueError(
                f"State '{state_name}' must define a V function, "
                "because it has actions as children."
            )
        for key in ("input_V", "output_V"):
            if state.get(key) is None:
                raise ValueError(
                    f"State '{state_name}' must define '{key}', "
                    "because it has actions as children."
                )

    # Identify states whose f receives an action (used for p).
    action_dependent_states = set()

    for state_name, state in generative_functions.items():
        if any(
            input_name in action_names
            or input_name in action_groups
            for input_name in (state.get("input_f") or [])
        ):
            action_dependent_states.add(state_name)

    # Recursively traverse all ancestors
    required_p_states = set()
    visited = set()

    def collect_ancestors(state_name):
        if state_name in visited:
            return

        visited.add(state_name)

        for parent_name in (
            generative_functions[state_name].get("parents") or []
        ):
            required_p_states.add(parent_name)
            collect_ancestors(parent_name)

    for state_name in action_dependent_states:
        collect_ancestors(state_name)

    # All relevant ancestors must define p
    for state_name in required_p_states:
        if generative_functions[state_name].get("p") is None:
            raise ValueError(
                f"State '{state_name}' must define a p function, "
                "because it is an ancestor of a state whose input_f "
                "contains an action."
            )

    return states





def make_distributions(mu, variance=None, covariance=None):
    """Builds distributions according to the shape of mu."""

    mu_t = torch.as_tensor(mu, dtype=torch.float32)

    if mu_t.ndim > 2:
        raise ValueError(
            "mu must be a scalar, a 1D list, "
            "or a 2D list."
        )

    if not torch.isfinite(mu_t).all():
        raise ValueError("mu contains non-finite values.")

    # Covariance is only valid for multivariate distributions
    if covariance is not None and mu_t.ndim != 2:
        raise ValueError(
            "covariance can only be defined "
            "for a 2D mu (MultivariateNormal)."
        )

    # Scalar and 1D distributions
    if mu_t.ndim <= 1:

        var_t = torch.as_tensor(
            1 if variance is None else variance,
            dtype=torch.float32,
        )

        if mu_t.shape != var_t.shape:
            raise ValueError(
                "mu and variance must have exactly "
                "the same shape."
            )

        if not torch.isfinite(var_t).all() or (var_t <= 0).any():
            raise ValueError(
                "All variances must be finite "
                "and strictly positive."
            )

        if mu_t.ndim == 0:
            return [
                dist.Normal(mu_t, torch.sqrt(var_t))
            ]

        return [
            dist.Normal(mu_t[i], torch.sqrt(var_t[i]))
            for i in range(mu_t.shape[0])
        ]

    # Multivariate distributions
    n_distributions, dimension = mu_t.shape

    if covariance is not None:

        cov_t = torch.as_tensor(
            covariance,
            dtype=torch.float32,
        )

        expected_shape = (
            n_distributions,
            dimension,
            dimension,
        )

        if tuple(cov_t.shape) != expected_shape:
            raise ValueError(
                f"covariance must have shape {expected_shape}, "
                f"received: {tuple(cov_t.shape)}."
            )

        if not torch.isfinite(cov_t).all():
            raise ValueError(
                "covariance contains non-finite values."
            )

        # Symmetry validation
        if not torch.allclose(
            cov_t,
            cov_t.transpose(-1, -2),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(
                "covariance matrices must be symmetric."
            )

        # Extract variances
        var_t = torch.diagonal(
            cov_t,
            dim1=-2,
            dim2=-1,
        )

        # Optional variance consistency validation
        if variance is not None:

            provided_var = torch.as_tensor(
                variance,
                dtype=torch.float32,
            )

            if provided_var.shape != mu_t.shape:
                raise ValueError(
                    "variance must have the same shape as mu."
                )

            if not torch.allclose(
                var_t,
                provided_var,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    "The diagonal of covariance must match "
                    "the variance values."
                )

    else:

        var_t = torch.as_tensor(
            1 if variance is None else variance,
            dtype=torch.float32,
        )

        if mu_t.shape != var_t.shape:
            raise ValueError(
                "mu and variance must have exactly "
                "the same shape."
            )

        cov_t = torch.diag_embed(var_t)

    # Variance validation
    if not torch.isfinite(var_t).all() or (var_t <= 0).any():
        raise ValueError(
            "All variances must be finite "
            "and strictly positive."
        )

    distributions = []

    for i in range(n_distributions):

        try:
            # Cholesky decomposition also validates
            # positive definiteness.
            torch.linalg.cholesky(cov_t[i])

            distributions.append(
                dist.MultivariateNormal(
                    loc=mu_t[i],
                    covariance_matrix=cov_t[i],
                )
            )

        except (ValueError, RuntimeError) as exc:
            raise ValueError(
                f"covariance[{i}] must be positive definite."
            ) from exc

    return distributions




def inverse_variance(value):
    """Computes the element-wise inverse of variances."""

    value_t = torch.as_tensor(
        value,
        dtype=torch.float32,
    )

    if not torch.isfinite(value_t).all() or (value_t <= 0).any():
        raise ValueError(
            "All variances must be finite "
            "and strictly positive."
        )

    result = 1.0 / value_t

    if result.ndim == 0:
        return result.item()

    return result


def init_distributions(D, states, pi_priors=None):
    """Initializes distributions and state parameters."""

    mu_dict = {}
    variance_dict = {}
    covariances = {}
    state_priors = {}
    Q_ = {}

    pi_priors = {} if pi_priors is None else dict(pi_priors)

    for state_name in states:

        params = D[state_name]

        mu = params.get("mu", 1)
        variance = params.get("variance")
        covariance = params.get("covariance")

        # Build distributions and validate parameters
        Q_[state_name] = make_distributions(
            mu,
            variance,
            covariance,
        )

        mu_t = torch.as_tensor(
            mu,
            dtype=torch.float32,
        )

        # Recover effective variance
        if covariance is not None:

            cov_t = torch.as_tensor(
                covariance,
                dtype=torch.float32,
            )

            variance_t = torch.diagonal(
                cov_t,
                dim1=-2,
                dim2=-1,
            )

            covariances[state_name] = cov_t

        else:

            variance_t = torch.as_tensor(
                1 if variance is None else variance,
                dtype=torch.float32,
            )

            # Store the equivalent covariance matrix
            if mu_t.ndim == 2:
                covariances[state_name] = torch.diag_embed(
                    variance_t
                )
            else:
                covariances[state_name] = None

        mu_dict[state_name] = mu_t
        variance_dict[state_name] = variance_t

        # Preserve aliases
        state_priors[state_name] = mu_t

        # Initialize priors if not provided
        if state_name not in pi_priors:
            pi_priors[state_name] = torch.as_tensor(
                inverse_variance(variance_t),
                dtype=torch.float32,
            )

    return (
        mu_dict,
        variance_dict,
        covariances,
        state_priors,
        Q_,
        pi_priors,
    )


def init_observations(states, pi_obs=None):
    """Initializes and validates observation precisions."""

    pi_obs = {} if pi_obs is None else pi_obs

    invalid_states = set(pi_obs.keys()) - set(states)

    if invalid_states:
        raise ValueError(
            "The following states in pi_obs are not "
            "declared in generative_functions: "
            f"{invalid_states}"
        )

    return {
        state: pi_obs.get(state, 1)
        for state in states
    }


def init_graph(current_graph, generative_functions, functions_map, observations, actions):
    """Builds the graph if necessary."""

    if current_graph is None:
        current_graph = graph.build_graph(
            generative_functions,
            observations,
            actions,
        )

        navigator = graph.GraphNavigator(current_graph)
    else:
        navigator = None

    return current_graph, navigator, functions_map



def init_p(generative_functions, pi_priors, pi_p=None, inputs_map=None):
    """Initializes the precision and error of p functions."""

    pi_p = {} if pi_p is None else dict(pi_p)
    inputs_map = {} if inputs_map is None else inputs_map

    for state_name, state in generative_functions.items():

        # p is optional: skip states that do not define it
        if state.get("p") is None:
            continue

        # Default precision is 1
        precision = pi_p.setdefault(state_name, torch.tensor(1, dtype=torch.float32))

        # Initialize the inputs associated with this state
        inputs_map[f"pi_p_{state_name}"] = precision
        inputs_map[f"error_{state_name}"] = precision
        inputs_map[f"pi_{state_name}"] = pi_priors[state_name]

    return pi_p, inputs_map


def actions_index(actions_dict):
    # 1. Create a list of all unique individual actions in order
    list_actions = []
    for actions in actions_dict.values():
        for action in actions:
            if action not in list_actions:
                list_actions.append(action)
    
    # 2. Generate the dictionary {action: index}
    # Example: {'forward': 0, 'back': 1, 'stay still': 2, 'yes': 3, 'no': 4}
    action_to_idx = {action: i for i, action in enumerate(list_actions)}
    
    # 3. Generate the reverse dictionary {index: action}
    # Example: {0: 'forward', 1: 'back', ...}
    idx_to_action = {i: action for action, i in action_to_idx.items()}
    
    return action_to_idx, idx_to_action