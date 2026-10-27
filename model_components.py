from torch.func import jacrev
import preprocessing
import torch

def get_state_prediction(navigator, functions_map, inputs_map, state, Jacobian=None):
    """
    Computes the prediction g of a state as a flattened vector.

    The association of outputs follows output_g. Flattening is performed
    here only because this function produces a numerical vector.
    If Jacobian=True, returns (predicted_vector, Jacobian).
    """
    g_function_name = navigator.function(state, "g")
    if g_function_name is None:
        raise ValueError(f"No g function is defined for state '{state}'.")

    input_names = preprocessing.get_io_names(navigator, state, "g", "input")
    dict_inputs = preprocessing.prepare_inputs(input_names, inputs_map)

    def flattened_g(inputs):
        return preprocessing.execute_flat_function(
            navigator, functions_map, inputs, state, "g"
        )

    predictions = flattened_g(dict_inputs)

    if Jacobian:
        J = jacrev(flattened_g)(dict_inputs)
        return predictions, J

    return predictions




def get_actual(navigator, Q_, state, observations):
    """
    Retrieves the actual values of the children of a state.

    - Child of type 'state': retrieves the Q.loc values of its distributions.
    - Child of type 'observation': retrieves its observed value.
    - If both types exist, retrieves both.

    Returns
    -------
    torch.Tensor
        Vector containing the actual values of all children.
    """

    import torch

    actual = []

    children = navigator.children(state)
    for child in children:
        child_node = child["node"]
        node_type = navigator.type(child_node)
        # 1. Child corresponding to a state
        if node_type == "state":
            if child_node not in Q_:
                raise KeyError(
                    f"No Q_ distribution is defined for state "
                    f"'{child_node}'."
                )

            # Q_[child_node] contains a list of distributions
            for q in Q_[child_node]:
                value = q.loc

                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(
                        value, dtype=torch.float32
                    )
                else:
                    value = value.to(dtype=torch.float32)

                actual.append(value.reshape(-1))

        # 2. Child corresponding to an observation
        elif node_type == "observation":
            #print("child_node:", child_node)
            #print("observations:", observations)

            # Find the category containing this observation
            observation_key = next(
                (
                    key
                    for key, variables in observations.items()
                    if isinstance(variables, dict) and child_node in variables
                ),
                None
            )

            if observation_key is None:
                raise KeyError(
                    f"Observation '{child_node}' is missing from the observations dictionary."
                )

            # Retrieve the actual value
            value = observations[observation_key][child_node]
            # Convert to PyTorch if it is not already a tensor
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(
                    value, dtype=torch.float32
                )
            else:
                value = value.to(dtype=torch.float32)
            
            actual.append(value.reshape(-1))

    # 3. Concatenate all values
    if not actual:
        return torch.empty(0, dtype=torch.float32)

    return torch.cat(actual)




def get_actual_precision(navigator, pi_obs, graph, Q_, state):
    """
    Builds a flattened precision vector for the children of a state.

    - Child of type state: individual precision of each dimension
      of each Q_ distribution (inverse of the variance).
    - Child of type observation: all observations in the same
      group share the same precision.
    - The order of elements follows the order of the graph's children.

    Returns
    -------
    torch.Tensor
        1D precision vector.
    """

    dtype = torch.float32
    device = torch.device("cpu")
    precision_values = []

    children = navigator.children(state)
    i = 0

    while i < len(children):
        child_node = children[i]["node"]
        node_type = navigator.type(child_node)

        # --------------------------------------------------
        # 1. Child corresponding to a state
        # --------------------------------------------------
        if node_type == "state":

            if child_node not in Q_:
                raise KeyError(
                    f"No Q_ distribution is defined for '{child_node}'."
                )

            for q in Q_[child_node]:

                # Case of a full covariance matrix:
                # extract only the marginal variances.
                if hasattr(q, "covariance_matrix"):
                    covariance = torch.as_tensor(
                        q.covariance_matrix,
                        dtype=dtype,
                        device=device
                    )

                    if covariance.ndim == 0:
                        variances = covariance.reshape(1)
                    elif covariance.ndim == 1:
                        variances = covariance
                    else:
                        variances = torch.diagonal(covariance)

                # Case where standard deviations are accessible via scale.
                elif hasattr(q, "scale"):
                    scale = torch.as_tensor(
                        q.scale,
                        dtype=dtype,
                        device=device
                    ).reshape(-1)

                    variances = scale.square()

                else:
                    raise TypeError(
                        f"Unable to retrieve the variances "
                        f"of the distribution for '{child_node}'."
                    )

                if torch.any(variances <= 0):
                    raise ValueError(
                        f"The variances of '{child_node}' "
                        "must be strictly positive."
                    )

                # One precision value per dimension, without a matrix.
                precision_values.append(variances.reciprocal().reshape(-1))

            i += 1

        # --------------------------------------------------
        # 2. Child corresponding to an observation
        # --------------------------------------------------
        elif node_type == "observation":

            group = graph[child_node].get("group", child_node)
            group_children = []

            # Collect consecutive observations belonging to the same group.
            while i < len(children):
                candidate = children[i]["node"]

                if navigator.type(candidate) != "observation":
                    break

                candidate_group = graph[candidate].get(
                    "group", candidate
                )

                if candidate_group != group:
                    break

                group_children.append(candidate)
                i += 1

            # Retrieve the group's precision.
            n = len(group_children)

            if isinstance(pi_obs, dict) and group in pi_obs:
                precision_value = torch.as_tensor(
                    pi_obs[group],
                    dtype=dtype,
                    device=device
                )

                if precision_value.numel() == n:
                    # The vector already has the correct size: keep it as is.
                    precision_values.append(precision_value.reshape(-1))

                elif precision_value.numel() == 1:
                    # Scalar: repeat it for each observation.
                    precision_values.append(
                        precision_value.reshape(()).expand(n).clone()
                    )

                else:
                    raise ValueError(
                        f"Group '{group}' contains {n} observations, "
                        f"but pi_obs contains {precision_value.numel()} elements."
                    )

            else:
                # No precision specified: default precision is 1.
                precision_values.append(
                    torch.ones(n, dtype=dtype, device=device)
                )

        else:
            # Skip actions and other node types.
            i += 1

    # --------------------------------------------------
    # 3. Concatenate all precision values into a 1D vector
    # --------------------------------------------------
    if not precision_values:
        return torch.empty(0, dtype=dtype, device=device)

    return torch.cat(precision_values)





def get_futur_state_reward(navigator, functions_map, inputs_map, state):
    """
    Executes V and returns a 1D vector, with outputs flattened
    and concatenated in the order declared by output_V.
    """
    if navigator.function(state, "V") is None:
        return None

    return preprocessing.execute_flat_function(
        navigator, functions_map, inputs_map, state, "V"
    )


def get_transition_prediction(
    navigator, functions_map, inputs_map, state, recursive_input=None
):
    """
    Executes f and returns a 1D vector, with outputs flattened
    and concatenated in the order declared by output_f.

    When provided, recursive_input replaces the input named after
    the state before the arguments are prepared. inputs_map is not modified.
    """
    local_inputs = dict(inputs_map)
    if recursive_input is not None:
        has_values = (
            recursive_input.numel() > 0
            if isinstance(recursive_input, torch.Tensor)
            else len(recursive_input) > 0
            if hasattr(recursive_input, "__len__")
            else True
        )
        if has_values:
            local_inputs[state] = recursive_input

    if navigator.function(state, "f") is None:
        return None

    return preprocessing.execute_flat_function(
        navigator, functions_map, local_inputs, state, "f"
    )


def get_policy_variance(navigator, functions_map, inputs_map, state, std, obs_variance):
    _, J = get_state_prediction(
        navigator, functions_map, inputs_map, state, Jacobian=True
    )

    # jacrev of a vector output with a dictionary of inputs returns a
    # dictionary keyed by input name. Flatten each input's Jacobian block.
    J_matrix = torch.cat([
        value.reshape(-1) for value in J.values()
    ]).reshape(1, -1)

    J_transposed = J_matrix.T
    return torch.sum((J_matrix * (std ** 2)) * J_transposed + obs_variance)


def get_policy_candidates(navigator, functions_map, inputs_map):
    policy_candidates = {}

    for state, node in navigator.graph.items():
        if node.get("type") != "state":
            continue
        if navigator.function(state, "p") is None:
            continue

        outputs = preprocessing.execute_named_function(
            navigator, functions_map, inputs_map, state, "p"
        )

        policy_candidates[state] = next(iter(outputs.values()))

    return policy_candidates


def get_transition_cost(navigator, functions_map, inputs_map, state):
    """
    Executes V and returns a 1D vector, with outputs flattened
    and concatenated in the order declared by output_V.
    Returns None if no V function is defined.
    """
    if navigator.function(state, "V") is None:
        return None

    return preprocessing.execute_flat_function(
        navigator, functions_map, inputs_map, state, "V"
    )