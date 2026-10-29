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




def get_actual_precision(navigator, pi_obs, Q_, state):
    """
    Builds the precision vector following navigator.output_g(state).

    State outputs:
        Inverse marginal variances from Q_.

    Individual observation outputs:
        Use pi_obs[observation], or 1 if missing.

    Observation group outputs:
        Retrieve group observations and use their individual
        precisions from pi_obs, or 1 if missing.

    Unknown outputs are ignored.
    """

    dtype = torch.float32
    device = torch.device("cpu")
    precision_values = []

    outputs_g = navigator.output_g(state)

    for output in outputs_g:

        # --------------------------------------------------
        # 1. State output
        # --------------------------------------------------
        if output in Q_:

            for q in Q_[output]:

                if hasattr(q, "covariance_matrix"):
                    covariance = torch.as_tensor(
                        q.covariance_matrix,
                        dtype=dtype,
                        device=device
                    )

                    if covariance.ndim <= 1:
                        variances = covariance.reshape(-1)
                    else:
                        variances = torch.diagonal(covariance)

                elif hasattr(q, "scale"):
                    scale = torch.as_tensor(
                        q.scale,
                        dtype=dtype,
                        device=device
                    )
                    variances = scale.square().reshape(-1)

                else:
                    raise TypeError(
                        f"Cannot retrieve variances for '{output}'."
                    )

                if torch.any(variances <= 0):
                    raise ValueError(
                        f"Variances for '{output}' must be positive."
                    )

                precision_values.append(
                    variances.reciprocal().reshape(-1)
                )

            continue

        # --------------------------------------------------
        # 2. Individual observation
        # --------------------------------------------------
        try:
            output_type = navigator.type(output)
        except KeyError:
            output_type = None

        if output_type == "observation":
            observations = [output]

        else:
            # --------------------------------------------------
            # 3. Observation group
            # --------------------------------------------------
            observations = navigator.get_observations_by_group(output)

            if not observations:
                continue

        # --------------------------------------------------
        # 4. Retrieve each observation's precision
        # --------------------------------------------------
        for observation in observations:

            precision = (
                pi_obs.get(observation)
                if isinstance(pi_obs, dict)
                else None
            )

            if isinstance(precision, torch.Tensor):
                precision = precision.to(
                    dtype=dtype,
                    device=device
                ).reshape(-1)
            else:
                precision = torch.ones(
                    1,
                    dtype=dtype,
                    device=device
                )

            precision_values.append(precision)

    # --------------------------------------------------
    # 5. Concatenate
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