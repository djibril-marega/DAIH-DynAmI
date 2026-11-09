from torch.func import jacrev
from . import preprocessing
import torch

def get_state_prediction(navigator, functions_map, inputs_map, state, states_list, Q_, Jacobian=None):
    """
    Computes g as a flattened numerical vector.

    Named outputs remain ordered by output_g. Optional masks are metadata and
    are not concatenated into the numerical prediction vector.
    """
    g_function_name = navigator.function(state, "g")
    if g_function_name is None:
        raise ValueError(f"No g function is defined for state '{state}'.")

    input_names = preprocessing.get_io_names(navigator, state, "g", "input")
    dict_inputs = preprocessing.prepare_inputs(input_names, inputs_map)

    def flattened_g(inputs):
        return preprocessing.execute_flat_function(
            navigator, functions_map, inputs, state, "g", states_list, "g", Q_
        )

    predictions = flattened_g(dict_inputs)

    if Jacobian:
        J = jacrev(flattened_g)(dict_inputs)
        return predictions, J

    return predictions


def get_actual(navigator, Q_, state, observations):
    """
    Retrieves actual child values in the exact order declared by output_g.

    State outputs contribute every Q_[state].loc value.
    Individual observations and observation groups contribute their observed
    values in the order returned by the navigator.
    """
    actual = []

    outputs_g = (
        navigator.output_g(state)
        if hasattr(navigator, "output_g")
        else [child["node"] for child in navigator.children(state)]
    )

    for output in outputs_g:
        # State output
        if output in Q_:
            for q in Q_[output]:
                value = q.loc
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value, dtype=torch.float32)
                else:
                    value = value.to(dtype=torch.float32)
                actual.append(value.reshape(-1))
            continue

        # Individual observation or observation group
        try:
            output_type = navigator.type(output)
        except (KeyError, TypeError, AttributeError):
            output_type = None

        if output_type == "observation":
            observation_names = [output]
        else:
            observation_names = navigator.get_observations_by_group(output)

        if not observation_names:
            raise KeyError(
                f"Output '{output}' declared in output_g is neither a "
                "known state nor an observation or observation group."
            )

        for observation_name in observation_names:
            observation_key = next(
                (
                    key for key, variables in observations.items()
                    if isinstance(variables, dict) and observation_name in variables
                ),
                None,
            )
            if observation_key is None:
                raise KeyError(
                    f"Observation '{observation_name}' is missing from "
                    "the observations dictionary."
                )

            value = observations[observation_key][observation_name]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, dtype=torch.float32)
            else:
                value = value.to(dtype=torch.float32)
            actual.append(value.reshape(-1))

    if not actual:
        return torch.empty(0, dtype=torch.float32)

    return torch.cat(actual)


def get_actual_precision(navigator, pi_obs, Q_, state):
    """
    Builds precision values in output_g order.

    State outputs use inverse marginal variances from Q_. Individual
    observations and observation groups use pi_obs, defaulting to 1.
    """
    dtype = torch.float32
    device = torch.device("cpu")
    precision_values = []

    outputs_g = navigator.output_g(state)

    for output in outputs_g:
        if output in Q_:
            for q in Q_[output]:
                if hasattr(q, "covariance_matrix"):
                    covariance = torch.as_tensor(
                        q.covariance_matrix, dtype=dtype, device=device
                    )
                    variances = (
                        covariance.reshape(-1)
                        if covariance.ndim <= 1
                        else torch.diagonal(covariance)
                    )
                elif hasattr(q, "scale"):
                    scale = torch.as_tensor(
                        q.scale, dtype=dtype, device=device
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
                precision_values.append(variances.reciprocal().reshape(-1))
            continue

        try:
            output_type = navigator.type(output)
        except KeyError:
            output_type = None

        if output_type == "observation":
            observation_names = [output]
        else:
            observation_names = navigator.get_observations_by_group(output)

        if not observation_names:
            raise KeyError(
                f"Output '{output}' is neither a state, an observation, "
                "nor a valid observation group."
            )

        for observation in observation_names:
            precision = (
                pi_obs.get(observation)
                if isinstance(pi_obs, dict)
                else None
            )
            if isinstance(precision, torch.Tensor):
                precision = precision.to(dtype=dtype, device=device).reshape(-1)
            else:
                precision = torch.ones(1, dtype=dtype, device=device)
            precision_values.append(precision)

    if not precision_values:
        return torch.empty(0, dtype=dtype, device=device)

    return torch.cat(precision_values)


def get_transition_prediction(
    navigator, functions_map, inputs_map, state, states_list, Q_,
    recursive_input=None, policy=None
):
    """
    Executes f and returns a 1D vector, with outputs flattened and
    concatenated in output_f order.
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

    if policy:
        return next(iter(preprocessing.execute_named_function(
            navigator, functions_map, local_inputs, state, "f",
            states_list, "f", Q_
        ).values()))
    return preprocessing.execute_flat_function(
        navigator, functions_map, local_inputs, state, "f",
        states_list, "f", Q_
    )


def get_policy_candidates(navigator, functions_map, inputs_map, state_list, Q_):
    policy_candidates = {}
    for state, node in navigator.graph.items():
        if node.get("type") != "state":
            continue
        if navigator.function(state, "p") is None:
            continue
        outputs = preprocessing.execute_named_function(
            navigator, functions_map, inputs_map, state, "p", state_list, "p", Q_
        )
        policy_candidates[state] = outputs
    return policy_candidates


def get_transition_cost(navigator, functions_map, inputs_map, state, states_list, Q_):
    """
    Executes V and returns a 1D vector, with outputs flattened and
    concatenated in output_V order. Returns None if no V is defined.
    """
    if navigator.function(state, "V") is None:
        return None
    return preprocessing.execute_flat_function(
        navigator, functions_map, inputs_map, state, "V", states_list, "V", Q_
    )
