import model_components, utils
import torch
from collections.abc import Mapping

def preprocessing_layer_for_free_energy(state_name, navigator, functions_map, inputs_map, Q_, pi_priors, pi_obs, observations):
    # 2. Update state by state
    #  - state_prior = output of function f for the state (must have the same size as Q)
    state_prior = model_components.get_transition_prediction(navigator, functions_map, inputs_map, state_name)
    #  - state_posterior = Q of the function
    state_posterior = inputs_map[state_name]
    #  - D_posterior = shifted Q (takes Q, removes the first element, and replaces the last one with 0)
    D_posterior = utils.decaler(state_posterior, dim=0)
    #  - pi_prior = inverse of the covariance of Q (here Q uses standard deviations; they must be squared to obtain the inverse)
    pi_prior = pi_priors[state_name]
    #  - actual = posterior of child states or observations
    actual = model_components.get_actual(navigator, Q_, state_name, observations)
    # - prediction: prediction of g
    prediction = model_components.get_state_prediction(navigator, functions_map, inputs_map, state_name)
    #print("prediction: ", prediction)
    #  - pi_obs = precision of function g
    pi_obs_t = model_components.get_actual_precision(navigator, pi_obs, Q_, state_name)

    # Reshape prior
    state_prior_original = state_prior.reshape_as(state_posterior)
    return  {
        "state_prior": state_prior_original,
        "state_posterior": state_posterior,
        "D_posterior": D_posterior,
        "pi_prior": pi_prior,
        "actual": actual,
        "prediction": prediction,
        "pi_obs": pi_obs_t
    }


def sync_state_inputs(D, Q_, inputs_map=None):
    """
    Synchronize the inputs corresponding to states with Q[i].loc.
    The other entries in inputs_map are preserved.
    """

    if inputs_map is None:
        inputs_map = {}

    state_names = list(D.keys())

    if len(state_names) != len(Q_):
        raise ValueError(
            f"{len(state_names)} states in D, "
            f"but {len(Q_)} Q distributions."
        )

    for state_name in Q_:
        list_loc = [
            state.loc
            for state in Q_[state_name]
        ]

        inputs_map[state_name] = torch.stack(list_loc)

    return inputs_map


def actions_vectors_for_f(all_action_of_state, action_name, navigator):

    action_groups = navigator.get_groups_by_actions(all_action_of_state) 
    dict_one_hot = {}
    for action_group in action_groups:
        action_for_group = navigator.get_actions_by_group(action_group)
        dict_one_hot[action_group] = utils.one_hot(action_for_group, action_name)

    return dict_one_hot

def get_one_hot_action_parents(action_one_hot, actions_id, navigator):
    index = torch.argmax(action_one_hot).item()
    action_name = utils.get_name_by_index(actions_id, index)
    
    state_parents = navigator.parent_nodes(action_name, node_type="state")

    return state_parents, action_name

def preprocessing_for_excepted_free_energy(all_action_of_state, action_name, parent_of_action, local_inputs_map, pi_priors, navigator, functions_map, Q_):
    f_action_inputs = actions_vectors_for_f(all_action_of_state, action_name, navigator)
    local_inputs_map |= f_action_inputs
    temporal_pred = model_components.get_transition_prediction(navigator, functions_map, local_inputs_map, parent_of_action, recursive_input=None)
    outputs_f = navigator.output_f(parent_of_action)
    if len(outputs_f) == 1:
        local_inputs_map[outputs_f[0]] = temporal_pred
    else:
        for output_f in outputs_f:
            local_inputs_map[output_f] = temporal_pred

    transition_cost = model_components.get_transition_cost(navigator, functions_map, local_inputs_map, parent_of_action)
    pi_priors = pi_priors[parent_of_action]

    list_co_variances = []
    for q in Q_[parent_of_action]:
        list_co_variances.append(utils.extract_variance(q))

    co_variances = torch.stack(list_co_variances)

    return {
        "temporal_pred": temporal_pred,
        "transition_cost": transition_cost,
        "pi_priors": pi_priors,
        "co_variances": co_variances
    }


"""
Helpers for executing graph functions with explicitly ordered inputs/outputs.

The graph is the source of truth for names and ordering:
    input_f / output_f
    input_g / output_g
    input_p / output_p
    input_V / output_V

Values are not flattened or converted unless a numerical vector is explicitly
needed by a consumer (for example, get_state_prediction).
"""


def prepare_inputs(input_names, inputs_map):
    """Build keyword arguments in graph-declared order without modifying values."""
    missing = [name for name in input_names if name not in inputs_map]
    if missing:
        raise KeyError(
            f"Missing inputs: {missing}. "
            f"Expected inputs (graph order): {list(input_names)}"
        )

    return {name: inputs_map[name] for name in input_names}


def map_outputs(output_names, output):
    """
    Map function return values to graph-declared output names.

    A single declared output receives the whole returned object, even if that
    object is a list, matrix, tensor, or nested structure. Multiple positional
    outputs must be returned as a tuple/list. Mapping outputs are matched by
    name and returned in graph-declared order.
    """
    output_names = list(output_names)
    n_outputs = len(output_names)

    if n_outputs == 0:
        if output is not None:
            raise ValueError(
                "The function returns a value, but no output "
                "is declared in the graph."
            )
        return {}

    if n_outputs == 1:
        name = output_names[0]
        if isinstance(output, Mapping):
            if name not in output:
                raise KeyError(
                    f"Output '{name}' is missing from the returned dictionary."
                )
            if len(output) != 1:
                raise ValueError(
                    f"Only one output is declared ({name}), but "
                    "multiple named outputs are returned."
                )
            return {name: output[name]}
        return {name: output}

    if isinstance(output, Mapping):
        missing = [name for name in output_names if name not in output]
        extra = [name for name in output if name not in output_names]
        if missing or extra:
            raise ValueError(
                "Outputs are incompatible with the graph. "
                f"Missing: {missing}. Unexpected: {extra}."
            )
        return {name: output[name] for name in output_names}

    if isinstance(output, (tuple, list)):
        if len(output) != n_outputs:
            raise ValueError(
                f"The function returns {len(output)} objects, but the graph "
                f"declares {n_outputs} outputs: {output_names}."
            )
        return dict(zip(output_names, output))

    raise TypeError(
        f"The function must return a tuple, a list, or a dictionary "
        f"to map {n_outputs} outputs: {output_names}. "
        f"Received type: {type(output).__name__}."
    )


def execute_named_function(
    navigator, functions_map, inputs_map, state, function_name
):
    """Execute f/g/p/V and return {declared_output_name: returned_object}."""
    function_ref = navigator.function(state, function_name)
    if function_ref is None:
        return {}

    if function_ref not in functions_map:
        raise KeyError(
            f"Function '{function_ref}' associated with '{state}.{function_name}' "
            "is missing from functions_map."
        )

    function = functions_map[function_ref]
    input_names = get_io_names(navigator, state, function_name, "input")
    output_names = get_io_names(navigator, state, function_name, "output")
    kwargs = prepare_inputs(input_names, inputs_map)
    result = function(**kwargs)
    return map_outputs(output_names, result)


def get_io_names(navigator, state, function_name, direction):
    """Support both the generic Navigator API and the existing input_f methods."""
    if function_name not in ("f", "g", "p", "V"):
        raise ValueError("function_name must be 'f', 'g', 'p', or 'V'.")

    generic_name = "input_names" if direction == "input" else "output_names"
    if hasattr(navigator, generic_name):
        return getattr(navigator, generic_name)(state, function_name)

    # Compatibility with the GraphNavigator supplied in the project.
    method_name = f"{direction}_{function_name}"
    if hasattr(navigator, method_name):
        return getattr(navigator, method_name)(state)

    # Fallback for navigators exposing only node()/graph.
    node = navigator.node(state) if hasattr(navigator, "node") else navigator.graph[state]
    return node.get(f"{direction}_{function_name}", [])


def as_flat_tensor(value, *, dtype=torch.float32):
    """Convert one output value to a flat tensor for numerical calculations."""
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)
    if isinstance(value, (int, float, bool)):
        return torch.as_tensor([value], dtype=dtype)

    # Lists/tuples may themselves contain tensors. Concatenate recursively
    # rather than using as_tensor(list_of_tensors), which is not reliable.
    if isinstance(value, (tuple, list)):
        if not value:
            return torch.empty(0, dtype=dtype)
        try:
            return torch.as_tensor(value, dtype=dtype).reshape(-1)
        except (TypeError, ValueError, RuntimeError):
            parts = [as_flat_tensor(item, dtype=dtype) for item in value]
            return torch.cat(parts) if len(parts) > 1 else parts[0]

    try:
        return torch.as_tensor(value, dtype=dtype).reshape(-1)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(
            f"Unable to convert output of type "
            f"{type(value).__name__} to a numerical tensor."
        ) from exc


def flatten_named_outputs(named_outputs):
    """Concatenate named outputs in their declared order."""
    if not named_outputs:
        return torch.empty(0, dtype=torch.float32)

    values = [as_flat_tensor(value) for value in named_outputs.values()]
    return torch.cat(values) if len(values) > 1 else values[0]



def execute_flat_function(
    navigator, functions_map, inputs_map, state, function_name
):
    """
    Execute f/g/p/V and return a single 1D vector.

    Outputs are first associated with the names declared in the graph,
    then each output is flattened and concatenated in output_* order.
    """
    named_outputs = execute_named_function(
        navigator, functions_map, inputs_map, state, function_name
    )
    return flatten_named_outputs(named_outputs)