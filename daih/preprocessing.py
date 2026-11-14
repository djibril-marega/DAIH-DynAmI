from . import model_components, utils
import torch, math, copy
from collections.abc import Mapping

def preprocessing_layer_for_free_energy(state_name, navigator, functions_map, inputs_map, Q_, pi_priors, pi_obs, observations, states_list):
    """Prepare the quantities required for the variational free energy."""
    state_prior = model_components.get_parent_g_predictions(
        navigator, functions_map, inputs_map, state_name, states_list, Q_
    )
    f_transition = model_components.get_transition_prediction(
        navigator, functions_map, inputs_map, state_name, states_list, Q_
    ).reshape(1, -1)

    state_posterior = inputs_map[state_name].reshape(1, -1)
    pi_prior = model_components.get_parent_g_precisions(
        navigator, state_name, pi_priors, Q_
    )

    actual = model_components.get_actual(navigator, Q_, state_name, observations)
    prediction = model_components.get_state_prediction(
        navigator, functions_map, inputs_map, state_name, states_list, Q_
    )
    pi_obs_t = model_components.get_actual_precision(
        navigator, pi_obs, Q_, state_name
    )

    if state_prior.size(0) == 0:
        state_prior = torch.zeros(state_posterior.shape)
        pi_prior = torch.full(state_posterior.shape, fill_value=1e-7)

    return {
        "state_prior": state_prior,
        "state_posterior": state_posterior,
        "state_posterior_init_size": inputs_map[state_name],
        "f_transition": f_transition,
        "pi_prior": pi_prior,
        "actual": actual,
        "prediction": prediction,
        "pi_obs": pi_obs_t,
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

def preprocessing_for_excepted_free_energy(
        all_action_of_state, action_name, parent_of_action, local_inputs_map, pi_states, navigator, functions_map, Q_, states_list
    ):
    f_action_inputs = actions_vectors_for_f(all_action_of_state, action_name, navigator)
    local_inputs_map |= f_action_inputs
    temporal_pred = model_components.get_transition_prediction(navigator, functions_map, local_inputs_map, parent_of_action, states_list, Q_, policy=True)
    outputs_f = navigator.output_f(parent_of_action)
    if len(outputs_f) == 1:
        local_inputs_map[outputs_f[0]] = temporal_pred
    else:
        for output_f in outputs_f:
            local_inputs_map[output_f] = temporal_pred

    transition_cost = model_components.get_transition_cost(navigator, functions_map, local_inputs_map, parent_of_action, states_list, Q_)
    pi_state = pi_states[parent_of_action]

    list_co_variances = []
    for q in Q_[parent_of_action]:
        list_co_variances.append(utils.extract_variance(q))

    co_variances = torch.stack(list_co_variances)

    return {
        "temporal_pred": temporal_pred,
        "transition_cost": transition_cost,
        "pi_priors": pi_state,
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



def map_outputs(output_names, output, navigator):
    """
    Map function outputs to graph-declared output names.

    WorldModel has already consumed any local masks before this function
    receives the result. This function only associates final output matrices
    with their graph-declared names.

    Rules:
        - One output: returned unchanged.
        - Multi-output form: one final matrix per declared name.
        - Output order follows output_names.
        - Legacy flat tensors are still split across observation outputs.
    """

    output_names = list(output_names)
    n_outputs = len(output_names)

    # No outputs
    if n_outputs == 0:
        if output is not None:
            raise ValueError(
                "The function returns a value, but no output "
                "is declared in the graph."
            )
        return {}

    # Single output: state
    if n_outputs == 1:
        return {
            output_names[0]: output
        }

    # Multiple outputs: one final matrix per declared name. Any local masks
    # have already been consumed by WorldModel.
    if isinstance(output, Mapping):
        missing = [name for name in output_names if name not in output]
        extra = [name for name in output if name not in output_names]
        if missing or extra:
            raise ValueError(
                f"Returned output mapping does not match output_names; "
                f"missing={missing}, extra={extra}."
            )
        return {name: output[name] for name in output_names}

    if isinstance(output, (tuple, list)) and len(output) == n_outputs:
        return dict(zip(output_names, output))

    output_sizes = {}

    for name in output_names:

        # Check whether the name is an individual observation.
        if (
            name in navigator.graph
            and navigator.graph[name].get("type") == "observation"
        ):
            output_sizes[name] = 1

        else:
            # Otherwise, interpret the name as an observation group.
            observations = navigator.get_observations_by_group(name)

            if not observations:
                raise KeyError(
                    f"'{name}' is neither an individual observation "
                    "nor a valid observation group."
                )

            output_sizes[name] = len(observations)

    # Total number of observations expected.
    expected_size = sum(output_sizes.values())

    # Split Tensor along its last dimension.
    if isinstance(output, torch.Tensor):

        if output.shape[-1] != expected_size:
            raise ValueError(
                f"The output tensor contains {output.shape[-1]} "
                f"elements along its last dimension, but "
                f"{expected_size} were expected."
            )

        chunks = torch.split(
            output,
            list(output_sizes.values()),
            dim=-1
        )

    # Split list or tuple.
    elif isinstance(output, (tuple, list)):

        if len(output) != expected_size:
            raise ValueError(
                f"The function returns {len(output)} elements, "
                f"but {expected_size} were expected."
            )

        chunks = []
        index = 0

        for size in output_sizes.values():
            chunks.append(output[index:index + size])
            index += size

    else:
        raise TypeError(
            f"Unsupported output type: {type(output).__name__}."
        )

    return dict(zip(output_names, chunks))



def execute_named_function(
    navigator, functions_map, inputs_map, state, function_name, states_list, function_type, Q_
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
    #kwargs = prepare_inputs(input_names, inputs_map)
    kwargs = prepare_world_model_inputs(
        navigator, input_names, inputs_map, state, states_list,
        function_type, Q_, output_names=output_names
    )
    result = function(**kwargs)
    if function_type != "p":
        return map_outputs(output_names, result.values, navigator)
    else:
        return result


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

    values = []
    for value in named_outputs.values():
        values.append(as_flat_tensor(value))
    return torch.cat(values) if len(values) > 1 else values[0]



def execute_flat_function(
    navigator, functions_map, inputs_map, state, function_name, states_list, function_type, Q_
):
    """
    Execute f/g/p/V and return a single 1D vector.

    WorldModel returns final matrices only. Outputs are associated with the
    names declared in the graph, then flattened and concatenated in output_*
    order.
    """
    named_outputs = execute_named_function(
        navigator, functions_map, inputs_map, state, function_name, states_list, function_type, Q_
    )
    return flatten_named_outputs(named_outputs)



def get_distribution_width(distribution):
    """
    Determines the number of columns represented by a distribution.

    Normal            -> 1 column
    MultivariateNormal -> number of dimensions of event_shape
    """
    event_shape = distribution.event_shape

    if len(event_shape) == 0:
        return 1

    return math.prod(event_shape)



def get_output_shape(navigator, state_name, Q_):
    """
    Determines the output shape based on the children of the current state.

    - observation : one observation = one column.
    - hierarchy : dimensions are determined from Q_.
    - action : ignored in dimension calculations.

    Observational and hierarchical connections
    can coexist.
    """

    children = navigator.children(state_name)

    if not children:
        raise ValueError(
            f"State '{state_name}' has no children."
        )

    # Separate children based on their connection type.
    observation_children = [
        child for child in children
        if child["connection"] == "observation"
    ]

    hierarchy_children = [
        child for child in children
        if child["connection"] == "hierarchy"
    ]

    action_children = [
        child for child in children
        if child["connection"] == "action"
    ]

    # Check for unknown connections.
    known_connections = {"observation", "hierarchy", "action"}

    unknown_connections = {
        child["connection"]
        for child in children
        if child["connection"] not in known_connections
    }

    if unknown_connections:
        raise ValueError(
            f"Unknown connections for '{state_name}': "
            f"{unknown_connections}"
        )

    # Actions do not contribute to the output shape.
    child_shapes = []

    # Observational children: one column per observation.
    if observation_children:
        child_shapes.append((1, len(observation_children)))

    # Hierarchical children: dimensions retrieved from Q_.
    for child in hierarchy_children:

        child_state = child["node"]

        if child_state not in Q_:
            raise KeyError(
                f"Child state '{child_state}' is missing from Q_."
            )

        distributions = Q_[child_state]

        if not distributions:
            raise ValueError(
                f"No distribution found for '{child_state}'."
            )

        rows = len(distributions)
        cols = get_distribution_width(distributions[0])

        # Check width consistency.
        for distribution in distributions[1:]:
            width = get_distribution_width(distribution)

            if width != cols:
                raise ValueError(
                    f"Incompatible dimensions in Q_['{child_state}']."
                )

        child_shapes.append((rows, cols))

    if not child_shapes:
        raise ValueError(
            f"State '{state_name}' has no observational "
            "or hierarchical children to determine "
            "its shape."
        )

    # All children contributing to the output must have
    # the same number of rows.
    rows = child_shapes[0][0]

    if any(shape[0] != rows for shape in child_shapes):
        raise ValueError(
            f"Children of '{state_name}' "
            "do not have the same number of rows."
        )

    # Concatenate columns.
    cols = sum(shape[1] for shape in child_shapes)

    return (rows, cols)




def get_named_output_shapes(navigator, output_names, Q_):
    """
    Return one (rows, columns) shape per declared output name, in graph order.

    A state uses the dimensions of its Q_ distributions. An individual
    observation uses one cell, and an observation group uses one cell per
    observation in that group.
    """
    shapes = []

    for name in output_names:
        node = navigator.graph.get(name)
        node_type = node.get("type") if node is not None else None

        if node_type == "state":
            if name not in Q_:
                raise KeyError(
                    f"Output state '{name}' is missing from Q_."
                )
            distributions = Q_[name]
            if not distributions:
                raise ValueError(
                    f"No distributions found in Q_['{name}']."
                )
            width = get_distribution_width(distributions[0])
            for distribution in distributions[1:]:
                if get_distribution_width(distribution) != width:
                    raise ValueError(
                        f"Incompatible dimensions in Q_['{name}']."
                    )
            shapes.append((len(distributions), width))
            continue

        if node_type == "observation":
            shapes.append((1, 1))
            continue

        observations = navigator.get_observations_by_group(name)
        if not observations:
            raise KeyError(
                f"Output '{name}' is neither a state, an individual "
                "observation, nor a valid observation group."
            )
        shapes.append((1, len(observations)))

    return tuple(shapes)


def prepare_world_model_inputs(
    navigator,
    input_names,
    inputs_map,
    state_name,
    states_list,
    function_type,
    Q_,
    output_names=None,
):
    """
    Prepares arguments intended for WorldModel.forward().

    WorldModel conventions are independent
    of the input names declared in the graph.

    Actions can be:
        - An individual action.
        - An action group.
    """

    # Current state
    if state_name not in inputs_map:
        raise KeyError(
            f"State '{state_name}' is missing from inputs_map."
        )

    state = inputs_map[state_name]

    # State identifier
    try:
        state_id = states_list.index(state_name)+1
    except ValueError:
        raise ValueError(
            f"State '{state_name}' is missing from states_list."
        )

    # Layer identifier
    layer_id = navigator.get_state_layer(state_name)

    kwargs = {
        "state": state,
        "state_id": state_id,
        "layer_id": layer_id,
        "function_type": function_type,
    }

    # Search for individual actions and action groups.
    detected_actions = []

    for name in input_names:

        # Individual action
        is_individual_action = (
            name in navigator.graph
            and navigator.graph[name].get("type") == "action"
        )

        # Action group
        actions_in_group = navigator.get_actions_by_group(name)
        is_action_group = bool(actions_in_group)

        if is_individual_action or is_action_group:

            if name not in inputs_map:
                raise KeyError(
                    f"Action or action group '{name}' "
                    "is missing from inputs_map."
                )

            detected_actions.append(name)

    # WorldModel accepts only a single action or a single group.
    if len(detected_actions) > 1:
        raise ValueError(
            "WorldModel.forward() accepts only a single action, "
            f"but multiple were detected: {detected_actions}"
        )

    if detected_actions:
        action_name = detected_actions[0]
        kwargs["action"] = inputs_map[action_name]

    # Optional parameters
    for parameter in ("max_steps", "deterministic"):
        if parameter in input_names and parameter in inputs_map:
            kwargs[parameter] = inputs_map[parameter]

    # g outputs are shaped independently, in the exact order declared by
    # output_g. Keep output_shape for single-output compatibility.
    if function_type == "g":
        if output_names is None:
            output_names = get_io_names(navigator, state_name, "g", "output")
        output_shapes = get_named_output_shapes(navigator, output_names, Q_)
        kwargs["output_shapes"] = output_shapes
        if len(output_shapes) == 1:
            kwargs["output_shape"] = output_shapes[0]
    elif function_type == "V":
        kwargs["output_shape"] =  (1, 1)
    elif function_type == "p":
        kwargs["temperature"] = inputs_map[f"error_{state_name}"]

    return kwargs


def handle_missing_observations(observations_actual, pi_obs, pi_obs_substitued, low_precision=1e-7, substitute_value=0.0):
    """
    Handles missing observations and restores their original precisions.

    Args:
        observations_actual (dict): Observations grouped by name.
        pi_obs (dict): Precision of each individual observation.
        pi_obs_substitued (dict): Backup of original precisions for
            currently missing observations.
        low_precision (float): Precision assigned to missing observations.
        substitute_value (float): Numerical value replacing None.

    Returns:
        tuple:
            - observations_actual
            - pi_obs
            - pi_obs_substitued
    """

    for group, observations in observations_actual.items():

        for observation, value in observations.items():

            # --------------------------------------------------
            # 1. Missing observation
            # --------------------------------------------------
            if value is None:

                # Save the original precision only once.
                if observation not in pi_obs_substitued:

                    if observation in pi_obs:
                        pi_obs_substitued[observation] = copy.deepcopy(
                            pi_obs[observation]
                        )

                # Reduce its influence during inference.
                pi_obs[observation] = torch.tensor(
                    low_precision,
                    dtype=torch.float32
                )

                # Replace None with a numerical value.
                observations[observation] = substitute_value

            # --------------------------------------------------
            # 2. Observation available again
            # --------------------------------------------------
            elif observation in pi_obs_substitued:

                # Restore the original precision if it was saved.
                pi_obs[observation] = pi_obs_substitued.pop(observation)

    return observations_actual, pi_obs, pi_obs_substitued



