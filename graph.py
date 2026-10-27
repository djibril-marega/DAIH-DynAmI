
class GraphNavigator:

    def __init__(self, graph):
        self.graph = graph

    # =========================================================
    # Retrieve a node
    # =========================================================

    def node(self, node_name):
        """
        Returns the node matching the given name.
        """
        if node_name not in self.graph:
            raise KeyError(f"Node '{node_name}' does not exist.")

        return self.graph[node_name]
    
    # =========================================================
    # Retrieve a node
    # =========================================================
    def position(self, node_name):
        return list(self.graph.keys()).index(node_name)
    
    # =========================================================
    # General information
    # =========================================================

    def get(self, node_name, element=None):
        """
        Retrieves a specific element from a node.

        Examples:
            get("state_1")
            get("state_1", "g")
            get("state_1", "f")
            get("state_1", "input_g")
            get("state_1", "output_g")
            get("state_1", "parents")
            get("state_1", "children")
        """

        node = self.node(node_name)

        if element is None:
            return node

        if element not in node:
            raise KeyError(
                f"Element '{element}' does not exist "
                f"in node '{node_name}'."
            )

        return node[element]

    # =========================================================
    # Node type
    # =========================================================

    def type(self, node_name):
        """
        Returns the node type:
            'state'
            'observation'
        """
        return self.node(node_name)["type"]

    # =========================================================
    # Parents
    # =========================================================

    def parents(self, node_name):
        """
        Returns all parents of a node.
        """
        return self.node(node_name)["parents"]

    # =========================================================
    # Children
    # =========================================================

    def children(self, node_name):
        """
        Returns all children of a node.
        """
        return self.node(node_name)["children"]

    # =========================================================
    # Filter parents
    # =========================================================

    def parent_nodes(self, node_name, node_type=None):
        """
        Returns the names of the parents.

        node_type can be:
            "state"
            "observation"
            None
        """

        parents = self.parents(node_name)

        result = []

        for parent in parents:

            parent_name = parent["node"]

            if node_type is None:
                result.append(parent_name)

            elif self.type(parent_name) == node_type:
                result.append(parent_name)

        return result

    # =========================================================
    # Filter children
    # =========================================================

    def child_nodes(self, node_name, node_type=None):
        """
        Returns the names of the children.

        node_type can be:
            "state"
            "observation"
            "action"
        """

        children = self.children(node_name)

        result = []

        for child in children:

            child_name = child["node"]

            if node_type is None:
                result.append(child_name)

            elif self.type(child_name) == node_type:
                result.append(child_name)

        return result

    # =========================================================
    # Functions
    # =========================================================

    
    def function(self, node_name, function_name):
        """
        Returns the name of an f, g, V, or p function.
        """

        if function_name not in ("f", "g", "V", "p"):
            raise ValueError(
                "function_name must be 'f', 'g', 'V', or 'p'."
            )

        node = self.node(node_name)

        if node["type"] != "state":
            raise TypeError(
                f"Node '{node_name}' is not a state."
            )

        function = node.get(function_name)

        if function is None or (
            isinstance(function, str) and not function.strip()
        ):
            return None

        return function

    # =========================================================
    # Inputs / Outputs of g
    # =========================================================

    def input_g(self, node_name):
        return self.node(node_name)["input_g"]

    def output_g(self, node_name):
        return self.node(node_name)["output_g"]

    # =========================================================
    # Inputs / Outputs of f
    # =========================================================

    def input_f(self, node_name):
        return self.node(node_name)["input_f"]

    def output_f(self, node_name):
        return self.node(node_name)["output_f"]

    
    # =========================================================
    # Inputs / Outputs of V
    # =========================================================

    def input_V(self, node_name):
        return self.node(node_name)["input_V"]

    def output_V(self, node_name):
        return self.node(node_name)["output_V"]

    # =========================================================
        # Inputs / Outputs of p
    # =========================================================
    
    def input_p(self, node_name):
        return self.node(node_name)["input_p"]

    def output_p(self, node_name):
        return self.node(node_name)["output_p"]

    # =========================================================
    # Connection used to reach a child
    # =========================================================

    def child_connections(self, node_name):
        """
        Returns the children with their connection.

        Example:
        [
            {
                "node": "state_2",
                "connection": "x"
            }
        ]
        """

        return self.children(node_name)

    # =========================================================
    # Connection used to reach a parent
    # =========================================================

    def parent_connections(self, node_name):
        return self.parents(node_name)

    # =========================================================
    # Action
    # =========================================================

    def action_nodes(self, state_name):
        """Returns the action nodes available from a state."""
        return [
            child["node"]
            for child in self.children(state_name)
            if self.type(child["node"]) == "action"
        ]


    def available_actions(self, state_name):
        """Returns the names of the actions available from a state."""
        return [
            self.get(node, "name")
            for node in self.action_nodes(state_name)
        ]


    def is_action(self, node_name):
        """Indicates whether a node is an action."""
        return self.type(node_name) == "action"
    

    def get_actions_by_group(self, group_name: str) -> list:
        """
        Returns the list of action names belonging to the specified group.
        """
        return [
            node_name for node_name, properties in self.graph.items()
            if properties.get('type') == 'action' and properties.get('group') == group_name
        ]

    def get_groups_by_actions(self, actions: list[str]) -> list[str]:
        """
        Returns the groups to which the given actions belong,
        without duplicates.
        """
        groups = {
            properties.get('group')
            for node_name, properties in self.graph.items()
            if properties.get('type') == 'action'
            and node_name in actions
            and properties.get('group') is not None
        }

        return list(groups)

    
    # =========================================================
    # Layers
    # =========================================================

    def states_by_layer(self, layer):
        """
        Returns the names of the states belonging to a layer.

        Example:
            navigator.states_by_layer(2)
        """
        return [
            name
            for name, node in self.graph.items()
            if node.get("type") == "state"
            and node.get("layer") == layer
        ]

    def states_at_or_above_layer(self, layer):
        """
        Returns the states whose layer number
        is greater than or equal to the specified one.

        Example:
            navigator.states_at_or_above_layer(2)
        """
        return [
            name
            for name, node in self.graph.items()
            if node.get("type") == "state"
            and node.get("layer") is not None
            and node["layer"] >= layer
        ]

    def states_at_or_below_layer(self, layer):
        """
        Returns the states whose layer number
        is less than or equal to the specified one.
        """
        return [
            name
            for name, node in self.graph.items()
            if node.get("type") == "state"
            and node.get("layer") is not None
            and node["layer"] <= layer
        ]

    def layers(self):
        """
        Returns the available layers,
        in descending order.
        """
        return sorted(
            {
                node["layer"]
                for node in self.graph.values()
                if node.get("type") == "state"
                and node.get("layer") is not None
            },
            reverse=True
        )

    def nodes_by_layer(self, layer, node_type=None):
        """
        Returns the nodes in a layer.
        By default, only state layers exist.

        node_type allows filtering by type.
        """
        if node_type not in (None, "state"):
            raise ValueError(
                "Currently, only states have a layer."
            )

        return self.states_by_layer(layer)

    def layers(self):
        """
        Returns the available layers,
        in ascending order.
        """
        return sorted({
            node["layer"]
            for node in self.graph.values()
            if node.get("type") == "state"
            and node.get("layer") is not None
        })

    
    # =========================================================
    # Motors
    # =========================================================

    def states_with_motors(self):
        """
        Returns the names of states that have at least one motor.

        A motor is an observation whose group also belongs
        to an action.
        """

        # Action groups
        action_groups = {
            node.get("group")
            for node in self.graph.values()
            if node.get("type") == "action"
            and node.get("group") is not None
        }

        states = []

        for state_name, state in self.graph.items():

            if state.get("type") != "state":
                continue

            # Observations accessible from the state
            observations = self.child_nodes(
                state_name,
                node_type="observation"
            )

            for observation_name in observations:

                observation = self.node(observation_name)
                observation_group = observation.get("group")

                if observation_group in action_groups:
                    states.append(state_name)
                    break

        return states


    def motors_by_state(self, state_name):
        """
        Returns the observations (motors) of a state.

        A motor is an observation whose group matches
        the group of at least one action in the graph.
        """

        state = self.node(state_name)

        if state.get("type") != "state":
            raise TypeError(
                f"Node '{state_name}' is not a state."
            )

        # Action groups
        action_groups = {
            node.get("group")
            for node in self.graph.values()
            if node.get("type") == "action"
            and node.get("group") is not None
        }

        # Observations accessible from the state
        observations = self.child_nodes(
            state_name,
            node_type="observation"
        )

        # Keep observations whose group
        # matches that of an action
        return [
            observation_name
            for observation_name in observations
            if self.node(observation_name).get("group")
            in action_groups
        ]

    # =========================================================
    # Inputs/Outputs
    # =========================================================
    
    def input_names(self, node_name, function_name):
        """Returns the inputs declared for a function."""
        if function_name not in ("f", "g", "p", "V"):
            raise ValueError(
                "function_name must be 'f', 'g', 'p', or 'V'."
            )

        return self.node(node_name).get(
            f"input_{function_name}", []
        )

    def output_names(self, node_name, function_name):
        """Returns the outputs declared for a function."""
        if function_name not in ("f", "g", "p", "V"):
            raise ValueError(
                "function_name must be 'f', 'g', 'p', or 'V'."
            )

        return self.node(node_name).get(
            f"output_{function_name}", []
        )








def build_graph(generative_functions, observations, actions=None):
    """
    Builds the generative model graph.

    Hierarchical relationships are defined exclusively
    by the parents field. The layer attribute is metadata
    that does not determine the connections.

    The functions f, g, V, and p, along with their inputs/outputs,
    are preserved as metadata on state nodes.

    Actions and observations are represented as
    individual nodes. Groups remain metadata.

    Motors are observations belonging to a group
    also declared in actions. They must be referenced
    individually in output_g and comply with the rule
    for validating hierarchical parents.

    Each edge is recorded in both directions:
    children contains the children, and parents contains the parents.
    """
    graph = {}
    actions = actions or {}
    observations = observations or {}

    # ---------------------------------------------------------
    # 1. Utility functions
    # ---------------------------------------------------------

    def add_node(name, node_type, **attributes):
        if name in graph:
            raise ValueError(
                f"Node '{name}' already exists."
            )

        graph[name] = {
            "type": node_type,
            **attributes,
            "parents": [],
            "children": []
        }

    def add_edge(parent, child, connection):
        """Adds a directed relationship without duplicates."""

        if parent not in graph:
            raise KeyError(
                f"Parent node '{parent}' does not exist."
            )

        if child not in graph:
            raise KeyError(
                f"Child node '{child}' does not exist."
            )

        if not any(
            edge["node"] == child
            for edge in graph[parent]["children"]
        ):
            graph[parent]["children"].append({
                "node": child,
                "connection": connection
            })

        if not any(
            edge["node"] == parent
            for edge in graph[child]["parents"]
        ):
            graph[child]["parents"].append({
                "node": parent,
                "connection": connection
            })

    # ---------------------------------------------------------
    # 2. Index actions and observations
    # ---------------------------------------------------------

    action_to_group = {}
    observation_to_group = {}

    for group_name, action_names in actions.items():
        for action_name in action_names:
            if action_name in action_to_group:
                raise ValueError(
                    f"Action '{action_name}' belongs "
                    "to multiple groups."
                )

            action_to_group[action_name] = group_name

    for group_name, observation_names in observations.items():
        for observation_name in observation_names:
            if observation_name in observation_to_group:
                raise ValueError(
                    f"Observation '{observation_name}' "
                    "belongs to multiple groups."
                )

            observation_to_group[observation_name] = group_name

    # A motor is an observation whose group also exists
    # in the actions dictionary.
    motor_groups = set(actions) & set(observations)

    motor_to_group = {
        motor_name: group_name
        for group_name in motor_groups
        for motor_name in observations[group_name]
    }

    # ---------------------------------------------------------
    # 3. Create state nodes
    # ---------------------------------------------------------

    for state_name, state in generative_functions.items():

        if "layer" not in state:
            raise ValueError(
                f"State '{state_name}' does not have a layer."
            )

        
        add_node(
            state_name,
            "state",
            layer=state["layer"],
            f=state.get("f"),
            g=state.get("g"),
            V=state.get("V"),
            p=state.get("p"),
            input_f=state.get("input_f") or [],
            output_f=state.get("output_f") or [],
            input_g=state.get("input_g") or [],
            output_g=(
                state.get("output_g")
                or state.get("outpu_g")
                or []
            ),
            input_V=state.get("input_V") or [],
            output_V=state.get("output_V") or [],
            input_p=state.get("input_p") or [],
            output_p=state.get("output_p") or [],
            state_parents=state.get("parents") or [],
        )


    # ---------------------------------------------------------
    # 4. Create action nodes
    # ---------------------------------------------------------

    for group_name, action_names in actions.items():
        for action_name in action_names:
            add_node(
                action_name,
                "action",
                group=group_name
            )

    # ---------------------------------------------------------
    # 5. Create observation nodes
    # ---------------------------------------------------------

    for group_name, observation_names in observations.items():
        for observation_name in observation_names:
            add_node(
                observation_name,
                "observation",
                group=group_name,
                is_motor=(group_name in motor_groups)
            )

    # ---------------------------------------------------------
    # 6. Build hierarchical relationships
    # ---------------------------------------------------------

    for state_name, state in generative_functions.items():

        for parent_name in state.get("parents") or []:

            if parent_name not in generative_functions:
                raise KeyError(
                    f"Parent '{parent_name}' of '{state_name}' "
                    "does not exist."
                )

            # No layer check:
            # the relationship is determined exclusively by parents.
            add_edge(
                parent_name,
                state_name,
                "hierarchy"
            )

    # ---------------------------------------------------------
    # 7. Connect actions via input_f
    # ---------------------------------------------------------

    for state_name, state in generative_functions.items():

        for input_name in state.get("input_f") or []:

            # An individual action.
            if input_name in action_to_group:
                add_edge(
                    state_name,
                    input_name,
                    "action"
                )

            # An action group: connect all its actions.
            elif input_name in actions:
                for action_name in actions[input_name]:
                    add_edge(
                        state_name,
                        action_name,
                        "action"
                    )

            # An input corresponding to a state:
            # no implicit hierarchical relationship.
            elif input_name in generative_functions:
                continue

            # An observation: no relationship is created here.
            elif input_name in observation_to_group:
                continue

            # An intermediate variable produced by f.
            elif any(
                input_name in (
                    candidate.get("output_f") or []
                )
                for candidate in generative_functions.values()
            ):
                continue

            else:
                raise KeyError(
                    f"Unknown input '{input_name}' in "
                    f"input_f of '{state_name}'."
                )

    # ---------------------------------------------------------
    # 8. Connect observations via output_g
    # ---------------------------------------------------------

    for state_name, state in generative_functions.items():

        for output_name in (
            state.get("output_g")
            or state.get("outpu_g")
            or []
        ):

            # A motor group cannot be referenced
            # directly: motors must be referenced individually.
            if output_name in motor_groups:
                raise ValueError(
                    f"Motor '{output_name}' must be "
                    "referenced individually in output_g, "
                    "not by its group."
                )

            # An individual observation, including motors.
            if output_name in observation_to_group:

                # Motor-specific validation.
                if output_name in motor_to_group:

                    motor_group = motor_to_group[output_name]
                    state_parents = state.get("parents") or []

                    eligible = any(
                        motor_group in (
                            generative_functions[parent_name]
                            .get("input_f") or []
                        )
                        for parent_name in state_parents
                    )

                    if not eligible:
                        raise ValueError(
                            f"Motor '{output_name}' cannot "
                            f"be connected to '{state_name}': "
                            "none of its hierarchical parents "
                            f"uses the factor '{motor_group}' "
                            "in input_f."
                        )

                add_edge(
                    state_name,
                    output_name,
                    "observation"
                )

            # A group of regular observations.
            elif output_name in observations:

                for observation_name in observations[output_name]:
                    add_edge(
                        state_name,
                        observation_name,
                        "observation"
                    )

            # Any other output remains an intermediate variable.

    return graph
