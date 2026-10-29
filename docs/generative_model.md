# Generative Model Definition Guide

## Objective

This guide describes how to configure the generative model using
`generative_functions`, `actions`, and `observations`, as well as the
relationships created by `build_graph()`.

The graph contains three types of nodes:

-   `state`: states declared in `generative_functions`;
-   `action`: individual actions declared in `actions`;
-   `observation`: individual observations declared in `observations`.

Action and observation groups, function names, and intermediate
variables remain metadata; they do not automatically become graph nodes.

> **Key distinction:** `parents` defines explicit hierarchical
> relationships. `layer` describes the state's temporal level and
> expected update policy. In the supplied `build_graph()`, `layer` is
> stored as an attribute, but this function alone does not execute any
> layer-based temporal update mechanism.

## 1. Declaring a State

Each state is an entry in `generative_functions`. It must define `layer`
and the fields required by the functions it uses.

``` python
generative_functions = {
    "mouse_position": {
        "layer": 1,
        "parents": [],
        "f": "predict_mouse_position",
        "input_f": ["mouse_position"],
        "output_f": ["next_mouse_position"],
        "g": "observe_mouse_position",
        "input_g": ["mouse_position"],
        "output_g": ["mouse_position_obs"],
    }
}
```

  -----------------------------------------------------------------------
  Field                               Role
  ----------------------------------- -----------------------------------
  `layer`                             The state's temporal level in the
                                      model architecture.

  `parents`                           List of parent states in the
                                      hierarchy.

  `f`                                 Name of the transition or temporal
                                      prediction function.

  `input_f`, `output_f`               Declared inputs and outputs of `f`.

  `g`                                 Name of the observation-generative
                                      function.

  `input_g`, `output_g`               Declared inputs and outputs of `g`.

  `p`                                 Policy function; optional except
                                      where required by model validation.

  `input_p`, `output_p`               Inputs and outputs of `p`.

  `V`                                 Cost function associated with the
                                      temporal transition.

  `input_V`, `output_V`               Inputs and outputs of `V`.
  -----------------------------------------------------------------------

Function names are metadata resolved by the execution system. Including
them in the dictionary does not create graph nodes.

## 2. Understanding `layer` and `parents`

These fields complement each other but serve different purposes.

### `parents`: the hierarchy

Hierarchical relationships are declared explicitly through `parents`.
For a child state, every name in this list must correspond to an
existing state in `generative_functions`.

``` python
generative_functions = {
    "survival_quality": {
        "layer": 3,
        "parents": [],
    },
    "predator_dynamics": {
        "layer": 2,
        "parents": ["survival_quality"],
    },
    "mouse_position": {
        "layer": 1,
        "parents": ["predator_dynamics"],
    },
}
```

The graph contains these links:

``` text
survival_quality (layer 3)
          |
          v
predator_dynamics (layer 2)
          |
          v
mouse_position (layer 1)
```

In `build_graph()`, a hierarchical link is directed from parent to
child. There is no requirement that layers be consecutive: a parent in
layer 5 can be linked to a child in layer 1 if that relationship is
declared.

A state may have multiple parents, and a parent state may have multiple
children. Two states in the same layer are not linked automatically.

### `layer`: temporal organization of updates

In the model convention, layers describe the temporal organization of
the dynamics:

-   states in the lowest layers are updated continuously or most
    frequently;
-   higher levels depend on errors or on lower levels' inability to
    stabilize free energy;
-   states in the same layer share an update regime, conditioned on the
    stability achieved at lower levels.

This temporal interpretation belongs to the model's semantics. **The
supplied `build_graph()` only records `layer`; it does not itself
trigger these updates.** The scheduling mechanism must be implemented in
the component that executes updates.

Remember:

-   `parents` answers: "Which states depend hierarchically on this
    state?"
-   `layer` answers: "At which temporal level is this state updated?"
-   a layer does not, by itself, create a hierarchical link.

## 3. Transition Function `f`

`f` describes the state's transition or temporal prediction.

``` python
"predator_dynamics": {
    "layer": 2,
    "parents": ["survival_quality"],
    "f": "pred_predator_dynamics",
    "input_f": ["predator_dynamics", "action_movement"],
    "output_f": ["next_predator_dyn"],
}
```

`input_f` may reference:

-   a state declared in `generative_functions`;
-   an individual action;
-   an action group declared in `actions`;
-   an individual observation declared in `observations`;
-   an intermediate variable produced by an `f` function.

According to `build_graph()`:

-   if the input is an individual action, an `action` link is created
    between the state and that action;
-   if the input is an action group, the state is linked to every
    individual action in the group;
-   if the input is a state, no additional hierarchical relationship is
    created;
-   if the input is an observation, this rule creates no link;
-   if the input is a variable declared in an `output_f`, it is treated
    as an intermediate variable;
-   any other unknown input raises a `KeyError`.

The hierarchy is defined by `parents`, not by `input_f`.

`output_f` describes the functional outputs of `f`. `build_graph()` does
not create nodes for these outputs: they remain metadata unless the same
name is separately declared as a state, action, or observation.

## 4. Observation-Generative Function `g`

`g` describes the generation or prediction of observations associated
with a state.

``` python
"cat_position": {
    "layer": 1,
    "parents": ["predator_dynamics"],
    "g": "observe_cat_position",
    "input_g": ["cat_position"],
    "output_g": ["cat_position_obs"],
}
```

Observations are defined in `observations` as groups of individual
observation names:

``` python
observations = {
    "cat_position_obs": ["x_cat", "y_cat"],
    "cheese_position_obs": ["x_cheese", "y_cheese"],
}
```

If `output_g` contains the name of an ordinary observation group, the
state is linked to each individual observation in that group. If
`output_g` contains the name of an individual observation, only that
node is linked.

``` text
cat_position
    |------> x_cat
    `------> y_cat
```

Outputs that match neither an observation group nor an individual
observation remain intermediate variables and do not create nodes.

The code also accepts the misspelled field `outpu_g` for compatibility,
but using `output_g` exclusively is recommended.

### Individual observations and precision

Each observation listed in a group is a distinct graph node and is
addressed individually during inference. Observation precision follows
the same individual-observation convention: `pi_obs` maps an observation
name such as `x_cat` or `x_cheese` to its own precision value. It is not
indexed by the observation group or by the state that predicts the
observation.

Missing-value handling is an inference-time concern, not a
graph-construction rule. If an incoming observation is `None`, the
inference handler substitutes a numeric placeholder and temporarily
reduces that observation's precision. When a real value becomes
available again, its previously saved precision is restored. This does
not create or remove graph nodes and does not change the `observations`
group declaration.

A numerical value of `0` is a valid observation. Only the explicit
missing-value representation (`None` in the current handler) triggers
this treatment.

## 5. Policy Function `p`

`p` is the policy function: it determines a policy or action-related
output from the declared inputs. It is optional for an isolated state,
but `validate_model()` requires an ancestor state to define `p` when it
is an ancestor of a state whose `input_f` contains an individual action
or an action group.

``` python
"survival_quality": {
    "layer": 3,
    "parents": [],
    "f": "pred_survival_quality",
    "input_f": ["survival_quality"],
    "output_f": ["next_survival_quality"],
    "g": "pred_relational_dynamics",
    "input_g": ["survival_quality"],
    "output_g": ["predator_pred", "ressource_pred", "satiety_pred"],
    "p": "policy_for_survival",
    "input_p": [
        "survival_quality",
        "pi_survival_quality",
        "error_survival_quality",
    ],
    "output_p": ["policy_survival"],
}
```

Rules:

-   if `p` is defined, `input_p` and `output_p` must also be defined;
-   the inputs and outputs of `p` are metadata used by the model's
    execution system;
-   `validate_model()` recursively searches the ancestors of states
    whose `input_f` depends on an action and requires a `p` function for
    each such ancestor.

The presence of `p` does not mean that `build_graph()` executes the
function. Function resolution and execution belong to the model engine.

## 6. Temporal Cost Function `V`

`V` is the cost function associated with a temporal transition. For
example, it may evaluate a predicted transition based on
`next_predator_dyn`.

``` python
"predator_dynamics": {
    "layer": 2,
    "parents": ["survival_quality"],
    "f": "pred_predator_dynamics",
    "input_f": ["predator_dynamics", "action_movement"],
    "output_f": ["next_predator_dyn"],
    "g": "pred_danger_movement",
    "input_g": ["predator_dynamics"],
    "output_g": ["cat_movement_pred", "self_movement_pred"],
    "V": "predator_dynamics_value",
    "input_V": ["next_predator_dyn"],
    "output_V": ["predator_value"],
}
```

Rules:

-   if `V` is defined, `input_V` and `output_V` must also be defined;
-   `build_graph()` stores these fields as state metadata;
-   `validate_model()` requires `V`, `input_V`, and `output_V` for every
    state whose `input_f` contains an individual action or an action
    group.

This last rule reflects the supplied code: it applies to the state whose
`f` function directly depends on an action. It is distinct from the rule
requiring `p` on its hierarchical ancestors.

## 7. Declaring Actions

Each key in `actions` is a group, and each item in its list is an
individual action.

``` python
actions = {
    "action_movement": [
        "forward",
        "back",
        "left",
        "right",
        "diagonal",
    ]
}
```

Groups are metadata; individual actions become nodes of type `action`.

To link a state to every action in a group, use the group name in
`input_f`:

``` python
"predator_dynamics": {
    "layer": 2,
    "parents": ["survival_quality"],
    "f": "pred_predator_dynamics",
    "input_f": ["predator_dynamics", "action_movement"],
    "output_f": ["next_predator_dyn"],
}
```

The graph then creates `action` links to `forward`, `back`, `left`,
`right`, and `diagonal`.

An individual action must belong to only one group. `build_graph()`
raises a `ValueError` if an action appears in multiple groups.

## 8. Declaring Observations and Motors

Each key in `observations` is a group; each item is an individual
observation.

``` python
observations = {
    "cat_position_obs": ["x_cat", "y_cat"],
    "action_movement": ["x_mouse", "y_mouse"],
}
```

Here, the observations `x_mouse` and `y_mouse` are **motors** under the
graph convention, because their group `action_movement` is also declared
in `actions`.

The nodes remain distinct:

-   `forward`, `back`, and so on are `action` nodes;
-   `x_mouse` and `y_mouse` are `observation` nodes;
-   `action_movement` remains a group name and is not a node.

### Special rule for motors

To connect a state to a motor:

1.  declare the group in `actions`;
2.  declare the motor observations under the same group name in
    `observations`;
3.  ensure that at least one direct hierarchical parent of the state
    uses this group in its `input_f`;
4.  reference the motors individually in `output_g`.

Example:

``` python
"predator_dynamics": {
    "layer": 2,
    "parents": ["survival_quality"],
    "input_f": ["action_movement"],
},
"mouse_position": {
    "layer": 1,
    "parents": ["predator_dynamics"],
    "output_g": ["x_mouse", "y_mouse"],
}
```

The direct parent `predator_dynamics` uses `action_movement` in
`input_f`, which permits the motor link under `build_graph()`'s
validation rule.

Do not write `output_g: ["action_movement"]` to connect motors: a motor
group cannot be referenced directly in `output_g`. Each motor
observation must be named individually.

> **Important scope detail:** the supplied code checks direct
> hierarchical parents (`parents`), not all ancestors. If no direct
> parent uses the expected motor group in `input_f`, the graph raises a
> `ValueError`.

An individual observation may belong to only one group.

## 9. Node and Relationship Structure

Each node includes, among other fields:

-   `type`;
-   `parents`;
-   `children`.

A state node also retains metadata: `layer`, `f`, `g`, `p`, `V`,
`input_f`, `output_f`, `input_g`, `output_g`, `input_p`, `output_p`,
`input_V`, `output_V`, and `state_parents`.

The attribute `a` is no longer part of the structure; it has no role in
the model described here.

Simplified example:

``` python
{
    "mouse_position": {
        "type": "state",
        "layer": 1,
        "f": None,
        "g": None,
        "p": None,
        "V": None,
        "input_f": [],
        "output_f": [],
        "input_g": [],
        "output_g": [],
        "input_p": [],
        "output_p": [],
        "input_V": [],
        "output_V": [],
        "state_parents": ["predator_dynamics"],
        "parents": [
            {
                "node": "predator_dynamics",
                "connection": "hierarchy"
            }
        ],
        "children": []
    }
}
```

Relationships are stored in both directions: the child appears in the
parent's `children`, and the parent appears in the child's `parents`.
The link types used are `hierarchy`, `action`, and `observation`.

Intermediate variables do not become nodes unless they are explicitly
declared as states, actions, or observations.

## 10. Convention for Representing Derivatives

In the model's dynamic representation, each element represents a
derivative of a given order. A representation such as:

``` python
[
    [0, 0],  # order 0
    [0, 0],  # order 1
    [0, 0],  # order 2
]
```

can represent three derivative orders, each with two components:

-   row 0: state value (order 0);
-   row 1: first derivative;
-   row 2: second derivative.

The same convention can be applied to scalars or lists, provided the
model functions interpret the structure consistently.

## 11. Checks Before Calling `build_graph()`

Check that:

-   every state has `layer`;
-   every parent exists in `generative_functions`;
-   state names, individual action names, and individual observation
    names do not overlap;
-   every individual action belongs to only one group;
-   every individual observation belongs to only one group;
-   `input_f` entries correspond to a state, an action, an action group,
    an observation, or a variable produced by `output_f`;
-   `output_g` entries correspond to a group or individual observation
    when an observation link is expected;
-   motors are referenced individually and satisfy the direct-parent
    rule;
-   `p` has `input_p` and `output_p` when defined;
-   `V` has `input_V` and `output_V` when defined;
-   every state whose `input_f` uses an action defines `V` with its
    inputs and outputs;
-   hierarchical ancestors of states that depend on actions define `p`.

Separately, before running inference, check that `pi_obs` contains a
precision for every individual observation whose precision is used. If
missing observations are represented by `None`, verify that the
inference handler can temporarily lower and later restore each
observation's precision.

The supplied code does not check whether the hierarchical graph is
acyclic. If an acyclic hierarchy is required, this validation must be
added separately.

## 12. Recommended Procedure for Adding a State

1.  Choose a unique name.
2.  Define `layer` according to the expected temporal regime.
3.  Declare parent states in `parents`.
4.  Define `f`, `input_f`, and `output_f`.
5.  Define `g`, `input_g`, and `output_g`.
6.  Add `p`, `input_p`, and `output_p` if a policy is required.
7.  Add `V`, `input_V`, and `output_V` if the state's `input_f` directly
    depends on an action.
8.  Declare any required actions and observations in their dictionaries.
9.  Add or verify the individual `pi_obs` entry for each observation
    used by inference.
10. Check the special constraints for motors.
11. Build the graph and fix any validation errors.

## Summary

-   `parents` defines the explicit hierarchy.
-   `layer` describes the expected temporal update level; the scheduling
    mechanism must be implemented outside `build_graph()`.
-   `f` describes the temporal transition.
-   `g` describes observation generation.
-   `p` describes the policy.
-   `V` describes temporal transition cost.
-   `actions` and `observations` group individual nodes, but the groups
    themselves are not nodes.
-   `pi_obs` is associated with individual observation names, not groups
    or states.
-   Missing observations are handled during inference by substituting a
    numeric placeholder and temporarily lowering precision; this does
    not change graph structure.
-   `a` is no longer used.
-   The derivative convention must be reconciled explicitly with the
    shape expected by `make_distributions()`.

**General principle:** the configuration declares states, functions,
dependencies, distributions, and node groups. The initialization code
checks part of their consistency and builds internal structures;
temporal rules and actual function execution must be provided by the
model engine.
