# README — Generative Model Configuration and Initialization

## 1. Purpose

The `model()` method initializes the generative model from its
functional definition, its prior distributions, and its action and
observation groups.

The main parameters and structures are:

-   `generative_functions`: states, functions `f`, `g`, `p`, and `V`,
    and their dependencies;
-   `D`: means, variances, and covariances used to initialize state
    distributions;
-   `observations`: groups of individual observations;
-   `actions`: groups of individual actions;
-   `pi_obs`: a mapping from **individual observation names** to
    observation precisions, managed by the inference setup rather than
    initialized by the current `model()` method shown;
-   `pi_p`: optional precisions associated with `p` functions;
-   `pi_priors`: prior precisions, initialized from variances if they
    are not already present.

The method validates the configuration, builds PyTorch distributions,
initializes the graph if necessary, and prepares internal structures.

> **Important:** `build_graph()` records `layer` as metadata. The
> layer-based temporal update behavior described by the model must be
> implemented by the execution mechanism; graph construction alone does
> not implement it.

## 2. General Architecture

  -----------------------------------------------------------------------
  Element                             Role
  ----------------------------------- -----------------------------------
  `generative_functions`              Declares states and functions `f`,
                                      `g`, `p`, and `V`, along with their
                                      inputs and outputs.

  `D`                                 Defines the prior-distribution
                                      parameters for each state.

  `observations`                      Declares groups of individual
                                      observations.

  `actions`                           Declares groups of individual
                                      actions.

  `pi_obs`                            Maps individual observation names
                                      to their precisions.

  `pi_p`                              Defines precisions associated with
                                      `p` functions.

  `pi_priors`                         Stores prior precisions derived
                                      from variances or supplied
                                      beforehand.
  -----------------------------------------------------------------------

The keys in `D` must match the keys in `generative_functions` exactly.
States are matched by name, not by declaration order.

## 3. Defining States and Their Functions

Each state must define `f`, `input_f`, `output_f`, `g`, `input_g`, and
`output_g`. `p` and `V` are conditional, but their metadata is required
whenever either function is declared.

  -----------------------------------------------------------------------
  Field                               Rule
  ----------------------------------- -----------------------------------
  `layer`                             Required by `build_graph()`;
                                      indicates the expected temporal
                                      update level.

  `parents`                           Hierarchical references to existing
                                      states.

  `f`, `input_f`, `output_f`          Required according to
                                      `validate_model()`.

  `g`, `input_g`, `output_g`          Required according to
                                      `validate_model()`.

  `p`, `input_p`, `output_p`          If `p` is defined, `input_p` and
                                      `output_p` must also be defined.

  `V`, `input_V`, `output_V`          If `V` is defined, `input_V` and
                                      `output_V` must also be defined.
  -----------------------------------------------------------------------

Example:

``` python
generative_functions = {
    "survival_quality": {
        "layer": 3,
        "parents": [],
        "f": "pred_survival_quality",
        "input_f": ["survival_quality"],
        "output_f": ["next_survival_quality"],
        "g": "pred_relational_dynamics",
        "input_g": ["survival_quality"],
        "output_g": [
            "predator_pred",
            "ressource_pred",
            "satiety_pred",
        ],
        "p": "policy_for_survival",
        "input_p": [
            "survival_quality",
            "pi_survival_quality",
            "error_survival_quality",
        ],
        "output_p": ["policy_survival"],
    },
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
    },
}
```

### Rules for `p`

`p` represents the policy function. It is optional for some states.
However, `validate_model()` traverses the parents and ancestors of
states whose `input_f` contains an individual action or an action group;
each ancestor found must define `p`.

### Rules for `V`

`V` represents the temporal transition cost function. According to
`validate_model()`, any state whose `input_f` directly contains an
individual action or an action group must define `V`, `input_V`, and
`output_V`.

These requirements are distinct:

-   `V` is required on the state that directly uses the action in
    `input_f`;
-   `p` is required on the hierarchical ancestors of states that depend
    on an action.

Function names must be resolvable by the execution system. Graph
construction stores their names and inputs/outputs as metadata; it does
not execute them.

### Hierarchy and timing

`parents` defines hierarchical links. `layer` describes the expected
temporal update regime: lower layers are updated more continuously,
while higher layers depend on errors and on the ability of lower layers
to stabilize **free energy**.

The supplied `build_graph()` does not implement this scheduling; it only
stores the `layer` value. Do not confuse the conceptual convention with
an execution mechanism that is already active.

## 4. Defining Prior Distributions with `D`

`D` contains the initial statistical parameters for each state:

``` python
D = {
    "position": {
        "mu": [[0, 0]],
        "variance": [[100, 100]],
        # "covariance": [...]  # optional
    }
}
```

-   `mu`: mean or mean structure;
-   `variance`: variances, with exactly the same shape as `mu`;
-   `covariance`: optional covariance matrices for multivariate
    distributions.

In `init_distributions()`, if `mu` or `variance` is missing, its default
value is `1`. Supplying both explicitly is recommended. Values are
converted to PyTorch `float32` tensors.

### Exact interpretation of shapes in the current code

`make_distributions()` interprets the shape of `mu` as follows:

  -----------------------------------------------------------------------
  Shape of `mu`                       Current interpretation
  ----------------------------------- -----------------------------------
  Scalar, e.g. `0`                    One scalar `Normal` distribution.

  1D list, e.g. `[0, 0, 0]`           Three scalar `Normal`
                                      distributions.

  2D list,                            Two `MultivariateNormal`
  e.g. `[[0, 0, 0], [0, 0, 0]]`       distributions of dimension 3.

  2D list with one row,               One `MultivariateNormal`
  e.g. `[[0, 0, 0, 0, 0, 0]]`         distribution of dimension 6.
  -----------------------------------------------------------------------

Thus, a 1D list represents multiple scalar distributions, while a 2D
list represents one multivariate distribution per row.

Example: six independent scalar distributions.

``` python
D = {
    "state_a": {
        "mu": [0, 0, 0, 0, 0, 0],
        "variance": [100, 100, 100, 100, 100, 100],
    }
}
```

Example: one multivariate distribution of dimension 3.

``` python
D = {
    "state_a": {
        "mu": [[0, 0, 0]],
        "variance": [[100, 100, 100]],
    }
}
```

### Important: derivatives and distributions use different conventions

In the model's dynamic representation, a structure such as:

``` python
[
    [0, 0],  # order 0
    [0, 0],  # order 1
    [0, 0],  # order 2
]
```

may represent the state value, its first derivative, and its second
derivative. Under this convention, each row represents a derivative
order, for both scalars and lists. This interpretation must remain
consistent with the shape interpretation used by `make_distributions()`.

## 5. Choosing Between `Normal` and `MultivariateNormal`

For a scalar or a 1D list, the code creates `torch.distributions.Normal`
distributions, with standard deviation calculated as the square root of
the variance:

``` python
dist.Normal(
    loc=mu,
    scale=torch.sqrt(variance),
)
```

For each row of a 2D list, the code creates a
`torch.distributions.MultivariateNormal` distribution:

``` python
dist.MultivariateNormal(
    loc=mu,
    covariance_matrix=covariance,
)
```

A multivariate distribution can model covariance between components. Its
covariance matrix must be symmetric and positive definite.

## 6. Defining Covariances

If `covariance` is absent for a 2D entry, `make_distributions()` builds
a diagonal matrix using the values in `variance`.

For three components with variances `[100, 100, 100]`, the matrix is:

``` python
[
    [100,   0,   0],
    [  0, 100,   0],
    [  0,   0, 100],
]
```

Example with off-diagonal covariances:

``` python
D = {
    "state_a": {
        "mu": [[0, 0, 0]],
        "variance": [[100, 100, 100]],
        "covariance": [
            [
                [100, 50, 0],
                [50, 100, 20],
                [0, 20, 100],
            ]
        ],
    }
}
```

Constraints checked by `make_distributions()`:

1.  `mu` and `variance` have exactly the same shape;
2.  `mu` has no more than two dimensions;
3.  values in `mu` and `variance` are finite;
4.  variances are strictly positive;
5.  for a 2D entry, `covariance` has shape
    `(number_of_distributions, dimension, dimension)`;
6.  the diagonal of each covariance matrix matches `variance`;
7.  each matrix is symmetric and valid for `MultivariateNormal`,
    i.e. positive definite.

Providing a covariance for a scalar or 1D list raises a `ValueError`.

## 7. Accessing `self.Q_`

After initialization, `self.Q_` is a dictionary indexed by state name.
Each value is a list of PyTorch distributions.

``` python
q = self.Q_["state_a"]
```

If `state_a` has two multivariate distributions, they are accessible as
`q[0]` and `q[1]`. If its configuration contains six scalar
distributions, they are accessible from `q[0]` to `q[5]`.

For a multivariate distribution:

``` python
q[0].mean
q[0].covariance_matrix
q[0].sample()
```

For a scalar distribution:

``` python
q[0].mean
q[0].variance
q[0].sample()
```

The number of distributions is `len(self.Q_["state_a"])`.

## 8. Prior Precisions: `pi_priors`

If no precision is already defined for a state, `init_distributions()`
initializes it from the inverse of the variance:

$$
\pi = \frac{1}{\mathrm{variance}}
$$

For example, variances `[100, 25, 4]` produce precisions
`[0.01, 0.04, 0.25]`.

The inversion is recursive for nested lists. Precisions already present
in `pi_priors` are preserved and are not recalculated automatically.

Implementation note: `inverse_variance()` checks that values are
positive, but does not explicitly check their finiteness before
inversion. The variances used to create distributions are checked by
`make_distributions()`.

## 9. Observation Precisions: `pi_obs`

`pi_obs` is keyed by **individual observation name**, not by state name
or observation-group name. Each entry specifies the precision associated
with one observation.

Example:

``` python
pi_obs = {
    "y_cheese": torch.tensor(1.0),
    "x_cat": torch.tensor(1.0),
}
```

The values shown are illustrative. Choose precisions appropriate to the
scale and reliability of each observation. A higher precision gives an
observation more influence in the observation-error term; a lower
precision reduces its influence.

In the current implementation described here, the `model()` method no
longer calls `init_observations()` and does not accept `pi_obs` in its
shown signature. The surrounding application must therefore ensure that
`self.pi_obs` is initialized before `inference()` is called. The keys
must match the individual observations that inference may process.

### Missing observations

The inference path handles an observation whose value is `None` before
calling `beliefs_update()`:

1.  it saves the original precision the first time the observation is
    found missing;
2.  it temporarily assigns a low precision (currently `1e-7` by
    default);
3.  it replaces `None` with a numeric substitute (currently `0.0` by
    default) so downstream tensor calculations can proceed;
4.  when the observation becomes available again, it restores the saved
    precision.

The substitute is only a computational placeholder. Its influence is
reduced by the low precision; it must not be interpreted as a real
measurement. A real observation equal to zero remains a valid
observation and must not be treated as missing.

The backup dictionary, named `pi_obs_substitued` in the current code,
tracks the original precisions for observations that are currently
missing. Keep its lifecycle aligned with `pi_obs`. Ensure the
precision-update code does not overwrite the temporary low precision
before the missing observation is processed, and verify that all
precision tensors have compatible device and dtype with the
calculations.

## 10. Precisions and Errors Associated with `p`

`init_p()` initializes parameters for states that declare a `p`
function. States without `p` are ignored.

For each relevant state:

-   `pi_p[state_name]` receives the precision for `p`, defaulting to
    `1`;
-   `inputs_map["pi_p_<state_name>"]` receives this precision;
-   `inputs_map["error_<state_name>"]` is initially assigned the same
    precision;
-   `inputs_map["pi_<state_name>"]` receives the state's prior precision
    from `pi_priors[state_name]`.

Example:

``` python
pi_p = {
    "survival_quality": 5
}
```

Initialization assigns `5` to `pi_p_survival_quality` and
`error_survival_quality`. It also assigns `pi_survival_quality` the
state's prior precision, which is not necessarily equal to `5`.

Corresponding call:

``` python
self.pi_p, self.inputs_map = model.init_p(
    generative_functions,
    self.pi_priors,
    pi_p=pi_p,
    inputs_map=self.inputs_map,
)
```

The argument order must match the actual signature of
`init_p(generative_functions, pi_priors, pi_p=None, inputs_map=None)`.

## 11. Initializing the Graph, Actions, and Observations

`init_graph()` builds the graph only if no current graph is supplied.
When it builds the graph, it also creates a `GraphNavigator`. If a graph
already exists, the function returns `navigator = None`.

Actions and observations are organized into groups:

``` python
actions = {
    "action_movement": ["forward", "back", "left", "right"],
    "eat": ["yes", "no"],
}

observations = {
    "position_obs": ["x_position", "y_position"],
    "action_movement": ["x_motor", "y_motor"],
}
```

A group present in both `actions` and `observations` is interpreted by
`build_graph()` as a motor group. Individual actions and observations
remain separate nodes. To connect motors to a state, reference the
individual motor observations in `output_g` and satisfy the direct
hierarchical-parent validation rule.

The graph:

-   links states according to `parents`;
-   links states to actions according to `input_f`;
-   links states to observations according to `output_g`;
-   stores `f`, `g`, `p`, `V`, and their inputs/outputs as metadata.

Group names do not become nodes.

## 12. Initialization Example

``` python
actions = {
    "move": ["forward", "backward"]
}

observations = {
    "position_obs": ["x_position", "y_position"]
}

generative_functions = {
    "position": {
        "layer": 1,
        "parents": [],
        "f": "predict_position",
        "input_f": ["position", "move"],
        "output_f": ["next_position"],
        "g": "predict_position_obs",
        "input_g": ["position"],
        "output_g": ["position_obs"],
    }
}

D = {
    "position": {
        "mu": [[0, 0]],
        "variance": [[100, 100]],
    }
}

pi_obs = {
    "x_position": torch.tensor(1.0),
    "y_position": torch.tensor(1.0),
}
```

Then, according to the current `model()` signature:

``` python
model_instance.model(
    generative_functions=generative_functions,
    functions_map=functions_map,
    observations=observations,
    actions=actions,
)
```

`functions_map` must contain the mappings needed to resolve the
functions. The example assumes `torch` is imported and that `pi_obs` is
assigned to the agent (or otherwise initialized by the application)
before inference. The current `model()` signature shown does not take
`D` or `pi_obs` as arguments; `D` is supplied to the agent through its
own configuration/initialization path.

After initialization:

``` python
q_position = model_instance.Q_["position"]

print(len(q_position))       # 1
print(q_position[0].mean)    # tensor([0., 0.])
print(q_position[0].sample())
```

Here, `mu = [[0, 0]]` creates one multivariate distribution of dimension
2.

## 13. Validation Errors and Limitations

According to the supplied code, `validate_model()` may report:

-   `D` or `generative_functions` is missing;
-   state keys differ between `D` and `generative_functions`;
-   a required field is missing from `f` or `g`;
-   an unknown hierarchical parent;
-   missing `input_p` or `output_p` when `p` is defined;
-   missing `input_V` or `output_V` when `V` is defined;
-   missing `V` for a state whose `input_f` directly uses an action;
-   missing `p` for an ancestor of a state whose `input_f` uses an
    action.

`make_distributions()` may report:

-   different shapes for `mu` and `variance`;
-   non-finite values;
-   non-positive variances;
-   more than two dimensions for `mu`;
-   a covariance with an incorrect shape, one that is not symmetric, or
    one that is not positive definite;
-   a covariance diagonal that differs from `variance`.

`build_graph()` checks, among other things, that layers exist, parent
references are valid, group membership is not duplicated, node names do
not collide, `input_f` references are valid, and motor rules are
satisfied.

Limitations to keep in mind:

-   the graph does not check for cycles in the hierarchy;
-   `layer` does not itself schedule updates in the code shown;
-   functions named in metadata are not executed by `build_graph()`;
-   the derivative convention must be reconciled with the shape
    interpretation used by `make_distributions()`;
-   the current `model()` method does not initialize observation
    precisions; the application must initialize `pi_obs` consistently
    before inference;
-   the missing-observation handler assumes that a missing observation's
    original precision can be backed up and restored by individual
    observation name.

## 14. Key Rules to Remember

1.  Every state in `generative_functions` must have a corresponding
    entry in `D`.
2.  `mu` determines the number and dimension of distributions under the
    current implementation.
3.  `variance` must have exactly the same shape as `mu`.
4.  A 1D list represents multiple scalar normal distributions.
5.  A 2D list represents one multivariate distribution per row.
6.  Without an explicit covariance, multivariate covariance matrices are
    diagonal.
7.  `parents` defines hierarchical links.
8.  `layer` describes the expected temporal update level, but
    `build_graph()` does not guarantee its execution.
9.  `p` is the policy function; required ancestors must define it
    according to the validation rules.
10. `V` is the temporal transition cost function; it is mandatory for
    states whose `input_f` directly contains an action.
11. Action and observation groups are metadata, not nodes.
12. `pi_obs` is indexed by individual observation names and must be
    initialized before inference.
13. A missing observation is replaced with a placeholder and assigned
    low precision temporarily; its original precision is restored when
    it returns.
14. The derivative convention must be separated explicitly from, or
    implemented in, the distribution structure if it is to apply to `D`.

**General principle:** the configuration declares states, functions,
dependencies, distributions, and node groups. The initialization code
checks part of their consistency and builds internal structures;
temporal rules and actual function execution must be provided by the
model engine.
