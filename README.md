# DAIH-DynAmI

**Deep Active Inference Hybrid Framework with Dynamic Planning Horizon
and Amortized Active Inference**

DAIH-DynAmI is an experimental hybrid active inference framework. It
aims to combine continuous belief updating, prediction of action
consequences, amortized policy generation, and reflex motor control. Its
guiding principle is to find an action that is sufficiently consistent
with the internal model within a limited computational budget, rather
than exhaustively exploring every possible policy.

> **Project status:** implemented prototype under testing and
> validation. The core architecture and intended mechanisms are present
> in the codebase. The project is undergoing testing, debugging, and
> validation to verify that the implementation behaves as expected and
> remains consistent with the theoretical framework. Further refinements
> may be introduced as testing progresses.

## Table of Contents

-   [Objectives](#objectives)
-   [Model Principles](#model-principles)
-   [Software Architecture](#software-architecture)
-   [Preprocessing and Output Mapping](#preprocessing-and-output-mapping)
-   [Inference and Belief Updating](#inference-and-belief-updating)
-   [Policy Selection](#policy-selection)
-   [Reflex Motor Control](#reflex-motor-control)
-   [Model Configuration](#model-configuration)
-   [Execution](#execution)
-   [Main Parameters](#main-parameters)
-   [Limitations and Items to Verify](#limitations-and-items-to-verify)
-   [Roadmap](#roadmap)
-   [Additional Documentation](#additional-documentation)
-   [License and Citation](#license-and-citation)

## Objectives

The project explores an agent architecture based on the following ideas:

1.  **Continuous perceptual inference:** adjust beliefs to reduce
    discrepancies between observations and predictions from the
    generative model.
2.  **Hierarchical states:** organize states into dependency levels and
    temporal update regimes.
3.  **Generalized coordinates of motion:** represent a state and its
    time derivatives within a single structure.
4.  **Amortized decision-making:** use policy-generating functions to
    propose candidates without systematically traversing the entire
    policy space.
5.  **Critical evaluation:** simulate the predicted consequences of a
    policy and compare its expected score with its reevaluated score.
6.  **Fast motor control:** adjust motor outputs using gradient descent
    on variational free energy.

The design hypothesis is that an agent operating in a dynamic
environment must be able to make decisions on time and with limited
resources. A sufficiently good and consistent solution may be preferable
to a more expensive exhaustive search.

## Model Principles

### Generative model

The model is described in terms of states and associated functions:

-   `f`: state transition or temporal prediction;
-   `g`: prediction of observations from states;
-   `p`: policy generation;
-   `V`: cost or value associated with a temporal transition;
-   `parents`: explicit hierarchical dependencies;
-   `layer`: expected temporal update level.

Actions and observations are declared as groups, but their individual
elements become nodes in the graph. The functions named in the
configuration are references: their execution is handled by the model
engine, not by graph construction alone.

### Generalized coordinates of motion

A state can be represented by several derivative orders:

$$
\tilde{\mu} =
\begin{bmatrix}
\mu^{(0)} \\
\mu^{(1)} \\
\vdots \\
\mu^{(n)}
\end{bmatrix}
$$

The conceptual dynamics of a generalized coordinate are:

$$
\mu^{(i)}_{t+\Delta t} \approx \mu^{(i)}_t + \left( D\mu^{(i)}_t - \frac{\partial F}{\partial \mu^{(i)}_t} \right)\Delta t
$$

where $D\mu^{(i)}$ denotes the next derivative order. The final
order is closed by setting its next derivative to zero. This expression
describes the intended principle; the exact tensor shapes and their
interpretation must remain consistent across the functions that prepare
inputs, calculate $F$, and perform updates.

For a state with three orders, the temporal error term can be written
as:

$$
\begin{bmatrix}
\dot{\mu}_0 + \mu_1 - x_1 \\
\dot{\mu}_1 + \mu_2 - x_2 \\
\dot{\mu}_2 - x_3
\end{bmatrix}
$$

where the $x_i$ represent prior outputs of the transition function `f`,
according to the model's convention.

## Software Architecture

The main code defines the `Agent_DAIH` class. It relies on several
specialized modules.

  -----------------------------------------------------------------------
  Module / component                  General responsibility
  ----------------------------------- -----------------------------------
  `Agent_DAIH`                        Coordinates initialization,
                                      inference, decision-making, and
                                      motor commands.

  `model`                             Validates the configuration,
                                      initializes distributions, and
                                      builds the graph.

  `preprocessing`                     Prepares state inputs and
                                      quantities required for the $F$ and
                                      $G$ calculations.

  `inference`                         Provides update, differentiation,
                                      error-calculation,
                                      missing-observation handling, and
                                      policy-management operations.

  `model_components`                  Produces policy candidates.

  `utils`                             Groups utilities, including
                                      gradient-descent updates and action
                                      initialization.

  `GraphNavigator`                    Traverses states, layers,
                                      relationships, and graph engines.
  -----------------------------------------------------------------------

### General cycle

``` text
Model configuration
        |
        v
Initialize distributions, precisions, and graph
        |
        v
Receive observations
        |
        v
Handle missing observations and update beliefs, layer by layer
        |
        +----> Reflex adjustment of motor outputs
        |
        v
Generate candidate policies
        |
        v
Simulate and calculate posterior G
        |
        v
Consistency / tolerance test
        |
        +----> Rejection: adjust error and retry
        |
        v
Select and record the action
```

In the current implementation, `inference(observations)` first handles
missing observations, then runs `beliefs_update(observations)`, yields
any motor commands, and proceeds to policy selection where applicable.

## Preprocessing and Output Mapping

The preprocessing layer bridges the graph representation and the
`WorldModel.forward()` interface. It prepares the current state, its
identifiers, the function type, optional action information, and any
function-specific parameters.

### Preparing WorldModel inputs

`prepare_world_model_inputs(...)` builds a keyword-argument dictionary. The
state must exist in `inputs_map`, and its identifier is obtained from its
position in `states_list`:

$$
\text{state\_id} = \operatorname{index}(\text{state\_name}) + 1
$$

The layer identifier is obtained from
`navigator.get_state_layer(state_name)`. The common arguments are:

-   `state`: current state value from `inputs_map`;
-   `state_id`: one-based position of the state in `states_list`;
-   `layer_id`: layer returned by the graph navigator;
-   `function_type`: one of the model function types.

Actions are detected from the declared `input_names`. A name is treated as
an action when it is an individual action node or an action group recognized
by `navigator.get_actions_by_group(...)`. The corresponding value must
exist in `inputs_map`.

`WorldModel.forward()` accepts at most one action argument: either one
individual action or one action group. If more than one action or group is
detected, preprocessing raises an error. The action value is passed under
the `action` keyword.

The optional parameters `max_steps` and `deterministic` are forwarded only
when they are declared in `input_names` and are present in `inputs_map`.

Function-specific arguments are prepared as follows:

  -----------------------------------------------------------------------
  Function type   Additional argument
  -------------   -------------------------------------------------------
  `g`             `output_shape`, calculated from observational and
                  hierarchical children of the state
  `V`             `output_shape = (1, 1)`
  `p`             `temperature`, read from
                  `inputs_map[f"error_{state_name}"]`
  `f`             No additional argument is added by this preprocessing
                  function
  -----------------------------------------------------------------------

### Computing output dimensions

`get_distribution_width(distribution)` determines how many columns a
distribution represents. A scalar `Normal` distribution has an empty
`event_shape` and therefore represents one column. A multivariate
distribution uses the product of its event dimensions:

$$
w(d) =
\begin{cases}
1 & \text{if } \operatorname{event\_shape}(d) = () \\
\displaystyle\prod_{k=1}^{m} s_k
& \text{if } \operatorname{event\_shape}(d) = (s_1,\ldots,s_m)
\end{cases}
$$

The product is implemented with `math.prod(event_shape)`.

`get_output_shape(...)` examines the children of the current state and
separates them by connection type:

-   **Observation children:** contribute one row and one column per
    individual observation. For \(n_o\) observations, their shape is
    \((1,n_o)\).
-   **Hierarchy children:** their distributions are retrieved from `Q_`.
    The number of distributions determines the row count, and the event
    shape of the first distribution determines the column count. All
    distributions belonging to that child must have the same width.
-   **Action children:** are ignored when calculating the output shape.

Let the contributing child shapes be \((R_1,C_1),\ldots,(R_n,C_n)\). They
must all have the same row count \(R\). The resulting shape concatenates
their columns:

$$
\operatorname{output\_shape}
=
\left(R,\sum_{i=1}^{n} C_i\right)
$$

An error is raised when the state has no children, no observational or
hierarchical child contributes to the dimensions, a referenced child is
missing from `Q_`, distributions have inconsistent widths, or the
contributing children have different row counts.

This means observational and hierarchical children can coexist, but their
row counts must be compatible. In particular, an observation contribution
has one row, so it can coexist with a hierarchical contribution only when
that contribution also has one row.

### Mapping function outputs

`map_outputs(output_names, output, navigator)` maps a function result to the
names declared in the graph.

-   With no declared output names, the function must return `None`;
    otherwise an error is raised.
-   With exactly one output name, the complete returned object is mapped
    to that name unchanged. This is the state-output convention.
-   With multiple output names, every name must identify either an
    individual observation or an observation group. Groups are expanded
    according to the number of individual observations they contain.

For multiple outputs, the expected number of observation values is:

$$
N_{\text{expected}} = \sum_{j=1}^{n} s_j
$$

where \(s_j=1\) for an individual observation and \(s_j\) is the number of
individual observations in a group.

For a tensor, the last dimension must equal \(N_{\text{expected}}`. The
tensor is split along that dimension while preserving the declared output
order:

$$
\text{chunks}
=
\operatorname{split}
\left(
\text{output},
(s_1,\ldots,s_n),
\text{dim}=-1
\right)
$$

For a tuple or list, its length must equal \(N_{\text{expected}}\), and
consecutive slices are assigned to the declared output names. A
mapping/dictionary is not accepted when multiple outputs are declared.

Unsupported output types, unknown observation names, and size mismatches
raise explicit errors.

The resulting dictionary maps each declared output name to its corresponding
tensor chunk or sequence slice. The order of names in `output_names` is
therefore significant.

## Inference and Belief Updating

### Variational free energy $F$

Variational free energy combines a complexity term related to prior
beliefs and an accuracy term related to observations. The conceptual
formulation used by this project is:

$$
F(\tilde{\mu}) = \left[
\frac{1}{2}
(\dot{\tilde{\mu}} + D\tilde{\mu} - \tilde{\mu}_{\text{prior}})^2
\Pi_{\text{prior}}
-\frac{1}{2}\ln|\Pi_{\text{obs}}|
-\frac{1}{2}\ln|\Pi_{\text{prior}}|
\right] - \left[
-\frac{1}{2}
(\tilde{y}-g(\tilde{\mu}))^2\Pi_{\text{obs}}
\right]
$$

The implementation calculates free energy through `free_energy(...)`.
Its sign conventions, normalization constants, tensor shapes, and
argument names must remain consistent with the theoretical formulation
as the model evolves.

The main symbols are:

-   $\tilde{\mu}$: generalized belief or estimated state;
-   $D\tilde{\mu}$: next generalized derivative;
-   $\tilde{\mu}_{\text{prior}}$: prior prediction from
    the dynamics;
-   $\Pi_{\text{prior}}$: precision of prior beliefs;
-   $\tilde{y}$: available observation;
-   $g(\tilde{\mu})$: predicted observation;
-   $\Pi_{\text{obs}}$: precision associated with an
    individual observation.

### Belief updating

The `beliefs_update(observations)` method traverses the layers returned
by the graph navigator. For each state, it prepares predictions,
beliefs, and precisions; calculates $F$; produces motor commands where
relevant; updates the posterior mean using an Euler step; updates
observation and prior precisions; estimates precision from the second
derivative of $F$; and aggregates free energy by layer.

Observation precision is maintained per individual observation. When an
observation is missing (`None`), the inference preprocessing replaces it
with a numerical substitute and temporarily lowers its precision so that
the placeholder has negligible influence. When the observation becomes
available again, its saved precision is restored. See the configuration
and model-definition guides for the relevant data conventions.

### Precisions and distributions

The model uses PyTorch `Normal` or `MultivariateNormal` distributions,
depending on the shape of the initial parameters. Prior precisions can
be initialized from variances:

$$
\Pi_{\text{prior}} = \frac{1}{\sigma^2}
$$

A high precision corresponds to lower uncertainty. Multivariate
covariance matrices must have valid shapes, be symmetric, and be
positive definite.

## Policy Selection

### Expected free energy $G$

Action selection uses **Expected Free Energy** ($G$) to estimate the
consequences of a policy. The conceptual formulation provided by this
project is:

$$
G(\pi) = \frac{1}{2}\ln(2\pi e)
-\frac{1}{2}\ln(\Pi_{\text{prior}})
-\frac{1}{2}\ln(2\pi e,\Sigma^2)
-V(f(\mu))
$$

This expression must be interpreted according to the precision,
variance, and transition-cost conventions adopted by the model. The
current code calculates the score using `execepted_free_energy(...)`,
separating an ambiguity term from a transition-risk/cost term. The
term-by-term correspondence with the theoretical equation should be
verified as the project evolves.

### Amortized policy generation $P$

The theoretical architecture proposes that one or more $P$ functions
quickly generate a candidate policy and its expected score:

$$
P_t(S_t,\Pi_{\text{prior}},\epsilon) \longrightarrow (\pi,G_{\text{prior}})
$$

where $S_t$ represents the relevant internal state,
$\Pi_{\text{prior}}$ the associated precisions,
$\epsilon$ the prediction error used to adapt generation,
$\pi$ the candidate policy, and $G_{\text{prior}}$ its
expected score.

Policy length may depend on precision or uncertainty: when confidence
decreases, the agent should favor a shorter, more cautious horizon. This
is an architectural objective; its exact behavior depends on the
implementation of the `p` functions and candidate generator.

### Tolerance loop

The `policy_selection()` method retrieves candidates through
`model_components.get_policy_candidates(...)`, simulates actions in
local copies of beliefs and inputs, and calculates a reevaluated score
$G_{\text{posterior}}$.

The error described by the model is:

$$
\epsilon = \Pi_P(G_{\text{prior}}-G_{\text{posterior}})^2
$$

The theoretical tolerance rule is:

-   if $\epsilon \leq 1$, the candidate is considered
    consistent with the prediction;
-   otherwise, the candidate is rejected and the error is fed back into
    the generator inputs to encourage another proposal;
-   precision $\Pi_P$ is adjusted to change tolerance and,
    eventually, the planning horizon.

The current code implements a threshold comparison, an update to `pi_p`,
and a selection among accepted candidates. A bounded retry strategy and
the behavior when no candidate is accepted remain important validation
points.

### Learning during wakefulness and sleep

The proposed architecture distinguishes two modes:

-   **Wakefulness:** locally adjust the policy generator to reduce the
    gap between the predicted score and the reevaluated score.
-   **Rest / sleep:** replay past trajectories and train the generative
    functions using a global loss based on $F$.

This separation is an architectural proposal. The supplied main code
does not, by itself, show a complete sleep or trajectory-replay loop.

## Reflex Motor Control

Motors are represented by motor observations associated with an action
group bearing the same name. The `motor_command(...)` method checks
whether a state is connected to motors, then adjusts the predicted
output of `g` using gradient descent on $F$.

The conceptual rule is:

$$
y_{\text{new}} = y_{\text{pred}} - \Delta t \, k \, \frac{\partial F}{\partial y_{\text{pred}}}
$$

The motor update uses the step size supplied to the gradient-descent
operation. The time step and update coefficient therefore determine the
magnitude of each adjustment. Their values must be evaluated for control
stability and convergence.

## Model Configuration

The model is defined using dictionaries. A schematic example:

``` python
actions = {
    "action_movement": ["forward", "back", "left", "right"]
}

observations = {
    "position_obs": ["x_position", "y_position"],
    "action_movement": ["x_motor", "y_motor"],
}

generative_functions = {
    "position": {
        "layer": 1,
        "parents": [],
        "f": "predict_position",
        "input_f": ["position", "action_movement"],
        "output_f": ["next_position"],
        "g": "predict_position_observation",
        "input_g": ["position"],
        "output_g": ["position_obs"],
    }
}

D = {
    "position": {
        "mu": [[0.0, 0.0]],
        "variance": [[100.0, 100.0]],
    }
}
```

This example illustrates the general structure and must be adapted to
the actual function signatures. The names referenced by `f`, `g`, `p`,
and `V` must be available to the function-resolution mechanism.

### Field roles

-   `parents` defines hierarchical links between states.
-   `layer` indicates the expected temporal regime; it is not sufficient
    on its own to schedule updates.
-   `input_f` / `output_f` declare the inputs and outputs of the
    dynamics.
-   `input_g` / `output_g` declare the inputs and outputs of observation
    prediction.
-   `input_p` / `output_p` are required when a `p` function is declared.
-   `input_V` / `output_V` are required when a `V` function is declared.
-   `D` contains initial means, variances, and, if necessary,
    covariances.
-   `pi_obs` maps individual observation names to their precisions; it
    is used by inference-time observation handling and is not
    initialized by the current `model()` method shown in the code.
-   `pi_p` optionally configures precisions associated with policy
    generators.

### Rules to follow

1.  Every state declared in `generative_functions` must have a
    corresponding entry in `D`.
2.  The keys in `D` and `generative_functions` must match.
3.  Each parent must refer to an existing state.
4.  State, action, and observation names must be distinguishable.
5.  An individual action or observation must belong to only one group.
6.  When a state directly depends on an action in `input_f`, comply with
    the `V` validation requirements.
7.  Hierarchical ancestors involved in a dependency on an action must
    define `p`, according to the model's validation rules.
8.  To connect motors, declare the group in both `actions` and
    `observations`, then reference the individual motor observations in
    `output_g` according to the graph rules.

## Execution

The conceptual entry point is an instance of `Agent_DAIH`:

``` python
agent = Agent_DAIH()

agent.model(
    generative_functions=generative_functions,
    functions_map=functions_map,
    observations=observations,
    actions=actions,
    pi_p=pi_p,
)

for result in agent.inference(current_observations):
    print(result)
```

The variables `functions_map`, `pi_p`, and `current_observations` must
be defined for the use case. In the current method signature shown,
`model()` does not accept `pi_obs`; ensure `agent.pi_obs` is initialized
by the surrounding application before inference. Check the repository's
exact signatures and required modules before using this example: it is a
starting template, not a guaranteed runnable example.

## Main Parameters

The default values visible in `Agent_DAIH.__init__` include:

  ---------------------------------------------------------------------------
  Parameter                            Initial value Visible purpose
  --------------------- ---------------------------- ------------------------

  `max_policies_iter`                           `100` Intended limit for policies
                                                     iterations.

  ---------------------------------------------------------------------------

The time step `dt=0.1` remains hard-coded in several update calls.

## Limitations and Items to Verify

Before interpreting results as validation of the model, check the
following:

-   **Convention for $F$:** verify signs, normalization constants,
    dimensions, and gradients for each term.
-   **Selection loop:** add a stopping strategy or maximum number of
    attempts if no candidate passes the threshold.
-   **Updating `pi_p`:** ensure that the error and precision used
    correspond to the correct state and candidate.
-   **Numerical stability:** check for finite values, strictly positive
    precisions, and valid covariance matrices.
-   **Layer scheduling:** `layer` is graph metadata; the temporal
    mechanism must be implemented during execution.
-   **Hierarchical graph:** secondary documentation indicates that the
    current construction does not check for cycles.
-   **Observation precision lifecycle:** verify that individual `pi_obs`
    entries are initialized for all observations used by inference,
    remain positive, and are not unintentionally overwritten during
    precision updates.
-   **Unused parameters:** verify iteration and tolerance attributes
    before assuming they effectively control execution.

## Roadmap

### Priority 1 — Testing and experimental validation

-   add unit tests for $F$, $G$, belief updates, and motor control;
-   test missing values, tensors with different shapes, and extreme
    precisions;
-   compare fixed-step and adaptive-step updates;
-   record scores and actions in reproducible scenarios;
-   document limitations and experimental results.

### Priority 2 — Learning during sleep

-   replay trajectories stored during the agent's interactions;
-   adjust the generative model and policy generator using past
    experiences;
-   reduce prediction errors and improve policy selection;
-   stabilize learned parameters before interactions resume;
-   compare performance before and after the sleep phase.

## Additional Documentation

The repository's secondary guides complement this main README:

-   **[Generative Model Configuration and Initialization](./docs/config.md):** `D`
    parameters, distributions, covariances, prior precisions, and
    individual observation precisions.
-   **[Generative Model Definition Guide](./docs/generative_model.md):** states, `parents`, `layer`,
    `f`, `g`, `p`, and `V` functions, actions, observations, and
    graph-construction rules.

These documents detail the configuration contracts. This README explains
their place in the overall DAIH-DynAmI architecture.

## License and Citation

### License

This project is distributed under the MIT License. See the
[`LICENSE`](LICENSE) file for the terms of use, modification, and
redistribution.

### Theoretical inspirations

The development of this project draws in particular on the following
works on active inference, dynamic planning, and sophisticated
inference:

-   Parr, T., Pezzulo, G. & Friston, K. J. --- **Active Inference: The
    Free Energy Principle in Mind, Brain, and Behavior.** [MIT
    Press](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)
-   **Dynamic Planning in Active Inference.**
    [arXiv:2402.11658](https://arxiv.org/abs/2402.11658)
-   **Sophisticated Inference.**
    [arXiv:2006.04120](https://arxiv.org/abs/2006.04120)

------------------------------------------------------------------------

**DAIH-DynAmI**: an experimental hybrid active inference prototype
exploring coordination between generalized beliefs, amortized policy
generation, and reflex motor control.
