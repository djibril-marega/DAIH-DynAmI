# DAIH-DynAmI

**Deep Active Inference Hybrid Framework with Dynamic Planning Horizon
and Amortized Active Inference**

DAIH-DynAmI is an experimental hybrid active inference framework. It
combines continuous belief updating, prediction of action consequences,
amortized policy generation, reflex motor control, and a dedicated offline
learning process. Its guiding principle is to find actions that are
sufficiently consistent with the internal model within a limited
computational budget, rather than exhaustively exploring every possible
policy.

> **Project status:** implemented prototype under testing and validation.
> The online inference pipeline and the offline learning architecture are
> implemented in the codebase. The project remains experimental: numerical
> behavior, learning stability, and the effects of automatic architectural
> changes must be evaluated in the target environment.

## Table of Contents

-   [Objectives](#objectives)
-   [Model Principles](#model-principles)
-   [Software Architecture](#software-architecture)
-   [Part I — Online Inference](#part-i--online-inference)
    -   [Preprocessing and Output Mapping](#preprocessing-and-output-mapping)
    -   [Inference and Belief Updating](#inference-and-belief-updating)
    -   [Policy Selection](#policy-selection)
    -   [Reflex Motor Control](#reflex-motor-control)
    -   [Observation Groups and Vector Observations](#observation-groups-and-vector-observations)
    -   [Model Configuration](#model-configuration)
    -   [Running Inference](#running-inference)
-   [Part II — Offline Learning](#part-ii--offline-learning)
    -   [Purpose and Data Memories](#purpose-and-data-memories)
    -   [The Three Offline Phases](#the-three-offline-phases)
    -   [Architecture Adaptation](#architecture-adaptation)
    -   [Compatibility After Restructuring](#compatibility-after-restructuring)
    -   [Offline Configuration and Execution](#offline-configuration-and-execution)
    -   [Checkpoints and Reports](#checkpoints-and-reports)
-   [Main Parameters](#main-parameters)
-   [Limitations and Items to Verify](#limitations-and-items-to-verify)
-   [Additional Documentation](#additional-documentation)
-   [License and Citation](#license-and-citation)

## Objectives

The project explores an agent architecture based on the following ideas:

1.  **Continuous perceptual inference:** adjust beliefs to reduce
    discrepancies between observations and predictions from the generative
    model.
2.  **Hierarchical states:** organize states through explicit dependencies
    and layer metadata.
3.  **Generalized coordinates of motion:** represent a state and its time
    derivatives within a single structure.
4.  **Amortized decision-making:** use policy-generating functions to
    propose candidates without systematically traversing the entire policy
    space.
5.  **Critical evaluation:** simulate the predicted consequences of a
    policy and compare its expected score with its reevaluated score.
6.  **Fast motor control:** adjust motor outputs using gradient descent on
    variational free energy.
7.  **Offline model learning:** use memories collected during inference to
    optimize hierarchical predictions, temporal predictions, and the
    architecture itself.

The design hypothesis is that an agent operating in a dynamic environment
must make decisions on time and with limited resources. A sufficiently good
and consistent solution may be preferable to a more expensive exhaustive
search.

## Model Principles

### Generative model

The model is described in terms of states and associated functions:

-   `f`: state transition or temporal prediction;
-   `g`: prediction of observations from states;
-   `p`: policy generation;
-   `V`: cost or value associated with a temporal transition;
-   `parents`: explicit hierarchical dependencies;
-   `layer`: temporal update-level metadata.

Actions and observations are declared as groups. A group can contain
several scalar nodes, or it can declare one vector-valued observation.
The graph stores the individual nodes and their group metadata. The
functions named in the configuration are references: their execution is
handled by the model engine, not by graph construction alone.

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

where $D\mu^{(i)}$ denotes the internal dynamic transition function `f`
that predicts the next derivative order from the generalized state. The
final derivative order is structurally closed by setting its subsequent
velocity to zero. Tensor shapes, padding, and their spatial interpretation
must remain consistent across the functions that prepare inputs,
calculate $F$, and perform updates.

For a state with three orders, the temporal error term can be written as:

$$
\begin{bmatrix}
\dot{\mu}_0 + \mu_1 - x_1 \\
\dot{\mu}_1 + \mu_2 - x_2 \\
\dot{\mu}_2 - x_3
\end{bmatrix}
$$

where the $x_i$ elements represent the top-down target values projected
from the generative function `g` of the parent layer, serving as the
hierarchical prior for the current state's internal dynamics according
to the model's convention.

## Software Architecture

The main code defines the `Agent_DAIH` class. It relies on specialized
modules.

  -----------------------------------------------------------------------
  Module / component                  General responsibility
  ----------------------------------- -----------------------------------
  `Agent_DAIH`                        Coordinates model initialization,
                                      inference, decision-making, motor
                                      commands, memory collection, and
                                      access to offline training.

  `model`                             Validates the configuration,
                                      initializes distributions and
                                      numerical bounds, and builds the
                                      graph.

  `graph`                             Builds the state/action/observation
                                      graph and provides graph navigation.

  `preprocessing`                     Prepares state inputs and quantities
                                      required for the $F$ and $G$
                                      calculations, including observation
                                      groups and vector observations.

  `inference`                         Provides belief updates,
                                      differentiation, error calculation,
                                      missing-observation handling, and
                                      precision management.

  `world_generative_model`                       Executes the shared generative
                                      neural model.

  `model_components`                  Produces policy candidates and
                                      supports their evaluation.

  `policy_generator`                  Provides policy-generation
                                      components.

  `utils`                              Groups utilities, including
                                      gradient-descent updates and action
                                      initialization.

  `offline_training/offline.py`       Implements the offline learning
                                      controller and optimization phases.

  `offline_training/offline_architecture.py`
                                      Applies and validates structural
                                      changes to the model.

  `offline_training/persistence.py`   Saves and restores deployable agent
                                      checkpoints.

  -----------------------------------------------------------------------

# Part I — Online Inference

Online inference is the interaction-time process. It receives observations,
updates beliefs, adjusts motor outputs where applicable, selects a policy,
and records experience for later offline learning. Offline optimization
is a separate process described in Part II.

## Preprocessing and Output Mapping

The preprocessing layer bridges the graph representation and the
`WorldGenerativeModel.forward()` interface. It prepares the current state, its
identifiers, the function type, optional action information, and any
function-specific parameters.

### Preparing WorldGenerativeModel inputs

`prepare_world_generative_model_inputs(...)` builds a keyword-argument dictionary.
The state must exist in `inputs_map`, and its identifier is obtained from
its position in `states_list`:

$$
\mathrm{state\_id} = \mathrm{index}(\mathrm{state\_name}) + 1
$$

The layer identifier is obtained from
`navigator.get_state_layer(state_name)`. The common arguments are:

-   `state`: current state value from `inputs_map`;
-   `state_id`: one-based position of the state in `states_list`;
-   `layer_id`: layer returned by the graph navigator;
-   `function_type`: one of the model function types.

Actions are detected from the declared `input_names`. A name is treated as
an action when it is an individual action node or an action group
recognized by `navigator.get_actions_by_group(...)`. The corresponding
value must exist in `inputs_map`.

`WorldGenerativeModel.forward()` accepts at most one action argument: either one
individual action or one action group. If more than one action or group is
detected, preprocessing raises an error. The action value is passed under
the `action` keyword.

Optional parameters such as `max_steps` and `deterministic` are forwarded
only when they are declared in `input_names` and present in `inputs_map`.

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
1 & \text{if } \mathrm{event\_shape}(d) = () \\
\displaystyle\prod_{k=1}^{m} s_k
& \text{if } \mathrm{event\_shape}(d) = (s_1,\ldots,s_m)
\end{cases}
$$

The product is implemented with `math.prod(event_shape)`.

`get_output_shape(...)` examines the children of the current state and
separates them by connection type. Observation children contribute their
declared dimensions; hierarchical children contribute the widths of their
distributions in `Q_`; action children are ignored when calculating the
output shape. Contributing child dimensions must have compatible row
counts. The resulting shape concatenates their columns:

$$
\mathrm{output\_shape} =
\left(R,\sum_{i=1}^{n} C_i\right)
$$

An error is raised when no child contributes to the dimensions, a
referenced child is missing from `Q_`, distributions have inconsistent
widths, or contributing children have incompatible row counts.

### Mapping function outputs

`map_outputs(output_names, output, navigator)` maps a function result to
the names declared in the graph.

-   With no declared output names, the function must return `None`;
    otherwise an error is raised.
-   With exactly one output name, the complete returned object is mapped
    to that name unchanged. This is the state-output convention.
-   With multiple output names, outputs are mapped in declared order.
    Individual observations and observation groups are expanded according
    to their dimensions.

For tensor outputs, the last dimension must equal the total expected
dimension. The tensor is split along that dimension while preserving the
declared output order:

$$
\mathrm{chunks} = \mathrm{split}\left(\mathrm{output}, (s_1,\ldots,s_n), \mathrm{dim}=-1\right)
$$

For tuple or list outputs, the number and order of returned values must
match the declared output structure. A mapping/dictionary is accepted
when its keys match the declared output names.

Unsupported output types, unknown observation names, and size mismatches
raise explicit errors.

## Inference and Belief Updating

### Variational free energy $F$

Variational free energy combines a complexity term related to prior
beliefs and an accuracy term related to observations. The conceptual
formulation used by this project is:

$$
F(\tilde{\mu}) = \left[
\frac{1}{2}
(\dot{\tilde{\mu}} + f(\tilde{\mu}) - \tilde{\mu}_{\text{prior}})^2
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
argument names must remain consistent with the theoretical formulation.

The main symbols are:

-   $\tilde{\mu}$: generalized belief or estimated state;
-   $f(\tilde{\mu})$: dynamic transition prediction of the next
    generalized derivative;
-   $\tilde{\mu}_{\text{prior}}$: prior prediction from the dynamics;
-   $\Pi_{\text{prior}}$: precision of prior beliefs;
-   $\tilde{y}$: available observation;
-   $g(\tilde{\mu})$: predicted observation;
-   $\Pi_{\text{obs}}$: precision associated with an observation.

### Belief updating

The `beliefs_update(observations)` method traverses the layers returned
by the graph navigator. For each state, it prepares predictions,
beliefs, and precisions; calculates $F$; produces motor commands where
relevant; updates the posterior mean using an Euler step; updates
observation and prior precisions; estimates precision from the second
derivative of $F$; and aggregates free energy by layer.

Observation precision is maintained per observation node. Missing
observations (`None`) are replaced during preprocessing by numerical
substitutes and their precision is temporarily lowered so that the
placeholder has negligible influence. When the observation becomes
available again, its saved precision is restored.

### Precisions, distributions, and numerical bounds

The model uses PyTorch `Normal` or `MultivariateNormal` distributions,
depending on the shape of the initial parameters. Prior precisions can be
initialized from variances:

$$
\Pi_{\text{prior}} = \frac{1}{\sigma^2}
$$

A high precision corresponds to lower uncertainty. Multivariate
covariance matrices must have valid shapes, be symmetric, and be
positive definite.

`Agent_DAIH` also exposes numerical limits for state values, observation
values, rewards, and precisions. These limits protect inference and
offline calculations against invalid or extreme numerical values.
Observation limits can be configured globally with
`min_observation_value` and `max_observation_value`, or overridden per
observation group with `observation_bounds`. For a vector observation,
the group's bounds are applied elementwise to the entire vector.

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
separating an ambiguity term from a transition-risk/cost term.

### Amortized policy generation $P$

The theoretical architecture proposes that one or more `p` functions
quickly generate a candidate policy and its expected score:

$$
P_t(S_t,\Pi_{\text{prior}},\epsilon) \longrightarrow (\pi,G_{\text{prior}})
$$

where $S_t$ represents the relevant internal state,
$\Pi_{\text{prior}}$ the associated precisions,
$\epsilon$ the prediction error used to adapt generation,
$\pi$ the candidate policy, and $G_{\text{prior}}$ its expected score.

Policy length may depend on precision or uncertainty: when confidence
decreases, the agent can favor a shorter, more cautious horizon. The
actual horizon is determined by the policy candidate returned by the
configured policy-generation functions.

### Candidate evaluation and tolerance

The `policy_selection()` method retrieves candidates through
`model_components.get_policy_candidates(...)`, simulates actions in
local copies of beliefs and inputs, and calculates a reevaluated score
$G_{\text{posterior}}$.

The model's prediction error is:

$$
\epsilon = \Pi_P(G_{\text{prior}}-G_{\text{posterior}})^2
$$

The tolerance rule is:

-   if $\epsilon \leq 1$, the candidate is considered consistent with
    the prediction;
-   otherwise, the candidate is rejected and the error is fed back into
    the generator inputs to encourage another proposal;
-   policy precision $\Pi_P$ is adjusted to change tolerance.

The implementation limits policy-selection attempts with
`max_policies_iter` and raises an error if the limit is reached before a
candidate is accepted.

## Reflex Motor Control

Motors are represented by motor observations associated with an action
group bearing the same name. A group is recognized as a motor group when
it is declared in both `actions` and `observations`. The `motor_command(...)`
method checks whether a state is connected to motors, then adjusts the
predicted output of `g` using gradient descent on $F$.

The conceptual rule is:

$$
y_{\text{new}} = y_{\text{pred}} - \Delta t \, k \, \frac{\partial F}{\partial y_{\text{pred}}}
$$

The motor update uses the step size supplied to the gradient-descent
operation. The time step and update coefficient determine the magnitude
of each adjustment. Their values must be evaluated for control stability
and convergence.

## Observation Groups and Vector Observations

DAIH supports two observation declaration formats.

### Multiple scalar observations in one group

The historical format declares a group containing several named scalar
observations. Each name becomes an individual graph node:

```python
observations = {
    "position_obs": ["x_position", "y_position"],
}
```

The runtime input is a mapping from the individual observation names
to their values:

```python
current_observations = {
    "x_position": 10.0,
    "y_position": 20.0,
}
```

### One vector-valued observation in one group

A group can instead declare a single vector-valued observation by
containing one positive integer dimension:

```python
observations = {
    "image": [600],
}
```

This creates **one observation node** named `image`, with a vector
dimension of 600. It does not create 600 graph nodes.

The runtime input uses the group/node name and supplies the complete
vector:

```python
current_observations = {
    "image": image_vector,  # one-dimensional vector with 600 elements
}
```

The same convention applies to the output of `g`: declare the vector
observation group name in `output_g`, and return a tensor whose final
dimension matches the declared vector dimension.

A model can mix both formats:

```python
observations = {
    "position_obs": ["x_position", "y_position"],
    "image": [600],
    "self_position": ["x_self", "y_self"],
}
```

Vector observations remain a single node throughout graph construction,
preprocessing, output mapping, inference, and offline processing. Their
dimension is used when validating and preparing model outputs.

### Observation bounds

Bounds are configured per observation group. For example:

```python
agent = Agent_DAIH(
    observation_bounds={
        "image": {"min": 0.0, "max": 255.0},
        "position_obs": {"min": -100.0, "max": 100.0},
    }
)
```

The `image` bounds apply to every element of the 600-dimensional vector.
The same group-level rule applies to all scalar members of a legacy
multi-observation group. Groups not explicitly configured use the global
observation limits.

## Model Configuration

The model is defined using dictionaries. A schematic example:

```python
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

The names referenced by `f`, `g`, `p`, and `V` must be available to
the function-resolution mechanism.

### Field roles

-   `parents` defines hierarchical links between states.
-   `layer` indicates the expected temporal regime; it is graph metadata
    used by the inference update process.
-   `input_f` / `output_f` declare the inputs and outputs of the
    dynamics.
-   `input_g` / `output_g` declare the inputs and outputs of observation
    prediction.
-   `input_p` / `output_p` are required when a `p` function is declared.
-   `input_V` / `output_V` are required when a `V` function is declared.
-   `D` contains initial means, variances, and, if necessary,
    covariances.
-   `pi_obs` maps observation node names to their precisions.
-   `pi_p` optionally configures precisions associated with policy
    generators.

### Rules to follow

1.  Every state declared in `generative_functions` must have a
    corresponding entry in `D`.
2.  The keys in `D` and `generative_functions` must match.
3.  Each parent must refer to an existing state.
4.  State, action, and observation names must be distinguishable.
5.  An individual action or observation must belong to only one group.
6.  An observation declaration must contain either one positive integer
    dimension (one vector node) or a non-empty list of observation names
    (scalar nodes).
7.  When a state directly depends on an action in `input_f`, comply with
    the `V` validation requirements.
8.  Hierarchical ancestors involved in a dependency on an action must
    define `p`, according to the model's validation rules.
9.  To connect motors, declare the group in both `actions` and
    `observations`, then reference the individual motor observation
    names in `output_g`.

## Running Inference

The entry point is an instance of `Agent_DAIH`:

```python
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

`functions_map` and `current_observations` must be defined for the use
case. `inference()` yields motor-command messages when motor outputs are
updated, followed by the selected policy result. The agent also records
the observations and transitions needed by the offline subsystem.

# Part II — Offline Learning

Offline learning is a DAIH package feature. It is not an application
trainer and does not contain application-specific workflows. It operates
on an initialized `Agent_DAIH` and on memories recorded by online
inference, with optional additional observation batches supplied by the
caller.

The offline process does not replace online inference. It optimizes the
generative model and policy components between interaction periods. Its
three macro-phases are hierarchical prediction learning, temporal
prediction learning, and architectural restructuring.

## Purpose and Data Memories

During online inference, DAIH maintains two separate bounded memories.

### Observation memory

The observation memory stores records containing:

-   the observation frame;
-   the architecture present when the frame was recorded;
-   a surprise score derived from the online free-energy values.

It is used by **Phase I — Hierarchical Predictions**. Records are ranked
by surprise, and the current Top-K selection determines which contexts
are used for the hierarchical objective.

### Transition memory

The transition memory stores awake transitions containing:

-   the initial agent state and beliefs;
-   the selected action or policy;
-   the next agent state and beliefs;
-   architecture metadata for the initial and next snapshots;
-   a surprise score when available.

It is used by **Phase II — Temporal Predictions** and policy optimization.
Temporal learning uses compatible contiguous transition chains. The
policy sequence itself determines the rollout horizon; the legacy
`temporal_horizon` setting does not determine the active horizon.

The two memories are stored separately and have independent capacities
and paths. Online free-energy reference statistics are also accumulated
in the observation-memory file, independently of the bounded observation
records.

## The Three Offline Phases

The offline controller runs the following cycle.

### Phase I — Hierarchical Predictions

Phase I optimizes the shared `WorldGenerativeModel` using observation memory.

1.  Rebuild temporary belief contexts from stored observation frames.
2.  Rank the contexts by surprise/free energy.
3.  Select the current Top-K contexts.
4.  Optimize the shared generative model against the hierarchical
    free-energy objective.
5.  Evaluate the hierarchical objective and record its history.

The Top-K value can increase as the offline process compares the
current offline free energy with the mean free energy accumulated during
online inference. This allows the training set to expand progressively
when the offline result remains above the online reference by more than
the configured global precision.

### Phase II — Temporal Predictions

Phase II learns from transition memory, not from the observation-frame
batch used by Phase I.

1.  Find compatible transition sequences in the stored awake experience.
2.  Generate policy candidates from compatible initial snapshots.
3.  Use each candidate policy sequence to define the temporal rollout
    horizon.
4.  Dream the trajectory through the generative model and calculate
    temporal prediction losses.
5.  Optimize the shared parameters of the temporal generative world model.
6.  Optimize policy generators using expected-free-energy and
    prior/posterior alignment objectives on compatible transition starts.

The temporal objective uses contiguous transitions. If a required state
or time step is incompatible or missing, the rollout is rejected rather
than silently shortened. The policy loss is evaluated using the
compatible stored transition initial state.

### Joint convergence and transition to Phase III

Phases I and II are evaluated together. Both objectives are minimized.
The controller waits until sufficient history has accumulated, then
compares the change across the configured patience window against their
respective precision thresholds.

-   **Converged:** both objectives have stabilized within their
    precision thresholds.
-   **Diverged:** one objective improves beyond its threshold while the
    other worsens beyond its threshold.
-   **Active:** neither condition is met, or there is not yet enough
    history.

When the joint status is converged or diverged, the controller moves to
Phase III if structural learning is enabled. The phase status is based
on the offline objectives and configured precision values, not on an
application-specific trainer or an external model-accuracy estimate.

### Phase III — Architectural Restructuring

Phase III is a persistent two-part process. Its controller state is stored
with the offline trainer so that an interrupted run resumes at the same
structural level.

**A. Simultaneous cleanup**

DAIH first builds the complete set of currently eligible deletion mutations:
obsolete weights, neurons, intermediate layers, coordinates, under-states,
states, and hierarchy layers. The mutations are evaluated from the same
pre-mutation architecture and submitted to `OfflineArchitecture.apply` in
one transaction. The resulting architecture is checked against validation
free energy. If the collective mutation is rejected, the original agent
snapshot is restored. If it is accepted, Phases I and II are restarted
before expansion begins.

**B. Capacity relaunch**

Expansion is performed in ordered families. Each candidate starts from an
identical snapshot, receives a short offline adaptation on observation
memory (and policy adaptation only through transition memory), and is
evaluated on the validation observations. The candidate with the greatest
positive validation-F gain is committed. Candidate gradients are measured
on newly introduced parameter regions and filtered using a moving gradient
reference (`structural_gradient_beta`, `structural_gradient_min`).

The order is deliberate:

1. World Generative Model capacity: add a neuron, add a one-neuron
   intermediate layer, add both together, or expand policy hidden capacity.
2. Generalized coordinates.
3. Under-states.
4. States.
5. Hierarchy layers.

The generative family, including policy capacity, must be exhausted before
the state representation is expanded. Within a family, a committed
candidate becomes the new baseline; Phases I and II run again before the
next structural search. If no candidate produces a validation-F gain above
`global_precision` and a sufficient new-parameter gradient, DAIH advances
to the next family. When every family is exhausted, Phase III is complete.

New coordinates and states are initialized through the normal model
rebuild path. Historical observation/transition records are not fabricated
for them: incompatible historical components remain unavailable, while
the new elements participate in current inference and can adapt through
the free-energy objective.

### Adaptive Learning Rates

The offline optimizer uses separate parameter groups for leaf layers of
the shared World Generative Models and for policy-generator layers. Each
group has a persistent learning rate in the offline controller state.
Existing groups retain their rates when the optimizer is rebuilt; newly
created groups receive the configured default rate and can subsequently
adapt independently. Policy groups use `policy_learning_rate` when set,
otherwise the base rate scaled by `agent.lr_policies`.

## Compatibility After Restructuring

A structural change does not automatically invalidate all stored
experience.

Each observation record and transition snapshot carries architecture
metadata describing the states and distribution shapes present when
the record was created. During offline replay, DAIH checks compatibility
against the current architecture.

-   Existing states and components are reused when their stored data
    remains compatible.
-   Incompatible or removed components are skipped for that replay.
-   Newly created or modified components without compatible historical
    data wait for subsequent online inference to generate transitions
    under the new architecture.
-   Those new transitions can then participate in later offline cycles.

This preserves usable historical experience instead of discarding an
entire memory after every restructuring.

## Offline Configuration and Execution

`OfflineConfig` controls optimization, memory, convergence, structural
constraints, and checkpoint behavior.

A minimal example:

```python
from daih import Agent_DAIH, OfflineConfig

agent = Agent_DAIH()

agent.model(
    generative_functions=generative_functions,
    functions_map=functions_map,
    observations=observations,
    actions=actions,
    pi_p=pi_p,
)

config = OfflineConfig(
    inference_memory_path="offline_artifacts/observation_memory.pt",
    transition_memory_path="offline_artifacts/transition_memory.pt",
    inference_memory_capacity=256,
    transition_memory_capacity=256,
    initial_top_k=1,
    top_k_step=1,
    macro_patience=5,
    macro_relative_tolerance=1e-3,
    global_precision=1e-3,
    enable_structural_phase=True,
)

result = agent.training_offline(
    observations=recent_observation_batch,
    config=config,
)
```

The `observations` argument supplies additional observation frames to
merge into observation memory for the offline call. It is not the source
of Phase II temporal training. Phase II reads transition memory.
`test_observations` can be supplied separately to evaluate the current
model on a caller-selected batch.

The short alias `agent.offline(...)` calls the same method.

### Motor training

Motor-function training is disabled by default because the motor
function is user-owned. To enable it, set `train_motor=True` and provide
both `motor_fn` and `motor_train_fn`. The callback `motor_train_fn`
receives the collected motor trace and observation frames. DAIH does not
assume or impose an application-specific motor-training implementation.

## Checkpoints and Reports

The offline trainer can write:

-   a checkpoint before an offline update;
-   periodic post-update checkpoints;
-   full-agent checkpoints at the end of each offline phase;
-   a best-agent checkpoint based on the evaluated free energy;
-   an `offline_plan.json` report describing the completed step,
    objective values, phase status, memory usage, Top-K progression,
    structural operation, and checkpoint paths.

The checkpoint system in `offline_training/persistence.py` also provides
`save_agent(...)` and `load_agent(...)` for deployable agent snapshots.
The function map and any user-defined local callables must be restored
through the factories required by `load_agent(...)`.

Offline learning temporarily constructs beliefs for replay and restores
the persistent online posterior beliefs afterward. The returned report
marks this behavior with `q_updated_offline: False`: offline parameter
optimization does not replace the agent's current online posterior state.

# Main Parameters

The default values visible in `Agent_DAIH.__init__` include:

| Parameter | Default value | Description |
|---|---:|---|
| `max_grad_norm` | `5.0` | Maximum gradient norm used by inference updates. |
| `min_state_value` / `max_state_value` | `-1e3` / `1e3` | Numerical bounds for state values. |
| `min_observation_value` / `max_observation_value` | `-1e3` / `1e3` | Default numerical bounds for observations. |
| `min_reward_value` / `max_reward_value` | `-1e2` / `1e2` | Numerical bounds for reward/cost values. |
| `min_precision` / `max_precision` | `1e-4` / `1e4` | General precision bounds. |
| `min_obs_precision` / `max_obs_precision` | `1e-4` / `1e4` | Observation precision bounds. |
| `min_policy_precision` / `max_policy_precision` | `1e-4` / `1e4` | Policy precision bounds. |
| `max_policy_error` | `10` | Maximum policy prediction error used by numerical safeguards. |
| `numerical_eps` | `1e-6` | Numerical epsilon. |

The principal `OfflineConfig` defaults include:

| Parameter | Default value | Description |
|---|---:|---|
| `learning_rate` | `1e-3` | Base offline optimizer learning rate. |
| `weight_decay` | `1e-5` | AdamW weight decay. |
| `gradient_clip` | `5.0` | Offline parameter gradient clipping. |
| `macro_patience` | `5` | Number of objective-history samples required for joint status evaluation. |
| `global_precision` | `1e-3` | Precision threshold for the hierarchical objective and structural decisions. |
| `temporal_precision` | `None` | Optional temporal objective precision; defaults to `global_precision`. |
| `initial_top_k` | `1` | Initial number of surprising records selected. |
| `top_k_step` | `1` | Top-K increment when the online/offline free-energy criterion is met. |
| `inference_memory_capacity` | `256` | Maximum retained observation records. |
| `transition_memory_capacity` | `256` | Maximum retained transition records. |
| `enable_structural_phase` | `True` | Enables Phase III. |
| `structural_gradient_beta` | `0.9` | EMA coefficient for the Phase-III candidate-gradient reference. |
| `structural_gradient_min` | `1e-10` | Minimum new-parameter gradient magnitude for a candidate to qualify. |
| `weights_only` | `False` | Disables architecture/phase mutations when enabled. |

See the `OfflineConfig` dataclass in
`daih/offline_training/offline.py` for the complete list of options,
including per-state and graph constraints, memory paths, checkpoint
settings, and policy-loss weights.

# Limitations and Items to Verify

DAIH is an experimental framework. Before interpreting results as
validation of the model, check the following:

-   **Convention for $F$ and $G$:** verify signs, normalization
    constants, dimensions, and gradients for each term.
-   **Generative-function contracts:** ensure that user-provided
    `f`, `g`, `p`, and `V` functions return tensors with dimensions
    consistent with their declarations.
-   **Vector observations:** verify vector lengths, output ordering, and
    group-level bounds for every configured observation.
-   **Policy selection:** evaluate convergence, retry behavior, and the
    quality of accepted candidates in representative environments.
-   **Offline convergence:** joint convergence/divergence is based on
    finite objective histories and configured precision thresholds; it
    is not proof that the learned model is globally optimal.
-   **Transition replay:** temporal learning requires compatible,
    contiguous transition chains and policy candidates with usable
    horizons.
-   **Structural changes:** automatic pruning, additions, and graph
    mutations must be validated against the application's function
    contracts and state constraints.
-   **Numerical stability:** check for finite values, strictly positive
    precisions, valid covariance matrices, and bounded gradients.
-   **Memory lifecycle:** ensure that observation and transition memory
    paths are persistent and writable, and that their capacities match
    the available storage and intended replay diversity.

## Additional Documentation

The repository's secondary guides complement this main README:

-   **[Generative Model Configuration and Initialization](./docs/config.md):**
    `D` parameters, distributions, covariances, prior precisions,
    observation groups, and numerical initialization.
-   **[Generative Model Definition Guide](./docs/generative_model.md):**
    states, `parents`, `layer`, `f`, `g`, `p`, and `V` functions,
    actions, observations, and graph-construction rules.

These documents detail the configuration contracts. This README explains
how the online inference and offline learning subsystems fit together
inside DAIH.

## License and Citation

### License

This project is distributed under the MIT License. See the
[`LICENSE`](LICENSE) file for the terms of use, modification, and
redistribution.

### Theoretical inspirations

The development of this project draws in particular on the following
works on active inference, dynamic planning, and sophisticated inference:

-   Parr, T., Pezzulo, G. & Friston, K. J. — **Active Inference: The
    Free Energy Principle in Mind, Brain, and Behavior.**
    [MIT Press](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)
-   **Dynamic Planning in Active Inference.**
    [arXiv:2402.11658](https://arxiv.org/abs/2402.11658)
-   **Sophisticated Inference.**
    [arXiv:2006.04120](https://arxiv.org/abs/2006.04120)

------------------------------------------------------------------------

**DAIH-DynAmI**: an experimental hybrid active inference framework
combining online inference, amortized policy generation, reflex motor
control, and offline architectural learning.
