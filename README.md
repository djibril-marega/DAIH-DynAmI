# DAIH-DynAmI

**Deep Active Inference Hybrid Framework with Dynamic Planning Horizon and Amortized Active Inference**

DAIH-DynAmI is an experimental hybrid active inference framework. It aims to combine continuous belief updating, prediction of action consequences, amortized policy generation, and reflex motor control. Its guiding principle is to find an action that is sufficiently consistent with the internal model within a limited computational budget, rather than exhaustively exploring every possible policy.

> **Project status:** prototype under development. This README distinguishes the intended theoretical principles, the mechanisms visible in the supplied code, and possible future developments. It does not claim that every theoretical mechanism has already been fully implemented.

## Table of Contents

- [Objectives](#objectives)
- [Model Principles](#model-principles)
- [Software Architecture](#software-architecture)
- [Inference and Belief Updating](#inference-and-belief-updating)
- [Policy Selection](#policy-selection)
- [Reflex Motor Control](#reflex-motor-control)
- [Model Configuration](#model-configuration)
- [Execution](#execution)
- [Main Parameters](#main-parameters)
- [Limitations and Items to Verify](#limitations-and-items-to-verify)
- [Roadmap](#roadmap)
- [Additional Documentation](#additional-documentation)
- [License and Citation](#license-and-citation)

## Objectives

The project explores an agent architecture based on the following ideas:

1. **Continuous perceptual inference:** adjust beliefs to reduce the discrepancy between observations and predictions from the generative model.
2. **Hierarchical states:** organize states into dependency levels and temporal update regimes.
3. **Generalized coordinates of motion:** represent a state and its time derivatives within a single structure.
4. **Amortized decision-making:** use policy-generating functions to propose candidates without systematically traversing the entire policy space.
5. **Critical evaluation:** simulate the predicted consequences of a policy and compare its expected score with its reevaluated score.
6. **Fast motor control:** adjust motor outputs using gradient descent on variational free energy.
7. **Learning and adaptation:** evolve precisions and, eventually, learning rates according to prediction quality.

The design hypothesis is that an agent operating in a dynamic environment must be able to make decisions on time and with limited resources. A sufficiently good and consistent solution may be preferable to a more expensive exhaustive search.

## Model Principles

### Generative model

The model is described in terms of states and associated functions:

- `f`: state transition or temporal prediction;
- `g`: prediction of observations from states;
- `p`: policy generation;
- `V`: cost or value associated with a temporal transition;
- `parents`: explicit hierarchical dependencies;
- `layer`: expected temporal update level.

Actions and observations are declared as groups, but their individual elements become nodes in the graph. The functions named in the configuration are references: their execution is handled by the model engine, not by graph construction alone.

### Generalized coordinates of motion

A state can be represented by several derivative orders:

\[
\tilde{\mu} =
\begin{bmatrix}
\mu^{(0)}\\
\mu^{(1)}\\
\vdots\\
\mu^{(n)}
\end{bmatrix}
\]

The conceptual dynamics of a generalized coordinate are:

\[
\mu^{(i)}_{t+\Delta t}
\approx \mu^{(i)}_t +
\left(D\mu^{(i)}_t -
\frac{\partial F}{\partial \mu^{(i)}_t}\right)\Delta t
\]

where \(D\mu^{(i)}\) denotes the next derivative order. The final order is closed by setting its next derivative to zero. This expression describes the intended principle; the exact tensor shapes and their interpretation must remain consistent across the functions that prepare inputs, calculate \(F\), and perform updates.

For a state with three orders, for example, the temporal error term can be written as:

\[
\begin{bmatrix}
\dot{\mu}_0-x_1-\mu_1\\
\dot{\mu}_1-x_2-\mu_2\\
\dot{\mu}_2-x_3
\end{bmatrix}
\]

where the \(x_i\) represent the prior outputs of the transition function `f`, according to the model's convention.

## Software Architecture

The supplied main code defines the `Agent_DAIH` class. It relies on several specialized modules.

| Module / component | General responsibility |
|---|---|
| `Agent_DAIH` | Coordinates initialization, inference, decision-making, and motor commands. |
| `model` | Validates the configuration, initializes distributions, and builds the graph. |
| `preprocessing` | Prepares state inputs and quantities required for the \(F\) and \(G\) calculations. |
| `inference` | Provides update, differentiation, error-calculation, and policy-management operations. |
| `model_components` | Produces policy candidates. |
| `utils` | Groups utilities, including gradient-descent updates and action initialization. |
| `GraphNavigator` | Traverses states, layers, relationships, and graph engines. |

### General cycle

```text
Model configuration
        |
        v
Initialize distributions, precisions, and graph
        |
        v
Receive observations
        |
        v
Update beliefs, layer by layer
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

In the supplied code, `inference(observations)` first runs `beliefs_update(observations)`, sends any motor commands, and then calls `policy_selection()`.

## Inference and Belief Updating

### Variational free energy \(F\)

Variational free energy combines a complexity term related to prior beliefs and an accuracy term related to observations. In the supplied code, the calculation is organized around:

```python
def free_energy(
    self,
    state_prior,
    state_posterior,
    D_poresterior,
    pi_prior,
    actual,
    pi_obs,
    gamma_obs,
    prediction,
):
    complexity = (
        torch.sum(
            0.5 * pi_prior
            * (state_posterior - D_poresterior - state_prior) ** 2
        )
        - torch.sum(0.5 * gamma_obs)
        - torch.sum(0.5 * torch.log(pi_prior))
    )

    accuracy = torch.sum(-0.5 * pi_obs * (actual - prediction) ** 2)
    return complexity - accuracy
```

The main symbols are:

- \(\mu\): belief or estimated state;
- \(D\mu\): next generalized derivative;
- \(\mu_{\text{prior}}\): prior prediction from the dynamics;
- \(\Pi_{\text{prior}}\): precision of prior beliefs;
- \(y\): available observation;
- \(g(\mu)\): predicted observation;
- \(\Pi_{\text{obs}}\): observation precision.

The reference mathematical formula and the Python function should be considered together: sign conventions, normalization constants, and argument names should be checked as the model evolves.

### Belief updating

The `beliefs_update(observations)` method traverses the layers returned by the graph navigator. For each state, it:

1. prepares predictions, beliefs, and precisions;
2. calculates \(F\);
3. produces a motor command if the state is connected to motors;
4. updates the posterior mean using an Euler step;
5. updates observation precision;
6. estimates precision from the second derivative of \(F\) and updates the distribution;
7. updates prior precision;
8. aggregates free energy by layer and updates the associated threshold.

The inputs used by the generative functions are then resynchronized with the updated beliefs.

### Precisions and distributions

The model uses PyTorch `Normal` or `MultivariateNormal` distributions, depending on the shape of the initial parameters. Prior precisions can be initialized from variances:

\[
\Pi_{\text{prior}} = \frac{1}{\sigma^2}
\]

A high precision corresponds to lower uncertainty. Multivariate covariance matrices must have valid shapes, be symmetric, and be positive definite.

## Policy Selection

### Expected free energy \(G\)

Action selection uses **Expected Free Energy** (\(G\)) to estimate the consequences of a policy. The conceptual formulation provided by this project is:

\[
G(\pi)=
\frac{1}{2}\ln(2\pi e)
-\frac{1}{2}\ln(\Pi_{\text{prior}})
-\frac{1}{2}\ln(2\pi e\,\Sigma^2)
-V(f(\mu_{\text{prior}}))
\]

This expression should be interpreted according to the precision, variance, and transition-cost conventions adopted by the model. The current code calculates the score using `execepted_free_energy(...)`, separating an ambiguity term from a transition-risk/cost term. The term-by-term correspondence with the theoretical equation should be verified as the project evolves.

### Amortized policy generation \(P\)

The theoretical architecture proposes that one or more \(P\) functions quickly generate a candidate policy and its expected score:

\[
P_t(S_t,\Pi_{\text{prior}},\epsilon)
\longrightarrow (\pi,G_{\text{prior}})
\]

where:

- \(S_t\) represents the relevant internal state;
- \(\Pi_{\text{prior}}\) represents the associated precisions;
- \(\epsilon\) represents the prediction error used to adapt generation;
- \(\pi\) is the candidate policy;
- \(G_{\text{prior}}\) is its expected score.

Policy length may depend on precision or uncertainty: when confidence decreases, the agent should favor a shorter, more cautious horizon. This relationship is an architectural objective; its exact behavior depends on the implementation of the `p` functions and the candidate generator.

### Tolerance loop

The `policy_selection()` method retrieves candidates through `model_components.get_policy_candidates(...)`, simulates actions in local copies of beliefs and inputs, and then calculates a reevaluated score \(G_{\text{posterior}}\).

The error described by the model is:

\[
\epsilon = \Pi_P
\left(G_{\text{prior}}-G_{\text{posterior}}\right)^2
\]

The theoretical tolerance rule is:

- if \(\epsilon \leq 1\), the candidate is considered consistent with the prediction;
- otherwise, the candidate is rejected and the error is fed back into the generator inputs to encourage another proposal;
- precision \(\Pi_P\) is adjusted to change tolerance and, eventually, the planning horizon.

In the current code, `policy_selection()` implements a threshold comparison, an update to `pi_p`, and a selection among accepted candidates.

### Learning during wakefulness and sleep

The proposed architecture distinguishes two modes:

- **Wakefulness:** locally adjust the policy generator to reduce the gap between the predicted score and the reevaluated score.
- **Rest / sleep:** replay past trajectories and train the generative functions using a global loss based on \(F\).

This separation is an architectural proposal. The supplied main code does not, by itself, show a complete sleep or trajectory-replay loop.

## Reflex Motor Control

Motors are represented by motor observations associated with an action group bearing the same name. The `motor_command(...)` method checks whether a state is connected to motors, then adjusts the predicted output of `g` using gradient descent on \(F\).

The conceptual rule is:

\[
y_{\text{new}} =
y_{\text{pred}} -
\Delta t\,k\,
\frac{\partial F}{\partial y_{\text{pred}}}
\]

In the supplied code, the call uses `utils.update_wih_grad_desc(g_output, F, dt=0.1, k=1.0)`, then returns only the components associated with the motors of the relevant state. The step size \(k\), time step, and stopping criteria are important parameters for control stability and speed.

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
        # Add V and its metadata if the state directly depends
        # on an action in input_f, according to the validation rules.
    }
}

D = {
    "position": {
        "mu": [[0.0, 0.0]],
        "variance": [[100.0, 100.0]],
    }
}
```

This example illustrates the general structure and must be adapted to the actual signatures of the project's functions. The names referenced by `f`, `g`, `p`, and `V` must be available to the function-resolution mechanism.

### Field roles

- `parents` defines hierarchical links between states.
- `layer` indicates the expected temporal regime; it is not sufficient on its own to schedule updates.
- `input_f` / `output_f` declare the inputs and outputs of the dynamics.
- `input_g` / `output_g` declare the inputs and outputs of observation prediction.
- `input_p` / `output_p` are required when a `p` function is declared.
- `input_V` / `output_V` are required when a `V` function is declared.
- `D` contains initial means, variances, and, if necessary, covariances.
- `pi_obs` optionally configures observation precisions.
- `pi_p` optionally configures the precisions associated with policy generators.

### Rules to follow

1. Every state declared in `generative_functions` must have a corresponding entry in `D`.
2. The keys in `D` and `generative_functions` must match.
3. Each parent must refer to an existing state.
4. State, action, and observation names must be distinguishable.
5. An individual action or observation must belong to only one group.
6. When a state directly depends on an action in `input_f`, comply with the `V` validation requirements.
7. Hierarchical ancestors involved in a dependency on an action must define `p`, according to the model's validation rules.
8. To connect motors, declare the group in both `actions` and `observations`, then reference the individual motor observations in `output_g` according to the graph rules.

## Execution

The conceptual entry point is an instance of `Agent_DAIH`:

```python
agent = Agent_DAIH()

agent.model(
    generative_functions=generative_functions,
    functions_map=functions_map,
    observations=observations,
    pi_obs=pi_obs,
    actions=actions,
    pi_p=pi_p,
)

for result in agent.inference(current_observations):
    print(result)
```

The variables `functions_map`, `pi_obs`, `pi_p`, and `current_observations` must be defined for the use case. Check the exact signatures and required modules in the repository before using this example: it is a starting template, not an example guaranteed to run without adaptation.

## Main Parameters

The default values visible in `Agent_DAIH.__init__` include:

| Parameter | Initial value | Visible purpose |
|---|---:|---|
| `learning_rate` | `0.1` | General parameter intended for learning; actual usage should be verified. |
| `num_iterations` | `100` | Intended number of iterations; actual usage should be verified. |
| `tolerance_F` | `1e-5` | Intended tolerance for free energy. |
| `tolerance_Q` | `1e-5` | Intended tolerance for beliefs/distributions. |
| `gamma` | `1` | General model parameter. |
| `max_motor_iters` | `100` | Intended limit for motor iterations. |
| `lr_motor` | `0.1` | Intended motor learning rate. |
| `tolerance_F_motor` | `1e-5` | Intended tolerance for motor control. |
| `tolerance_action` | `1e-5` | Intended tolerance for actions. |

Several calls currently use hard-coded values, including `dt=0.1`, `k=1.0` for certain updates, and `k=100` for updating `pi_p`. Configuration attributes therefore do not necessarily control every calculation at this stage.

## Limitations and Items to Verify

Before interpreting results as validation of the model, the following points should be checked:

- **Missing or null observations:** define explicit semantics for unavailable observations rather than treating them as ordinary observed values.
- **Convention for \(F\):** verify signs, normalization constants, dimensions, and gradients for each term.
- **Selection loop:** add a stopping strategy or maximum number of attempts if no candidate passes the threshold.
- **Updating `pi_p`:** ensure that the error and precision used correspond to the correct state and candidate.
- **Numerical stability:** check for finite values, strictly positive precisions, and valid covariance matrices.
- **Layer scheduling:** `layer` is graph metadata; the temporal mechanism must be implemented during execution.
- **Hierarchical graph:** secondary documentation indicates that the current construction does not check for cycles.
- **Unused parameters:** verify iteration, learning-rate, and tolerance attributes before assuming they effectively control execution.

## Roadmap

### Priority 1 — Handle missing observations

**Objective:** accept null or unavailable observations without producing invalid gradients or updates.

Suggested approach: reduce the precision of the affected observation to a minimum value of `1e-8`. It is important to distinguish an observation that is genuinely equal to zero from a missing observation (`None`, a mask, or a sentinel value). Zero can be a valid observation and should not automatically be treated as missing.

Validation criteria:

- a missing observation does not cause `NaN` or `Inf`;
- its contribution to observation error becomes negligible;
- other observations continue to influence beliefs;
- precisions remain within a defined and documented numerical range.

### Priority 2 — Dynamically adapt learning steps \(k\)

**Objective:** make updates faster when error is stable and more cautious when updates oscillate or diverge.

Ideas to evaluate:

- separate learning rates for belief updates, precision updates, `pi_p`, and motors;
- reduce the step size when free energy increases or updates become unstable;
- cautiously increase the step size when progress is steady;
- bound each step with minimum and maximum values;
- plot \(F\), \(G\), precisions, and learning steps to compare behavior.

Validation criteria:

- more stable convergence;
- no explosion of gradients or precisions;
- measured settling time;
- comparison against a fixed learning rate as a baseline.

### Priority 3 — Strengthen policy selection

- explicitly determine whether selection should minimize or maximize \(G\);
- limit the number of generation and evaluation cycles;
- define behavior when no policy meets the tolerance;
- test whether feeding back the error actually leads to a different proposal;
- verify adaptation of `pi_p` and the planning horizon.

### Priority 4 — Testing and experimental validation

- add unit tests for \(F\), \(G\), belief updates, and motor control;
- test null values, tensors with different shapes, and extreme precisions;
- compare fixed-step and adaptive-step updates;
- record scores and actions in reproducible scenarios;
- document limitations and experimental results.

### Priority 5 — Learning during sleep

- replay trajectories stored during the agent's interactions;
- adjust the generative model and policy generator using past experiences;
- reduce prediction errors and improve policy selection;
- stabilize learned parameters before interactions resume;
- compare performance before and after the sleep phase.

## Additional Documentation

The repository's secondary guides complement this main README:

- **Generative model configuration and initialization:** `D` parameters, distributions, covariances, and precisions.
- **Guide to defining the generative model:** states, `parents`, `layer`, `f`, `g`, `p`, and `V` functions, actions, observations, and graph-construction rules.

These documents detail the configuration contracts. This README explains their place in the overall DAIH-DynAmI architecture.

## License and Citation

### License

This project is distributed under the MIT License. See the [`LICENSE`](LICENSE) file for the terms of use, modification, and redistribution.

### Theoretical inspirations

The development of this project draws in particular on the following works on active inference, dynamic planning, and sophisticated inference:

- Parr, T., Pezzulo, G. & Friston, K. J. — *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior.* [MIT Press](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)
- *Dynamic Planning in Active Inference.* [arXiv:2402.11658](https://arxiv.org/abs/2402.11658)
- *Sophisticated Inference.* [arXiv:2006.04120](https://arxiv.org/abs/2006.04120)

---

**DAIH-DynAmI —** an experimental hybrid active inference prototype exploring coordination between generalized beliefs, amortized policy generation, and reflex motor control.
