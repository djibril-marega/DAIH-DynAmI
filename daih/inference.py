import torch
import torch.distributions as dist
from . import utils
import copy

def euler_update(mu, dt, F_value, max_grad_norm=100.0):
    """Finite, norm-clipped Euler update of generalized coordinates."""
    if not torch.isfinite(F_value).all():
        return mu.detach().clone()
    gradient = torch.autograd.grad(
        F_value, mu, retain_graph=True, allow_unused=True
    )[0]
    if gradient is None or not torch.isfinite(gradient).all():
        return mu.detach().clone()

    norm = torch.linalg.vector_norm(gradient)
    if torch.isfinite(norm) and norm > max_grad_norm:
        gradient = gradient * (max_grad_norm / norm.clamp_min(1e-12))

    D_mu = torch.zeros_like(mu)
    D_mu[:-1] = mu[1:]
    updated = mu + float(dt) * (D_mu - gradient)
    if not torch.isfinite(updated).all():
        return mu.detach().clone()
    return updated



def second_derivative_F(mu, F_value):
    """
    Computes one Hessian matrix per group.

    Parameters
    ----------
    mu : torch.Tensor
        Input tensor of shape (a, b) or (a,).

    F_value : torch.Tensor
        Scalar objective per group:
            (a,)   : one scalar per group.
            (a,b)  : outputs summed within each group.
            ()     : one global scalar objective.

    Returns
    -------
    torch.Tensor
        Hessian blocks of shape (a, b, b).
    """

    if not mu.requires_grad:
        raise ValueError("mu must require gradients.")

    # Normalize mu to (a, b)
    if mu.ndim == 1:
        mu_view = mu.unsqueeze(-1)
    elif mu.ndim == 2:
        mu_view = mu
    else:
        raise ValueError("mu must have 1 or 2 dimensions.")

    a, b = mu_view.shape

    # Build one scalar objective per group
    if F_value.ndim == 0:
        objectives = [F_value] * a

    elif F_value.shape == (a,):
        objectives = [F_value[i] for i in range(a)]

    elif F_value.shape == (a, b):
        objectives = [
            F_value[i].sum()
            for i in range(a)
        ]

    else:
        raise ValueError(
            f"F_value must have shape (), ({a},) "
            f"or ({a}, {b})."
        )

    hessians = []

    for i, F_i in enumerate(objectives):

        # First derivative
        grad = torch.autograd.grad(
            F_i,
            mu,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]

        if grad is None:
            H_i = torch.zeros(
                (b, b),
                dtype=mu.dtype,
                device=mu.device
            )
            hessians.append(H_i)
            continue

        grad_i = grad.reshape(a, b)[i]

        rows = []

        # Second derivative
        for j in range(b):

            if not grad_i[j].requires_grad:
                second_grad = None
            else:
                second_grad = torch.autograd.grad(
                    grad_i[j],
                    mu,
                    retain_graph=True,
                    allow_unused=True
                )[0]

            if second_grad is None:
                row = torch.zeros(
                    b,
                    dtype=mu.dtype,
                    device=mu.device
                )
            else:
                row = second_grad.reshape(a, b)[i]

            rows.append(row)

        H_i = torch.stack(rows)

        # Numerical symmetrization
        H_i = 0.5 * (H_i + H_i.T)

        hessians.append(H_i)

    return torch.stack(hessians)





def update_distributions(hessian, posterior, Q, eps=1e-6):
    """Numerically robust posterior-distribution update.

    Non-finite posterior/Hessian values are treated as a rejected update: the
    previous distribution is retained rather than constructing Normal(NaN).
    Hessians are symmetrized and projected to a positive-definite precision.
    """
    if eps <= 0:
        raise ValueError("eps must be strictly positive.")
    if hessian.ndim != 3:
        raise ValueError("hessian must have shape (a, b, b).")
    if hessian.shape[0] != len(Q):
        raise ValueError("Number of Hessian matrices must match Q.")

    for i, q in enumerate(Q):
        H = hessian[i]
        if not torch.isfinite(H).all():
            continue
        loc = posterior[i]
        if not torch.isfinite(loc).all():
            continue

        H = 0.5 * (H + H.T)
        # Work in float64 for the eigendecomposition/inversion. Negative and
        # near-zero curvature cannot define a valid Gaussian precision.
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(H.to(torch.float64))
            eigenvalues = eigenvalues.clamp(min=max(float(eps), 1e-6), max=1e8)
            covariance = (eigenvectors * eigenvalues.reciprocal().unsqueeze(0)) @ eigenvectors.T
            covariance = 0.5 * (covariance + covariance.T)
            covariance = covariance.to(dtype=loc.dtype, device=loc.device)
            covariance = torch.nan_to_num(covariance, nan=1.0, posinf=1e4, neginf=1e-6)
        except RuntimeError:
            continue

        if isinstance(q, dist.MultivariateNormal):
            try:
                # Keep covariance strictly positive definite after casting.
                covariance = covariance + max(float(eps), 1e-6) * torch.eye(
                    covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
                )
                Q[i] = dist.MultivariateNormal(loc=loc, covariance_matrix=covariance)
            except (ValueError, RuntimeError):
                continue
        elif isinstance(q, dist.Normal):
            if H.shape != (1, 1):
                raise ValueError(
                    f"Normal distribution at index {i} requires a (1, 1) Hessian, got {H.shape}."
                )
            variance = covariance[0, 0].clamp(min=1e-8, max=1e6)
            scale = torch.sqrt(variance).clamp(min=1e-4, max=1e3)
            Q[i] = dist.Normal(loc=loc, scale=scale)
        else:
            raise TypeError(f"Unsupported distribution: {type(q)}")
    return Q



def update_pi_prior(pi_states, pi_priors, hessian, state_name, eps=1e-6):
    """Update a state's precision and propagate it to its incoming g relations.

    ``pi_states[state_name]`` is the state-level precision. Every nested
    ``pi_priors[parent][state_name]`` is then refreshed from that value.
    """
    if eps <= 0:
        raise ValueError("eps must be strictly positive.")
    if state_name not in pi_states:
        raise KeyError(f"No state precision is defined for '{state_name}'.")

    prior = pi_states[state_name]
    if not torch.isfinite(hessian).all():
        return pi_states, pi_priors
    if hessian.ndim == 3:
        precision = torch.diagonal(hessian, dim1=-2, dim2=-1)
    elif hessian.ndim == 2:
        precision = torch.diagonal(hessian)
    elif hessian.ndim == 1:
        precision = hessian
    else:
        raise ValueError(f"Unsupported Hessian shape: {hessian.shape}")

    if precision.numel() != prior.numel():
        raise ValueError(
            f"Incompatible dimensions for state '{state_name}': "
            f"Hessian precision has {precision.numel()} elements, "
            f"but state precision has {prior.numel()}."
        )

    precision = torch.nan_to_num(
        precision, nan=eps, posinf=1e8, neginf=eps
    ).reshape(prior.shape).to(
        device=prior.device, dtype=prior.dtype
    ).clamp(min=eps, max=1e8).detach()
    pi_states[state_name] = precision

    # Update only the incoming edges of this child. No graph traversal is
    # required during inference; the nested relation map is already built.
    for parent_state, child_precisions in pi_priors.items():
        if state_name in child_precisions:
            child_precisions[state_name] = precision.clone()

    return pi_states, pi_priors


def update_pi_obs(navigator, pi_obs, gamma_obs, F, Q_, state_name, dt=0.1, lr_obs=None):
    """
    Updates observation precisions using gradient descent.

    The function:
        1. Retrieves the learning rate.
        2. Updates gamma_obs using gradient descent.
        3. Converts gamma_obs into actual precisions.
        4. Updates pi_obs using observation entries only.

    State outputs are skipped because their precisions are already
    determined by Q_.

    Observation outputs:
        - Individual observations: update pi_obs[observation_name].
        - Observation groups: update each individual observation
          belonging to the group.

    The ordering follows navigator.output_g(state_name) and the same
    ordering used by get_actual_precision.
    """

    # --------------------------------------------------
    # 2. Update gamma_obs using gradient descent
    # --------------------------------------------------
    new_gamma_obs = utils.update_wih_grad_desc(
        gamma_obs,
        F,
        dt=dt,
        k=lr_obs,
    )

    # --------------------------------------------------
    # 3. Convert gamma_obs into actual precisions
    # --------------------------------------------------
    new_gamma_obs = torch.nan_to_num(
        new_gamma_obs, nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(min=-20.0, max=20.0)
    new_pi_actual = torch.exp(new_gamma_obs).clamp(min=1e-9, max=1e8).detach()

    new_pi_actual = torch.as_tensor(new_pi_actual).reshape(-1)

    # --------------------------------------------------
    # 4. Update observation precisions in pi_obs
    # --------------------------------------------------
    outputs_g = navigator.output_g(state_name)

    idx = 0

    for output in outputs_g:

        # --------------------------------------------------
        # State output: skip its precision entries
        # --------------------------------------------------
        if output in Q_:

            for q in Q_[output]:

                if hasattr(q, "covariance_matrix"):
                    covariance = torch.as_tensor(q.covariance_matrix)

                    if covariance.ndim <= 1:
                        n = covariance.numel()
                    else:
                        n = min(covariance.shape)

                elif hasattr(q, "scale"):
                    n = torch.as_tensor(q.scale).numel()

                else:
                    raise TypeError(
                        f"Cannot retrieve dimensions for '{output}'."
                    )

                idx += n

            continue

        # --------------------------------------------------
        # Individual observation or group of observations
        # --------------------------------------------------
        try:
            output_type = navigator.type(output)
        except KeyError:
            output_type = None

        if output_type == "observation":
            observations = [output]
        else:
            observations = navigator.get_observations_by_group(output)

            if not observations:
                continue

        # --------------------------------------------------
        # Update each individual observation
        # --------------------------------------------------
        for observation in observations:

            if idx >= new_pi_actual.numel():
                raise ValueError(
                    "new_pi_actual does not contain enough precision values."
                )

            pi_obs[observation] = new_pi_actual[idx].clone()
            idx += 1

    return pi_obs


def seconde_derivative_from_output(f_out, mu, Pi_bruit=None, eps=1e-6):
    """
    Computes one propagated precision matrix per group.

    For each group a:

        J_a = d f_out[a] / d mu[a]

        Pi_a = J_a.T @ Pi_bruit[a] @ J_a + eps * I

    Parameters
    ----------
    f_out : torch.Tensor
        Output already computed and connected to the autograd graph.
        Shape: (a, b) or (a,).

    mu : torch.Tensor
        Input variable connected to the autograd graph.
        Must have the same shape as f_out.

    Pi_bruit : torch.Tensor, optional
        Noise precision in the output space.
        Accepted shapes:
            (b, b)       : shared by all groups.
            (a, b, b)    : one matrix per group.
        Defaults to identity matrices.

    eps : float
        Strictly positive regularization term.

    Returns
    -------
    torch.Tensor
        Propagated precision matrices.
        Shape: (a, b, b).
    """

    # --------------------------------------------------
    # 1. Validation
    # --------------------------------------------------

    if f_out.shape != mu.shape:
        raise ValueError(
            "f_out and mu must have the same shape."
        )

    if f_out.ndim not in (1, 2):
        raise ValueError(
            "f_out must have 1 or 2 dimensions."
        )

    if not mu.requires_grad:
        raise ValueError(
            "mu must require gradients."
        )

    if not f_out.requires_grad or f_out.grad_fn is None:
        raise ValueError(
            "f_out is not connected to an autograd graph."
        )

    if eps <= 0:
        raise ValueError("eps must be strictly positive.")

    # --------------------------------------------------
    # 2. Normalize dimensions
    # --------------------------------------------------

    if f_out.ndim == 1:
        f_out = f_out.unsqueeze(-1)
        mu = mu.unsqueeze(-1)

    a, b = f_out.shape

    # --------------------------------------------------
    # 3. Compute one Jacobian per group
    # --------------------------------------------------

    jacobians = []

    for group in range(a):

        rows = []

        for j in range(b):

            output_scalar = f_out[group, j]

            # Derivative of one output component
            # with respect to the complete input mu.

            if output_scalar.requires_grad:

                grad_mu = torch.autograd.grad(
                    outputs=output_scalar,
                    inputs=mu,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )[0]

            else:
                grad_mu = None

            if grad_mu is None:
                grad_group = torch.zeros(
                    b,
                    dtype=mu.dtype,
                    device=mu.device
                )
            else:
                # Extract only the corresponding group.
                grad_group = grad_mu[group].reshape(-1)

            rows.append(grad_group)

        J_a = torch.stack(rows, dim=0)

        jacobians.append(J_a)

    J = torch.stack(jacobians, dim=0)

    # J.shape = (a, b, b)

    # --------------------------------------------------
    # 4. Noise precision
    # --------------------------------------------------

    if Pi_bruit is None:

        Pi_bruit = torch.eye(
            b,
            dtype=J.dtype,
            device=J.device
        ).expand(a, b, b)

    else:

        Pi_bruit = Pi_bruit.to(
            dtype=J.dtype,
            device=J.device
        )

        if Pi_bruit.shape == (b, b):
            Pi_bruit = Pi_bruit.expand(a, b, b)

        elif Pi_bruit.shape != (a, b, b):
            raise ValueError(
                f"Pi_bruit must have shape "
                f"({b}, {b}) or ({a}, {b}, {b})."
            )

    # --------------------------------------------------
    # 5. Propagate precision
    # --------------------------------------------------

    # Batched matrix multiplication:
    # Pi[a] = J[a].T @ Pi_bruit[a] @ J[a]

    Pi = J.transpose(-2, -1) @ Pi_bruit @ J

    # Numerical symmetrization
    Pi = 0.5 * (Pi + Pi.transpose(-2, -1))

    # Regularization
    I = torch.eye(
        b,
        dtype=Pi.dtype,
        device=Pi.device
    )

    Pi = Pi + eps * I.unsqueeze(0)

    return Pi



def update_threshold(list_pi_priors, F_list, pi_threshold, dt=0.1, k=1.0):
    list_grad = []
    for pi_prior, F in zip(list_pi_priors, F_list):
        grad_F = torch.autograd.grad(
            F, pi_prior, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        if grad_F is not None and torch.isfinite(grad_F).all():
            list_grad.append(torch.sum(grad_F))
    if not list_grad:
        return pi_threshold.detach()
    grad_sum = torch.stack(list_grad).sum()
    updated = pi_threshold - k * dt * grad_sum
    return updated.detach() if not torch.isfinite(updated).all() else updated



def policy_error(G_prior, G_posterior, pi_p):
    return torch.clamp(pi_p*((G_prior - G_posterior) ** 2), max=1e+9)


def update_current_action(best_candidat, actions_id, navigator):
    index = torch.argmax(best_candidat[0]).item()
    action_name = utils.get_name_by_index(actions_id, index)
    action_group_name = navigator.get_groups_by_actions(action_name)
    actions = navigator.get_actions_by_group(action_group_name[0])
    action_one_hot = utils.one_hot(actions, action_name)
    return action_group_name, action_one_hot



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


