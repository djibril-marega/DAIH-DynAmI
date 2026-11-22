import torch
import torch.distributions as dist
from . import utils


def euler_update(mu, dt, F_value, max_grad_norm):
    """Finite Euler update with smoothly bounded gradient."""
    if not torch.isfinite(F_value).all():
        return mu.detach().clone()

    gradient = torch.autograd.grad(
        F_value,
        mu,
        retain_graph=True,
        allow_unused=True
    )[0]

    if gradient is None or not torch.isfinite(gradient).all():
        return mu.detach().clone()

    gradient = utils.smooth_norm_limit(gradient, max_grad_norm)

    D_mu = torch.zeros_like(mu)
    D_mu[:-1] = mu[1:]

    if not torch.isfinite(D_mu).all():
        return mu.detach().clone()

    updated = mu + float(dt) * (D_mu - gradient)

    if not torch.isfinite(updated).all():
        return mu.detach().clone()

    return updated


def second_derivative_F(mu, F_value):
    """
    Computes one Hessian matrix per group.

    mu: (a, b) or (a,)
    F_value: (), (a,) or (a, b)
    Returns: (a, b, b)
    """
    if not mu.requires_grad:
        raise ValueError("mu must require gradients.")

    if mu.ndim == 1:
        mu_view = mu.unsqueeze(-1)
    elif mu.ndim == 2:
        mu_view = mu
    else:
        raise ValueError("mu must have 1 or 2 dimensions.")

    a, b = mu_view.shape

    if F_value.ndim == 0:
        objectives = [F_value] * a
    elif F_value.shape == (a,):
        objectives = [F_value[i] for i in range(a)]
    elif F_value.shape == (a, b):
        objectives = [F_value[i].sum() for i in range(a)]
    else:
        raise ValueError(
            f"F_value must have shape (), ({a},) or ({a}, {b})."
        )

    hessians = []

    for i, F_i in enumerate(objectives):
        grad = torch.autograd.grad(
            F_i,
            mu,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]

        if grad is None:
            hessians.append(torch.zeros(
                (b, b), dtype=mu.dtype, device=mu.device
            ))
            continue

        grad_i = grad.reshape(a, b)[i]
        rows = []

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
                row = torch.zeros(b, dtype=mu.dtype, device=mu.device)
            else:
                row = second_grad.reshape(a, b)[i]

            rows.append(row)

        H_i = torch.stack(rows)
        H_i = 0.5 * (H_i + H_i.T)
        hessians.append(H_i)

    return torch.stack(hessians)


def update_distributions(
    hessian,
    posterior,
    Q,
    min_state_value,
    max_state_value,
    min_precision,
    max_precision,
    eps
):
    """Update posterior distributions from Hessian-derived precision."""
    if eps <= 0:
        raise ValueError("eps must be strictly positive.")
    if min_state_value >= max_state_value:
        raise ValueError("Invalid state bounds.")
    if min_precision <= 0 or min_precision >= max_precision:
        raise ValueError("Invalid precision bounds.")
    if hessian.ndim != 3:
        raise ValueError("hessian must have shape (a, b, b).")

    for i, q in enumerate(Q):
        H = hessian[i]

        if not torch.isfinite(H).all():
            continue

        loc = utils.smooth_bound(
            posterior[i], min_state_value, max_state_value
        )
        if not torch.isfinite(loc).all():
            continue

        H = 0.5 * (H + H.T)

        try:
            H64 = H.to(torch.float64)
            eigenvalues, eigenvectors = torch.linalg.eigh(H64)
            bounded_eigenvalues = utils.smooth_bound(
                eigenvalues, min_precision, max_precision
            )

            covariance = (
                eigenvectors
                * bounded_eigenvalues.reciprocal().unsqueeze(0)
            ) @ eigenvectors.T

            covariance = 0.5 * (covariance + covariance.T)
            covariance = covariance.to(dtype=loc.dtype, device=loc.device)

            if not torch.isfinite(covariance).all():
                continue
        except RuntimeError:
            continue

        if isinstance(q, dist.MultivariateNormal):
            try:
                Q[i] = dist.MultivariateNormal(
                    loc=loc, covariance_matrix=covariance
                )
            except (ValueError, RuntimeError):
                continue

        elif isinstance(q, dist.Normal):
            if H.shape != (1, 1):
                raise ValueError(
                    f"Normal distribution at index {i} requires "
                    f"a (1, 1) Hessian, got {H.shape}."
                )

            variance = covariance[0, 0]
            if not torch.isfinite(variance):
                continue

            Q[i] = dist.Normal(loc=loc, scale=torch.sqrt(variance))
        else:
            raise TypeError(f"Unsupported distribution: {type(q)}")

    return Q


def update_pi_prior(
    pi_states,
    pi_priors,
    hessian,
    state_name,
    min_precision,
    max_precision,
    eps
):
    """Update state precision and propagate it to incoming g relations."""
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

    if not torch.isfinite(precision).all():
        return pi_states, pi_priors

    precision = precision.reshape(prior.shape).to(
        device=prior.device, dtype=prior.dtype
    )
    precision = utils.smooth_bound(
        precision, min_precision, max_precision
    ).detach()

    for parent_state, child_precisions in pi_priors.items():
        if state_name in child_precisions:
            child_precisions[state_name] = precision.clone()

    return pi_states, pi_priors


def update_pi_obs(
    navigator,
    pi_obs,
    gamma_obs,
    F,
    Q_,
    state_name,
    min_precision,
    max_precision,
    dt=0.1,
    lr_obs=None
):
    """Update observation precisions while skipping state-output entries."""
    new_gamma_obs = utils.update_wih_grad_desc(
        gamma_obs, F, dt=dt, k=lr_obs
    )

    if not torch.isfinite(new_gamma_obs).all():
        return pi_obs

    new_gamma_obs = utils.smooth_bound(
        new_gamma_obs,
        torch.log(torch.tensor(
            min_precision,
            dtype=new_gamma_obs.dtype,
            device=new_gamma_obs.device
        )),
        torch.log(torch.tensor(
            max_precision,
            dtype=new_gamma_obs.dtype,
            device=new_gamma_obs.device
        ))
    )

    new_pi_actual = torch.exp(new_gamma_obs).detach().reshape(-1)
    outputs_g = navigator.output_g(state_name)
    idx = 0

    for output in outputs_g:
        if output in Q_:
            for q in Q_[output]:
                if hasattr(q, "covariance_matrix"):
                    covariance = torch.as_tensor(q.covariance_matrix)
                    n = covariance.numel() if covariance.ndim <= 1 else min(
                        covariance.shape
                    )
                elif hasattr(q, "scale"):
                    n = torch.as_tensor(q.scale).numel()
                else:
                    raise TypeError(
                        f"Cannot retrieve dimensions for '{output}'."
                    )
                idx += n
            continue

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

        for observation in observations:
            dimension = navigator.get_observation_size(observation)
            end_idx = idx + dimension
            if end_idx > new_pi_actual.numel():
                raise ValueError(
                    "new_pi_actual does not contain enough precision values."
                )
            precision = new_pi_actual[idx:end_idx].clone()
            pi_obs[observation] = (
                precision if dimension > 1 or navigator.node(observation).get("is_vector", False)
                else precision[0]
            )
            idx = end_idx

    return pi_obs


def seconde_derivative_from_output(
    f_out,
    mu,
    min_precision,
    max_precision,
    eps,
    Pi_bruit=None
):
    """Compute one Jacobian-propagated precision matrix per group."""
    if f_out.ndim not in (1, 2):
        raise ValueError("f_out must have 1 or 2 dimensions.")
    if not mu.requires_grad:
        raise ValueError("mu must require gradients.")
    if not f_out.requires_grad or f_out.grad_fn is None:
        raise ValueError("f_out is not connected to an autograd graph.")
    if eps <= 0:
        raise ValueError("eps must be strictly positive.")

    if f_out.ndim == 1:
        f_out = f_out.unsqueeze(-1)
        mu = mu.unsqueeze(-1)

    a, b = f_out.shape
    jacobians = []

    for group in range(a):
        rows = []
        for j in range(b):
            output_scalar = f_out[group, j]
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
                    b, dtype=mu.dtype, device=mu.device
                )
            else:
                grad_group = grad_mu[group].reshape(-1)
            rows.append(grad_group)

        jacobians.append(torch.stack(rows, dim=0))

    J = torch.stack(jacobians, dim=0)

    if Pi_bruit is None:
        Pi_bruit = torch.eye(
            b, dtype=J.dtype, device=J.device
        ).expand(a, b, b)
    else:
        Pi_bruit = Pi_bruit.to(dtype=J.dtype, device=J.device)
        if Pi_bruit.shape == (b, b):
            Pi_bruit = Pi_bruit.expand(a, b, b)
        elif Pi_bruit.shape != (a, b, b):
            raise ValueError(
                f"Pi_bruit must have shape ({b}, {b}) or ({a}, {b}, {b})."
            )

    if not torch.isfinite(Pi_bruit).all():
        raise ValueError("Pi_bruit contains non-finite values.")

    Pi_bruit = torch.stack([
        utils.smooth_positive_matrix(
            Pi_bruit[i], min_precision, max_precision
        )
        for i in range(a)
    ])

    Pi = J.transpose(-2, -1) @ Pi_bruit @ J
    Pi = 0.5 * (Pi + Pi.transpose(-2, -1))

    identity = torch.eye(b, dtype=Pi.dtype, device=Pi.device)
    Pi = Pi + eps * identity.unsqueeze(0)

    Pi = torch.stack([
        utils.smooth_positive_matrix(
            Pi[i], min_precision, max_precision
        )
        for i in range(a)
    ])

    return Pi


def update_threshold(list_pi_priors, F_list, pi_threshold, dt=0.1, k=1.0):
    list_grad = []
    for pi_prior, F in zip(list_pi_priors, F_list):
        grad_F = torch.autograd.grad(
            F,
            pi_prior,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        if grad_F is not None and torch.isfinite(grad_F).all():
            list_grad.append(torch.sum(grad_F))

    if not list_grad:
        return pi_threshold.detach()

    grad_sum = torch.stack(list_grad).sum()
    updated = pi_threshold - k * dt * grad_sum
    return updated.detach() if not torch.isfinite(updated).all() else updated


def policy_error(G_prior, G_posterior, pi_p, max_policy):
    delta_G = G_prior - G_posterior
    return pi_p * delta_G.pow(2)


def update_current_action(best_candidat, actions_id, navigator):
    index = torch.argmax(best_candidat[0]).item()
    action_name = utils.get_name_by_index(actions_id, index)
    action_group_name = navigator.get_groups_by_actions(action_name)
    actions = navigator.get_actions_by_group(action_group_name[0])
    action_one_hot = utils.one_hot(actions, action_name)
    return action_group_name, action_one_hot
