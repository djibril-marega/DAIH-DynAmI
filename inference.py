import torch
import torch.distributions as dist
import utils
import copy

def euler_update(mu, dt, F_value):
    # Gradient of F with respect to each mu_i
    dF_dmu = torch.autograd.grad(F_value, mu, retain_graph=True)[0]
    # D_mu: time derivative of each order
    D_mu = torch.zeros_like(mu)
    D_mu[:-1] = mu[1:]

    # Euler update for all orders
    mu_next = mu + dt * (D_mu - dF_dmu)

    return mu_next



def second_derivative_F(mu, F_value):
    # First derivative, preserving the computation graph
    dF_dmu = torch.autograd.grad(
        F_value,
        mu,
        create_graph=True,
        retain_graph=True
    )[0]

    # Compute the Hessian
    second_derivatives = []

    for i in range(dF_dmu.numel()):
        d2F_dmu2 = torch.autograd.grad(
            dF_dmu.reshape(-1)[i],
            mu,
            retain_graph=True,
            allow_unused=True
        )[0]

        second_derivatives.append(
            torch.zeros_like(mu) if d2F_dmu2 is None else d2F_dmu2
        )

    hessian = torch.stack(second_derivatives).reshape(
        *mu.shape, *mu.shape
    )

    return hessian



def update_distributions(hessian, posterior, Q, eps=1e-6):
    for i, q in enumerate(Q):
        if isinstance(q, dist.MultivariateNormal):
            H_i = hessian[i, :, i, :]
            H_i = 0.5 * (H_i + H_i.T)

            eye = torch.eye(
                H_i.shape[-1],
                device=H_i.device,
                dtype=H_i.dtype
            )
            cov_i = torch.linalg.inv(H_i + eps * eye)

            Q[i] = dist.MultivariateNormal(
                loc=posterior[i],
                covariance_matrix=cov_i
            )

        elif isinstance(q, dist.Normal):
            # Scalar precision of component i
            precision_i = hessian[i, i]

            # Variance = inverse of precision
            variance_i = 1.0 / (precision_i + eps)
            scale_i = torch.sqrt(variance_i)

            Q[i] = dist.Normal(
                loc=posterior[i],
                scale=scale_i
            )

        else:
            raise TypeError(f"Unsupported distribution: {type(q)}")

    return Q


def update_pi_prior(pi_prior, hessian, state_name, eps=1e-6):
    prior = pi_prior[state_name]

    if hessian.ndim == 4:
        # Hessian (n, d, n, d): extract the diagonal blocks
        n, d = hessian.shape[:2]

        blocks = torch.stack([
            hessian[i, :, i, :] for i in range(n)
        ])

        # Diagonal of each block: (n, d)
        precision = torch.diagonal(
            blocks, dim1=-2, dim2=-1
        )

    elif hessian.ndim == 2:
        # Standard Hessian (N, N)
        precision = torch.diagonal(hessian)

    elif hessian.ndim == 1:
        # Already in diagonal form
        precision = hessian

    else:
        raise ValueError(
            f"Unsupported Hessian shape: {hessian.shape}"
        )

    # Check dimension compatibility
    if precision.numel() != prior.numel():
        raise ValueError(
            f"Incompatible dimensions: precision={precision.shape}, "
            f"prior={prior.shape}"
        )

    precision = precision.reshape(prior.shape)

    # Ensure strictly positive precisions
    pi_prior[state_name] = precision.clamp_min(eps)

    return pi_prior


def update_pi_obs(navigator, pi_obs, gamma_obs, F, Q_, state_name, pi_of_g, dt=0.1,iter=1):
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
    # 1. Retrieve the learning rate
    # --------------------------------------------------
    lr = get_lr(pi_of_g, iter=iter)

    # --------------------------------------------------
    # 2. Update gamma_obs using gradient descent
    # --------------------------------------------------
    new_gamma_obs = utils.update_wih_grad_desc(
        gamma_obs,
        F,
        dt=dt,
        k=lr,
    )

    # --------------------------------------------------
    # 3. Convert gamma_obs into actual precisions
    # --------------------------------------------------
    new_pi_actual = torch.clamp(
        torch.exp(new_gamma_obs),
        min=1e-9,
    ).detach()

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


def jacobian_from_output(f_out, mu, Pi_bruit=None, eps=1e-6):
    """
    Computes the propagated precision:
        Pi = J.T @ Pi_bruit @ J + eps * I

    f_out: output already computed and connected to the autograd graph.
    mu: input variable.
    Pi_bruit: noise precision in the output space.
              Defaults to the identity matrix.
    eps: strictly positive regularization term.

    Returns a matrix of shape (mu.numel(), mu.numel()).
    """

    if not f_out.requires_grad or f_out.grad_fn is None:
        raise ValueError(
            "F_out is not connected to an autograd graph."
        )

    if eps <= 0:
        raise ValueError("eps must be strictly positive.")

    # Compute the Jacobian
    rows = []

    for i in range(f_out.numel()):
        grad_i = torch.autograd.grad(
            outputs=f_out.reshape(-1)[i],
            inputs=mu,
            retain_graph=True,
            create_graph=False,
            allow_unused=False
        )[0]

        rows.append(grad_i.reshape(-1))

    J = torch.stack(rows)

    # Noise precision in the output space
    if Pi_bruit is None:
        Pi_bruit = torch.eye(
            f_out.numel(),
            dtype=J.dtype,
            device=J.device
        )
    else:
        Pi_bruit = Pi_bruit.to(
            dtype=J.dtype,
            device=J.device
        )

    if Pi_bruit.shape != (f_out.numel(), f_out.numel()):
        raise ValueError(
            f"Pi_bruit must have shape "
            f"({f_out.numel()}, {f_out.numel()})."
        )

    # Propagated precision
    Pi = J.T @ Pi_bruit @ J

    # Numerical symmetrization
    Pi = 0.5 * (Pi + Pi.T)

    # Background noise to avoid singularity
    I = torch.eye(
        mu.numel(),
        dtype=Pi.dtype,
        device=Pi.device
    )

    Pi = Pi + eps * I

    return Pi


def update_threshold(list_pi_priors, F_list, pi_threshold, dt=0.1, k=1.0):
    list_grad = []

    for pi_prior, F in zip(list_pi_priors, F_list):
        grad_F = torch.autograd.grad(
            F,
            pi_prior,
            create_graph=True,
            retain_graph=True
        )[0]

        list_grad.append(torch.sum(grad_F))

    grad_sum = torch.stack(list_grad).sum()   
    pi_threshold_new = pi_threshold - k * dt * grad_sum

    return pi_threshold_new


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

def get_lr(precision, iter=1):
    return precision*iter


