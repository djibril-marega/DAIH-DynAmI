import torch
import torch.distributions as dist

def derivative_x(function, order, argnums=0):
        def derivative(*args):
            x = args[argnums]
            result = function(*args)

            for _ in range(order):
                result = torch.autograd.grad(
                    outputs=result,
                    inputs=x,
                    grad_outputs=torch.ones_like(result),
                    create_graph=True,
                    retain_graph=True
                )[0]

            return result

        return derivative

def get_vectors(d):
        return torch.cat(list(d.values()))

def one_hot(actions: list, action: str) -> torch.Tensor:
    vector = torch.zeros(len(actions))

    if action not in actions:
        return vector

    index = actions.index(action)
    vector[index] = 1
    return vector

def init_actions(actions, data):
    if data is None:
        data = {}
    for key, values in actions.items():
        data[key] = torch.zeros(len(values))

    return data

def decaler(x, dim=0):
    y = torch.zeros_like(x)
    src = [slice(None)] * x.ndim
    dst = [slice(None)] * x.ndim

    src[dim] = slice(1, None)
    dst[dim] = slice(None, -1)

    y[tuple(dst)] = x[tuple(src)]
    return y

def get_name_by_index(inverted_dict, index):
    """
    Returns the name corresponding to the index from a reversed dictionary {index: name}.
    Supports Python integers and PyTorch tensors.
    """
    # If the index comes from PyTorch (e.g., tensor(1)), extract its integer value
    if hasattr(index, 'item'):
        index = index.item()
        
    # Direct and highly efficient lookup in the reversed dictionary
    return inverted_dict.get(index, f"Unknown index {index}")


def extract_variance(q):
    if isinstance(q, dist.Normal):
        return q.variance

    elif isinstance(q, dist.MultivariateNormal):
        return q.covariance_matrix

    else:
        raise TypeError(f"Unsupported distribution: {type(q)}")

def update_wih_grad_desc(x, F, dt=0.1, k=1.0):


    # Gradient of F with respect to the observations
    grad_F = torch.autograd.grad(
        F,
        x,
        create_graph=True,
        retain_graph=True
    )[0]

    # A single Euler step
    next_x = x - dt * k * grad_F

    return next_x