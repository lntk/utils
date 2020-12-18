import torch
import numpy as np
import sys

MAX_FLOAT = 1e30


def sample_data(data, num_sample=None, replace=False, weights=None):
    n = len(data)

    if weights is None:
        weights = np.ones(shape=(n,)) / n

    if not replace:
        random_ids = np.nonzero(np.random.binomial(n=1, p=weights, size=n))[0].astype(int)

        if num_sample is None:
            samples = np.take(data, random_ids)
        else:
            samples = np.take(data, random_ids)[:num_sample]

        return samples
    else:
        raise NotImplementedError


def get_sub_matrix(X, row_ids, col_ids):
    if isinstance(X, np.ndarray):
        return np.take(np.take(X, row_ids, axis=0), col_ids, axis=1)
    elif isinstance(X, torch.Tensor):
        return X[row_ids].T[col_ids].T
    else:
        raise NotImplementedError


def jacobian_in_batch(out, inp):
    """
    Compute the Jacobian matrix in batch form.

    :param out:
    :param inp:
    :return: (B, D_y, D_x)
    """

    batch = out.shape[0]
    single_y_size = int(np.prod(out.shape[1:]))
    out = out.view(batch, -1)
    vector = torch.ones(batch).to(out)

    # Compute Jacobian row by row.
    # dy_i / dx -> dy / dx
    # (B, D) -> (B, 1, D) -> (B, D, D)

    jac = [torch.autograd.grad(out[:, i], inp,
                               grad_outputs=vector,
                               retain_graph=True,
                               create_graph=True)[0].view(batch, -1)
           for i in range(single_y_size)]
    jac = torch.stack(jac, dim=1)

    return jac


def get_jacobian_tungnd(model, batched_inp, out_dim):
    batch_size = batched_inp.size(0)

    inp = batched_inp.unsqueeze(1)  # batch_size, 1, input_dim
    inp = inp.repeat(1, out_dim, 1)  # batch_size, output_dim, input_dim

    out = model(inp)
    grad_inp = torch.eye(out_dim).reshape(1, out_dim, out_dim).repeat(batch_size, 1, 1).cuda()

    jacobian = torch.autograd.grad(out, [inp], [grad_inp], create_graph=True, retain_graph=True)[0]

    return jacobian


def jacobian_torch(func, inp):
    return torch.autograd.functional.jacobian(func, inp, create_graph=True, strict=True)


def check_invalid_torch(x):
    if torch.sum(torch.isinf(x)):
        raise Exception("Having Inf.")
    if torch.sum(torch.isnan(x)):
        raise Exception("Having NaN.")


def max_norm(X):
    if isinstance(X, np.ndarray):
        if len(X) > 1:
            return np.linalg.norm(X.reshape(-1, ), ord=np.inf)

        return np.linalg.norm(X, ord=np.inf)

    elif isinstance(X, torch.Tensor):
        if len(X) > 1:
            return torch.norm(X.reshape(-1, ), p=float('inf'))

        return torch.norm(X, p=float('inf'))
    else:
        raise NotImplementedError


def is_small(x, verbose=False):
    for type_x in [np.float64]:
        if isinstance(x, type_x):
            if verbose:
                print(f"Comparing {x} of type {type_x} with machine epsilon {np.finfo(type_x).eps}")
            return x < np.finfo(type_x).eps


def nan_to_zero(x):
    if type(x) is np.ndarray:
        x[~ np.isfinite(x)] = 0
    elif torch.is_tensor(x):
        x[~ torch.isfinite(x)] = 0
    else:
        raise NotImplementedError

    return x


def is_psd(matrix, checker="cholesky", symmetric=False, added=1e-15):
    if symmetric:
        matrix = (matrix + matrix.T) / 2

    if checker == "cholesky":
        try:
            np.linalg.cholesky(matrix + added * np.eye(matrix.shape[0]))
        except np.linalg.LinAlgError:
            return False

        return True
    elif checker == "eigs":
        return np.all(np.linalg.eigvals(matrix) >= 0)
