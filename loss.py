from utils.general import normalise_matrices
from utils.metric import get_inner_distances, euclidean_distance, cosine_similarity
import torch
from torch.nn import MSELoss, BCELoss
from optra.gromov_wasserstein import entropic_gromov_wasserstein


def entropic_norm_gromov_wasserstein_loss(D_x, D_g, epsilon, niter, loss_fun='square_loss', coupling=True, cuda=False):
    p = torch.ones((D_x[0].shape[1],))
    p /= torch.numel(p)
    q = torch.ones((D_g[0].shape[1],))
    q /= torch.numel(q)

    gw_x_x, _ = entropic_gromov_wasserstein(D_x, D_x, p, p,
                                            loss_fun, epsilon, niter,
                                            coupling=coupling, cuda=cuda)
    gw_x_g, T = entropic_gromov_wasserstein(D_x, D_g, p, q,
                                            loss_fun, epsilon, niter,
                                            coupling=coupling, cuda=cuda)
    gw_g_g, _ = entropic_gromov_wasserstein(D_g, D_g, q, q,
                                            loss_fun, epsilon, niter,
                                            coupling=coupling, cuda=cuda)

    # compute normalized Gromov-Wasserstein distance
    return 2 * gw_x_g - gw_x_x - gw_g_g, T


def reconstruction_loss(name="mse", **kwargs):
    if name == "gromov":
        return gwae_batch_loss(x=kwargs["x"], y=kwargs["x_recon"], adversary=kwargs["adversary"])
    elif name == "mse":
        return MSELoss()(kwargs["x"], kwargs["x_recon"])
    elif name == "bce":
        return BCELoss(reduction="mean")(kwargs["x"], kwargs["x_recon"])
    else:
        raise NotImplementedError


def gwae_batch_loss(x1, x2, y1, y2, adversary=None, normalized=False, metric="euclidean"):
    batch_size = x1.shape[0]

    if adversary is None:
        f_x1 = x1.view(batch_size, -1)
        f_x2 = x2.view(batch_size, -1)

        f_y1 = y1.view(batch_size, -1)
        f_y2 = y2.view(batch_size, -1)
    else:
        f_x1, f_y1 = adversary(x1, y1)
        f_x2, f_y2 = adversary(x2, y2)

    if metric == "euclidean":
        D_x = euclidean_distance(f_x1, f_x2)
        D_y = euclidean_distance(f_y1, f_y2)
    elif metric == "cosine":
        D_x = cosine_similarity(f_x1, f_x2)
        D_y = cosine_similarity(f_y1, f_y2)
    else:
        raise NotImplementedError

    if normalized:
        D_x = normalise_matrices(D_x)
        D_y = normalise_matrices(D_y)

    loss = torch.mean((D_x - D_y) ** 2)

    if adversary is None:
        return loss
    else:
        return loss, f_x1, f_x2, f_y1, f_y2


def bce_loss(x, y):
    return BCELoss(reduction="mean")(x, y)


def mse_loss(x, y, reduction="sum"):
    return MSELoss(reduction=reduction)(x, y)


def get_alignment_transform(target, source):
    m = torch.mm(torch.t(target), source)
    u, _, v = torch.svd(m.cpu())
    p = torch.mm(u, torch.t(v))

    return p


def compute_alignment_loss(target, source, lam=1.):
    """

    :param lam: weighting coefficient
    :param target: target shape
    :param source: source shape
    :return: alignment loss
    """

    dimension = target.shape[1]

    mean_target = torch.mean(target, dim=0, keepdim=True)
    mean_source = torch.mean(source, dim=0, keepdim=True)

    p = get_alignment_transform(
        target=(target - mean_target),
        source=(source - mean_source)
    ).cuda()

    translation_offset = mean_target - torch.mm(mean_source, torch.t(p))

    loss_translation = torch.norm(translation_offset)
    loss_ortho = torch.norm(p - torch.eye(dimension).cuda())

    print(f"loss_translation: {loss_translation}")
    print(f"loss_ortho: {loss_ortho}")

    loss_align = loss_translation + lam * loss_ortho

    return loss_align
