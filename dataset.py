import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sklearn.datasets import make_swiss_roll, make_s_curve
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FashionMNIST, MNIST, SVHN, CIFAR10
from torchvision.transforms import transforms
from PIL import Image

from tqdm import tqdm
import io


####################################################################################
# LANGUAGE DATASET
# Author: Khang Le
####################################################################################
def load_vec(emb_path, nmax=None):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in tqdm(enumerate(f)):
            word, vec = line.rstrip().split(' ', 1)
            vec = np.fromstring(vec, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vec)
            word2id[word] = len(word2id)

            if nmax is not None:
                if len(word2id) == nmax:
                    break

    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


def get_wordvec_path(language, data_dir="./data"):
    if language == "en":
        return f"{data_dir}/wiki.en.vec"
    elif language == "fr":
        return f"{data_dir}/wiki.fr.vec"
    else:
        raise NotImplementedError


class WordVec(Dataset):
    def __init__(self, language="en", nmax=None, data_dir="./data"):
        self.embeddings, self.id2word, self.word2id = load_vec(get_wordvec_path(language=language, data_dir=data_dir), nmax=nmax)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]).float(), self.id2word[idx]


def WordVec_loader(language, batch_size, train, data_dir="./data"):
    shuffle = train

    dataset = WordVec(language=language, data_dir=data_dir, nmax=200000)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             drop_last=True,
                             shuffle=shuffle)

    return data_loader


####################################################################################
# TOY DATASET
# Author: Khang Le
####################################################################################

# Source: https://gist.github.com/ctralie/128cc07da67f1d2e10ea470ee2d23fe8
def getGreedyPerm(D):
    """
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points
    Return
    ------
    tuple (list, list)
        (permutation (N-length array of indices),
        lambdas (N-length array of insertion radii))
    """

    N = D.shape[0]
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return perm, lambdas


def get_toy_anchors(data_x, data_y, num_anchor=10, noise=0.05):
    if data_x == "scurve" and data_y == "swissroll":
        x, tx = get_toy_data(data=data_x, num_point=num_anchor, noise=noise)
        xx = x[:, 0]
        xy = x[:, 1]
        xz = x[:, 2]

        ty = 1.5 * np.pi * (1 + 2 * (tx / (3 * np.pi) + 0.5))
        yx = ty * np.cos(ty)
        yy = xy * 11.5
        yz = ty * np.sin(ty)

        yx = np.expand_dims(yx, axis=1)
        yy = np.expand_dims(yy, axis=1)
        yz = np.expand_dims(yz, axis=1)

        y = np.concatenate((yx, yy, yz), axis=1)

        return (x, tx), (y, ty)
    elif data_x == "swissroll" and data_y == "scurve":
        x, tx = get_toy_data(data=data_y, num_point=num_anchor, noise=noise)
        xx = x[:, 0]
        xy = x[:, 1]
        xz = x[:, 2]

        ty = 1.5 * np.pi * (1 + 2 * (tx / (3 * np.pi) + 0.5))
        yx = ty * np.cos(ty)
        yy = xy * 11.5
        yz = ty * np.sin(ty)

        yx = np.expand_dims(yx, axis=1)
        yy = np.expand_dims(yy, axis=1)
        yz = np.expand_dims(yz, axis=1)

        y = np.concatenate((yx, yy, yz), axis=1)

        return (y, ty), (x, tx)
    if data_x == "scurve" and data_y == "swissroll_aligned":
        x, tx = get_toy_data(data=data_x, num_point=num_anchor, noise=noise)
        xx = x[:, 0]
        xy = x[:, 1]
        xz = x[:, 2]

        ty = 1.5 * np.pi * (1 + 2 * (tx / (3 * np.pi) + 0.5))
        yx = ty * np.cos(ty)
        yy = xy
        yz = ty * np.sin(ty)

        yx = np.expand_dims(yx, axis=1)
        yy = np.expand_dims(yy, axis=1)
        yz = np.expand_dims(yz, axis=1)

        y = np.concatenate((yx, yy, yz), axis=1)

        y = y / 9.482798076150805

        return (x, tx), (y, ty)
    elif data_x == "swissroll_aligned" and data_y == "scurve":
        x, tx = get_toy_data(data=data_y, num_point=num_anchor, noise=noise)
        xx = x[:, 0]
        xy = x[:, 1]
        xz = x[:, 2]

        ty = 1.5 * np.pi * (1 + 2 * (tx / (3 * np.pi) + 0.5))
        yx = ty * np.cos(ty)
        yy = xy
        yz = ty * np.sin(ty)

        yx = np.expand_dims(yx, axis=1)
        yy = np.expand_dims(yy, axis=1)
        yz = np.expand_dims(yz, axis=1)

        y = np.concatenate((yx, yy, yz), axis=1)

        y = y / 9.482798076150805

        return (y, ty), (x, tx)
    else:
        raise NotImplementedError


def sampling(distribution, size, variance=1.0):
    if distribution == "gaussian":
        return torch.randn(*size) * variance
    elif distribution == "uniform":
        return torch.rand(*size)
    elif distribution == "sphere":
        if len(size) > 2:
            batch_size = size[0]
            dim_z = size[1]

            z = torch.randn(*size)
            norm_z = torch.norm(z.view(batch_size, dim_z), dim=-1, p=2, keepdim=True)

            while torch.min(norm_z) == 0:
                z = torch.randn(*size)
                norm_z = torch.norm(z.view(batch_size, dim_z), dim=-1, p=2, keepdim=True)

            z_norm = z / norm_z.view(batch_size, 1, 1)
        elif len(size) == 2:
            z = torch.randn(*size)
            norm_z = torch.norm(z, dim=-1, p=2, keepdim=True)

            while torch.min(norm_z) == 0:
                z = torch.randn(*size)
                norm_z = torch.norm(z, dim=-1, p=2, keepdim=True)

            z_norm = z / norm_z
        else:
            raise NotImplementedError

        # uniform_vec = torch.ones(*size[1:]) / torch.norm(torch.ones(*size[1:]), dim=-1, p=2, keepdim=True)
        # z_norm[torch.isnan(z_norm)] = torch.ones(*size[1:]) / tor  # skip batch dimension

        # print(torch.min(z_norm))
        # print(torch.max(z_norm))
        # print(torch.sum(torch.isnan(z_norm)))
        return z_norm
    else:
        raise NotImplementedError


def make_2d_manifold(n_samples, noise=0.0):
    r = Rotation.from_euler("xyz", [15, 30, 45], degrees=True)
    t = np.random.rand(n_samples, 2)
    x = np.array(t[:, :1]) * 4
    y = np.array(t[:, 1:2]) * 2
    z = np.zeros_like(x)
    t = t[:, 0]

    X = np.concatenate((x, y, z), axis=-1)  # combine (x, y, z)
    X = r.apply(X)  # rotate
    X += noise * np.random.rand(n_samples, 3)

    return X, t


def make_swiss_roll_with_hole(n_samples=100, noise=0.0):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    x = t * np.cos(t)
    y = 21 * np.random.rand(1, n_samples)
    z = t * np.sin(t)

    X = np.concatenate((x, y, z))
    X += noise * np.random.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)

    return X, t


def gaussians_4mode(sample_size, scale=5., zoom=1.):
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    dataset = []
    y = []
    for i in range(sample_size):
        point = np.random.randn(2) * .2
        index = random.randint(0, len(centers) - 1)
        center = centers[index]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
        y.append(index)

    dataset = np.array(dataset) * zoom
    y = np.array(y)

    return dataset, y


def gaussians_8mode(sample_size):
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    dataset = []
    y = []
    for i in range(sample_size):
        point = np.random.randn(2) * .2
        index = random.randint(0, len(centers) - 1)
        center = centers[index]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
        y.append(index)
    return np.array(dataset), np.array(y)


def gaussians_5mode(sample_size):
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (0, 0)
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    dataset = []
    y = []
    for i in range(sample_size):
        point = np.random.randn(2) * .2
        index = random.randint(0, len(centers) - 1)
        center = centers[index]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
        y.append(index)
    return np.array(dataset), np.array(y)


def gaussians_25mode(sample_size):
    scale = 3.
    centers = list()
    for i in range(-2, 3):
        for j in range(-2, 3):
            centers.append((i, j))

    centers = [(scale * x, scale * y) for x, y in centers]
    dataset = []
    y = []
    for i in range(sample_size):
        point = np.random.randn(2) * .2
        index = random.randint(0, len(centers) - 1)
        center = centers[index]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
        y.append(index)
    return np.array(dataset), np.array(y)


def gaussians_3d_4mode(sample_size):
    scale = 2.
    centers = [
        (0, 1, -1),
        (0, 1, 1),
        (0, -1, 1),
        (0, -1, -1),
    ]
    centers = [(scale * x, scale * y, scale * z) for x, y, z in centers]
    dataset = []
    y = []
    for i in range(sample_size):
        point = np.random.randn(3) * .2
        index = random.randint(0, len(centers) - 1)
        center = centers[index]
        point[0] += center[0]
        point[1] += center[1]
        point[2] += center[2]
        dataset.append(point)
        y.append(index)
    return np.array(dataset), np.array(y)


class GaussianData(Dataset):
    def __init__(self, sample_size, mode="4mode"):
        if mode == "4mode":
            x, y = gaussians_4mode(sample_size=sample_size)
        elif mode == "5mode":
            x, y = gaussians_5mode(sample_size=sample_size)
        elif mode == "8mode":
            x, y = gaussians_8mode(sample_size=sample_size)
        elif mode == "3d_4mode":
            x, y = gaussians_3d_4mode(sample_size=sample_size)
        else:
            raise NotImplementedError

        self.x = torch.from_numpy(x).float().cuda()
        self.y = torch.from_numpy(y).float().cuda()

    def __len__(self):
        L, _ = self.x.shape
        return L

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def farthest_point_sampling(data, num_point, noise=0.05):
    from graph.utils import compute_distance_matrix

    num_coverage = 1000
    x, label_x = get_toy_data(data=data, num_point=num_coverage, noise=noise)
    Dx = compute_distance_matrix(x)
    perm, _ = getGreedyPerm(Dx)

    x = np.take(x, indices=perm[:num_point], axis=0)
    label_x = np.take(label_x, indices=perm[:num_point], axis=0)

    return x, label_x


def get_toy_data(data, num_point, noise=0.05, fps=False):
    """

    :param data:
    :param num_point:
    :param noise: float, default=0.05.
    :param fps: bool, default=False. [Whether to use **Farthest point sampling**.]
    :return:
    """

    if data == "swissroll":
        if fps:
            return farthest_point_sampling(data, num_point, noise)
        else:
            return make_swiss_roll(num_point, noise=noise)
    elif data == "scurve":
        if fps:
            return farthest_point_sampling(data, num_point, noise)
        else:
            return make_s_curve(num_point, noise=noise)
    elif data == "swissroll_aligned":
        if fps:
            x, label_x = farthest_point_sampling(data, num_point, noise)
        else:
            x, label_x = make_swiss_roll(num_point, noise=noise)

        x[1, :] = x[1, :] / 10.5
        x = x / 9.482798076150805

        return x, label_x

    elif data == "plane2d":
        return make_2d_manifold(num_point, noise=noise)
    elif data == "gaussian25":
        return gaussians_25mode(num_point)
    elif data == "gaussian4":
        return gaussians_4mode(num_point)
    elif data == "gaussian5":
        return gaussians_5mode(num_point)
    elif data == "gaussian8":
        return gaussians_8mode(num_point)
    else:
        raise NotImplementedError


####################################################################################
# IMAGE DATASET
# Author: Khang Le
####################################################################################
def get_image_anchors(data_x, data_y, num_anchor, data_dir="data", random_sample=False):
    data_iter_x = iter(get_data_loader(data=data_x, batch_size=1, train=True, data_dir=data_dir))
    data_iter_y = iter(get_data_loader(data=data_y, batch_size=1, train=True, data_dir=data_dir))

    anchor_x = list()
    anchor_y = list()

    if random_sample:
        for i in range(num_anchor):
            tmp_x, label_tmp_x = next(data_iter_x)
            while True:
                tmp_y, label_tmp_y = next(data_iter_y)
                if label_tmp_y == label_tmp_x:
                    anchor_x.append(tmp_x)
                    anchor_y.append(tmp_y)
                    break
    else:
        for _ in range(int(num_anchor / 10)):
            found = list()

            while True:
                if len(found) == 10:
                    break

                tmp_x, label_tmp_x = next(data_iter_x)
                if label_tmp_x in found:
                    continue
                else:
                    while True:
                        tmp_y, label_tmp_y = next(data_iter_y)
                        if label_tmp_y == label_tmp_x:
                            anchor_x.append(tmp_x)
                            anchor_y.append(tmp_y)
                            found.append(label_tmp_x)
                            break

    anchor_x = torch.cat(anchor_x, dim=0)
    anchor_y = torch.cat(anchor_y, dim=0)

    return anchor_x, anchor_y


def get_image_data_same_label(data, label, num_data, data_dir="data", train=False):
    data_iter = iter(get_data_loader(data=data, batch_size=1, train=train, data_dir=data_dir))
    selected = list()
    count = 1

    while True:
        tmp, label_tmp = next(data_iter)
        if label_tmp == label:
            selected.append(tmp)
            count += 1

        if count > num_data:
            break

    selected = torch.cat(selected, dim=0)

    return selected


def infinite_data_loader(data, batch_size, train=True, data_dir="./data", get_labels=False):
    data_loader = get_data_loader(data, batch_size, train, data_dir)

    while True:
        for images, target in iter(data_loader):
            if get_labels:
                yield images, target
            else:
                yield images


def get_data_loader(data, batch_size, train=True, data_dir="./data", data_transform=None, return_dataset=False):
    if data == "mnist":
        return MNIST_loader(batch_size=batch_size, train=train, data_dir=data_dir)
    elif data == "mnist64":
        return MNIST_loader(batch_size=batch_size, train=train, size=64, data_dir=data_dir)
    elif data == "bmnist":
        return BoldMNIST_loader(batch_size=batch_size, train=train, size=32, data_dir=data_dir)
    elif data == "rmnist":
        return RotatedMNIST_loader(batch_size=batch_size, train=train, size=32, data_dir=data_dir)
    elif data == "fmnist":
        return FashionMNIST_loader(batch_size=batch_size, train=train, data_dir=data_dir)
    elif data == "fmnist64":
        return FashionMNIST_loader(batch_size=batch_size, train=train, size=64, data_dir=data_dir)
    elif data == "svhn":
        return SVHN_loader(batch_size=batch_size, train=train, data_dir=data_dir)
    elif data == "svhn_gray":
        return SVHN_GRAY_loader(batch_size=batch_size, train=train, data_dir=data_dir)
    elif data == "svhn64":
        return SVHN_loader(batch_size=batch_size, train=train, size=64, data_dir=data_dir)
    elif data == "wordvec_en":
        return WordVec_loader(language="en", batch_size=batch_size, train=train, data_dir=data_dir)
    elif data == "wordvec_fr":
        return WordVec_loader(language="fr", batch_size=batch_size, train=train, data_dir=data_dir)
    elif data == "cifar_gray":
        return CIFAR_GRAY_loader(batch_size=batch_size, train=train, data_dir=data_dir)
    elif data == "cifar":
        return CIFAR_loader(batch_size=batch_size, train=train, data_dir=data_dir, data_transform=data_transform, return_dataset=return_dataset)
    else:
        raise NotImplementedError


def MNIST_loader(batch_size, train, size=32, data_dir="./data", return_dataset=False):
    shuffle = train

    dataset = MNIST(data_dir,
                    train=train,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        # transforms.Normalize([0.5], [0.5]),
                    ]))

    if return_dataset:
        return dataset

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             drop_last=True,
                             shuffle=shuffle)

    return data_loader


def BoldMNIST_loader(batch_size, train, size=32, data_dir="./data"):
    from utils.transform import Dilate

    shuffle = train

    dataset = MNIST(data_dir,
                    train=train,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize(size),
                        Dilate(kernel_size=3),
                        transforms.ToTensor(),
                        # transforms.Normalize([0.5], [0.5]),
                    ]))

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             drop_last=True,
                             shuffle=shuffle)

    return data_loader


def RotatedMNIST_loader(batch_size, train, size=32, data_dir="./data"):
    shuffle = train

    dataset = MNIST(data_dir, train=train, download=True,
                    transform=transforms.Compose([
                        transforms.Resize(size),
                        transforms.RandomRotation(degrees=(45, 45), fill=(0,)),
                        transforms.ToTensor(),
                    ]))

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             drop_last=True,
                             shuffle=shuffle)

    return data_loader


def FashionMNIST_loader(batch_size, train, size=32, data_dir="./data"):
    shuffle = train

    dataset = FashionMNIST(data_dir,
                           train=train,
                           download=True,
                           transform=transforms.Compose([
                               transforms.Resize(size),
                               transforms.ToTensor(),
                           ]))

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             drop_last=True,
                             shuffle=shuffle)

    return data_loader


def SVHN_loader(batch_size, train, size=32, data_dir="./data"):
    shuffle = train

    if train:
        split = "train"
    else:
        split = "tests"

    dataset = SVHN(data_dir,
                   split=split,
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize(size),
                       transforms.ToTensor(),
                       # transforms.Normalize([0.5], [0.5]),
                   ]))

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             drop_last=True,
                             shuffle=shuffle)

    return data_loader


def SVHN_GRAY_loader(batch_size, train, size=32, data_dir="./data"):
    shuffle = train

    if train:
        split = "train"
    else:
        split = "tests"

    dataset = SVHN(data_dir,
                   split=split,
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize(size),
                       transforms.ToTensor(),
                       transforms.Grayscale(),
                       # transforms.Normalize([0.5], [0.5]),
                   ]))

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             drop_last=True,
                             shuffle=shuffle)

    return data_loader


def CIFAR_GRAY_loader(batch_size, train, size=32, data_dir="./data"):
    shuffle = train

    dataset = CIFAR10(data_dir,
                      train=train,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Grayscale(num_output_channels=1),
                          transforms.Resize(size),
                          transforms.ToTensor(),
                      ]))

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             drop_last=True,
                             shuffle=shuffle)

    return data_loader


def CIFAR_loader(batch_size, train, size=32, data_dir="./data", data_transform=None, return_dataset=False):
    shuffle = train

    if data_transform is None:
        data_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    dataset = CIFAR10(data_dir,
                      train=train,
                      download=True,
                      transform=data_transform)

    if return_dataset:
        return dataset

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             drop_last=True,
                             shuffle=shuffle)

    return data_loader


class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset, train, data, label, label_numpy):
        self.mnist_dataset = dataset
        self.train = train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = label
            self.train_data = data
            self.labels_set = set(label_numpy)
            self.label_to_indices = {label: np.where(label_numpy == label)[0] for label in self.labels_set}
        else:
            self.test_labels = label
            self.test_data = data
            # generate fixed triplets for testing
            self.labels_set = set(label_numpy)
            self.label_to_indices = {label: np.where(label_numpy == label)[0] for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[np.random.choice(
                             list(self.labels_set - set([self.test_labels[i].item()]))
                         )])]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        if type(img1).__module__ == np.__name__:
            if len(img1.shape) < 3:
                mode = "L"
            else:
                mode = "RGB"
                img1 = np.transpose(img1, (1, 2, 0))
                img2 = np.transpose(img2, (1, 2, 0))
                img3 = np.transpose(img3, (1, 2, 0))
        else:
            if len(img1.numpy().shape) < 3:
                mode = "L"
            else:
                mode = "RGB"
                img1 = np.transpose(img1.numpy(), (1, 2, 0))
                img2 = np.transpose(img2.numpy(), (1, 2, 0))
                img3 = np.transpose(img3.numpy(), (1, 2, 0))

        img1 = Image.fromarray(img1, mode=mode)
        img2 = Image.fromarray(img2, mode=mode)
        img3 = Image.fromarray(img3, mode=mode)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


def triplet_loader(data, batch_size, train, size=32, data_dir="./data"):
    shuffle = train

    if data == "mnist":
        mnist_dataset = MNIST(data_dir,
                              train=train,
                              download=True,
                              transform=transforms.Compose([
                                  transforms.Resize(size),
                                  transforms.ToTensor(),
                              ]))

        if train:
            kwargs = {"train": train, "data": mnist_dataset.data, "label": mnist_dataset.train_labels, "label_numpy": mnist_dataset.train_labels.numpy()}
        else:
            kwargs = {"train": train, "data": mnist_dataset.test_data, "label": mnist_dataset.test_labels, "label_numpy": mnist_dataset.test_labels.numpy()}

        dataset = TripletDataset(dataset=mnist_dataset, **kwargs)
    elif data == "svhn":
        if train:
            split = "train"
        else:
            split = "tests"

        svhn_dataset = SVHN(data_dir,
                            split=split,
                            download=True,
                            transform=transforms.Compose([
                                transforms.Resize(size),
                                transforms.ToTensor(),
                            ]))

        if train:
            kwargs = {"train": train, "data": svhn_dataset.data, "label": svhn_dataset.labels, "label_numpy": svhn_dataset.labels}
        else:
            kwargs = {"train": train, "data": svhn_dataset.data, "label": svhn_dataset.labels, "label_numpy": svhn_dataset.labels}

        dataset = TripletDataset(dataset=svhn_dataset, **kwargs)
    else:
        raise NotImplementedError

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             drop_last=True,
                             shuffle=shuffle)

    return data_loader
