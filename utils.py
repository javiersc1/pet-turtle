import random
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import linear_sum_assignment

def permuteLabels(trueLabels, estLabels):

    Inf = math.inf

    trueLabelVals = np.unique(trueLabels)
    kTrue = len(trueLabelVals)
    estLabelVals = np.unique(estLabels)
    kEst = len(estLabelVals)

    cost_matrix = np.zeros([kEst, kTrue])
    for ii in range(kEst):
        inds = np.where(estLabels == estLabelVals[ii])
        for jj in range(kTrue):
            cost_matrix[ii,jj] = np.size(np.where(trueLabels[inds] == trueLabelVals[jj]))
    
    rInd, cInd = linear_sum_assignment(-cost_matrix)

    outLabels = Inf * np.ones(np.size(estLabels)).reshape(np.size(trueLabels), 1)

    for ii in range(rInd.size):
        outLabels[estLabels == estLabelVals[rInd[ii]]] = trueLabelVals[cInd[ii]]

    outLabelVals = np.unique(outLabels)
    if np.size(outLabelVals) < max(outLabels):
        lVal = 1
        for ii in range(np.size(outLabelVals)):
            outLabels[outLabels == outLabelVals[ii]] = lVal
            lVal += 1       
    return outLabels
    
def missRate(trueLabels, estLabels):
    estLabels = permuteLabels(trueLabels, estLabels)
    err = np.sum(trueLabels != estLabels) / np.size(trueLabels)
    return err, estLabels

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cluster_acc(y_pred, y_true, return_matching=False):
    """
    Calculate clustering accuracy and clustering mean per class accuracy.
    Requires scipy installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        Accuracy in [0,1]
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    match = np.array(list(map(lambda i: col_ind[i], y_pred)))

    mean_per_class = [0 for i in range(D)]
    for c in range(D):
        mask = y_true == c
        mean_per_class[c] = np.mean((match[mask] == y_true[mask]))
    mean_per_class_acc = np.mean(mean_per_class)

    if return_matching:
        return w[row_ind, col_ind].sum() / y_pred.size, mean_per_class_acc, match
    else:
        return w[row_ind, col_ind].sum() / y_pred.size, mean_per_class_acc

def get_nmi(y_pred, y_true):
    """
    Calculate normalized mutual information. Require scikit-learn installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        NMI in [0,1]
    """
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred, average_method='geometric')
    return nmi

def get_ari(y_pred, y_true):
    """
    Calculate adjusted rand index. Require scikit-learn installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        ARI in [0,1]
    """
    return metrics.adjusted_rand_score(y_true, y_pred)

datasets = [
    "food101",
    "food101lt",
    "cifar10",
    "cifar100",
    "birdsnap",
    "sun397",
    "cars",
    "aircraft",
    "dtd",
    "pets",
    "caltech101",
    "flowers",
    "mnist",
    "fer2013",
    "stl10",
    "eurosat",
    "resisc45",
    "gtsrb",
    "kitti",
    "country211",
    "pcam",
    "ucf101",
    "kinetics700",
    "clevr",
    "hatefulmemes",
    "sst",
    "imagenet",
    "cub",
    "cifar10lt",
    "adni2smri",
    "PathMNIST",
    "DermaMNIST",
    "OCTMNIST",
    "BloodMNIST",
    "OrganAMNIST",
    "TissueMNIST",
    "PathMNIST",
    "ChestMNIST",
    "iNaturalist",
    "iNaturalistFull"
]

datasets_to_c = {
    "food101": 101,
    "food101lt": 101,
    "cifar10": 10,
    "cifar100": 100,
    "cifar10020": 20,
    "birdsnap": 500,
    "sun397": 397,
    "cars": 196,
    "aircraft": 100,
    "dtd": 47,
    "pets": 37,
    "caltech101": 102,
    "flowers": 102,
    "mnist": 10,
    "fer2013": 7,
    "stl10": 10,
    "eurosat": 10,
    "resisc45": 45,
    "gtsrb": 43,
    "kitti": 4,
    "country211": 211,
    "pcam": 2,
    "ucf101": 101,
    "kinetics700": 700,
    "clevr": 8,
    "hatefulmemes": 2,
    "sst": 2,
    "imagenet": 1000,
    "cifar10lt": 10,
    "adni2smri": 2,
    "PathMNIST": 9,
    "DermaMNIST": 7,
    "OCTMNIST": 4,
    "BloodMNIST": 8,
    "OrganAMNIST": 11,
    "TissueMNIST": 8,
    "BreastMNIST": 2,
    "PneumoniaMNIST": 2,
    "ChestMNIST": 2,
    "iNaturalist": 13,
    "iNaturalistFull": 5089
}

# food101         training set  75750, test set 25250
# cifar10         training set  50000, test set 10000
# cifar100        training set  50000, test set 10000
# birdsnap        training set  37221, test set 2500
# sun397          training set  19850, test set 19850
# cars            training set   8144, test set 8041
# aircraft        training set   6667, test set 3333
# dtd             training set   3760, test set 1880
# pets            training set   3680, test set 3669
# caltech101      training set   3060, test set 6084
# flowers         training set   2040, test set 6149
# mnist           training set  60000, test set 10000
# fer2013         training set  28709, test set 3589
# stl10           training set   5000, test set 8000
# eurosat         training set  10000, test set 5000
# resisc45        training set  25200, test set 6300
# gtsrb           training set  26640, test set 12630
# kitti           training set   5985, test set 1496
# country211      training set  42200, test set 21100
# pcam            training set 294912, test set 32768
# ucf101          training set   9537, test set 3783
# kinetics700     training set 536485, test set 33966
# clevr           training set   2000, test set 500
# hatefulmemes    training set   8500, test set 500
# sst             training set   7792, test set 1821
# imagenet        training set 1281167, test set 50000


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None, device=torch.device("cpu")):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()
        self.device = device
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=self.device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input
