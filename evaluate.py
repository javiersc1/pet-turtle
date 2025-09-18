import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import pandas as pd

from utils import datasets_to_c, get_cluster_acc
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

def _parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to evaluate TURTLE", required=True)
    parser.add_argument('--phis', type=str, default=["clipvitL14", "dinov2"], nargs='+', help="Representation spaces to evaluate TURTLE",
                            choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 'clipvitB32', 'clipvitB16', 'clipvitL14', 'dinov2'])
    parser.add_argument('--root_dir', type=str, default='/home/javier/Desktop/turtle/data', help='Root dir to store everything')
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--ckpt_path', type=str, default="model", help="Path to the checkpoint to evaluate")
    return parser.parse_args(args)

if __name__ == '__main__':
    args = _parse_args()

    # Load pre-computed representations
    Zs_val = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_val.npy").astype(np.float32) for phi in args.phis]
    y_gt_val = np.load(f"{args.root_dir}/labels/{args.dataset}_val.npy")

    print(f'Load dataset {args.dataset}')
    print(f'Representations of {args.phis}: ' + ' '.join(str(Z_val.shape) for Z_val in Zs_val))

    C = datasets_to_c[args.dataset]
    feature_dims = [Z_val.shape[1] for Z_val in Zs_val]

    # Task encoder
    task_encoder = [nn.utils.parametrizations.weight_norm(nn.Linear(d, C)).to(args.device) for d in feature_dims]
    ckpt = torch.load(f"data/task_checkpoints/1space/{args.phis[0]}/{args.dataset}/{args.ckpt_path}.pt", weights_only=True)
    for task_phi, ckpt_phi in zip(task_encoder, ckpt.values()):
        task_phi.load_state_dict(ckpt_phi)

    for task_phi in task_encoder:
        #nn.utils.remove_weight_norm(task_phi)
        nn.utils.parametrize.remove_parametrizations(task_phi, "weight")
    # Evaluate clustering accuracy
    label_per_space = [F.softmax(task_phi(torch.from_numpy(Z_val).to(args.device)), dim=1) for task_phi, Z_val in zip(task_encoder, Zs_val)] # shape of (N, K, C)
    labels = torch.mean(torch.stack(label_per_space), dim=0) # shape of (N, C)

    y_pred = labels.argmax(dim=-1).detach().cpu().numpy()
    cluster_acc, _ = get_cluster_acc(y_pred, y_gt_val)

    phis = '_'.join(args.phis)
    print(f'{args.dataset:12}, {phis:20}, Number of found clusters {len(np.unique(y_pred))}, Cluster Acc: {cluster_acc:.4f}')

    # cm = confusion_matrix(y_gt_val, y_pred)
    # indexes = linear_sum_assignment(_make_cost_m(cm))
    # indexes = np.asarray(indexes)
    # indexes = np.transpose(indexes)
    # js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    # cm2 = cm[:, js]

    # np.save(f"/home/javier/Desktop/turtle/figures/confusion_{args.ckpt_path}.npy", cm2)


