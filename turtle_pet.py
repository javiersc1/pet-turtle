import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score

from utils import seed_everything, get_cluster_acc, datasets_to_c
from utils import Sparsemax

def power_law_distribution(size: int, exponent: float):
    """Returns a power law distribution summing up to 1."""
    k = torch.arange(1, size + 1)
    power_dist = k ** (-exponent)
    power_dist = power_dist / power_dist.sum()
    return power_dist

def _parse_args(args):
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', type=str, help="Dataset to run TURTLE", required=True)
    parser.add_argument('--phis', type=str, default=["clipvitL14", "dinov2"], nargs='+', help="Representation spaces to run TURTLE",
                            choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 'clipvitB32', 'clipvitB16', 'clipvitL14', 'dinov2'])
    # training
    parser.add_argument('--gamma', type=float, default=100.0, help='Hyperparameter for entropy regularization in Eq. (12)')
    parser.add_argument('--power', type=float, default=0.5, help='Hyperparameter for Power Law Distribution')

    parser.add_argument('--T', type=int, default=6000, help='Number of outer iterations to train task encoder')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='Learning rate for inner loop')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='Learning rate for task encoder')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--warm_start', action='store_true', help="warm start = initialize inner learner from previous iteration, cold start = initialize randomly, cold-start is used by default")
    parser.add_argument('--M', type=int, default=10, help='Number of inner steps at each outer iteration')
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--root_dir', type=str, default="data", help='Root dir to store everything')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args(args)

def run(args=None):
    args = _parse_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    sparsemax = Sparsemax(dim=1, device=device)
    # Load pre-computed representations
    Zs_train = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_train.npy").astype(np.float32) for phi in args.phis]
    Zs_val = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_val.npy").astype(np.float32) for phi in args.phis]
    y_gt_val = np.load(f"{args.root_dir}/labels/{args.dataset}_val.npy")
    print(f'Load dataset {args.dataset}')
    print(f'Representations of {args.phis}: ' + ' '.join(str(Z_train.shape) for Z_train in Zs_train))

    n_tr, C = Zs_train[0].shape[0], datasets_to_c[args.dataset]
    feature_dims = [Z_train.shape[1] for Z_train in Zs_train]
    batch_size = min(args.batch_size, n_tr)
    print("Number of training samples:", n_tr)

    if args.dataset == "ignore":
        targetDistribution = torch.tensor(np.load("/home/javier/Desktop/turtle/data/labels/food101lt_distribution.npy")).to(args.device)
    else:
        targetDistribution = power_law_distribution(C, args.power).to(args.device)

    # Define task encoder
    #task_encoder = [nn.utils.weight_norm(nn.Linear(d, C)).to(args.device) for d in feature_dims]
    task_encoder = [nn.utils.parametrizations.weight_norm(nn.Linear(d, C)).to(args.device) for d in feature_dims]

    def task_encoding(Zs):
        assert len(Zs) == len(task_encoder)
        # Generate labeling by the average of $\sigmoid(\theta \phi(x))$, Eq. (9) in the paper
        label_per_space = [sparsemax(task_phi(z)) for task_phi, z in zip(task_encoder, Zs)] # shape of (K, N, C)

        labels = torch.mean(torch.stack(label_per_space), dim=0) # shape of (N, C)
        return labels, label_per_space

    # we use Adam optimizer for faster convergence, other optimziers such as SGD could also work
    optimizer = torch.optim.Adam(sum([list(task_phi.parameters()) for task_phi in task_encoder], []), lr=args.outer_lr, betas=(0.9, 0.999))

    # Define linear classifiers for the inner loop
    def init_inner():
        W_in = [nn.Linear(d, C).to(args.device) for d in feature_dims]
        inner_opt = torch.optim.Adam(sum([list(W.parameters()) for W in W_in], []), lr=args.inner_lr, betas=(0.9, 0.999))

        return W_in, inner_opt

    W_in, inner_opt = init_inner()

    best_acc = 0.0

    num_spaces = len(args.phis)
    phis = '_'.join(args.phis)
    exp_path = f"{args.root_dir}/task_checkpoints/{num_spaces}space/{phis}/{args.dataset}"

    # start training
    iters_bar = tqdm(range(args.T))
    for i in iters_bar:
        optimizer.zero_grad()
        # load batch of data
        indices = np.random.choice(n_tr, size=batch_size, replace=False)
        Zs_tr = [torch.from_numpy(Z_train[indices]).to(args.device) for Z_train in Zs_train]

        labels, label_per_space = task_encoding(Zs_tr)

        # init inner
        if not args.warm_start:
            # cold start, re-init every time
            W_in, inner_opt = init_inner()
        # else, warm start, keep previous

        # inner loop: update linear classifiers
        for idx_inner in range(args.M):
            inner_opt.zero_grad()
            # stop gradient by "labels.detach()" to perform first-order hypergradient approximation, i.e., Eq. (13) in the paper
            loss = sum([F.cross_entropy(w_in(z_tr), labels.detach()) for w_in, z_tr in zip(W_in, Zs_tr)])
            loss.backward()
            inner_opt.step()

        # update task encoder
        optimizer.zero_grad()
        pred_error = sum([F.cross_entropy(w_in(z_tr).detach(), labels) for w_in, z_tr in zip(W_in, Zs_tr)])

        # entropy regularization
        #entr_reg = sum([torch.special.entr(l.mean(0)).sum() for l in label_per_space])
        estimatedDistribution = label_per_space[0].mean(0)
        estimatedDistribution,_ = torch.sort(estimatedDistribution, descending=True)

        dist_reg = nn.functional.kl_div(estimatedDistribution.log(), targetDistribution, reduction='batchmean')

        # final loss, Eq. (12) in the paper
        (pred_error + args.gamma * dist_reg).backward()
        optimizer.step()

        # evaluation, compute clustering accuracy on test split
        if (i+1) % 20 == 0 or (i+1) == args.T:
            labels_val, _ = task_encoding([torch.from_numpy(Z_val).to(args.device) for Z_val in Zs_val])
            preds_val = labels_val.argmax(dim=1).detach().cpu().numpy()
            cluster_acc, _ = get_cluster_acc(preds_val, y_gt_val)
            if cluster_acc > best_acc and len(np.unique(preds_val)) == C:
                print("best acc: "+str(cluster_acc)+" with clusters  "+str(len(np.unique(preds_val))))
                best_acc = cluster_acc
                torch.save({f'phi{i+1}': task_phi.state_dict() for i, task_phi in enumerate(task_encoder)}, f'{exp_path}/turtle_pet_model.pt')

            iters_bar.set_description(f'Training loss {float(pred_error.detach()):.3f}, reg {float(dist_reg.detach()):.3f}, found clusters {len(np.unique(preds_val))}/{C}, cluster acc {cluster_acc:.4f}')

    print(f'Training finished! ')
    print(f'Training loss {float(pred_error.detach()):.3f}, reg {float(dist_reg.detach()):.3f}, Number of found clusters {len(np.unique(preds_val))}/{C}, Cluster Acc {cluster_acc:.4f}')


    # save results
    inner_start = 'warmstart' if args.warm_start else 'coldstart'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    for task_phi in task_encoder:
        #nn.utils.remove_weight_norm(task_phi)
        nn.utils.parametrize.remove_parametrizations(task_phi, "weight")

    #task_path = f"turtle_{phis}_innerlr{args.inner_lr}_outerlr{args.outer_lr}_T{args.T}_M{args.M}_{inner_start}_gamma{args.gamma}_bs{args.batch_size}_seed{args.seed}"
    #torch.save({f'phi{i+1}': task_phi.state_dict() for i, task_phi in enumerate(task_encoder)}, f'{exp_path}/model.pt')


if __name__ == '__main__':
    run()
