import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils import seed_everything, get_cluster_acc, datasets_to_c

def _parse_args(args):
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', type=str, help="Dataset to run TURTLE", required=True)
    parser.add_argument('--phis', type=str, default=["clipvitL14", "dinov2"], nargs='+', help="Representation spaces to run TURTLE",
                            choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 'clipvitB32', 'clipvitB16', 'clipvitL14', 'dinov2', "auccl"])
    # training
    parser.add_argument('--gamma', type=float, default=1.0, help='Hyperparameter for entropy regularization in Eq. (12)')
    parser.add_argument('--T', type=int, default=6000, help='Number of outer iterations to train task encoder')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='Learning rate for inner loop')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='Learning rate for task encoder')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--warm_start', action='store_true', help="warm start = initialize inner learner from previous iteration, cold start = initialize randomly, cold-start is used by default")
    parser.add_argument('--M', type=int, default=10, help='Number of inner steps at each outer iteration')
    # others
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--root_dir', type=str, default="data", help='Root dir to store everything')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args(args)

def run(args=None):
    args = _parse_args(args)
    seed_everything(args.seed)

    # Load pre-computed representations
    Zs_train = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_train.npy").astype(np.float32) for phi in args.phis]
    Zs_val = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_val.npy").astype(np.float32) for phi in args.phis]
    y_gt_val = np.load(f"{args.root_dir}/labels/{args.dataset}_val.npy")

    n_tr, C = Zs_train[0].shape[0], datasets_to_c[args.dataset]
    feature_dims = [Z_train.shape[1] for Z_train in Zs_train]
    batch_size = min(args.batch_size, n_tr)

    if args.verbose:
        print(f'Load dataset {args.dataset}')
        print(f'Representations of {args.phis}: ' + ' '.join(str(Z_train.shape) for Z_train in Zs_train))
        print("Number of training samples:", n_tr)

    # Define task encoder
    #task_encoder = [nn.utils.weight_norm(nn.Linear(d, C)).to(args.device) for d in feature_dims]
    task_encoder = [nn.utils.parametrizations.weight_norm(nn.Linear(d, C)).to(args.device) for d in feature_dims]

    def task_encoding(Zs):
        assert len(Zs) == len(task_encoder)
        # Generate labeling by the average of $\sigmoid(\theta \phi(x))$, Eq. (9) in the paper
        label_per_space = [F.softmax(task_phi(z), dim=-1) for task_phi, z in zip(task_encoder, Zs)] # shape of (K, N, C)
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

    # start training
    num_spaces = len(args.phis)
    phis = '_'.join(args.phis)
    exp_path = f"{args.root_dir}/task_checkpoints/{num_spaces}space/{phis}/{args.dataset}"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    best_acc = 0.0

    if args.verbose:
        iters_bar = tqdm(range(args.T))
    else:
        iters_bar = range(args.T)
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
        entr_reg = sum([torch.special.entr(l.mean(0)).sum() for l in label_per_space])

        # final loss, Eq. (12) in the paper
        (pred_error - args.gamma * entr_reg).backward()
        optimizer.step()

        # evaluation, compute clustering accuracy on test split
        if (i+1) % 20 == 0 or (i+1) == args.T:
            labels_val, _ = task_encoding([torch.from_numpy(Z_val).to(args.device) for Z_val in Zs_val])
            preds_val = labels_val.argmax(dim=1).detach().cpu().numpy()
            cluster_acc, _ = get_cluster_acc(preds_val, y_gt_val)

            if cluster_acc > best_acc and len(np.unique(preds_val)) == C:
                if args.verbose:
                    print("best acc: "+str(cluster_acc)+" with clusters  "+str(len(np.unique(preds_val))))
                best_acc = cluster_acc
                torch.save({f'phi{i+1}': task_phi.state_dict() for i, task_phi in enumerate(task_encoder)}, f'{exp_path}/turtle_model.pt')
            if args.verbose:
                iters_bar.set_description(f'Training loss {float(pred_error.detach()):.3f}, entropy {float(entr_reg.detach()):.3f}, found clusters {len(np.unique(preds_val))}/{C}, cluster acc {cluster_acc:.4f}')

    if args.verbose:
        print(f'Training finished! ')
        print(f'Training loss {float(pred_error.detach()):.3f}, entropy {float(entr_reg.detach()):.3f}, Number of found clusters {len(np.unique(preds_val))}/{C}, Cluster Acc {cluster_acc:.4f}')

    # save results
    inner_start = 'warmstart' if args.warm_start else 'coldstart'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    for task_phi in task_encoder:
        #nn.utils.remove_weight_norm(task_phi)
        nn.utils.parametrize.remove_parametrizations(task_phi, "weight")

    return best_acc

    #task_path = f"turtle_{phis}_innerlr{args.inner_lr}_outerlr{args.outer_lr}_T{args.T}_M{args.M}_{inner_start}_gamma{args.gamma}_bs{args.batch_size}_seed{args.seed}"
    #torch.save({f'phi{i+1}': task_phi.state_dict() for i, task_phi in enumerate(task_encoder)}, f'{exp_path}/model.pt')


if __name__ == '__main__':
    val = run()
    print(val)
