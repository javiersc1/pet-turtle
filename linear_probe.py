import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset for Linear Probe evaluation", required=True)
    parser.add_argument('--phis', type=str, default="clipvitL14", help="Representation spaces to run Linear Probe", 
                        choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 'clipvitB32', 'clipvitB16', 'clipvitL14', 'dinov2'])
    parser.add_argument('--root_dir', type=str, default="data", help='Root dir to store everything')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args(args)

def run(args=None):
    args = _parse_args(args)

    Ztrain = np.load(f"{args.root_dir}/representations/{args.phis}/{args.dataset}_train.npy").astype(np.float32)
    Zval = np.load(f"{args.root_dir}/representations/{args.phis}/{args.dataset}_val.npy").astype(np.float32)
    ytrain = np.load(f"{args.root_dir}/labels/{args.dataset}_train.npy")
    yval = np.load(f"{args.root_dir}/labels/{args.dataset}_val.npy")

    ytrain = ytrain.flatten()
    yval = yval.flatten()
    if args.verbose:
        print(Ztrain.shape, ytrain.shape)
        print(len(np.unique(ytrain)))
    # Use given C=1. (sklearn: 'C' is the inverse of regularization strength)
    flag = 1 if args.verbose else 0
    clf = LogisticRegression(
            verbose=flag, 
            tol=1e-4, 
            C=args.C, 
            max_iter=1000,
            solver='lbfgs', 
            random_state=args.seed,
            n_jobs=8,
            #class_weight='balanced'
        )
    clf.fit(Ztrain, ytrain)
    train_acc = accuracy_score(ytrain, clf.predict(Ztrain))
    val_acc = accuracy_score(yval, clf.predict(Zval))

    if args.verbose:
        print(f"Train Accuracy: {train_acc * 100:.2f}")
        print(f"Val Accuracy: {val_acc * 100:.2f}")

    return val_acc

if __name__ == '__main__':
    val = run()
    print(val)