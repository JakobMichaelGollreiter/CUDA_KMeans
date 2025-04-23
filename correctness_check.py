#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

def load_labels(path, skip_header):
    data = np.loadtxt(path, delimiter=",", skiprows=1 if skip_header else 0)
    return data.astype(int) if data.ndim==1 else data[:,-1].astype(int)

def load_centers(path, skip_header):
    return np.loadtxt(path, delimiter=",", skiprows=1 if skip_header else 0)

def avg_center_dist(C_true, C_pred):
    dists = np.linalg.norm(C_true[:,None,:] - C_pred[None,:,:], axis=2)
    row, col = linear_sum_assignment(dists)
    return dists[row, col].mean()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("gt_labels")
    p.add_argument("pred_labels")
    p.add_argument("gt_centers")
    p.add_argument("pred_centers")
    p.add_argument("--header", action="store_true", help="skip CSV header")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="max average center L2 distance allowed")
    args = p.parse_args()

    # load
    y_true = load_labels(args.gt_labels, args.header)
    y_pred = load_labels(args.pred_labels, args.header)
    C_true = load_centers(args.gt_centers, args.header)
    C_pred = load_centers(args.pred_centers, args.header)

    # check labels
    ari = adjusted_rand_score(y_true, y_pred)
    if ari < 1.0:
        print(f"FAIL: ARI = {ari:.6f} < 1.0")
        sys.exit(1)

    # check centers
    avg_dist = avg_center_dist(C_true, C_pred)
    if avg_dist > args.tol:
        print(f"FAIL: avg center L2 = {avg_dist:.6e} > tol {args.tol}")
        sys.exit(1)

    print("PASS")
    sys.exit(0)

if __name__ == "__main__":
    main()
