#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

def load_labels(path, skip_header):
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1 if skip_header else 0)
    except ValueError as e:
        # If there's an error, try again with skiprows=1 (assuming there's a header)
        if not skip_header:
            print(f"Warning: Error loading {path} without header. Trying with header...")
            data = np.loadtxt(path, delimiter=",", skiprows=1)
        else:
            raise e
    return data.astype(int) if data.ndim==1 else data[:,-1].astype(int)

def load_centers(path, skip_header):
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1 if skip_header else 0)
    except ValueError as e:
        # If there's an error, try again with skiprows=1 (assuming there's a header)
        if not skip_header:
            print(f"Warning: Error loading {path} without header. Trying with header...")
            data = np.loadtxt(path, delimiter=",", skiprows=1)
        else:
            raise e
    return data

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

    print("shape y-true:", y_true.shape)
    print("shape y-pred:", y_pred.shape)

    print("shape C-true:", C_true.shape)
    print("shape C-pred:", C_pred.shape)

    # check labels
    ari = adjusted_rand_score(y_true, y_pred)
    ari_check = ari >= 0.99
    print(f"ARI check: {ari_check}, value = {ari:.6f}, threshold = 0.99")
    
    # check centers
    avg_dist = avg_center_dist(C_true, C_pred)
    dist_check = avg_dist <= args.tol
    print(f"Center distance check: {dist_check}, value = {avg_dist:.6e}, threshold = {args.tol}")
    
    # Final result
    if ari_check and dist_check:
        print("Correctness Check: PASS")
        sys.exit(0)
    else:
        print("Correctness Check: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()