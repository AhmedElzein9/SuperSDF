"""Stratified random train/validation/test split used in the SuperSDF paper.

This is a Python port of ``random_split_data.m``.  Sampling is performed
independently within each labeled class and has NO spatial constraint.

For every class c with n_c labeled pixels:
    n_train = max(5, round_MATLAB(train_ratio * n_c))
    n_val   = n_train
    n_test  = n_c - n_train - n_val

The small-class safety checks follow the MATLAB implementation exactly.
Indices returned by this function are zero-based indices into the vector of
labeled pixels (``gt > 0``), not zero-based full-image linear indices.
"""

from __future__ import annotations

import numpy as np


def _matlab_round_positive(x: float) -> int:
    """MATLAB-style round for non-negative values (0.5 rounds upward)."""
    return int(np.floor(float(x) + 0.5))


def random_split_data(
    features: np.ndarray,
    gt: np.ndarray,
    train_ratio: float,
    min_train_per_class: int = 5,
    seed: int | None = None,
    verbose: bool = False,
):
    """Create the class-wise random split used for SuperSDF.

    Parameters
    ----------
    features : ndarray, shape (rows, cols, n_features)
        Feature cube.  In the paper this is the fixed SuperSDF feature cube.
    gt : ndarray, shape (rows, cols)
        Ground-truth labels.  Label 0 is treated as background/unlabeled.
    train_ratio : float
        Per-class training fraction (0.05 Indian, 0.02 Pavia, 0.01 HanChuan).
    min_train_per_class : int, default=5
        Minimum number of training samples per class.  The validation subset
        initially uses the same number of samples as the training subset.
    seed : int or None
        Seed for the class-wise random permutations.
    verbose : bool
        Print per-class sample counts.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test,
    sample_map, train_indices, val_indices, test_indices, X_all_raw

    Notes
    -----
    NumPy and MATLAB use different random-number/permutation implementations,
    so the same numeric seed does not guarantee pixel-identical splits across
    languages.  The sampling protocol and seed schedule are the same.
    """
    if features.ndim != 3:
        raise ValueError("features must have shape (rows, cols, n_features)")
    if gt.ndim != 2:
        raise ValueError("gt must have shape (rows, cols)")
    if features.shape[:2] != gt.shape:
        raise ValueError("features and gt must have the same spatial size")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    rows, cols, n_feat = features.shape

    # order='F' follows MATLAB's column-major reshape/linear indexing.
    X_all_raw = features.reshape(rows * cols, n_feat, order="F")
    gt_flat = gt.reshape(-1, order="F")

    valid_mask = gt_flat > 0
    valid_indices = np.flatnonzero(valid_mask)
    valid_gt = gt_flat[valid_mask].astype(int, copy=False)
    valid_feat = X_all_raw[valid_mask, :]

    classes = np.unique(valid_gt)
    if classes.size == 0:
        raise ValueError("gt contains no labeled pixels (labels > 0)")
    rng = np.random.RandomState(seed)  # MT19937, closest NumPy analogue to MATLAB twister

    tr_parts: list[np.ndarray] = []
    vl_parts: list[np.ndarray] = []
    te_parts: list[np.ndarray] = []

    if verbose:
        print("\n" + "=" * 57)
        print(" random_split_data - stratified per-class random split")
        print("=" * 57)
        print(f" train ratio         : {100 * train_ratio:.1f}%")
        print(" spatial constraint  : none")
        print(f" minimum train/class : {min_train_per_class}")
        print("-" * 57)
        print(f"{'Class':<8}{'Total':<9}{'Ratio-N':<10}{'Train':<9}{'Val':<9}{'Test':<9}")

    for cls in classes:
        c_idx = np.flatnonzero(valid_gt == cls)
        n_tot = int(c_idx.size)

        n_ratio = _matlab_round_positive(train_ratio * n_tot)
        n_train = max(int(min_train_per_class), n_ratio)
        n_val = n_train

        # Same safety logic as random_split_data.m for very small classes.
        while (n_train + n_val) >= n_tot and n_val > 0:
            n_val -= 1
        while (n_train + n_val) >= n_tot and n_train > 1:
            n_train -= 1

        perm = rng.permutation(c_idx)
        tr = perm[:n_train]
        vl = perm[n_train:n_train + n_val]
        te = perm[n_train + n_val:]

        tr_parts.append(tr.astype(np.int64, copy=False))
        vl_parts.append(vl.astype(np.int64, copy=False))
        te_parts.append(te.astype(np.int64, copy=False))

        if verbose:
            print(f"{int(cls):<8}{n_tot:<9}{n_ratio:<10}{len(tr):<9}{len(vl):<9}{len(te):<9}")

    train_indices = np.concatenate(tr_parts)
    val_indices = np.concatenate(vl_parts)
    test_indices = np.concatenate(te_parts)

    # MATLAB shuffles the three aggregate index vectors after concatenation.
    train_indices = rng.permutation(train_indices)
    val_indices = rng.permutation(val_indices)
    test_indices = rng.permutation(test_indices)

    X_train = valid_feat[train_indices, :]
    y_train = valid_gt[train_indices]
    X_val = valid_feat[val_indices, :]
    y_val = valid_gt[val_indices]
    X_test = valid_feat[test_indices, :]
    y_test = valid_gt[test_indices]

    sample_flat = np.zeros(rows * cols, dtype=np.uint8)
    sample_flat[valid_indices[test_indices]] = 3
    sample_flat[valid_indices[val_indices]] = 2
    sample_flat[valid_indices[train_indices]] = 1
    sample_map = sample_flat.reshape(rows, cols, order="F")

    if verbose:
        print("-" * 57)
        print(
            f"Training: {len(y_train)} | Validation: {len(y_val)} | "
            f"Test: {len(y_test)}"
        )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        sample_map,
        train_indices,
        val_indices,
        test_indices,
        X_all_raw,
    )
