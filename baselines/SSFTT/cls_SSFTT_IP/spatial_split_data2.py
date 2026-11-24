# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 19:14:58 2026

@author: ahmed.e
"""

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def _pick_spatial(pool_idx, row_c, col_c, N, min_dist, existing_coords):
    if N <= 0:
        return np.array([], dtype=int), np.zeros((0, 2))

    pool_idx = np.array(pool_idx).flatten()
    pool_idx = pool_idx[np.random.permutation(len(pool_idx))]
    cand_rc  = np.column_stack([row_c[pool_idx], col_c[pool_idx]])

    occ = np.zeros((0, 2)) if (existing_coords is None or len(existing_coords) == 0) \
          else np.array(existing_coords, dtype=float)

    sel = []

    # Pass 1: strict distance
    for ii in range(len(pool_idx)):
        if len(sel) >= N:
            break
        cr, cc = cand_rc[ii]
        if len(occ) == 0:
            ok = True
        else:
            dv = occ - [cr, cc]
            ok = bool(np.all(dv[:, 0]**2 + dv[:, 1]**2 >= min_dist**2))
        if ok:
            sel.append(pool_idx[ii])
            occ = np.vstack([occ, [cr, cc]])

    # Pass 2: relaxed 70%
    if len(sel) < N:
        relaxed = max(1.0, min_dist * 0.7)
        used    = set(sel)
        mask    = np.array([p not in used for p in pool_idx])
        rem_idx = pool_idx[mask]
        rem_rc  = cand_rc[mask]
        for ii in range(len(rem_idx)):
            if len(sel) >= N:
                break
            cr, cc = rem_rc[ii]
            if len(occ) == 0:
                ok = True
            else:
                dv = occ - [cr, cc]
                ok = bool(np.all(dv[:, 0]**2 + dv[:, 1]**2 >= relaxed**2))
            if ok:
                sel.append(rem_idx[ii])
                occ = np.vstack([occ, [cr, cc]])

    # Pass 3: no constraint (fallback for tiny classes)
    if len(sel) < N:
        used     = set(sel)
        rem_pool = pool_idx[np.array([p not in used for p in pool_idx])]
        need     = N - len(sel)
        extra    = rem_pool[:min(need, len(rem_pool))]
        sel.extend(extra.tolist())

    sel_idx    = np.array(sel[:N], dtype=int)
    n_pre      = 0 if (existing_coords is None or len(existing_coords) == 0) \
                 else len(existing_coords)
    new_coords = occ[n_pre:] if len(occ) > n_pre else np.zeros((0, 2))
    return sel_idx, new_coords


# def _check_cross_distance(v_row, v_col, train_indices, test_indices, required):
#     if len(train_indices) == 0 or len(test_indices) == 0:
#         return
#     MAX    = 800
#     tr_sub = train_indices[np.random.choice(len(train_indices),
#                            min(MAX, len(train_indices)), replace=False)]
#     te_sub = test_indices[np.random.choice(len(test_indices),
#                           min(MAX, len(test_indices)), replace=False)]

    # tr_coords = np.column_stack([v_row[tr_sub], v_col[tr_sub]]).astype(float)
    # te_coords = np.column_stack([v_row[te_sub], v_col[te_sub]]).astype(float)
    # D         = cdist(tr_coords, te_coords)
    # min_d     = D.min()
    # status    = 'PASS' if min_d >= required else 'WARNING'
    # print(f'Cross-set check (sampled): min(train-test) = {min_d:.2f} px  [{status}]')


def _plot_splits(sample_map, gt, n_tr, n_vl, n_te):
    rows, cols = gt.shape
    n_cls = int(gt.max())
    cmap  = plt.cm.jet(np.linspace(0, 1, max(n_cls, 1)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, title, sid, count in zip(axes,
                                     ['Training', 'Validation', 'Test'],
                                     [1, 2, 3],
                                     [n_tr, n_vl, n_te]):
        rgb = np.zeros((rows, cols, 3))
        for k in range(n_cls):
            msk = (sample_map == sid) & (gt == k + 1)
            rgb[msk] = cmap[k, :3]
        if sid == 3:
            rgb[sample_map == 4] = [0.5, 0.5, 0.5]
        ax.imshow(rgb)
        ax.set_title(f'{title}  (n={count})', fontsize=10, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def spatial_split_data2(sdf_images, gt, train_ratio,
                        min_spatial_distance=5,
                        test_exclusion_radius=None,
                        min_train_per_class=5,
                        min_test_per_class=10,
                        visualize=True):
    """
    Spatially-constrained HSI train / val / test split.

    Train and val samples are spatially separated from each other (and
    within themselves) by min_spatial_distance. All remaining pixels become
    test; no test-exclusion filtering is applied (pool_te is NOT filtered),
    matching standard published HSI practice.

    Parameters
    ----------
    sdf_images : ndarray (rows, cols, features)
    gt         : ndarray (rows, cols), integer labels, 0 = background
    train_ratio: float  – fraction of labeled pixels per class for train AND val
    min_spatial_distance : int  – px separation enforced within train and val
    test_exclusion_radius: int  – diagnostic only (reported, not applied)
    min_train_per_class  : int  – floor on per-class training count
    min_test_per_class   : int  – kept for API compatibility, not actively used
    visualize            : bool – show sample-map figure

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test,
    sample_map, train_indices, val_indices, test_indices, X_all_raw
    """
    if test_exclusion_radius is None:
        test_exclusion_radius = min_spatial_distance

    print('\n=========================================================')
    print(' spatial_split_data2  -  spatially-constrained HSI split')
    print('=========================================================')
    print(f' train_ratio          = {train_ratio*100:.1f}%')
    print(f' min_spatial_distance = {min_spatial_distance} px')
    print(f' test_exclusion_radius= {test_exclusion_radius} px  (diagnostic only)')
    print(f' min_train_per_class  = {min_train_per_class}')
    print('---------------------------------------------------------')

    rows, cols, n_feat = sdf_images.shape
    X_all_raw = sdf_images.reshape(rows * cols, n_feat)

    col_coords, row_coords = np.meshgrid(np.arange(cols), np.arange(rows))
    row_coords = row_coords.flatten()
    col_coords = col_coords.flatten()

    valid_mask       = gt.flatten() > 0
    valid_indices    = np.where(valid_mask)[0]
    valid_gt         = gt.flatten()[valid_mask]
    valid_features   = X_all_raw[valid_mask]
    valid_row        = row_coords[valid_mask]
    valid_col        = col_coords[valid_mask]

    unique_classes = np.unique(valid_gt)

    TR_feat, TR_lab, TR_idx = [], [], []
    VL_feat, VL_lab, VL_idx = [], [], []
    TE_feat, TE_lab, TE_idx = [], [], []

    hdr = f'{"Class":<6} {"Total":<7} {"Ratio-N":<8} {"Floor?":<7} ' \
          f'{"Train":<6} {"Val":<6} {"Test":<8}'
    print(hdr)
    print('-' * 62)

    for cls in unique_classes:
        cmask = valid_gt == cls
        c_idx = np.where(cmask)[0]
        c_row = valid_row[cmask]
        c_col = valid_col[cmask]
        n_tot = len(c_idx)

        n_train_ratio = round(train_ratio * n_tot)
        n_train       = max(min_train_per_class, n_train_ratio)
        floor_hit     = n_train > n_train_ratio

        n_val = max(min_train_per_class, round(train_ratio * n_tot))
        n_val = min(n_val, n_train)

        while (n_train + n_val) >= n_tot and n_val > 0:
            n_val -= 1
        while (n_train + n_val) >= n_tot and n_train > 1:
            n_train -= 1
        if n_train >= n_tot:
            n_train = max(1, n_tot - 1)
            n_val   = 0

        pool_all = np.arange(n_tot)

        tr_local, tr_coords = _pick_spatial(
            pool_all, c_row, c_col, n_train, min_spatial_distance, None)

        pool_no_train = np.setdiff1d(pool_all, tr_local)
        vl_local, _   = _pick_spatial(
            pool_no_train, c_row, c_col, n_val, min_spatial_distance, tr_coords)

        pool_te = np.setdiff1d(pool_all, np.concatenate([tr_local, vl_local]))

        # Diagnostic only – never filters pool_te
        excl_applied = False
        if test_exclusion_radius > 0 and len(pool_te) > 0:
            parts = []
            if len(tr_local) > 0:
                parts.append(np.column_stack([c_row[tr_local], c_col[tr_local]]))
            if len(vl_local) > 0:
                parts.append(np.column_stack([c_row[vl_local], c_col[vl_local]]))
            if parts:
                tv_coords  = np.vstack(parts)
                cand_coords = np.column_stack([c_row[pool_te], c_col[pool_te]])
                D_mat      = cdist(cand_coords.astype(float), tv_coords.astype(float))
                min_D      = D_mat.min(axis=1)
                excl_applied = bool(np.sum(min_D < test_exclusion_radius) == 0)

        vf = valid_features[cmask]

        TR_idx.append(c_idx[tr_local]);  TR_feat.append(vf[tr_local])
        TR_lab.append(np.full(len(tr_local), cls, dtype=valid_gt.dtype))

        VL_idx.append(c_idx[vl_local]);  VL_feat.append(vf[vl_local])
        VL_lab.append(np.full(len(vl_local), cls, dtype=valid_gt.dtype))

        TE_idx.append(c_idx[pool_te]);   TE_feat.append(vf[pool_te])
        TE_lab.append(np.full(len(pool_te),  cls, dtype=valid_gt.dtype))

        print(f'{int(cls):<6} {n_tot:<7} {n_train_ratio:<8} '
              f'{"YES" if floor_hit else "no":<7} '
              f'{len(tr_local):<6} {len(vl_local):<6} {len(pool_te):<8} '
              )

    train_indices = np.concatenate(TR_idx)
    val_indices   = np.concatenate(VL_idx)
    test_indices  = np.concatenate(TE_idx)

    def _shuffle(feats, labs):
            X  = np.vstack(feats)          # full feature matrix  (N_total × features)
            y  = np.concatenate(labs)      # full label vector    (N_total,)
            sh = np.random.permutation(len(y))   # permute by total samples, not class count
            return X[sh], y[sh]

    X_train, y_train = _shuffle(TR_feat, TR_lab)
    X_val,   y_val   = _shuffle(VL_feat, VL_lab)
    X_test,  y_test  = _shuffle(TE_feat, TE_lab)

    print('-' * 62)
    print(f'Training: {len(y_train)}  |  Validation: {len(y_val)}  |  Test: {len(y_test)}')

    # if test_exclusion_radius > 0:
    #     _check_cross_distance(valid_row, valid_col,
    #                           train_indices, test_indices, test_exclusion_radius)

    # Build sample_map
    sample_map = np.zeros((rows, cols), dtype=int)
    for idx in test_indices:
        r, c = np.unravel_index(valid_indices[idx], (rows, cols))
        sample_map[r, c] = 3
    for idx in val_indices:
        r, c = np.unravel_index(valid_indices[idx], (rows, cols))
        sample_map[r, c] = 2
    for idx in train_indices:
        r, c = np.unravel_index(valid_indices[idx], (rows, cols))
        sample_map[r, c] = 1
    sample_map[(gt > 0) & (sample_map == 0)] = 4

    if visualize:
        _plot_splits(sample_map, gt, len(y_train), len(y_val), len(y_test))

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            sample_map, train_indices, val_indices, test_indices, X_all_raw)