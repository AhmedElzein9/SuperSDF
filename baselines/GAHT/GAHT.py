"""
GAHT_SDF.py — GAHT (Mei et al., TGRS 2022) under the SuperSDF evaluation protocol.

Place in the GAHT repo root (same level as train.py, models/, utils/).
Uses the same split, seeds, and run count as the SuperSDF experiments.

NOTE: utils/dataset.py sets undefined pixels to -1 and class indices start
at 0. All split logic below follows that convention.
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from utils.dataset import load_mat_hsi, HSIDataset
from utils.utils import metrics
from utils.scheduler import load_scheduler
from models.get_model import get_model
from train import train, test


# =========================================================================
# Split functions — match the MATLAB protocol (gt: -1 = background)
# =========================================================================
def random_split_gt(gt, train_ratio, min_train_per_class=5):
    """Stratified random split. Port of random_split_data.m"""
    train_gt = np.full_like(gt, -1)
    val_gt   = np.full_like(gt, -1)
    test_gt  = np.full_like(gt, -1)

    for cls in np.unique(gt):
        if cls < 0:
            continue
        idx   = np.argwhere(gt == cls)
        n_tot = len(idx)

        n_train = max(min_train_per_class, int(round(train_ratio * n_tot)))
        n_val   = n_train
        while (n_train + n_val) >= n_tot and n_val > 0:
            n_val -= 1
        while (n_train + n_val) >= n_tot and n_train > 1:
            n_train -= 1

        perm = np.random.permutation(n_tot)
        for i in perm[:n_train]:
            train_gt[idx[i][0], idx[i][1]] = cls
        for i in perm[n_train:n_train + n_val]:
            val_gt[idx[i][0], idx[i][1]] = cls
        for i in perm[n_train + n_val:]:
            test_gt[idx[i][0], idx[i][1]] = cls

    return train_gt, val_gt, test_gt


def spatial_split_gt(gt, train_ratio, min_dist=5, min_train_per_class=5):
    """Spatially-constrained split. Port of spatial_split_data2.m"""

    def pick_spatial(coords, n_want, min_d, existing):
        if n_want <= 0:
            return []
        order = np.random.permutation(len(coords))
        occ   = list(existing)
        sel   = []
        for thresh in (min_d, max(1.0, min_d * 0.7)):   # strict, then relaxed
            for ii in order:
                if len(sel) >= n_want:
                    break
                if ii in sel:
                    continue
                cr, cc = coords[ii]
                if occ:
                    o  = np.array(occ)
                    d2 = (o[:, 0] - cr) ** 2 + (o[:, 1] - cc) ** 2
                    ok = bool(np.all(d2 >= thresh ** 2))
                else:
                    ok = True
                if ok:
                    sel.append(ii)
                    occ.append((cr, cc))
            if len(sel) >= n_want:
                break
        for ii in order:                                 # unconstrained fallback
            if len(sel) >= n_want:
                break
            if ii not in sel:
                sel.append(ii)
        return sel[:n_want]

    train_gt = np.full_like(gt, -1)
    val_gt   = np.full_like(gt, -1)
    test_gt  = np.full_like(gt, -1)

    for cls in np.unique(gt):
        if cls < 0:
            continue
        idx    = np.argwhere(gt == cls)
        coords = [tuple(p) for p in idx]
        n_tot  = len(idx)

        n_train = max(min_train_per_class, int(round(train_ratio * n_tot)))
        n_val   = n_train
        while (n_train + n_val) >= n_tot and n_val > 0:
            n_val -= 1
        while (n_train + n_val) >= n_tot and n_train > 1:
            n_train -= 1

        tr_sel = pick_spatial(coords, n_train, min_dist, [])
        tr_pts = [coords[i] for i in tr_sel]

        remaining  = [i for i in range(n_tot) if i not in tr_sel]
        rem_coords = [coords[i] for i in remaining]
        vl_sel = [remaining[i] for i in
                  pick_spatial(rem_coords, n_val, min_dist, tr_pts)]

        te_sel = [i for i in range(n_tot)
                  if i not in tr_sel and i not in vl_sel]

        for i in tr_sel:
            train_gt[idx[i][0], idx[i][1]] = cls
        for i in vl_sel:
            val_gt[idx[i][0], idx[i][1]] = cls
        for i in te_sel:
            test_gt[idx[i][0], idx[i][1]] = cls

    return train_gt, val_gt, test_gt


# =========================================================================
# Main — GAHT only
# =========================================================================
if __name__ == "__main__":

    MODEL = 'ssftt'          # GAHT (the paper's own method) 'ssftt' proposed

    parser = argparse.ArgumentParser(
        description="GAHT under the SuperSDF evaluation protocol")
    parser.add_argument("--dataset_dir",  type=str,
        default="../../data")
    parser.add_argument("--dataset_name", type=str,   default="ip")
    parser.add_argument("--device",       type=str,   default="0")
    parser.add_argument("--patch_size",   type=int,   default=7)
    parser.add_argument("--num_run",      type=int,   default=1)
    parser.add_argument("--epoch",        type=int,   default=300)
    parser.add_argument("--bs",           type=int,   default=64)
    parser.add_argument("--ratio",        type=float, default=0.05)
    parser.add_argument("--protocol",     type=str,   default="random",
                        help="random | spatial")
    parser.add_argument("--min_dist",     type=int,   default=5)
    opts = parser.parse_args([])        # [] = Spyder-safe; use parse_args() in a terminal

    device = torch.device(f"cuda:{opts.device}"
                          if torch.cuda.is_available() else "cpu")

    print("=== GAHT — SuperSDF protocol ===")
    print(f"dataset    = {opts.dataset_name}")
    print(f"protocol   = {opts.protocol}")
    print(f"ratio      = {opts.ratio}  (train and val each, per class)")
    print(f"runs       = {opts.num_run}   epochs = {opts.epoch}   bs = {opts.bs}")
    print(f"device     = {device}\n")

    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)
    num_classes = len(labels)

    results, run_times = [], []

    for run in range(opts.num_run):
        seed = (run + 1) * 7 + 13          # same seed family as SuperSDF
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)

        print(f"\n--- run {run+1}/{opts.num_run}  (seed {seed}) ---")
        t0 = time.time()

        if opts.protocol == "random":
            train_gt, val_gt, test_gt = random_split_gt(gt, opts.ratio)
        elif opts.protocol == "spatial":
            train_gt, val_gt, test_gt = spatial_split_gt(
                gt, opts.ratio, min_dist=opts.min_dist)
        else:
            raise ValueError("protocol must be 'random' or 'spatial'")

        print(f"train={np.sum(train_gt >= 0)}  "
              f"val={np.sum(val_gt >= 0)}  "
              f"test={np.sum(test_gt >= 0)}")

        train_set = HSIDataset(image, train_gt,
                               patch_size=opts.patch_size, data_aug=True)
        val_set   = HSIDataset(image, val_gt,
                               patch_size=opts.patch_size, data_aug=False)

        train_loader = torch.utils.data.DataLoader(
            train_set, opts.bs, drop_last=False, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(
            val_set, opts.bs, drop_last=False, shuffle=False, num_workers=0)

        model = get_model(MODEL, opts.dataset_name, opts.patch_size).to(device)
        optimizer, scheduler = load_scheduler(MODEL, model)
        criterion = nn.CrossEntropyLoss()

        model_dir = (f"./checkpoints/{MODEL}/"
                     f"{opts.dataset_name}_{opts.protocol}/{run}")

        try:
            train(model, optimizer, criterion, train_loader, val_loader,
                  opts.epoch, model_dir, device, scheduler)
        except KeyboardInterrupt:
            print('training interrupted')

        probabilities = test(model, model_dir, image,
                             opts.patch_size, num_classes, device)
        prediction = np.argmax(probabilities, axis=-1)
        if run == 0:
            saved_prediction = prediction.copy()
        run_results = metrics(prediction, test_gt, n_classes=num_classes)
        results.append(run_results)
        # --- per-class accuracy for this run (diagonal / row sum) ---
        cm = run_results["Confusion matrix"]
        ca = np.diag(cm) / np.maximum(cm.sum(axis=1), 1) * 100
        if run == 0:
            CA_mc = np.zeros((opts.num_run, len(ca)))
        CA_mc[run, :] = ca
        run_times.append(time.time() - t0)

        print(f"OA={run_results['Accuracy']:.2f}%  "
              f"kappa={run_results['Kappa']:.4f}  "
              f"({run_times[-1]:.1f}s)")

        del model, train_set, train_loader, val_set, val_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()

# ---- summary ---------------------------------------------------------
    OA  = np.array([r["Accuracy"] for r in results])
    KAP = np.array([r["Kappa"] * 100 for r in results])
    AA  = CA_mc.mean(axis=1)          # AA per run = mean of that run's class accs

    print(f"\n{'='*68}")
    print(f"  GAHT — {opts.dataset_name.upper()}  "
          f"({opts.num_run} runs, {opts.protocol} protocol, "
          f"ratio={opts.ratio})")
    print(f"{'='*68}")
    print(f"{'Class':<6}  {'Name':<28}  {'Accuracy % (mean ± std)'}")
    print('-' * 68)
    for c in range(CA_mc.shape[1]):
        name = labels[c] if c < len(labels) else f'Class {c+1}'
        print(f"{c+1:<6}  {name:<28}  "
              f"{CA_mc[:, c].mean():6.2f} ± {CA_mc[:, c].std():5.2f}")
    print('-' * 68)
    print(f"{'OA(%)':<6}  {'':<28}  {OA.mean():6.2f} ± {OA.std():5.2f}")
    print(f"{'AA(%)':<6}  {'':<28}  {AA.mean():6.2f} ± {AA.std():5.2f}")
    print(f"{'κ(%)':<6}  {'':<28}  {KAP.mean():6.2f} ± {KAP.std():5.2f}")
    print('-' * 68)
    print(f"{'Time':<6}  {'':<28}  "
          f"{np.mean(run_times):6.1f} ± {np.std(run_times):5.1f} s/run  "
          f"(total {np.sum(run_times):.1f}s)")
    print(f"{'Device':<6}  {'':<28}  {device}")
    print('=' * 68)

    # ---- optional: save for building the paper table ---------------------
    np.savez(f"results_GAHT_{opts.dataset_name}_{opts.protocol}.npz",
             OA=OA, AA=AA, KAP=KAP, CA=CA_mc,
             times=np.array(run_times), labels=np.array(labels, dtype=object))
    print(f"\nSaved: results_GAHT_{opts.dataset_name}_{opts.protocol}.npz")
    
    # =====================================================================
    # CLASSIFICATION MAP
    # =====================================================================
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # prediction is 0-indexed over the whole image; shift to 1-indexed
    # and set background (gt < 0) to 0 so it renders black
    class_map = saved_prediction.astype(int) + 1
    class_map[gt < 0] = 0

    gt_map = gt.astype(int) + 1          # -1 background -> 0
    n_cls  = int(gt_map.max())

    # colors   = np.vstack([[0, 0, 0],
    #                       plt.cm.jet(np.linspace(0, 1, n_cls))[:, :3]])
    # cmap_cls = mcolors.ListedColormap(colors)
# Fixed class colors
    # class_colors = np.array([
    #     [0.00, 0.00, 0.75],   # class 1
    #     [0.00, 0.00, 1.00],   # class 2
    #     [0.00, 0.25, 1.00],   # class 3
    #     [0.00, 0.50, 1.00],   # class 4
    #     [0.00, 0.75, 1.00],   # class 5
    #     [0.00, 1.00, 1.00],   # class 6
    #     [0.25, 1.00, 0.75],   # class 7
    #     [0.50, 1.00, 0.50],   # class 8
    #     [0.75, 1.00, 0.25],   # class 9
    #     [1.00, 1.00, 0.00],   # class 10
    #     [1.00, 0.75, 0.00],   # class 11
    #     [1.00, 0.50, 0.00],   # class 12
    #     [1.00, 0.25, 0.00],   # class 13
    #     [1.00, 0.00, 0.00],   # class 14
    #     [0.75, 0.00, 0.00],   # class 15
    #     [0.50, 0.00, 0.00],   # class 16
    # ], dtype=float)
    
    class_colors = np.array([
    [0.0,       0.0,       1.0      ],  # class 1
    [0.0,       1.0/3.0,   1.0      ],  # class 2
    [0.0,       2.0/3.0,   1.0      ],  # class 3
    [0.0,       1.0,       1.0      ],  # class 4
    [1.0/3.0,   1.0,       2.0/3.0  ],  # class 5
    [2.0/3.0,   1.0,       1.0/3.0  ],  # class 6
    [1.0,       1.0,       0.0      ],  # class 7
    [1.0,       2.0/3.0,   0.0      ],  # class 8
    [1.0,       1.0/3.0,   0.0      ],  # class 9
], dtype=float)
    
    if n_cls > len(class_colors):
        raise ValueError(
            f"Dataset has {n_cls} classes, but only "
            f"{len(class_colors)} class colors are defined."
        )
    
    # 0 = black background
    # 1 = class_colors[0]
    # 2 = class_colors[1]
    # ...
    colors = np.vstack([
        [0.0, 0.0, 0.0],
        class_colors[:n_cls]
    ])
    
    cmap_cls = mcolors.ListedColormap(colors)
    
    # Force exact integer class-to-color mapping
    norm_cls = mcolors.BoundaryNorm(
        np.arange(-0.5, n_cls + 1.5, 1),
        cmap_cls.N
    )
    tag = f"{opts.dataset_name}_{opts.protocol}"

    # --- ground truth ---
    fig_gt, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(gt_map, cmap=cmap_cls, vmin=0, vmax=n_cls)
    ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax.axis('off')
    fig_gt.tight_layout()
    fig_gt.savefig(f'GT_{tag}.png', dpi=300, bbox_inches='tight')

    # --- classification map ---
    fig_cm, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(class_map, cmap=cmap_cls, vmin=0, vmax=n_cls)
    ax.set_title(f'GAHT — {opts.dataset_name.upper()}',
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    fig_cm.tight_layout()
    fig_cm.savefig(f'ClassMap_GAHT_{tag}.png', dpi=300, bbox_inches='tight')

    # --- side-by-side with legend ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(gt_map,   cmap=cmap_cls, vmin=0, vmax=n_cls)
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].imshow(class_map, cmap=cmap_cls, vmin=0, vmax=n_cls)
    axes[1].set_title(f'GAHT  (OA={OA.mean():.2f}%)',
                      fontsize=12, fontweight='bold')
    for a in axes:
        a.axis('off')

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[c + 1])
               for c in range(n_cls)]
    names = [labels[c] if c < len(labels) else f'Class {c+1}'
             for c in range(n_cls)]
    fig.legend(handles, names, loc='center left',
               bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f'Compare_GAHT_{tag}.png', dpi=300, bbox_inches='tight')

    plt.show()
    print(f"\nSaved maps: GT_{tag}.png, ClassMap_GAHT_{tag}.png, "
          f"Compare_GAHT_{tag}.png")