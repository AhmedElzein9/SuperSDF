"""
GAHT_SSFTT_Block.py — GAHT/SSFTT under the exact SuperSDF guarded-block protocol.

Place in the GAHT repo root (same level as train.py, models/, utils/).
Loads the exact MATLAB train/validation/test block masks used by SuperSDF; excluded labelled pixels remain ignored.

NOTE: utils/dataset.py sets undefined pixels to -1 and class indices start
at 0. All split logic below follows that convention.
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import scipy.io as sio
from pathlib import Path

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
# EXACT MATLAB GUARDED-BLOCK SPLITS
# =========================================================================
def _normalize_mask_stack(a, gt_shape, name):
    a = np.asarray(a).astype(bool)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"{name} must be HxWxR, got shape {a.shape}")
    if tuple(a.shape[:2]) != tuple(gt_shape):
        raise ValueError(
            f"{name} spatial shape {a.shape[:2]} does not match GAHT GT shape {gt_shape}."
        )
    return a


def load_matlab_block_splits(split_file, gt, num_run):
    """Load the EXACT MATLAB guarded-block masks used by SuperSDF.

    GAHT convention: gt==-1 is background; gt>=0 are classes 0..C-1.
    Excluded labelled pixels remain outside train/val/test and therefore do
    not enter OA/AA/Kappa.
    """
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"Block split file not found:\n  {split_file}")

    s = sio.loadmat(str(split_file))
    required = ["train_masks", "val_masks", "test_masks", "seeds"]
    missing = [k for k in required if k not in s]
    if missing:
        raise KeyError(f"{split_file.name} is missing required variables: {missing}")

    train_masks = _normalize_mask_stack(s["train_masks"], gt.shape, "train_masks")
    val_masks = _normalize_mask_stack(s["val_masks"], gt.shape, "val_masks")
    test_masks = _normalize_mask_stack(s["test_masks"], gt.shape, "test_masks")
    seeds = np.asarray(s["seeds"]).ravel().astype(int)

    n_available = min(train_masks.shape[2], val_masks.shape[2],
                      test_masks.shape[2], len(seeds))
    if n_available < num_run:
        raise ValueError(
            f"Requested num_run={num_run}, but {split_file.name} contains only {n_available} complete runs."
        )

    train_masks = train_masks[:, :, :num_run]
    val_masks = val_masks[:, :, :num_run]
    test_masks = test_masks[:, :, :num_run]
    seeds = seeds[:num_run]

    labelled = gt >= 0

    if "excluded_masks" in s:
        excluded_masks = _normalize_mask_stack(s["excluded_masks"], gt.shape, "excluded_masks")
        if excluded_masks.shape[2] < num_run:
            raise ValueError(
                f"excluded_masks contains only {excluded_masks.shape[2]} runs, but num_run={num_run}."
            )
        excluded_masks = excluded_masks[:, :, :num_run]
        explicit_excluded = True
    else:
        excluded_masks = np.zeros_like(train_masks, dtype=bool)
        explicit_excluded = False

    for i in range(num_run):
        tr = train_masks[:, :, i]
        vl = val_masks[:, :, i]
        te = test_masks[:, :, i]
        if np.any(tr & vl) or np.any(tr & te) or np.any(vl & te):
            raise ValueError(f"Train/val/test overlap in split run {i+1}.")
        retained = tr | vl | te
        if np.any(retained & ~labelled):
            raise ValueError(f"Background pixels occur in retained split run {i+1}.")

        if explicit_excluded:
            ex = excluded_masks[:, :, i]
            if np.any(ex & retained):
                raise ValueError(f"Excluded pixels overlap retained pixels in run {i+1}.")
            if np.any(ex & ~labelled):
                raise ValueError(f"excluded_masks contains background in run {i+1}.")
            if not np.array_equal(retained | ex, labelled):
                raise ValueError(
                    f"Run {i+1}: retained + excluded does not equal all labelled pixels."
                )
        else:
            excluded_masks[:, :, i] = labelled & ~retained

    expected_seeds = np.array([(i + 1) * 7 + 13 for i in range(num_run)], dtype=int)
    if not np.array_equal(seeds, expected_seeds):
        raise ValueError(
            "MATLAB split seeds do not match the SuperSDF seed family "
            f"{expected_seeds.tolist()}; got {seeds.tolist()}."
        )

    if not explicit_excluded:
        print("WARNING: excluded_masks not found; deriving excluded labelled pixels from the complement of train/val/test.")

    return train_masks, val_masks, test_masks, excluded_masks, seeds, s


def masks_to_gaht_gt(gt, train_mask, val_mask, test_mask):
    train_gt = np.full_like(gt, -1)
    val_gt = np.full_like(gt, -1)
    test_gt = np.full_like(gt, -1)
    train_gt[train_mask] = gt[train_mask]
    val_gt[val_mask] = gt[val_mask]
    test_gt[test_mask] = gt[test_mask]
    return train_gt, val_gt, test_gt


def resolve_block_split_file(split_dir, split_file, dataset_name):
    if split_file:
        p = Path(split_file).expanduser()
        if not p.is_absolute():
            p = Path(split_dir).expanduser() / p
        return p

    split_dir = Path(split_dir).expanduser()
    aliases = {
        "pu": ["BlockSplit_Pavia.mat", "BlockSplitPavia.mat", "SuperSDF_Splits_Pavia.mat"],
        "pavia": ["BlockSplit_Pavia.mat", "BlockSplitPavia.mat", "SuperSDF_Splits_Pavia.mat"],
        "ip": ["BlockSplit_Indian.mat", "BlockSplitIndian.mat", "SuperSDF_Splits_Indian.mat"],
        "indian": ["BlockSplit_Indian.mat", "BlockSplitIndian.mat", "SuperSDF_Splits_Indian.mat"],
        "hc": ["BlockSplit_HanChuan.mat", "BlockSplitHanChuan.mat", "SuperSDF_Splits_HanChuan.mat"],
        "hanchuan": ["BlockSplit_HanChuan.mat", "BlockSplitHanChuan.mat", "SuperSDF_Splits_HanChuan.mat"],
    }
    candidates = aliases.get(dataset_name.lower(), [
        f"BlockSplit_{dataset_name}.mat",
        f"BlockSplit{dataset_name}.mat",
        f"SuperSDF_Splits_{dataset_name}.mat",
    ])
    for name in candidates:
        p = split_dir / name
        if p.exists():
            return p
    mats = sorted(split_dir.glob("*.mat")) if split_dir.exists() else []
    if len(mats) == 1:
        return mats[0]
    raise FileNotFoundError(
        "Could not resolve a block split file.\n"
        f"Dataset: {dataset_name}\nFolder: {split_dir}\nTried: {candidates}\n"
        "Use --split_file with the exact .mat filename."
    )


# =========================================================================
# Main — GAHT only
# =========================================================================
if __name__ == "__main__":

    DEFAULT_MODEL = 'ssftt'   # Use --model ssftt or --model gaht

    parser = argparse.ArgumentParser(
        description="GAHT under the SuperSDF evaluation protocol")
    parser.add_argument("--dataset_dir",  type=str,
        default="../../data")
    parser.add_argument("--dataset_name", type=str,   default="pu")
    parser.add_argument("--model",        type=str,   default=DEFAULT_MODEL,
                        help="Model name accepted by models/get_model.py, e.g. ssftt or gaht")
    parser.add_argument("--device",       type=str,   default="0")
    parser.add_argument("--patch_size",   type=int,   default=7)
    parser.add_argument("--num_run",      type=int,   default=10)
    parser.add_argument("--epoch",        type=int,   default=300)
    parser.add_argument("--bs",           type=int,   default=64)
    parser.add_argument("--split_dir",    type=str,   default="./BlockSplit")
    parser.add_argument("--split_file",   type=str,   default="",
                        help="Exact MATLAB .mat filename; blank = resolve by dataset name")
    opts = parser.parse_args()
    MODEL = opts.model.lower()

    device = torch.device(f"cuda:{opts.device}"
                          if torch.cuda.is_available() else "cpu")

    print("=== GAHT/SSFTT — EXACT SuperSDF guarded-block protocol ===")
    print(f"model      = {MODEL}")
    print(f"dataset    = {opts.dataset_name}")
    print(f"runs       = {opts.num_run}   epochs = {opts.epoch}   bs = {opts.bs}")
    print(f"patch_size = {opts.patch_size}")
    print(f"device     = {device}\n")

    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)
    num_classes = len(labels)

    split_path = resolve_block_split_file(opts.split_dir, opts.split_file, opts.dataset_name)
    train_masks, val_masks, test_masks, excluded_masks, seeds, split_meta = \
        load_matlab_block_splits(split_path, gt, opts.num_run)

    print(f"split file = {split_path}")
    if "train_ratio" in split_meta:
        try:
            print(f"split ratio= {float(np.asarray(split_meta['train_ratio']).squeeze())*100:.2f}%")
        except Exception:
            pass
    for key in ("block_size", "block_buffer_px", "min_train", "min_test"):
        if key in split_meta:
            try:
                print(f"{key:11s}= {np.asarray(split_meta[key]).squeeze()}")
            except Exception:
                pass

    results, run_times = [], []

    for run in range(opts.num_run):
        seed = int(seeds[run])
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)

        print(f"\n--- run {run+1}/{opts.num_run}  (seed {seed}) ---")
        t0 = time.time()

        tr_mask = train_masks[:, :, run]
        vl_mask = val_masks[:, :, run]
        te_mask = test_masks[:, :, run]
        ex_mask = excluded_masks[:, :, run]
        train_gt, val_gt, test_gt = masks_to_gaht_gt(gt, tr_mask, vl_mask, te_mask)

        print(f"train={np.sum(tr_mask)}  "
              f"val={np.sum(vl_mask)}  "
              f"test={np.sum(te_mask)}  "
              f"excluded={np.sum(ex_mask)}")

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
                     f"{opts.dataset_name}_block/{run}")

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
          f"({opts.num_run} runs, exact guarded-block masks)")
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
    np.savez(f"results_{MODEL}_{opts.dataset_name}_block.npz",
             OA=OA, AA=AA, KAP=KAP, CA=CA_mc,
             times=np.array(run_times), labels=np.array(labels, dtype=object))
    print(f"\nSaved: results_{MODEL}_{opts.dataset_name}_block.npz")
    
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
    tag = f"{MODEL}_{opts.dataset_name}_block"

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
    ax.set_title(f'{MODEL.upper()} — {opts.dataset_name.upper()}',
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    fig_cm.tight_layout()
    fig_cm.savefig(f'ClassMap_{tag}.png', dpi=300, bbox_inches='tight')

    # --- side-by-side with legend ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(gt_map,   cmap=cmap_cls, vmin=0, vmax=n_cls)
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].imshow(class_map, cmap=cmap_cls, vmin=0, vmax=n_cls)
    axes[1].set_title(f'{MODEL.upper()}  (OA={OA.mean():.2f}%)',
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
    fig.savefig(f'Compare_{tag}.png', dpi=300, bbox_inches='tight')

    plt.show()
    print(f"\nSaved maps: GT_{tag}.png, ClassMap_{tag}.png, "
          f"Compare_{tag}.png")