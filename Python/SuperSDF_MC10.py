#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SuperSDF supplementary Python implementation.

Paper protocol implemented here
-------------------------------
* Feature extraction is label-free and is performed once on the complete HSI.
* Final fixed parameters selected using validation OA:
    Indian Pines      : k=35, Nsp=60,  D=25,  train=5%, val=5%
    Pavia University  : k=35, Nsp=240, D=240, train=2%, val=2%
    WHU-Hi-HanChuan   : k=35, Nsp=400, D=150, train=1%, val=1%
* Minimum training samples per class: 5; validation size equals training size.
* Remaining labeled pixels are the strictly held-out test set.
* No spatial exclusion/constraint is applied to the random split.
* Ten Monte-Carlo runs use seeds mc*7+13 -> 20,27,...,83.
* Common RBF-SVM setting: MATLAB KernelScale=1/sqrt(0.01), C=1e5,
  Standardize=true.  The equivalent scikit-learn RBF coefficient is 0.005.
* Gaussian smoothing is fixed for all datasets: sigma=0.95, size=55x55.

Important reproducibility note
------------------------------
This script follows the paper's algorithm and evaluation protocol, but exact
floating-point results need not be identical to MATLAB because scikit-image
SLIC, scikit-learn KMeans/SVM, SciPy filtering, and MATLAB implementations are
not numerically identical.

Expected MAT-file variables
---------------------------
Each dataset file must contain:
    img : HSI cube of shape (rows, cols, bands)
    gt  : integer ground-truth map of shape (rows, cols), with 0=background

Examples
--------
    python SuperSDF_Supplementary.py --dataset Indian
    python SuperSDF_Supplementary.py --dataset Pavia --file /path/PaviaUF.mat
    python SuperSDF_Supplementary.py --dataset HanChuan --plot-map

Dependencies
------------
    pip install numpy scipy scikit-learn scikit-image matplotlib
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from skimage.filters import threshold_otsu
from skimage.segmentation import slic

from random_split_data import random_split_data


# -------------------------------------------------------------------------
# Fixed paper settings
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    file: str
    k: int
    nsp: int
    d: int
    train_ratio: float


DATASETS = {
    "Indian": DatasetConfig("Indian_pines.mat", 35, 60, 25, 0.05),
    "Pavia": DatasetConfig("PaviaUF.mat", 35, 240, 240, 0.02),
    "HanChuan": DatasetConfig("HanChuan.mat", 35, 400, 150, 0.01),
}

N_MC = 10
MC_SEEDS = [mc * 7 + 13 for mc in range(1, N_MC + 1)]
MIN_TRAIN_PER_CLASS = 5

SIGMA_GAUSS = 0.95
GAUSS_FSIZE = 55
SLIC_COMPACTNESS = 0.1
GUIDED_EPS = 1e-4

SVM_C = 1e5
MATLAB_GAMMA_PARAMETER = 0.01
MATLAB_KERNEL_SCALE = 1.0 / np.sqrt(MATLAB_GAMMA_PARAMETER)  # 10
SKLEARN_GAMMA = 1.0 / (2.0 * MATLAB_KERNEL_SCALE**2)        # 0.005

KMEANS_SEED = 50
MAP_SEED = 50


# -------------------------------------------------------------------------
# MATLAB-like standardization for each one-vs-one binary SVM learner
# -------------------------------------------------------------------------
class MatlabStandardizer(BaseEstimator, TransformerMixin):
    """Center and scale predictors using sample standard deviation (N-1)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        if X.shape[0] > 1:
            self.scale_ = np.std(X, axis=0, ddof=1)
        else:
            self.scale_ = np.ones(X.shape[1], dtype=np.float64)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ == 0)] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_


def build_svm():
    """Approximate MATLAB fitcecoc + templateSVM(one-vs-one, Standardize=true)."""
    binary_learner = Pipeline(
        [
            ("standardize", MatlabStandardizer()),
            ("svm", SVC(C=SVM_C, kernel="rbf", gamma=SKLEARN_GAMMA)),
        ]
    )
    return OneVsOneClassifier(binary_learner, n_jobs=1)


# -------------------------------------------------------------------------
# SuperSDF feature extraction
# -------------------------------------------------------------------------
def guided_filter(guide: np.ndarray, src: np.ndarray, window_size: int, eps: float):
    """Gray-scale guided filter using box-filter statistics."""
    I = np.asarray(guide, dtype=np.float64)
    p = np.asarray(src, dtype=np.float64)
    ws = max(1, int(window_size))

    mean_I = uniform_filter(I, size=ws, mode="reflect")
    mean_p = uniform_filter(p, size=ws, mode="reflect")
    mean_Ip = uniform_filter(I * p, size=ws, mode="reflect")
    mean_II = uniform_filter(I * I, size=ws, mode="reflect")

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, size=ws, mode="reflect")
    mean_b = uniform_filter(b, size=ws, mode="reflect")
    return mean_a * I + mean_b


def extract_sdf(
    img: np.ndarray,
    k: int,
    nsp: int,
    d: int,
    sigma: float = SIGMA_GAUSS,
    fsize: int = GAUSS_FSIZE,
    compactness: float = SLIC_COMPACTNESS,
    guided_eps: float = GUIDED_EPS,
):
    """Extract the k SuperSDF feature maps described in the paper.

    Pipeline:
      correlation matrix -> k-means band grouping -> group mean image ->
      SLIC superpixels -> superpixel minimum map -> Otsu binary mask ->
      signed distance field -> guided filtering -> Gaussian smoothing.
    """
    if img.ndim != 3:
        raise ValueError("img must have shape (rows, cols, bands)")

    rows, cols, bands = img.shape
    if not (1 <= k <= bands):
        raise ValueError(f"k={k} must be between 1 and the number of bands ({bands})")

    # MATLAB: X = reshape(img, rows*cols, bands); R = corrcoef(X)
    X = img.reshape(rows * cols, bands, order="F")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    # MATLAB uses rng(50), kmeans(...,'Replicates',1).
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=1,
        max_iter=100,
        algorithm="lloyd",
        random_state=KMEANS_SEED,
    )
    labels = km.fit_predict(R)
    groups = [np.flatnonzero(labels == i) for i in range(k)]

    sdf_features = np.zeros((rows, cols, k), dtype=np.float64)
    truncate = (fsize - 1) / (2.0 * sigma)

    for i, band_idx in enumerate(groups):
        if band_idx.size == 0:
            raise RuntimeError(f"Empty spectral group produced for cluster {i + 1}")

        # One mean image per spectral group.
        group_img = np.mean(img[:, :, band_idx], axis=2)
        lo = float(np.min(group_img))
        hi = float(np.max(group_img))
        if hi > lo:
            norm_img = (group_img - lo) / (hi - lo)
        else:
            norm_img = np.zeros_like(group_img)

        # Grayscale SLIC. n_segments is a requested/approximate superpixel count.
        sp_labels = slic(
            norm_img,
            n_segments=int(nsp),
            compactness=float(compactness),
            start_label=0,
            channel_axis=None,
            convert2lab=False,
            enforce_connectivity=True,
        )

        # Replace every superpixel by the minimum normalized intensity in it.
        sp_min = np.zeros_like(norm_img)
        for sid in np.unique(sp_labels):
            mask = sp_labels == sid
            sp_min[mask] = np.min(norm_img[mask])

        # Paper implementation: Otsu threshold on the superpixel-minimum map.
        if np.all(sp_min == sp_min.flat[0]):
            binary = np.zeros_like(sp_min, dtype=bool)
        else:
            threshold = threshold_otsu(sp_min)
            binary = sp_min > threshold

        # MATLAB sign convention: bwdist(~B) - bwdist(B), positive inside B.
        if binary.all() or not binary.any():
            sdf = np.zeros_like(sp_min, dtype=np.float64)
        else:
            sdf = distance_transform_edt(binary) - distance_transform_edt(~binary)

        sdf = guided_filter(group_img, sdf, d, guided_eps)
        sdf = gaussian_filter(sdf, sigma=sigma, truncate=truncate, mode="reflect")
        sdf_features[:, :, i] = sdf

    return sdf_features


# -------------------------------------------------------------------------
# Metrics and evaluation
# -------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    total = int(cm.sum())
    oa = float(np.trace(cm) / max(total, 1)) * 100.0

    row_sum = cm.sum(axis=1)
    class_acc = np.divide(
        np.diag(cm),
        row_sum,
        out=np.zeros(len(classes), dtype=np.float64),
        where=row_sum > 0,
    ) * 100.0
    aa = float(np.mean(class_acc))

    col_sum = cm.sum(axis=0)
    po = oa / 100.0
    pe = float(np.dot(row_sum, col_sum)) / max(total**2, 1)
    kappa = ((po - pe) / max(1.0 - pe, 1e-12)) * 100.0
    return oa, aa, kappa, class_acc


def run_dataset(dataset: str, hsi_file: str | Path | None = None, plot_map: bool = False):
    cfg = DATASETS[dataset]
    path = Path(hsi_file if hsi_file is not None else cfg.file)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Use --file to provide the path to the {dataset} MAT file."
        )

    mat = sio.loadmat(path)
    if "img" not in mat or "gt" not in mat:
        visible_keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            "MAT file must contain variables named 'img' and 'gt'. "
            f"Found: {visible_keys}"
        )

    img = np.asarray(mat["img"], dtype=np.float64)
    gt = np.asarray(mat["gt"], dtype=np.int64)
    rows, cols, bands = img.shape
    classes = np.unique(gt[gt > 0])

    print("=" * 72)
    print(f"SuperSDF supplementary experiment - {dataset}")
    print("=" * 72)
    print(f"Image                  : {rows} x {cols} x {bands}")
    print(f"Final (k, Nsp, D)      : ({cfg.k}, {cfg.nsp}, {cfg.d})")
    print(f"Train / validation     : {cfg.train_ratio*100:.0f}% / {cfg.train_ratio*100:.0f}% per class")
    print("Test                   : remaining labeled samples")
    print(f"Minimum train/class    : {MIN_TRAIN_PER_CLASS}")
    print(f"Monte-Carlo runs       : {N_MC}")
    print(f"MC seeds               : {MC_SEEDS}")
    print(f"Gaussian sigma / size  : {SIGMA_GAUSS} / {GAUSS_FSIZE}x{GAUSS_FSIZE}")
    print(f"SVM C / KernelScale    : {SVM_C:.0e} / {MATLAB_KERNEL_SCALE:g}")
    print("Spatial split constraint: none")

    # Label-free feature extraction is performed once and reused in all MC runs.
    t0 = time.perf_counter()
    features = extract_sdf(img, cfg.k, cfg.nsp, cfg.d)
    feature_time = time.perf_counter() - t0
    print(f"\nFeature extraction time: {feature_time:.2f} s")

    oa_all = np.zeros(N_MC)
    aa_all = np.zeros(N_MC)
    kp_all = np.zeros(N_MC)
    ca_all = np.zeros((N_MC, len(classes)))
    clf_times = np.zeros(N_MC)

    for run, seed in enumerate(MC_SEEDS, start=1):
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            _sample_map,
            _tr_idx,
            _vl_idx,
            _te_idx,
            _X_all,
        ) = random_split_data(
            features,
            gt,
            cfg.train_ratio,
            min_train_per_class=MIN_TRAIN_PER_CLASS,
            seed=seed,
            verbose=False,
        )

        # Validation is intentionally NOT merged with training. It is a separate
        # subset in the paper protocol; the held-out test set is evaluated here.
        t_clf = time.perf_counter()
        model = build_svm()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        clf_times[run - 1] = time.perf_counter() - t_clf

        oa, aa, kp, ca = compute_metrics(y_test, y_pred, classes)
        oa_all[run - 1] = oa
        aa_all[run - 1] = aa
        kp_all[run - 1] = kp
        ca_all[run - 1, :] = ca
        print(f"MC {run:2d}/{N_MC} seed={seed:2d}: OA={oa:6.2f}%  AA={aa:6.2f}%  kappa={kp:6.2f}%")

    def ms(a):
        return float(np.mean(a)), float(np.std(a, ddof=1))

    print("\n" + "=" * 72)
    print("FINAL TEST RESULTS (mean +/- sample std over 10 MC runs)")
    print("=" * 72)
    for j, cls in enumerate(classes):
        m, s = ms(ca_all[:, j])
        print(f"Class {int(cls):2d}: {m:6.2f} +/- {s:5.2f}")

    oa_m, oa_s = ms(oa_all)
    aa_m, aa_s = ms(aa_all)
    kp_m, kp_s = ms(kp_all)
    tm_m, tm_s = ms(clf_times)

    print("-" * 72)
    print(f"OA (%)    : {oa_m:.2f} +/- {oa_s:.2f}")
    print(f"AA (%)    : {aa_m:.2f} +/- {aa_s:.2f}")
    print(f"Kappa (%) : {kp_m:.2f} +/- {kp_s:.2f}")
    print("\nTIMING (parameter search, metrics, and visualization excluded)")
    print(f"Feature extraction     : {feature_time:.2f} s")
    print(f"Mean SVM train+test    : {tm_m:.2f} +/- {tm_s:.2f} s")
    print(f"Reported method time   : {feature_time + tm_m:.2f} s")

    if plot_map:
        # A fixed reference split is used only to create a qualitative map.
        (
            X_train,
            y_train,
            _X_val,
            _y_val,
            _X_test,
            _y_test,
            _sample_map,
            _tr_idx,
            _vl_idx,
            _te_idx,
            X_all,
        ) = random_split_data(
            features,
            gt,
            cfg.train_ratio,
            min_train_per_class=MIN_TRAIN_PER_CLASS,
            seed=MAP_SEED,
            verbose=False,
        )
        model = build_svm()
        model.fit(X_train, y_train)
        pred_all = model.predict(X_all)
        class_map = pred_all.reshape(rows, cols, order="F")
        class_map = class_map.astype(float)
        class_map[gt == 0] = 0

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        axes[0].imshow(gt, cmap="jet", interpolation="nearest")
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        axes[1].imshow(class_map, cmap="jet", interpolation="nearest")
        axes[1].set_title(f"SuperSDF - {dataset}")
        axes[1].axis("off")
        fig.suptitle(f"k={cfg.k}, Nsp={cfg.nsp}, D={cfg.d}")
        fig.tight_layout()
        plt.show()

    return {
        "oa": oa_all,
        "aa": aa_all,
        "kappa": kp_all,
        "class_accuracy": ca_all,
        "feature_time": feature_time,
        "classifier_times": clf_times,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the final SuperSDF paper protocol.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="Indian",
        help="Dataset configuration to use (default: Indian).",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Optional path to MAT file; overrides the paper default filename.",
    )
    parser.add_argument(
        "--plot-map",
        action="store_true",
        help="Generate a qualitative full-scene classification map after MC evaluation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dataset(args.dataset, args.file, plot_map=args.plot_map)
