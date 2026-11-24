# SuperSDF

**Status:** the paper is under review.

Code for **SuperSDF: A training-free superpixel distance-field encoder for label-scarce hyperspectral crop and land-cover mapping** (A. Elzein, U. H. Atasever, M. Freeshah).

SuperSDF is a training-free, label-free spectral–spatial encoder for hyperspectral images. It works in five steps:
1. Group correlated bands with k-means and average each group into a mean image (k = 35).
2. Summarize each image by superpixel-wise minimum pooling (SLIC).
3. Binarize adaptively, using the Otsu level of the pooled map as the sensitivity.
4. Compute a signed distance field.
5. Refine the field with a guided filter and a Gaussian filter.

The resulting 35-channel feature cube is classified with an RBF-SVM.

## Repository layout

```
SuperSDF/
  matlab/
    supersdf_features.m                 SuperSDF feature extraction (paper model)
    SuperSDF_MC10.m                     10-run Monte Carlo evaluation (Protocol A)
    SuperSDF_Final_v2.m                 parameter selection (k x N_SP map, then D) on validation pixels
    random_split_data_MC.m              class-wise random split (Protocol A)
    spatial_block_split_data_guarded.m  guarded spatial block split (Protocol B)
    SuperSDF_Ablation.m                 component ablation (V1-V8), paired splits
    ablation_mcnemar.m                  exact McNemar test
    ablation_write_outputs.m            ablation tables / workbook
    experiments/                        design-choice, k-means seed, separability, timing and
                                        spatial-dependence experiments (entry point: SuperSDF_Phase1.m)
  python/                               Python port (for reference; see note below)
baselines/
  common/      split functions used with the baselines (same as SuperSDF/matlab)
  SuperPCA/    SuperPCA (original code) + SoperPCA_New.m / Run_MC_SoperPCA.m (evaluation under our protocol)
  S3PCA/       S3-PCA (original code) + FS3PCA_New.m (evaluation under our protocol)
  S3ULDA/      S3-ULDA (original code) + demo_S3ULDA_MC10.m (evaluation under our protocol)
  SSFTT/       SSFTT (PyTorch); cls_SSFTT_IP/SSFTT_SDF.py runs it with the SuperSDF seeds and metrics
  GAHT/        GAHT (PyTorch); GAHT.py: random split with the SuperSDF seeds;
               GAHT_SSFTT_Block.py: GAHT/SSFTT under the guarded block split
  MambaLG/     MambaLG (PyTorch, original code)
data/          put the dataset .mat files here (not included)
```

## Data

The datasets are not redistributed here. Download them from the original sources:

- **Indian Pines** and **Pavia University**: <https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes>
- **WHU-Hi-HanChuan**: <http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm>

The MATLAB scripts expect one `.mat` file per scene with variables `img` (rows × cols × bands) and `gt` (rows × cols, 0 = unlabeled), named `Indian_pines.mat`, `PaviaUF.mat` and `HanChuan.mat`. Put them in `data/` and add that folder to the MATLAB path, or copy them next to the scripts.

## Running SuperSDF (MATLAB)

Requirements: MATLAB R2024b or later, with the Image Processing Toolbox, the Statistics and Machine Learning Toolbox and, for `experiments/`, the Parallel Computing Toolbox (optional).

```matlab
addpath(genpath('SuperSDF/matlab')); addpath('data');
results = SuperSDF_MC10('Indian');    % 'Indian' | 'Pavia' | 'HanChuan'
```

Paper settings:

| Dataset | Labeled pixels per class | k | N_SP | D |
|---|---|---|---|---|
| Indian Pines | 5% | 35 | 60 | 25 |
| Pavia University | 2% | 35 | 240 | 240 |
| WHU-Hi-HanChuan | 1% | 35 | 400 | 150 |

All datasets use at least 5 training pixels per class. The other settings are:
- SLIC compactness 0.1
- guided filter ε = 1e-4
- Gaussian σ = 0.95 with a 55 × 55 kernel
- SVM: RBF kernel, C = 1e5, γ = 0.01, z-scored features, one-vs-one ECOC

Splits use seeds `7m+13` for m = 1…10, with `rng(seed,'twister')`. k-means uses `rng(50,'twister')`.

The Python port (`SuperSDF/python`) needs NumPy, SciPy, scikit-learn, scikit-image and Matplotlib. Its k-means, SLIC and SVM implementations differ from MATLAB's, so its numbers can differ slightly from the paper; the paper's results were produced with the MATLAB code.

## Baselines

The baseline folders contain third-party code from the original authors, plus our scripts that run each method under the same splits and evaluation as SuperSDF. Please cite the original papers and follow the licences in their folders:

- SuperPCA: Jiang et al., IEEE TGRS 2018
- S3-PCA: Zhang et al., IEEE TGRS 2021
- S3-ULDA: Lu et al., IEEE TGRS 2023
- SSFTT: Sun et al., IEEE TGRS 2022
- GAHT: Mei et al., IEEE TGRS 2022, doi:10.1109/TGRS.2022.3207933
- MambaLG: Pan et al., IEEE TGRS 2025, <https://ieeexplore.ieee.org/document/10812905>

SuperPCA, S3-PCA and S3-ULDA use the ERS superpixel MEX (`Entropy Rate Superpixel Segmentation/mex_ers.mexw64`, Windows). On other platforms, rebuild it with `make.m` in that folder. The LIBSVM copies bundled with the original repositories were left out; the evaluation scripts use MATLAB `fitcecoc`. Deep-model checkpoints, datasets and result files are not included.

## Citation

Please cite the paper if you use this code (reference to be added after publication).
