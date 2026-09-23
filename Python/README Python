SuperSDF supplementary Python code
==================================

Files
-----
1. SuperSDF_MC10.py
   Main feature-extraction and 10-run Monte-Carlo evaluation script.

2. random_split_data.py
   Python port of the paper's class-wise random train/validation/test split.

Final parameters reported in the paper
--------------------------------------
Dataset              k     Nsp     D      Train     Validation     Test
Indian Pines         35     60     25       5%          5%         rest
Pavia University     35    240    240       2%          2%         rest
WHU-Hi-HanChuan      35    400    150       1%          1%         rest

Common settings
---------------
Minimum training samples/class : 5
Validation size/class          : same as training size
Spatial split constraint       : none
Monte-Carlo runs               : 10
MC seeds                       : 20, 27, 34, 41, 48, 55, 62, 69, 76, 83
Gaussian sigma                 : 0.95
Gaussian filter size           : 55 x 55
SLIC compactness               : 0.1
Guided-filter regularization   : 1e-4
SVM C                          : 1e5
MATLAB KernelScale             : 1/sqrt(0.01) = 10

Expected dataset variables
--------------------------
Each MAT file must contain:
    img : rows x cols x bands
    gt  : rows x cols, with 0 for unlabeled/background pixels

Default filenames
-----------------
Indian     -> Indian_pines.mat
Pavia      -> PaviaUF.mat
HanChuan   -> HanChuan.mat

Examples
--------
python SuperSDF_MC10.py --dataset Indian
python SuperSDF_MC10.py --dataset Pavia --file /path/to/PaviaUF.mat
python SuperSDF_MC10.py --dataset HanChuan --plot-map

Dependencies
------------
pip install numpy scipy scikit-learn scikit-image matplotlib

Notes
-----
* SuperSDF feature extraction is label-free and is computed once on the complete
  HSI. Train/validation/test labels are used only for classifier training,
  validation during parameter selection, and final test evaluation.
* This final supplementary script does not repeat hyperparameter search. It uses
  the final (k, Nsp, D) values selected using validation OA in the paper.
* Exact numerical equality with MATLAB is not guaranteed because the Python and
  MATLAB implementations of SLIC, KMeans, guided filtering, Gaussian filtering,
  and multiclass SVM are not identical.
