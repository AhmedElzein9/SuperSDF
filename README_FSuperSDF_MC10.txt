SuperSDF MATLAB MC10

Files
-----
FSuperSDF_MC10.m
random_split_data_MC.m

Run
---
results = FSuperSDF_MC10('Indian');
results = FSuperSDF_MC10('Pavia');
results = FSuperSDF_MC10('HanChuan');

Dataset filenames expected by default
-------------------------------------
Indian_pines.mat
PaviaUF.mat
HanChuan.mat

Final paper settings
--------------------
Indian Pines : k=35, N_SP=60,  D=25,  train=5%, validation=5%
Pavia Univ.  : k=35, N_SP=240, D=240, train=2%, validation=2%
HanChuan     : k=35, N_SP=400, D=150, train=1%, validation=1%

Monte-Carlo seeds
-----------------
20, 27, 34, 41, 48, 55, 62, 69, 76, 83

Metrics are calculated only on the held-out test set.
Feature extraction is performed once because it is label-free.
