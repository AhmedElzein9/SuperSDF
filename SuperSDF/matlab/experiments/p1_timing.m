function S = p1_timing(DATASET_ID, nRep)
%P1_TIMING E7: stage-wise CPU time of SuperSDF feature extraction (median
%   of nRep repetitions, k-means included) and SVM train/predict time on the
%   first 3 paper splits. CPU only.
if nargin < 2 || isempty(nRep), nRep = 5; end
root = fileparts(mfilename('fullpath'));
cfg = p1_cfg(DATASET_ID);
L = load(fullfile(root, cfg.file), 'img', 'gt');
img = double(L.img); gt = double(L.gt); clear L
[rows, cols, ~] = size(img);
st = {'kmeans','mean','slic','pool','binarize','dist','gf','gauss'};
T = zeros(nRep, numel(st)); tot = zeros(nRep,1);
for r = 1:nRep
    q = tic; [F, info] = p1_features(img, cfg.k, cfg.N_SP, cfg.D, struct()); tot(r) = toc(q);
    for s = 1:numel(st), T(r,s) = info.t.(st{s}); end
end
valid_idx = find(gt(:) > 0); ph = zeros(rows, cols, 1);
A = reshape(F, rows*cols, []);
ttr = zeros(3,1); tpr = zeros(3,1);
for m = 1:3
    rng(m*7+13,'twister');
    [~,~,~,~,~,~,~,tri,~,tei] = random_split_data_MC(ph, gt, cfg.train_ratio, 5, false);
    tr = valid_idx(tri); te = valid_idx(tei);
    tpl = templateSVM('KernelFunction','rbf','KernelScale',10,'BoxConstraint',1e5,'Standardize',true);
    q = tic; mdl = fitcecoc(A(tr,:), gt(tr), 'Learners', tpl, 'Verbose', 0); ttr(m) = toc(q);
    q = tic; predict(mdl, A(te,:)); tpr(m) = toc(q);
end
S = struct('dataset',cfg.name,'short',cfg.short,'tag',cfg.tag,'stages',{st}, ...
    'stage_median',median(T,1),'feature_total_median',median(tot), ...
    'svm_train_median',median(ttr),'svm_predict_median',median(tpr), ...
    'n_train',numel(tr),'n_test',numel(te),'size',size(img), ...
    'cpu',getenv('PROCESSOR_IDENTIFIER'),'n_threads',maxNumCompThreads, ...
    'matlab_version',version,'computer',computer);
save(fullfile(root, sprintf('SuperSDF_P1_E7_%s.mat', cfg.tag)), '-struct', 'S');
fprintf('%s E7 feature=%.2fs (slic %.2f, gf %.2f, dist %.2f) svmTrain=%.2fs svmPred=%.2fs\n', cfg.short, ...
    S.feature_total_median, S.stage_median(3), S.stage_median(7), S.stage_median(6), ...
    S.svm_train_median, S.svm_predict_median);
end
