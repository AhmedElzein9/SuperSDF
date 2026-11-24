function S = p1_kmeans_seed(DATASET_ID, N_MC)
%P1_KMEANS_SEED E3: full pipeline with the k-means seed varied per MC run.
%   Run m: k-means with rng(FEATURE_SEEDS(m)) (same seeds as
%   SuperSDF_Final_v2.m: (1:10)*101+1000), features with the paper
%   parameters, split with rng(SPLIT_SEEDS(m)) (same splits as the paper),
%   SVM as SuperSDF_MC10.m. Also checks that SLIC is deterministic.
if nargin < 2 || isempty(N_MC), N_MC = 10; end
root = fileparts(mfilename('fullpath'));
cfg = p1_cfg(DATASET_ID);
split_seeds = (1:N_MC)*7 + 13;
feat_seeds  = (1:N_MC)*101 + 1000;
L = load(fullfile(root, cfg.file), 'img', 'gt');
img = double(L.img); gt = double(L.gt); clear L
[rows, cols, bands] = size(img);
valid_idx = find(gt(:) > 0); class_order = unique(gt(valid_idx));
ph = zeros(rows, cols, 1);

R = corrcoef(reshape(img, rows*cols, bands));
rng(50,'twister'); ref_lab = kmeans(R, cfg.k, 'Replicates', 1);

% SLIC determinism: same input twice -> identical labels?
g1 = mean(img(:,:,ref_lab == 1), 3); g1 = (g1 - min(g1(:))) / (max(g1(:)) - min(g1(:)));
La = superpixels(g1, cfg.N_SP, 'Method','slic','Compactness',0.1);
rng(12345,'twister');
Lb = superpixels(g1, cfg.N_SP, 'Method','slic','Compactness',0.1);
slic_det = isequal(La, Lb);

S = struct('dataset',cfg.name,'short',cfg.short,'tag',cfg.tag,'cfg',cfg, ...
    'split_seeds',split_seeds,'feat_seeds',feat_seeds,'slic_deterministic',slic_det);
S.OA = nan(N_MC,1); S.AA = nan(N_MC,1); S.kappa = nan(N_MC,1);
S.same_grouping_as_seed50 = false(N_MC,1); S.ARI_vs_seed50 = nan(N_MC,1);
S.n_const = nan(N_MC,1);
for m = 1:N_MC
    rng(feat_seeds(m),'twister');
    lab = kmeans(R, cfg.k, 'Replicates', 1);
    groups = arrayfun(@(i) find(lab == i), 1:cfg.k, 'UniformOutput', false);
    S.ARI_vs_seed50(m) = ari(lab, ref_lab);
    S.same_grouping_as_seed50(m) = S.ARI_vs_seed50(m) > 1 - 1e-12;
    [F, info] = p1_features(img, cfg.k, cfg.N_SP, cfg.D, struct(), groups);
    S.n_const(m) = info.n_constant_masks;
    rng(split_seeds(m),'twister');
    [~,~,~,~,~,~,~,tri,~,tei] = random_split_data_MC(ph, gt, cfg.train_ratio, 5, false);
    A = reshape(F, rows*cols, []); clear F
    tr = valid_idx(tri); te = valid_idx(tei);
    tpl = templateSVM('KernelFunction','rbf','KernelScale',10,'BoxConstraint',1e5,'Standardize',true);
    mdl = fitcecoc(A(tr,:), gt(tr), 'Learners', tpl, 'Verbose', 0);
    yp = predict(mdl, A(te,:)); clear A
    [S.OA(m), S.AA(m), S.kappa(m)] = p1_metrics(gt(te), yp, class_order);
    fprintf('%s E3 run %d kseed=%d ARI=%.3f OA=%6.2f AA=%6.2f K=%6.2f\n', cfg.short, m, ...
        feat_seeds(m), S.ARI_vs_seed50(m), S.OA(m), S.AA(m), S.kappa(m));
end
save(fullfile(root, sprintf('SuperSDF_P1_E3_%s_random_MC%d.mat', cfg.tag, N_MC)), '-struct', 'S');
fprintf('%s E3 OA=%.2f+-%.2f SLIC deterministic=%d\n', cfg.short, mean(S.OA), std(S.OA), slic_det);
end

function r = ari(a, b)
% Adjusted Rand index between two partitions.
[~,~,a] = unique(a); [~,~,b] = unique(b);
C = accumarray([a b], 1);
n = numel(a); c2 = @(x) x.*(x-1)/2;
sij = sum(c2(C(:))); sa = sum(c2(sum(C,2))); sb = sum(c2(sum(C,1)));
e = sa*sb/c2(n); mx = (sa+sb)/2;
r = (sij - e) / (mx - e);
end
