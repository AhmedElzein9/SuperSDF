function out = SuperSDF_Ablation(DATASET_ID, N_MC)
%SUPERSDF_ABLATION Paired cumulative + leave-one-out ablation of SuperSDF.
%
%   out = SuperSDF_Ablation('Indian'|'Pavia'|'HanChuan'|'all', N_MC)
%
%   Protocol (only one): class-wise stratified random split with
%   random_split_data_MC.m, train ratio = validation ratio, min 5 training
%   samples per class, rest = test. Validation is never used for training.
%   N_MC defaults to 10, seeds = (1:N_MC)*7+13, rng(seed,'twister').
%
%   Variants (K-means band grouping computed ONCE per dataset):
%     V1 K-means + SVM                      V5  full SuperSDF
%     V2 K-means + SP + SVM                 V6  no SDF
%     V3 K-means + SP + SDF + SVM           V7  no SP
%     V4 K-means + SP + SDF + GF + SVM      V8  no SP, no SDF
%     V5* = V5 plus the extra channel of SuperSDF_MC10.m line 175
%           (unnormalized mean image of the LAST band group, 36 channels)
%
%   Paired design: in each MC run the split is drawn ONCE and the same
%   train/test pixels are used for every variant. The script re-draws the
%   split from every variant's cube with the same seed and asserts that the
%   indices are identical.
%
%   Results are saved after every MC run to
%   SuperSDF_Ablation_<Dataset>_MC<N>.mat; a rerun resumes finished runs.
%   At the end ablation_write_outputs writes the xlsx and LaTeX table.

if nargin < 1 || isempty(DATASET_ID), DATASET_ID = 'all'; end
if nargin < 2 || isempty(N_MC), N_MC = 10; end

root = fileparts(mfilename('fullpath'));
addpath(root);

MIN_TRAIN = 5;
SVM_C     = 1e5;
SVM_GAMMA = 0.01;
seeds     = (1:N_MC) * 7 + 13;

names  = {'V1','V2','V3','V4','V5','V6','V7','V8','V5*'};
labels = {'K-means + SVM', ...
          'K-means + SP + SVM', ...
          'K-means + SP + SDF + SVM', ...
          'K-means + SP + SDF + GF + SVM', ...
          'K-means + SP + SDF + GF + GS + SVM (full)', ...
          'K-means + SP + GF + GS + SVM', ...
          'K-means + SDF + GF + GS + SVM', ...
          'K-means + GF + GS + SVM', ...
          'Full + last-group channel (MC10 line 175, 36 ch)'};
%          SP SDF GF GS
sw = [ 0  0   0  0;
       1  0   0  0;
       1  1   0  0;
       1  1   1  0;
       1  1   1  1;
       1  0   1  1;
       0  1   1  1;
       0  0   1  1;
       1  1   1  1 ];
nV = numel(names);
iV5 = 5; iV5s = 9;

if strcmpi(DATASET_ID, 'all')
    ids = {'Indian','Pavia','HanChuan'};
else
    ids = {DATASET_ID};
end

out = struct();
t_all = tic;

for di = 1:numel(ids)
    cfg = dataset_cfg(ids{di});
    outfile = fullfile(root, sprintf('SuperSDF_Ablation_%s_MC%d.mat', cfg.tag, N_MC));

    L = load(fullfile(root, cfg.file), 'img', 'gt');
    img = double(L.img);
    gt  = double(L.gt);
    clear L
    [rows, cols, bands] = size(img);

    % ---- K-means band grouping ONCE (identical to SuperSDF_MC10.m)
    tk = tic;
    R = corrcoef(reshape(img, rows*cols, bands));
    rng(50,'twister');
    km_labels = kmeans(R, cfg.k, 'Replicates', 1);
    groups = arrayfun(@(i) find(km_labels == i), 1:cfg.k, 'UniformOutput', false);
    kmeans_time = toc(tk);
    clear R
    fprintf('%s: %dx%dx%d, kmeans k=%d done in %.1f s\n', cfg.short, rows, cols, bands, cfg.k, kmeans_time);

    % ---- Feature extraction per variant
    feats = cell(nV,1);
    feat_time = zeros(nV,1);
    n_const = zeros(nV,1);
    for v = 1:8
        opts = struct('useSP',sw(v,1)==1,'useSDF',sw(v,2)==1, ...
                      'useGF',sw(v,3)==1,'useGauss',sw(v,4)==1);
        q = tic;
        [feats{v}, info] = supersdf_features(img, cfg.k, cfg.N_SP, cfg.D, opts, groups);
        feat_time(v) = toc(q);
        n_const(v) = info.n_constant_masks;
        fprintf('%s feat %-3s %7.1f s  ch=%d  constant masks=%d\n', ...
            cfg.short, names{v}, feat_time(v), size(feats{v},3), n_const(v));
    end
    % V5*: prepend group_img of the LAST loop iteration, as MC10 line 175
    q = tic;
    feats{iV5s} = cat(3, mean(img(:,:,groups{cfg.k}), 3), feats{iV5});
    feat_time(iV5s) = feat_time(iV5) + toc(q);
    n_const(iV5s) = n_const(iV5);
    fprintf('%s feat %-3s %7.1f s  ch=%d  (V5 + last-group channel)\n', ...
        cfg.short, names{iV5s}, feat_time(iV5s), size(feats{iV5s},3));
    clear img

    valid_idx = find(gt(:) > 0);
    class_order = unique(gt(valid_idx));
    nc = numel(class_order);

    % ---- Results container (resume if a compatible file exists)
    S = struct();
    S.dataset = cfg.name; S.short = cfg.short; S.tag = cfg.tag; S.cfg = cfg;
    S.protocol = 'Class-wise stratified random split (random_split_data_MC), val = train size, min 5 train/class, rest = test';
    S.N_MC = N_MC; S.seeds = seeds; S.min_train = MIN_TRAIN;
    S.svm = struct('C',SVM_C,'gamma',SVM_GAMMA,'KernelScale',1/sqrt(SVM_GAMMA),'Standardize',true);
    S.fixed = struct('compactness',0.1,'gf_lambda',1e-4,'gauss_sigma',0.95,'gauss_fsize',55,'kmeans_seed',50);
    S.variant_names = names; S.variant_labels = labels; S.switches = sw;
    S.size = [rows cols bands]; S.class_order = class_order;
    S.n_labeled = numel(valid_idx);
    S.groups = groups; S.kmeans_time = kmeans_time;
    S.feat_time = feat_time; S.n_const = n_const;
    S.n_channels = cellfun(@(x) size(x,3), feats);
    S.matlab_version = version; S.date = char(datetime('now','Format','yyyy-MM-dd HH:mm'));
    S.done = false(N_MC,1);
    S.OA = nan(N_MC,nV); S.AA = nan(N_MC,nV); S.kappa = nan(N_MC,nV);
    S.CA = nan(N_MC,nc,nV); S.clf_time = nan(N_MC,nV);
    S.train_counts = zeros(N_MC,nc); S.val_counts = zeros(N_MC,nc); S.test_counts = zeros(N_MC,nc);
    S.idx_identical = false(N_MC,1); S.mc10_equivalent = false(N_MC,1);
    S.idx_hash = zeros(N_MC,2);
    S.pred_run1 = cell(nV,1); S.ytest_run1 = [];
    S.wilcoxon_p = nan(nV,1); S.mcnemar_p = nan(nV,1); S.mcnemar_bc = nan(nV,2);

    if isfile(outfile)
        P = load(outfile);
        if isfield(P,'seeds') && isequal(P.seeds, seeds) && isequal(P.variant_names, names) ...
                && isequal(P.class_order, class_order)
            keep = {'done','OA','AA','kappa','CA','clf_time','train_counts','val_counts', ...
                    'test_counts','idx_identical','mc10_equivalent','idx_hash','pred_run1','ytest_run1'};
            for f = keep, S.(f{1}) = P.(f{1}); end
            fprintf('%s: resuming, %d/%d runs already done\n', cfg.short, sum(S.done), N_MC);
        end
        clear P
    end

    hashf = @(x) mod(sum(double(x(:)) .* (1:numel(x))'), 2^31);

    % ---- Monte-Carlo runs
    for m = 1:N_MC
        if S.done(m), continue; end
        seed = seeds(m);

        % Official split: drawn once, from the V5* cube exactly as MC10 does
        rng(seed,'twister');
        [Xtr_ref, y_tr, ~, y_vl, ~, y_te, ~, tri, vli, tei] = ...
            random_split_data_MC(feats{iV5s}, gt, cfg.train_ratio, MIN_TRAIN, false);

        % Check: the split does not depend on the feature cube
        same = true;
        for v = 1:nV
            rng(seed,'twister');
            [~,~,~,~,~,~,~,a,b,c] = random_split_data_MC(feats{v}, gt, cfg.train_ratio, MIN_TRAIN, false);
            same = same && isequal(a,tri) && isequal(b,vli) && isequal(c,tei);
        end
        assert(same, 'Split indices differ across variants (run %d)', m);

        lin_tr = valid_idx(tri);
        lin_te = valid_idx(tei);
        assert(isequal(gt(lin_tr), y_tr) && isequal(gt(lin_te), y_te), 'Index mapping error');

        A9 = reshape(feats{iV5s}, rows*cols, []);
        S.mc10_equivalent(m) = isequal(A9(lin_tr,:), Xtr_ref);
        clear A9 Xtr_ref

        S.idx_identical(m) = same;
        S.idx_hash(m,:) = [hashf(lin_tr) hashf(lin_te)];
        for ci = 1:nc
            S.train_counts(m,ci) = sum(y_tr == class_order(ci));
            S.val_counts(m,ci)   = sum(y_vl == class_order(ci));
            S.test_counts(m,ci)  = sum(y_te == class_order(ci));
        end
        fprintf('%s run %d/%d seed=%d tr=%d val=%d te=%d | split identical for all %d variants: %s | V5* input == MC10 X_train: %s | hash %d/%d\n', ...
            cfg.short, m, N_MC, seed, numel(lin_tr), numel(vli), numel(lin_te), nV, ...
            yesno(same), yesno(S.mc10_equivalent(m)), S.idx_hash(m,1), S.idx_hash(m,2));

        for v = 1:nV
            A = reshape(feats{v}, rows*cols, []);
            X_train = A(lin_tr,:);
            X_test  = A(lin_te,:);
            clear A

            cls_tic = tic;
            svm_template = templateSVM('KernelFunction','rbf', ...
                'KernelScale',1/sqrt(SVM_GAMMA), 'BoxConstraint',SVM_C, 'Standardize',true);
            model = fitcecoc(X_train, y_tr, 'Learners',svm_template, 'Verbose',0);
            y_pred = predict(model, X_test);
            S.clf_time(m,v) = toc(cls_tic);

            [S.OA(m,v), S.AA(m,v), S.kappa(m,v), ca] = metrics_mc10(y_te, y_pred, class_order);
            S.CA(m,:,v) = ca(:)';
            if m == 1
                S.pred_run1{v} = y_pred;
                S.ytest_run1 = y_te;
            end
            fprintf('%s run %d %-3s OA=%6.2f AA=%6.2f K=%6.2f clf=%.1fs\n', ...
                cfg.short, m, names{v}, S.OA(m,v), S.AA(m,v), S.kappa(m,v), S.clf_time(m,v));
        end

        S.done(m) = true;
        save(outfile, '-struct', 'S');
    end

    % ---- Significance vs V5
    for v = 1:nV
        if v == iV5, continue; end
        if N_MC >= 2
            S.wilcoxon_p(v) = signrank(S.OA(:,v), S.OA(:,iV5));
        end
        [S.mcnemar_p(v), bb, cc] = ablation_mcnemar(S.ytest_run1, S.pred_run1{v}, S.pred_run1{iV5});
        S.mcnemar_bc(v,:) = [bb cc];
    end
    save(outfile, '-struct', 'S');

    fprintf('%s V5 OA=%.2f+-%.2f (paper %.2f+-%.2f, diff %+.2f) | V5* OA=%.2f+-%.2f (diff %+.2f)\n', ...
        cfg.short, mean(S.OA(:,iV5)), std(S.OA(:,iV5)), cfg.paper_OA(1), cfg.paper_OA(2), ...
        mean(S.OA(:,iV5)) - cfg.paper_OA(1), mean(S.OA(:,iV5s)), std(S.OA(:,iV5s)), ...
        mean(S.OA(:,iV5s)) - cfg.paper_OA(1));

    out.(cfg.tag) = S;
    clear feats
end

ablation_write_outputs(root, N_MC);
fprintf('Total runtime %.1f s\n', toc(t_all));
end

% -------------------------------------------------------------------------
function [OA, AA, kappa, ca] = metrics_mc10(y_test, y_pred, class_order)
% Same confusionmat / OA / AA / kappa code as SuperSDF_MC10.m (lines 241-258)
cm = confusionmat(y_test, y_pred, 'Order', class_order);
OA = 100 * trace(cm) / sum(cm(:));
row_totals = sum(cm,2);
ca = 100 * diag(cm) ./ max(row_totals,1);
AA = mean(ca);
N = sum(cm(:));
po = trace(cm) / N;
pe = sum(sum(cm,2) .* sum(cm,1)') / (N^2);
if abs(1-pe) < eps
    kappa = 0;
else
    kappa = 100 * (po-pe) / (1-pe);
end
end

function s = yesno(t)
if t, s = 'YES'; else, s = 'NO'; end
end

function cfg = dataset_cfg(id)
switch lower(id)
    case {'indian','indianpines','ip'}
        cfg = struct('name','Indian Pines','short','IP','tag','IndianPines', ...
            'file','Indian_pines.mat','k',35,'N_SP',60,'D',25,'train_ratio',0.05, ...
            'paper_OA',[97.45 0.53]);
    case {'pavia','paviau','pu'}
        cfg = struct('name','Pavia University','short','PU','tag','PaviaUniversity', ...
            'file','PaviaUF.mat','k',35,'N_SP',240,'D',240,'train_ratio',0.02, ...
            'paper_OA',[99.05 0.23]);
    case {'hanchuan','hc'}
        cfg = struct('name','WHU-Hi-HanChuan','short','HC','tag','HanChuan', ...
            'file','HanChuan.mat','k',35,'N_SP',400,'D',150,'train_ratio',0.01, ...
            'paper_OA',[98.25 0.20]);
    otherwise
        error('Unknown DATASET_ID. Use Indian, Pavia, HanChuan or all.');
end
end
