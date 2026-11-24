function S = p1_run_paired(DATASET_ID, protocol, expName, variants, N_MC)
%P1_RUN_PAIRED Paired Monte-Carlo evaluation of several feature variants.
%   S = p1_run_paired(id, 'random'|'block', expName, variants, N_MC)
%   variants: struct array with fields name, label, opts (p1_features opts).
%   variants(1) is the reference (paper model V5); significance is vs it.
%   In each run the split is drawn ONCE from gt and used for every variant.
%   Random: random_split_data_MC with rng(seed,'twister') (as SuperSDF_MC10).
%   Block : spatial_block_split_data_guarded, geometry fixed per dataset.
%   SVM, metrics and seeds as SuperSDF_MC10.m. Saves after every run and
%   resumes finished runs.

if nargin < 5 || isempty(N_MC), N_MC = 10; end
root = fileparts(mfilename('fullpath'));
cfg = p1_cfg(DATASET_ID);
seeds = (1:N_MC)*7 + 13;
SVM_C = 1e5; SVM_GAMMA = 0.01; MIN_TRAIN = 5;
outfile = fullfile(root, sprintf('SuperSDF_P1_%s_%s_%s_MC%d.mat', expName, cfg.tag, protocol, N_MC));

L = load(fullfile(root, cfg.file), 'img', 'gt');
img = double(L.img); gt = double(L.gt); clear L
[rows, cols, bands] = size(img);

R = corrcoef(reshape(img, rows*cols, bands));
rng(50,'twister');
km = kmeans(R, cfg.k, 'Replicates', 1);
groups = arrayfun(@(i) find(km == i), 1:cfg.k, 'UniformOutput', false);
clear R

nV = numel(variants);
feats = cell(nV,1); ftime = zeros(nV,1); nconst = zeros(nV,1);
Lref = [];
for v = 1:nV
    o = variants(v).opts;
    if v == 1, o.keepL = true; end
    q = tic;
    [feats{v}, info] = p1_features(img, cfg.k, cfg.N_SP, cfg.D, o, groups);
    ftime(v) = toc(q); nconst(v) = info.n_constant_masks;
    if v == 1, Lref = info.L; end
    fprintf('%s %s feat %-10s %6.1f s ch=%d const=%d\n', cfg.short, expName, variants(v).name, ...
        ftime(v), size(feats{v},3), nconst(v));
end
clear img

valid_idx = find(gt(:) > 0);
class_order = unique(gt(valid_idx)); nc = numel(class_order);
ph = zeros(rows, cols, 1);

S = struct('dataset',cfg.name,'short',cfg.short,'tag',cfg.tag,'cfg',cfg,'protocol',protocol, ...
    'expName',expName,'variant_names',{{variants.name}},'variant_labels',{{variants.label}}, ...
    'seeds',seeds,'N_MC',N_MC,'class_order',class_order,'feat_time',ftime,'n_const',nconst, ...
    'n_channels',cellfun(@(x) size(x,3), feats),'groups',{groups}, ...
    'svm',struct('C',SVM_C,'gamma',SVM_GAMMA,'Standardize',true,'coding','onevsone'), ...
    'matlab_version',version,'date',char(datetime('now','Format','yyyy-MM-dd HH:mm')));
S.geometry = cfg.block;
S.done = false(N_MC,1);
S.OA = nan(N_MC,nV); S.AA = nan(N_MC,nV); S.kappa = nan(N_MC,nV);
S.CA = nan(N_MC,nc,nV); S.clf_time = nan(N_MC,nV);
S.train_counts = zeros(N_MC,nc); S.val_counts = zeros(N_MC,nc); S.test_counts = zeros(N_MC,nc);
S.excluded = zeros(N_MC,1);
S.leak = nan(N_MC,4);   % [SLIC sharing mean, test dist min, p05, mean] (px)
S.pred_run1 = cell(nV,1); S.ytest_run1 = [];
S.wilcoxon_p = nan(nV,1); S.mcnemar_p = nan(nV,1);

if isfile(outfile)
    P = load(outfile);
    if isequal(P.variant_names, S.variant_names) && isequal(P.seeds, seeds)
        for f = {'done','OA','AA','kappa','CA','clf_time','train_counts','val_counts', ...
                 'test_counts','excluded','leak','pred_run1','ytest_run1'}
            S.(f{1}) = P.(f{1});
        end
    end
    clear P
end

for m = 1:N_MC
    if S.done(m), continue; end
    seed = seeds(m);
    switch protocol
        case 'random'
            rng(seed,'twister');
            [~,~,~,~,~,~,~,tri,vli,tei] = random_split_data_MC(ph, gt, cfg.train_ratio, MIN_TRAIN, false);
            tr = valid_idx(tri); va = valid_idx(vli); te = valid_idx(tei);
        case 'block'
            g = cfg.block;
            [~,~,~,~,~,~,~,~,~,~,~,inf2] = spatial_block_split_data_guarded(ph, gt, cfg.train_ratio, ...
                g.B, g.gap, MIN_TRAIN, g.minTest, seed, false, struct('quiet',true));
            tr = inf2.fullTrainIndices; va = inf2.fullValIndices; te = inf2.fullTestIndices;
        otherwise
            error('protocol must be random or block');
    end
    y_tr = gt(tr); y_te = gt(te); y_va = gt(va);
    for ci = 1:nc
        S.train_counts(m,ci) = sum(y_tr == class_order(ci));
        S.val_counts(m,ci)   = sum(y_va == class_order(ci));
        S.test_counts(m,ci)  = sum(y_te == class_order(ci));
    end
    S.excluded(m) = numel(valid_idx) - numel(tr) - numel(va) - numel(te);

    % Leakage diagnostics (same definitions for both protocols)
    share = zeros(size(Lref,3),1);
    trva = [tr; va];
    for gi = 1:size(Lref,3)
        Lg = Lref(:,:,gi);
        share(gi) = mean(ismember(Lg(te), unique(Lg(trva))));
    end
    M = false(rows, cols); M(trva) = true;
    dt = bwdist(M); dte = dt(te);
    S.leak(m,:) = [mean(share), min(dte), prctile(dte,5), mean(dte)];

    for v = 1:nV
        A = reshape(feats{v}, rows*cols, []);
        Xtr = A(tr,:); Xte = A(te,:); clear A
        q = tic;
        tpl = templateSVM('KernelFunction','rbf','KernelScale',1/sqrt(SVM_GAMMA), ...
            'BoxConstraint',SVM_C,'Standardize',true);
        mdl = fitcecoc(Xtr, y_tr, 'Learners', tpl, 'Verbose', 0);
        yp = predict(mdl, Xte);
        S.clf_time(m,v) = toc(q);
        [S.OA(m,v), S.AA(m,v), S.kappa(m,v), ca] = p1_metrics(y_te, yp, class_order);
        S.CA(m,:,v) = ca(:)';
        if m == 1, S.pred_run1{v} = yp; S.ytest_run1 = y_te; end
        fprintf('%s %s %s run %d %-10s OA=%6.2f AA=%6.2f K=%6.2f clf=%.1fs\n', cfg.short, expName, ...
            protocol, m, variants(v).name, S.OA(m,v), S.AA(m,v), S.kappa(m,v), S.clf_time(m,v));
    end
    fprintf('%s %s %s run %d seed=%d tr=%d va=%d te=%d excl=%d share=%.3f dmean=%.1fpx\n', cfg.short, ...
        expName, protocol, m, seed, numel(tr), numel(va), numel(te), S.excluded(m), S.leak(m,1), S.leak(m,4));
    S.done(m) = true;
    save(outfile, '-struct', 'S');
end

for v = 2:nV
    if N_MC >= 2, S.wilcoxon_p(v) = signrank(S.OA(:,v), S.OA(:,1)); end
    S.mcnemar_p(v) = ablation_mcnemar(S.ytest_run1, S.pred_run1{v}, S.pred_run1{1});
end
save(outfile, '-struct', 'S');
fprintf('%s %s %s ref %s OA=%.2f+-%.2f\n', cfg.short, expName, protocol, variants(1).name, ...
    mean(S.OA(:,1)), std(S.OA(:,1)));
end
