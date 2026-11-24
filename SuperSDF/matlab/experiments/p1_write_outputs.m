function p1_write_outputs(root, N_MC)
%P1_WRITE_OUTPUTS Build SuperSDF_Phase1_Results.xlsx from the saved P1 .mat files.
if nargin < 1 || isempty(root), root = fileparts(mfilename('fullpath')); end
if nargin < 2 || isempty(N_MC), N_MC = 10; end
tags = {'IndianPines','PaviaUniversity','HanChuan'};
xfile = fullfile(root, 'SuperSDF_Phase1_Results.xlsx');
if isfile(xfile), xfile = fullfile(root, sprintf('SuperSDF_Phase1_Results_%s.xlsx', ...
        char(datetime('now','Format','yyyyMMdd_HHmm')))); end
PM = char(177);
f2 = @(x) sprintf('%.2f %s %.2f', mean(x), PM, std(x));
ld = @(n) loadif(fullfile(root, n));

% ---- Variants (E1 pooling, E2 binarization, E6a complement) and E10 block ablation
for spec = {{'VAR','random','Variants'}, {'ABL','block','E10_Block'}}
    s = spec{1}; T = {};
    for d = 1:3
        R = ld(sprintf('SuperSDF_P1_%s_%s_%s_MC%d.mat', s{1}, tags{d}, s{2}, N_MC));
        if isempty(R) || ~all(R.done), continue; end
        for v = 1:numel(R.variant_names)
            T(end+1,:) = {R.short, R.variant_names{v}, R.variant_labels{v}, R.n_channels(v), R.n_const(v), ...
                f2(R.OA(:,v)), f2(R.AA(:,v)), f2(R.kappa(:,v)), ...
                mean(R.OA(:,v)), std(R.OA(:,v)), mean(R.OA(:,v)) - mean(R.OA(:,1)), ...
                R.wilcoxon_p(v), R.mcnemar_p(v), R.feat_time(v), mean(R.clf_time(:,v))}; %#ok<AGROW>
        end
    end
    if ~isempty(T)
        writecell([{'Dataset','Variant','Description','Channels','ConstantMasks','OA','AA','Kappa', ...
            'OA_mean','OA_std','dOA_vs_V5','Wilcoxon_p_vs_V5','McNemar_p_run1_vs_V5','FeatTime_s','SVMTime_s'}; T], ...
            xfile, 'Sheet', s{3});
    end
end

% ---- Per-run OA, per-class (V5) for both protocols, counts, leakage
RUNS = {}; PC = {}; CNT = {}; LK = {};
for spec = {{'VAR','random'}, {'ABL','block'}}
    s = spec{1};
    for d = 1:3
        R = ld(sprintf('SuperSDF_P1_%s_%s_%s_MC%d.mat', s{1}, tags{d}, s{2}, N_MC));
        if isempty(R) || ~all(R.done), continue; end
        for v = 1:numel(R.variant_names)
            for m = 1:R.N_MC
                RUNS(end+1,:) = {R.short, s{2}, R.variant_names{v}, R.seeds(m), R.OA(m,v), R.AA(m,v), R.kappa(m,v)}; %#ok<AGROW>
            end
        end
        for c = 1:numel(R.class_order)
            ca = squeeze(R.CA(:,c,1));
            PC(end+1,:) = {R.short, s{2}, R.class_order(c), f2(ca), mean(ca), std(ca)}; %#ok<AGROW>
            tr = R.train_counts(:,c); va = R.val_counts(:,c); te = R.test_counts(:,c);
            tot = tr(1) + va(1) + te(1);
            CNT(end+1,:) = {R.short, s{2}, R.class_order(c), min(tr), max(tr), min(va), max(va), ...
                min(te), max(te), mean(tr), mean(te), ...
                100*mean(tr ./ (tr + va + te + 0)), numel(unique(tr)) == 1}; %#ok<AGROW>
        end
        g = R.geometry;
        LK(end+1,:) = {R.short, s{2}, g.B, g.gap, g.minTest, mean(R.leak(:,1)), min(R.leak(:,2)), ...
            mean(R.leak(:,3)), mean(R.leak(:,4)), mean(R.excluded), ...
            mean(sum(R.test_counts,2)), mean(sum(R.train_counts,2))}; %#ok<AGROW>
    end
end
if ~isempty(RUNS), writecell([{'Dataset','Protocol','Variant','Seed','OA','AA','Kappa'}; RUNS], xfile, 'Sheet', 'Runs'); end
if ~isempty(PC), writecell([{'Dataset','Protocol','Class','V5_acc','Mean','Std'}; PC], xfile, 'Sheet', 'PerClass_V5'); end
if ~isempty(CNT)
    writecell([{'Dataset','Protocol','Class','Train_min','Train_max','Val_min','Val_max','Test_min','Test_max', ...
        'Train_mean','Test_mean','Train_pct_of_retained_labels','Train_count_constant'}; CNT], xfile, 'Sheet', 'E8_Counts');
end
if ~isempty(LK)
    writecell([{'Dataset','Protocol','Block_B (block only)','Gap_px (block only)','MinTest (block only)', ...
        'SLIC_sharing_mean','TestDist_min_px','TestDist_p05_px','TestDist_mean_px','Excluded_labels', ...
        'Test_pixels_mean','Train_pixels_mean'}; LK], xfile, 'Sheet', 'Leakage');
end

% ---- E3
T = {};
for d = 1:3
    R = ld(sprintf('SuperSDF_P1_E3_%s_random_MC%d.mat', tags{d}, N_MC));
    V = ld(sprintf('SuperSDF_P1_VAR_%s_random_MC%d.mat', tags{d}, N_MC));
    if isempty(R), continue; end
    p = NaN; ref = ''; dd = NaN;
    if ~isempty(V) && all(V.done)
        p = signrank(R.OA, V.OA(:,1)); ref = f2(V.OA(:,1)); dd = mean(R.OA) - mean(V.OA(:,1));
    end
    T(end+1,:) = {R.short, f2(R.OA), f2(R.AA), f2(R.kappa), ref, dd, p, ...
        mean(R.ARI_vs_seed50), min(R.ARI_vs_seed50), sum(R.same_grouping_as_seed50), ...
        R.slic_deterministic, mat2str(R.feat_seeds)}; %#ok<AGROW>
end
if ~isempty(T)
    writecell([{'Dataset','OA_varied_kmeans_seed','AA','Kappa','OA_fixed_seed50 (V5)','dOA','Wilcoxon_p', ...
        'Mean_ARI_vs_seed50','Min_ARI','Runs_identical_grouping','SLIC_deterministic','kmeans_seeds'}; T], ...
        xfile, 'Sheet', 'E3_KmeansSeed');
end

% ---- E6b
T = {};
for d = 1:3
    R = ld(sprintf('SuperSDF_P1_E6b_%s.mat', tags{d}));
    if isempty(R), continue; end
    for v = 1:numel(R.names)
        T(end+1,:) = {R.short, R.names{v}, R.J(v), R.fisher_mean(v), R.nn1_loo(v), '', '', '', ''}; %#ok<AGROW>
    end
    T(end+1,:) = {R.short, 'Binary code of V5 masks', NaN, NaN, NaN, R.n_words, R.H_ratio, ...
        R.pair_hamming_min, R.pair_hamming_median}; %#ok<AGROW>
end
if ~isempty(T)
    writecell([{'Dataset','Features','J_trSb_over_trSw','MeanFisherRatio','1NN_LOO_acc', ...
        'Distinct_codewords','H(class|code)/H(class)','Min_pair_Hamming','Median_pair_Hamming'}; T], ...
        xfile, 'Sheet', 'E6b_Separability');
end

% ---- E7
T = {};
for d = 1:3
    R = ld(sprintf('SuperSDF_P1_E7_%s.mat', tags{d}));
    if isempty(R), continue; end
    T(end+1,:) = [{R.short, sprintf('%dx%dx%d', R.size)}, num2cell(R.stage_median), ...
        {R.feature_total_median, R.svm_train_median, R.svm_predict_median, R.n_train, R.n_test, ...
        R.cpu, R.n_threads, R.matlab_version}]; %#ok<AGROW>
end
if ~isempty(T)
    writecell([{'Dataset','Size','kmeans_s','groupmean_s','SLIC_s','pooling_s','binarize_s','distance_s', ...
        'guided_s','gauss_s','feature_total_s','SVM_train_s','SVM_predict_s','n_train','n_test', ...
        'CPU','threads','MATLAB'}; T], xfile, 'Sheet', 'E7_Timing');
end

% ---- Config
C = {'Parameters','Paper values from SuperSDF_MC10.m: IP (k,N_SP,D)=(35,60,25); PU (35,240,240); HC (35,400,150)';
     'Fixed','SLIC compactness 0.1; guided filter DegreeOfSmoothing 1e-4 (guide = unnormalized I_k); Gaussian sigma 0.95, 55x55';
     'k-means','corrcoef rows, k=35, Replicates 1, MATLAB default k-means++ init, rng(50,''twister''); E3 uses seeds (1:10)*101+1000';
     'SVM','fitcecoc one-vs-one, RBF, KernelScale 10 (gamma 0.01), BoxConstraint 1e5, Standardize true';
     'Split seeds','(1:10)*7+13, rng(seed,''twister'') before random_split_data_MC; block split gets the seed explicitly';
     'Random protocol','random_split_data_MC: n_train=max(5,round(r*N_c)), n_val=n_train, rest test';
     'Block protocol','spatial_block_split_data_guarded: PU/HC B=20, gap=5, min test 5; IP B=5, gap=2, min test 2 (B=20 infeasible on IP)';
     'Paired design','one split per run, same train/test pixels for every variant';
     'Leakage metrics','SLIC sharing = mean over 35 groups of the fraction of test pixels whose V5 superpixel contains a train/val pixel; test distance = Euclidean px to nearest train/val pixel';
     'Date', char(datetime('now','Format','yyyy-MM-dd HH:mm')); 'MATLAB', version};
writecell(C, xfile, 'Sheet', 'Config');
fprintf('Wrote %s\n', xfile);
end

function R = loadif(f)
if isfile(f), R = load(f); else, R = []; end
end
