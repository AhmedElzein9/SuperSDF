% =========================================================================
% SuperSDF_NestedMC10_k35_Contour.m
%
% Reviewer Comments 2 and 9:
% - Full k x N_SP validation grid is evaluated in EVERY outer MC run.
% - Final k is FIXED at 35 for all MC runs.
% - N_SP is selected only from the k=35 row of that run's contour.
% - D is then selected using k=35 and the selected N_SP.
% - The complete SuperSDF pipeline is rerun in every MC run with a
%   controlled run-specific feature seed.
% - Test labels are never used for parameter selection.
%
% Select random pixels, distance-guarded points, or guarded spatial blocks
% below. N_SP and D are reselected using validation labels in every run.
% Whole-scene feature extraction can still share unlabeled spatial context.
% =========================================================================

close all; clc; clear;

%% 0. USER CONFIGURATION
HSI_FILE   = 'HanChuan.mat'; % 'Indian_pines' | 'PaviaUF' | 'HanChuan' | 'LongKou'
DATASET_ID = 'HanChuan';       % 'Indian' | 'Pavia' | 'HanChuan' | 'LongKou'
TARGET_K   = 35;             % fixed final k in every MC run

% Run one combination at a time; use a separate output MAT file for each.
SPLIT_METHOD = 'block';     % 'random' | 'spatial' | 'block'
SP_POOLING   = 'max';        % 'min' | 'max' | 'mean' | 'median'
                            % 'average' is accepted as an alias for 'mean'.
% Split geometry is fixed before looking at classification accuracy.
SPATIAL_TRAIN_VAL_PX = 5;   % train-to-validation centre distance (pixels)  5
SPATIAL_TEST_PX      = 5;   % test-to-(train or validation) distance        5
BLOCK_SIZE           = 20;  % exclusive square tile side length (pixels)    10
BLOCK_BUFFER_PX      = 5;   % minimum between-set centre distance           5
PATCH_RADIUS         = 0;   % spatial splits: largest baseline patch radius 0
                            % random protocol keeps its original sampling
% The Indian Pines Oats class cannot meet 5 train, 5 validation, 10 test
% with a 5-pixel guard; the spatial functions will report infeasibility.

DS.Indian = struct('train_ratio',0.05,'max_bands',200, ...
                   'base_D',25,'nsp_grid',10:10:100);
DS.Pavia = struct('train_ratio',0.02,'max_bands',103, ...
                  'base_D',240,'nsp_grid',50:50:500);
DS.HanChuan = struct('train_ratio',0.01,'max_bands',274, ...
                     'base_D',150,'nsp_grid',50:50:500);
DS.LongKou = struct('train_ratio',0.01,'max_bands',270, ...
                     'base_D',150,'nsp_grid',50:50:500);

cfg = DS.(DATASET_ID);

% Full contour grids
K_GRID   = 35;%10:5:cfg.max_bands;
NSP_GRID = cfg.nsp_grid;

% D search after N_SP selection at k=35
D_GRID = 10:5:400;

% Fixed SuperSDF parameters
SIGMA_GAUSS = 0.95;
GAUSS_FSIZE = 55;
COMPACTNESS = 0.1;
LAMBDA_GF   = 1e-4;

% Fixed SVM parameters (Comment 8 is a separate issue)
SVM_C            = 1e5;
SVM_KERNEL_SCALE = 10;

% Monte Carlo settings
MIN_TRAIN_PER_CLASS = 5;
MIN_TEST_PER_CLASS  = 2;    % used by the two spatial split functions
N_MC = 10;
SPLIT_SEEDS   = (1:N_MC)*7   + 13;      % 20,27,...,83
FEATURE_SEEDS = (1:N_MC)*101 + 1000;    % controlled run-specific seeds

% Figures
MAKE_PER_RUN_CONTOURS = false;
MAKE_MEAN_CONTOUR     = true;
FIG_FONT     = 'Times New Roman';
FIG_FONTSIZE = 11;

SPLIT_METHOD = lower(char(SPLIT_METHOD));
SP_POOLING = lower(char(SP_POOLING));
if strcmp(SP_POOLING,'average'), SP_POOLING = 'mean'; end
if ~any(strcmp(SPLIT_METHOD,{'random','spatial','block'}))
    error('SPLIT_METHOD must be random, spatial, or block.');
end
if ~any(strcmp(SP_POOLING,{'min','max','mean','median'}))
    error('SP_POOLING must be min, max, mean/average, or median.');
end
OUT_FILE = sprintf('SuperSDF_NestedMC10_k35_%s_%s_%s.mat', ...
    DATASET_ID,SPLIT_METHOD,SP_POOLING);

%% 1. LOAD DATA
S = load(HSI_FILE);
if ~isfield(S,'img') || ~isfield(S,'gt')
    error('%s must contain variables img and gt.',HSI_FILE);
end

img = double(S.img);
gt  = S.gt;
[rows,cols,bands] = size(img);

K_GRID = K_GRID(K_GRID <= bands);
if ~ismember(TARGET_K,K_GRID)
    error('TARGET_K=%d must be included in K_GRID.',TARGET_K);
end

class_order = unique(gt(gt>0));
n_classes   = numel(class_order);

fprintf('\n============================================================\n');
fprintf('SuperSDF Nested MC10 - %s\n',DATASET_ID);
fprintf('Split method  : %s\n',SPLIT_METHOD);
fprintf('SP pooling    : %s\n',SP_POOLING);
fprintf('Final k       : %d (fixed)\n',TARGET_K);
fprintf('Contour grid  : %d k values x %d N_SP values\n', ...
    numel(K_GRID),numel(NSP_GRID));
fprintf('D grid        : %d values\n',numel(D_GRID));
fprintf('MC runs       : %d\n',N_MC);
fprintf('============================================================\n');

%% 2. PREALLOCATE RESULTS
OA_mc    = nan(N_MC,1);
AA_mc    = nan(N_MC,1);
Kappa_mc = nan(N_MC,1);
CA_mc    = nan(N_MC,n_classes);

best_nsp_mc     = nan(N_MC,1);
best_D_mc       = nan(N_MC,1);
best_nsp_val_mc = nan(N_MC,1);
best_D_val_mc   = nan(N_MC,1);

train_counts = zeros(N_MC,n_classes);
val_counts   = zeros(N_MC,n_classes);
test_counts  = zeros(N_MC,n_classes);
excluded_counts = zeros(N_MC,n_classes);
split_info_runs = cell(N_MC,1);
run_time     = nan(N_MC,1);

% Each run stores a full validation contour [numK x numNSP]
VAL_CONTOUR_MC = nan(numel(K_GRID),numel(NSP_GRID),N_MC);
VAL_D_MC       = nan(numel(D_GRID),N_MC);

%% 3. OUTER MONTE-CARLO LOOP
for mc = 1:N_MC

    fprintf('\n============================================================\n');
    fprintf('OUTER MC RUN %02d/%02d | split=%d | feature=%d\n', ...
        mc,N_MC,SPLIT_SEEDS(mc),FEATURE_SEEDS(mc));
    fprintf('============================================================\n');

    run_tic = tic;

    %% 3.1 Independent Train / Validation / Test split
    % All methods return FULL-IMAGE MATLAB linear indices here. The original
    % random_split_data split is preserved when SPLIT_METHOD='random'.
    [tr_idx,va_idx,te_idx,split_info] = select_experiment_split( ...
        gt,cfg.train_ratio,MIN_TRAIN_PER_CLASS,MIN_TEST_PER_CLASS, ...
        SPLIT_METHOD,SPLIT_SEEDS(mc),SPATIAL_TRAIN_VAL_PX, ...
        SPATIAL_TEST_PX,BLOCK_SIZE,BLOCK_BUFFER_PX,PATCH_RADIUS);
    split_info_runs{mc} = split_info;

    ytr = gt(tr_idx);
    yva = gt(va_idx);
    yte = gt(te_idx);
    for c = 1:n_classes
        cls = class_order(c);
        train_counts(mc,c) = sum(ytr==cls);
        val_counts(mc,c)   = sum(yva==cls);
        test_counts(mc,c)  = sum(yte==cls);
        excluded_counts(mc,c) = nnz(gt==cls) - train_counts(mc,c) ...
            - val_counts(mc,c) - test_counts(mc,c);
    end

    fprintf('Train / Val / Test = %d / %d / %d\n', ...
        numel(tr_idx),numel(va_idx),numel(te_idx));
    fprintf('Excluded labelled pixels = %d\n',sum(excluded_counts(mc,:)));

    %% 3.2 FULL k x N_SP validation grid for contour
    fprintf('\n[Run %02d] Full validation contour k x N_SP\n',mc);

    for ki = 1:numel(K_GRID)
        k_val = K_GRID(ki);

        for ni = 1:numel(NSP_GRID)
            nsp_val = NSP_GRID(ni);
            fprintf('  k=%-4d N_SP=%-4d ... ',k_val,nsp_val);

            try
                F = extract_sdf_seeded(img,k_val,nsp_val,cfg.base_D, ...
                    SIGMA_GAUSS,GAUSS_FSIZE,COMPACTNESS,LAMBDA_GF, ...
                    FEATURE_SEEDS(mc),SP_POOLING);

                val_oa = evaluate_validation(F,gt,tr_idx,va_idx, ...
                    SVM_C,SVM_KERNEL_SCALE);

                fprintf('Val OA=%.4f%%\n',val_oa);
            catch ME
                warning('Candidate failed k=%d N_SP=%d: %s', ...
                    k_val,nsp_val,ME.message);
                val_oa = NaN;
                fprintf('FAILED\n');
            end

            VAL_CONTOUR_MC(ki,ni,mc) = val_oa;
        end
    end

    %% 3.3 Select N_SP ONLY from the k=35 row
    target_row = find(K_GRID==TARGET_K,1);
    oa_k35 = squeeze(VAL_CONTOUR_MC(target_row,:,mc));

    if all(isnan(oa_k35))
        error('All validation results failed at k=%d in MC run %d.', ...
            TARGET_K,mc);
    end

    [best_nsp_oa,best_nsp_col] = max(oa_k35,[],'omitnan');
    best_nsp = NSP_GRID(best_nsp_col);

    best_nsp_mc(mc)     = best_nsp;
    best_nsp_val_mc(mc) = best_nsp_oa;

    fprintf('\nSelected from k=%d row only:\n',TARGET_K);
    fprintf('  N_SP=%d | Validation OA=%.4f%%\n',best_nsp,best_nsp_oa);

    if MAKE_PER_RUN_CONTOURS
        plot_validation_contour(K_GRID,NSP_GRID,VAL_CONTOUR_MC(:,:,mc), ...
            TARGET_K,best_nsp,sprintf('%s - %s - %s - MC %02d', ...
            DATASET_ID,SPLIT_METHOD,SP_POOLING,mc), ...
            FIG_FONT,FIG_FONTSIZE);
    end

    %% 3.4 D search using fixed k=35 and selected N_SP
    fprintf('\n[Run %02d] D search at k=%d, N_SP=%d\n', ...
        mc,TARGET_K,best_nsp);

    for di = 1:numel(D_GRID)
        D_val = D_GRID(di);
        fprintf('  D=%-4d ... ',D_val);

        try
            F = extract_sdf_seeded(img,TARGET_K,best_nsp,D_val, ...
                SIGMA_GAUSS,GAUSS_FSIZE,COMPACTNESS,LAMBDA_GF, ...
                FEATURE_SEEDS(mc),SP_POOLING);

            val_oa = evaluate_validation(F,gt,tr_idx,va_idx, ...
                SVM_C,SVM_KERNEL_SCALE);

            fprintf('Val OA=%.4f%%\n',val_oa);
        catch ME
            warning('D candidate failed D=%d: %s',D_val,ME.message);
            val_oa = NaN;
            fprintf('FAILED\n');
        end

        VAL_D_MC(di,mc) = val_oa;
    end

    if all(isnan(VAL_D_MC(:,mc)))
        error('All D candidates failed in MC run %d.',mc);
    end

    [best_D_oa,best_D_idx] = max(VAL_D_MC(:,mc),[],'omitnan');
    best_D = D_GRID(best_D_idx);

    best_D_mc(mc)     = best_D;
    best_D_val_mc(mc) = best_D_oa;

    fprintf('\nSelected D:\n');
    fprintf('  k=%d | N_SP=%d | D=%d | Validation OA=%.4f%%\n', ...
        TARGET_K,best_nsp,best_D,best_D_oa);

    %% 3.5 Re-run COMPLETE final pipeline for this MC realization
    F_final = extract_sdf_seeded(img,TARGET_K,best_nsp,best_D, ...
        SIGMA_GAUSS,GAUSS_FSIZE,COMPACTNESS,LAMBDA_GF, ...
        FEATURE_SEEDS(mc),SP_POOLING);

    %% 3.6 Train on TRAIN only and evaluate once on untouched TEST
    Xall = reshape(F_final,rows*cols,TARGET_K);
    Xtr  = Xall(tr_idx,:);
    Xte  = Xall(te_idx,:);

    svm_t = templateSVM('KernelFunction','rbf', ...
        'KernelScale',SVM_KERNEL_SCALE, ...
        'BoxConstraint',SVM_C, ...
        'Standardize',true);

    mdl = fitcecoc(Xtr,ytr,'Learners',svm_t,'Verbose',0);
    ypred = predict(mdl,Xte);

    cm = confusionmat(yte,ypred,'Order',class_order);

    OA_mc(mc) = 100*trace(cm)/sum(cm(:));

    row_totals = sum(cm,2);
    ca = 100*diag(cm)./max(row_totals,1);
    CA_mc(mc,:) = ca(:)';
    AA_mc(mc) = mean(ca);

    N = sum(cm(:));
    po = trace(cm)/N;
    pe = sum(sum(cm,2).*sum(cm,1)')/(N^2);

    if abs(1-pe) < eps
        Kappa_mc(mc) = 0;
    else
        Kappa_mc(mc) = 100*(po-pe)/(1-pe);
    end

    run_time(mc) = toc(run_tic);

    fprintf('\nFINAL TEST MC %02d\n',mc);
    fprintf('  k=%d | N_SP=%d | D=%d\n',TARGET_K,best_nsp,best_D);
    fprintf('  OA=%.2f%% | AA=%.2f%% | Kappa=%.2f%%\n', ...
        OA_mc(mc),AA_mc(mc),Kappa_mc(mc));

    %% 3.7 Checkpoint save
    results = build_results_struct(DATASET_ID,TARGET_K,mc, ...
        SPLIT_SEEDS,FEATURE_SEEDS,OA_mc,AA_mc,Kappa_mc,CA_mc, ...
        best_nsp_mc,best_D_mc,best_nsp_val_mc,best_D_val_mc, ...
        train_counts,val_counts,test_counts,VAL_CONTOUR_MC,VAL_D_MC, ...
        run_time,K_GRID,NSP_GRID,D_GRID,cfg,SIGMA_GAUSS,GAUSS_FSIZE, ...
        COMPACTNESS,LAMBDA_GF,SVM_C,SVM_KERNEL_SCALE);
    results.split_method = SPLIT_METHOD;
    results.sp_pooling = SP_POOLING;
    results.excluded_counts = excluded_counts;
    results.split_info_runs = split_info_runs;
    results.split_settings = struct('spatialTrainValPx',SPATIAL_TRAIN_VAL_PX, ...
        'spatialTestPx',SPATIAL_TEST_PX,'blockSize',BLOCK_SIZE, ...
        'blockBufferPx',BLOCK_BUFFER_PX,'patchRadius',PATCH_RADIUS, ...
        'minTestPerClass',MIN_TEST_PER_CLASS);

    save(OUT_FILE,'results','-v7.3');
end

%% 4. FINAL MC SUMMARY
fprintf('\n============================================================\n');
fprintf('FINAL NESTED MC10 RESULTS - %s\n',DATASET_ID);
fprintf('============================================================\n');
fprintf('Fixed k : %d\n',TARGET_K);
fprintf('OA      : %.2f +/- %.2f %%\n',mean(OA_mc),std(OA_mc));
fprintf('AA      : %.2f +/- %.2f %%\n',mean(AA_mc),std(AA_mc));
fprintf('Kappa   : %.2f +/- %.2f %%\n',mean(Kappa_mc),std(Kappa_mc));

fprintf('\nSelected parameters by run:\n');
for mc = 1:N_MC
    fprintf(['MC %02d: k=%d | N_SP=%d (Val=%.2f%%) | ' ...
             'D=%d (Val=%.2f%%) | Test OA=%.2f%%\n'], ...
        mc,TARGET_K,best_nsp_mc(mc),best_nsp_val_mc(mc), ...
        best_D_mc(mc),best_D_val_mc(mc),OA_mc(mc));
end

%% 5. PARAMETER FREQUENCY
fprintf('\nN_SP selection frequency:\n');
u_nsp = unique(best_nsp_mc);
for i = 1:numel(u_nsp)
    fprintf('  N_SP=%d selected %d/%d runs\n', ...
        u_nsp(i),sum(best_nsp_mc==u_nsp(i)),N_MC);
end

fprintf('\nD selection frequency:\n');
u_D = unique(best_D_mc);
for i = 1:numel(u_D)
    fprintf('  D=%d selected %d/%d runs\n', ...
        u_D(i),sum(best_D_mc==u_D(i)),N_MC);
end

%% 6. MEAN CONTOUR OVER ALL MC RUNS
% Recommended publication contour: average validation OA at each (k,N_SP)
% across the 10 outer MC realizations.
MEAN_CONTOUR = mean(VAL_CONTOUR_MC,3,'omitnan');
STD_CONTOUR  = std(VAL_CONTOUR_MC,0,3,'omitnan');

target_row = find(K_GRID==TARGET_K,1);
mean_oa_k35 = MEAN_CONTOUR(target_row,:);

[mean_best_nsp_oa,idx_nsp] = max(mean_oa_k35,[],'omitnan');
MEAN_BEST_NSP_AT_K35 = NSP_GRID(idx_nsp);

fprintf('\n============================================================\n');
fprintf('AGGREGATED CONTOUR SUMMARY\n');
fprintf('============================================================\n');
fprintf('Fixed final k                    : %d\n',TARGET_K);
fprintf('Best N_SP at k=35, mean contour : %d (Val OA=%.2f%%)\n', ...
    MEAN_BEST_NSP_AT_K35,mean_best_nsp_oa);

if MAKE_MEAN_CONTOUR
    plot_validation_contour(K_GRID,NSP_GRID,MEAN_CONTOUR, ...
        TARGET_K,MEAN_BEST_NSP_AT_K35, ...
        sprintf('%s - %s - %s - Mean Validation OA over %d MC runs', ...
        DATASET_ID,SPLIT_METHOD,SP_POOLING,N_MC), ...
        FIG_FONT,FIG_FONTSIZE);
end

%% 7. AGGREGATED D SEARCH AT THE FINAL k=35 AND FINAL N_SP
% This follows the exact sequence requested for the publication sensitivity
% analysis:
%       full contour -> fix k=35 -> choose best N_SP at k=35 -> sweep D.
%
% The D search below is based ONLY on Train/Validation labels. For every D,
% validation OA is averaged over the same 10 MC Train/Validation splits.
% These aggregated parameters are reported as the recommended/default
% configuration. They are NOT used to recompute the nested MC test results,
% because each outer MC test must remain protected from other runs' tuning.
AGG_D_VAL_MC = nan(numel(D_GRID),N_MC);

fprintf('\nAggregated D search using k=%d and N_SP=%d...\n', ...
    TARGET_K,MEAN_BEST_NSP_AT_K35);

for mc = 1:N_MC

    % Reuse the EXACT indices from this run; never generate another split.
    tr_idx = split_info_runs{mc}.fullTrainIndices;
    va_idx = split_info_runs{mc}.fullValIndices;
    te_idx = split_info_runs{mc}.fullTestIndices;

    ytr = gt(tr_idx);
    yva = gt(va_idx);
    yte = gt(te_idx);
    fprintf('  MC %02d/%02d\n',mc,N_MC);

    for di = 1:numel(D_GRID)
        D_val = D_GRID(di);

        try
            F = extract_sdf_seeded(img,TARGET_K,MEAN_BEST_NSP_AT_K35,D_val, ...
                SIGMA_GAUSS,GAUSS_FSIZE,COMPACTNESS,LAMBDA_GF, ...
                FEATURE_SEEDS(mc),SP_POOLING);

            AGG_D_VAL_MC(di,mc) = evaluate_validation( ...
                F,gt,tr_idx,va_idx,SVM_C,SVM_KERNEL_SCALE);

        catch ME
            warning('Aggregated D candidate failed: MC=%d D=%d: %s', ...
                mc,D_val,ME.message);
            AGG_D_VAL_MC(di,mc) = NaN;
        end
    end
end

MEAN_D_CURVE = mean(AGG_D_VAL_MC,2,'omitnan');
STD_D_CURVE  = std(AGG_D_VAL_MC,0,2,'omitnan');

[mean_best_D_oa,idx_D] = max(MEAN_D_CURVE,[],'omitnan');
MEAN_BEST_D = D_GRID(idx_D);

fprintf('\n============================================================\n');
fprintf('FINAL RECOMMENDED VALIDATION CONFIGURATION\n');
fprintf('============================================================\n');
fprintf('k       : %d\n',TARGET_K);
fprintf('N_SP    : %d\n',MEAN_BEST_NSP_AT_K35);
fprintf('D       : %d\n',MEAN_BEST_D);
fprintf('Mean validation OA at selected D: %.2f%%\n',mean_best_D_oa);
fprintf('============================================================\n');

%% 8. MEAN D CURVE FOR PUBLICATION
figure('Name','Mean_D_Validation','Units','centimeters', ...
       'Position',[2 2 12 8]);
errorbar(D_GRID,MEAN_D_CURVE,STD_D_CURVE,'-o', ...
    'LineWidth',1.5,'MarkerSize',5);
hold on;
plot(MEAN_BEST_D,mean_best_D_oa,'*','MarkerSize',12,'LineWidth',2);
xlabel('Guided Filter Window Size (D)', ...
    'FontName',FIG_FONT,'FontSize',FIG_FONTSIZE);
ylabel('Validation OA (%)', ...
    'FontName',FIG_FONT,'FontSize',FIG_FONTSIZE);
title(sprintf('%s - %s - %s - Mean Validation OA vs D (k=%d, N_{SP}=%d)', ...
    DATASET_ID,SPLIT_METHOD,SP_POOLING,TARGET_K,MEAN_BEST_NSP_AT_K35), ...
    'FontName',FIG_FONT,'FontSize',FIG_FONTSIZE+1);
set(gca,'FontName',FIG_FONT,'FontSize',FIG_FONTSIZE,'Box','on');
grid on;

%% 9. SAVE AGGREGATED RESULTS
results.mean_contour = MEAN_CONTOUR;
results.std_contour  = STD_CONTOUR;
results.mean_best_NSP_at_k35 = MEAN_BEST_NSP_AT_K35;
results.mean_best_NSP_val_OA = mean_best_nsp_oa;
results.aggregated_D_validation_runs = AGG_D_VAL_MC;
results.mean_D_curve = MEAN_D_CURVE;
results.std_D_curve  = STD_D_CURVE;
results.mean_best_D  = MEAN_BEST_D;
results.mean_best_D_val_OA = mean_best_D_oa;
results.recommended_parameters = struct( ...
    'k',TARGET_K,'N_SP',MEAN_BEST_NSP_AT_K35,'D',MEAN_BEST_D);
save(OUT_FILE,'results','-v7.3');

fprintf('\nFinal results saved to: %s\n',OUT_FILE);

%% ========================================================================
% LOCAL FUNCTION 0: SELECT ONE REPRODUCIBLE TRAIN / VAL / TEST SPLIT
% =========================================================================
function [tr_idx,va_idx,te_idx,split_info] = select_experiment_split( ...
    gt,ratio,min_train,min_test,method,split_seed, ...
    train_val_px,test_px,block_size,block_buffer_px,patch_radius)

    [rows,cols] = size(gt);
    % The split functions use only gt for selecting pixels. They require
    % this placeholder solely to preserve the original 11-output interface.
    feat_placeholder = zeros(rows,cols,1);
    opts = struct('patchRadius',patch_radius,'quiet',true);

    switch method
        case 'random'
            % Keep the original random-split RNG sequence unchanged.
            rng(split_seed,'twister');
            [~,~,~,~,~,~,~,tr_local,va_local,te_local,~] = ...
                random_split_data(feat_placeholder,gt,ratio,min_train,false);
            split_info = struct('method','random pixel', ...
                'seed',split_seed,'relaxed',[]);

        case 'spatial'
            [~,~,~,~,~,~,~,tr_local,va_local,te_local,~,split_info] = ...
                spatial_split_data4(feat_placeholder,gt,ratio, ...
                train_val_px,test_px,min_train,min_test,false,split_seed,opts);

        case 'block'
            opts.allowRelaxation = false; % Never change geometry per MC seed.
            [~,~,~,~,~,~,~,tr_local,va_local,te_local,~,split_info] = ...
                spatial_block_split_data_guarded(feat_placeholder,gt,ratio, ...
                block_size,block_buffer_px,min_train,min_test, ...
                split_seed,false,opts);

        otherwise
            error('Unknown SPLIT_METHOD: %s.',method);
    end

    % The split functions return positions into find(gt(:)>0).
    labelled = find(gt(:)>0);
    tr_idx = labelled(tr_local(:));
    va_idx = labelled(va_local(:));
    te_idx = labelled(te_local(:));

    split_info.fullTrainIndices = tr_idx;
    split_info.fullValIndices = va_idx;
    split_info.fullTestIndices = te_idx;
end

%% ========================================================================
% LOCAL FUNCTION 1: COMPLETE SUPERSDF EXTRACTION
% =========================================================================
function sdf_out = extract_sdf_seeded(img,k,N_SP,D,sigma,fsize, ...
                                      compact,lambda,feature_seed,pooling_method)
    [rows,cols,bands] = size(img);
    X = reshape(img,rows*cols,bands);
    R = corrcoef(X);

    rng(feature_seed,'twister');
    [band_labels,~] = kmeans(R,k,'Replicates',1,'Display','off');

    groups = cell(k,1);
    for i = 1:k
        groups{i} = find(band_labels==i);
        if isempty(groups{i})
            error('Empty spectral group for k=%d.',k);
        end
    end

    sdf_out = zeros(rows,cols,k);

    for i = 1:k
        group_img = mean(img(:,:,groups{i}),3);

        lo = min(group_img(:));
        hi = max(group_img(:));
        if hi > lo
            normalized_img = (group_img-lo)/(hi-lo);
        else
            normalized_img = zeros(size(group_img));
        end

        [L,N] = superpixels(normalized_img,N_SP, ...
            'Method','slic','Compactness',compact);

        idx_lists = label2idx(L);
        sp_pooled = zeros(size(normalized_img));
        for s = 1:N
            idx = idx_lists{s};
            values = normalized_img(idx);
            switch pooling_method
                case 'min'
                    pooled_value = min(values);
                case 'max'
                    pooled_value = max(values);
                case 'mean'
                    pooled_value = mean(values);
                case 'median'
                    pooled_value = median(values);
                otherwise
                    error('Unknown SP_POOLING: %s.',pooling_method);
            end
            sp_pooled(idx) = pooled_value;
        end

        % Preserve the original Otsu-derived ADAPTIVE sensitivity. This is
        % not equivalent to global Otsu binarization with imbinarize(I,T).
        sensitivity = graythresh(sp_pooled);
        binary_img = imbinarize(sp_pooled,'adaptive', ...
            'ForegroundPolarity','bright','Sensitivity',sensitivity);

        if all(binary_img(:)==0) || all(binary_img(:)==1)
            sdf = zeros(size(binary_img));
        else
            sdf = bwdist(~binary_img)-bwdist(binary_img);
        end

        sdf = imguidedfilter(sdf,group_img, ...
            'NeighborhoodSize',[D D], ...
            'DegreeOfSmoothing',lambda);

        sdf = imgaussfilt(sdf,sigma,'FilterSize',fsize);
        sdf_out(:,:,i) = sdf;
    end
end

%% ========================================================================
% LOCAL FUNCTION 2: VALIDATION-ONLY PERFORMANCE
% =========================================================================
function oa = evaluate_validation(sdf_cube,gt,tr_idx,va_idx,C,kernel_scale)
    [rows,cols,k] = size(sdf_cube);
    Xall = reshape(sdf_cube,rows*cols,k);

    Xtr = Xall(tr_idx,:);
    Xva = Xall(va_idx,:);
    ytr = gt(tr_idx);
    yva = gt(va_idx);

    svm_t = templateSVM('KernelFunction','rbf', ...
        'KernelScale',kernel_scale, ...
        'BoxConstraint',C, ...
        'Standardize',true);

    mdl = fitcecoc(Xtr,ytr,'Learners',svm_t,'Verbose',0);
    pred = predict(mdl,Xva);
    oa = 100*sum(pred==yva)/numel(yva);
end

%% ========================================================================
% % LOCAL FUNCTION 3: STRATIFIED RANDOM TRAIN / VAL / TEST SPLIT
% % =========================================================================
% function [tr_idx,va_idx,te_idx] = stratified_split_indices( ...
%     gt,train_ratio,min_train_per_class,split_seed)
% 
%     rng(split_seed,'twister');
%     labels = unique(gt(:));
%     labels(labels==0) = [];
% 
%     tr_idx = [];
%     va_idx = [];
%     te_idx = [];
% 
%     for ci = 1:numel(labels)
%         cls = labels(ci);
%         idx = find(gt(:)==cls);
%         n = numel(idx);
% 
%         if n < 3
%             error('Class %d has fewer than 3 labeled pixels.',cls);
%         end
% 
%         n_train = max(min_train_per_class,round(train_ratio*n));
%         n_val   = n_train;
% 
%         while n_train+n_val >= n && n_val > 1
%             n_val = n_val-1;
%         end
%         while n_train+n_val >= n && n_train > 1
%             n_train = n_train-1;
%         end
% 
%         if n_train+n_val >= n
%             error('Class %d has insufficient samples for train/val/test.',cls);
%         end
% 
%         p = idx(randperm(n));
%         tr_idx = [tr_idx; p(1:n_train)]; %#ok<AGROW>
%         va_idx = [va_idx; p(n_train+1:n_train+n_val)]; %#ok<AGROW>
%         te_idx = [te_idx; p(n_train+n_val+1:end)]; %#ok<AGROW>
%     end
% 
%     tr_idx = tr_idx(randperm(numel(tr_idx)));
%     va_idx = va_idx(randperm(numel(va_idx)));
%     te_idx = te_idx(randperm(numel(te_idx)));
% end

%% ========================================================================
% LOCAL FUNCTION 4: CONTOUR PLOT
% =========================================================================
function plot_validation_contour(K_GRID,NSP_GRID,Z,TARGET_K,selected_NSP, ...
                                 plot_title,font_name,font_size)
    [Kmesh,Nmesh] = meshgrid(K_GRID,NSP_GRID);
    Zplot = Z';

    figure('Name','Validation_Contour','Units','centimeters', ...
           'Position',[2 2 16 9]);
    contourf(Kmesh,Nmesh,Zplot,20,'LineStyle','none');
    colorbar;
    hold on;
    plot(TARGET_K,selected_NSP,'w*','MarkerSize',14,'LineWidth',2.5);

    xlabel('Number of Spectral Groups (k)', ...
        'FontName',font_name,'FontSize',font_size);
    ylabel('Number of Superpixels (N_{SP})', ...
        'FontName',font_name,'FontSize',font_size);
    title(plot_title,'FontName',font_name,'FontSize',font_size+1);
    set(gca,'FontName',font_name,'FontSize',font_size,'Box','on');
    grid on;
end

%% ========================================================================
% LOCAL FUNCTION 5: RESULTS STRUCT
% =========================================================================
function results = build_results_struct(dataset_id,target_k,completed_runs, ...
    split_seeds,feature_seeds,OA,AA,Kappa,CA,best_nsp,best_D, ...
    best_nsp_val,best_D_val,train_counts,val_counts,test_counts, ...
    val_contour_mc,val_D_mc,run_time,K_GRID,NSP_GRID,D_GRID,cfg, ...
    sigma,fsize,compactness,lambda,svm_c,svm_kernel_scale)

    results = struct();
    results.dataset_id = dataset_id;
    results.target_k = target_k;
    results.completed_runs = completed_runs;
    results.split_seeds = split_seeds;
    results.feature_seeds = feature_seeds;

    results.OA_runs = OA;
    results.AA_runs = AA;
    results.kappa_runs = Kappa;
    results.class_accuracy_runs = CA;

    results.best_NSP_runs = best_nsp;
    results.best_D_runs = best_D;
    results.best_NSP_validation_OA = best_nsp_val;
    results.best_D_validation_OA = best_D_val;

    results.train_counts = train_counts;
    results.val_counts = val_counts;
    results.test_counts = test_counts;

    results.validation_contour_runs = val_contour_mc;
    results.validation_D_runs = val_D_mc;
    results.run_time_sec = run_time;

    results.K_GRID = K_GRID;
    results.NSP_GRID = NSP_GRID;
    results.D_GRID = D_GRID;
    results.config = cfg;

    results.sigma_gauss = sigma;
    results.gauss_fsize = fsize;
    results.compactness = compactness;
    results.lambda_guided = lambda;
    results.svm_C = svm_c;
    results.svm_kernel_scale = svm_kernel_scale;

    done = 1:completed_runs;
    results.OA_mean = mean(OA(done),'omitnan');
    results.OA_std = std(OA(done),'omitnan');
    results.AA_mean = mean(AA(done),'omitnan');
    results.AA_std = std(AA(done),'omitnan');
    results.kappa_mean = mean(Kappa(done),'omitnan');
    results.kappa_std = std(Kappa(done),'omitnan');
end
