% =========================================================================
% demo_S3_PCA_MC10.m
%
% Modified from the original working demo_S3_PCA.m.
%
% IMPORTANT:
%   - S^3-PCA feature extraction and its fixed paper parameters are unchanged.
%   - Only the evaluation protocol is changed to 10 Monte-Carlo runs.
%   - random_split_data is used so the split protocol matches SuperSDF_New.
%   - The same MC seeds as SuperSDF_New are used: rng(mc*7 + 13).
%   - OA, AA, Kappa, and per-class accuracy are computed on TEST samples only.
%   - The full classification map is for visualization only.
%
% Fixed S^3-PCA parameters from the original demo:
%   Indian : num_PC=30, num_Pixel=75,  k=15
%   PaviaU : num_PC=30, num_Pixel=30,  k=13
%   Salinas: num_PC=30, num_Pixel=100, k=15
% =========================================================================

clc;
clear;
close all;

addpath(fullfile(fileparts(mfilename('fullpath')),'..','..','..','data'));  % repository data folder
addpath(genpath(cd));

%% =========================================================================
% 1. ORIGINAL S^3-PCA PARAMETERS
% =========================================================================
num_PC  = 35;      % Original optimal PCA dimension
N_MC    = 2;      % Monte-Carlo runs
MIN_TRAIN = 5;     % Same minimum-per-class rule as SuperSDF_New

database = 'HanChuan';   % 'Indian' | 'PaviaU' | 'Salinas' | 'HanChuan'

%% =========================================================================
% 2. LOAD DATASET + ORIGINAL FIXED PARAMETERS
% =========================================================================
if strcmp(database,'Indian')

    load Indian_pines_corrected;
    load Indian_pines_gt;
    load Indian_pines_randp;

    data3D   = indian_pines_corrected;
    label_gt = indian_pines_gt;

    num_Pixel  = 75;
    k          = 15;

    % Same label budget currently used by SuperSDF_New
    train_ratio = 0.05;

elseif strcmp(database,'Salinas')

    load Salinas_corrected;
    load Salinas_gt;
    load Salinas_randp;

    data3D   = salinas_corrected;
    label_gt = salinas_gt;

    num_Pixel  = 100;
    k          = 15;

    % Same label budget currently used by SuperSDF_New
    train_ratio = 0.01;

elseif strcmp(database,'PaviaU')

    S = load('PaviaUF');
    data3D   = S.img;
    label_gt = S.gt;

    num_Pixel  = 30;
    k          = 13;

    % Same label budget currently used by SuperSDF_New
    train_ratio = 0.02;
elseif strcmp(database,'LongKou')

    S = load('LongKou.mat');
    if ~isfield(S,'img') || ~isfield(S,'gt')
        error('LongKou.mat must contain variables img and gt.');
    end

    data3D   = S.img;
    label_gt = S.gt;

    num_PC      = 35;
    num_Pixel   = 15;
    k           = 15;
    train_ratio = 0.01;
elseif strcmp(database,'HanChuan')

    S = load('HanChuan.mat');
    if ~isfield(S,'img') || ~isfield(S,'gt')
        error('HanChuan.mat must contain variables img and gt.');
    end

    data3D   = S.img;
    label_gt = S.gt;

    num_PC      = 35;
    num_Pixel   = 15;
    k           = 15;
    train_ratio = 0.05;

else
    
    error('Unsupported database: %s', database);
end

data3D = double(data3D);
label_gt = double(label_gt);

% Original demo normalization
data3D = data3D ./ max(data3D(:));

[rows, cols, bands] = size(data3D);

fprintf('\n============================================================\n');
fprintf(' S^3-PCA - ORIGINAL PARAMETERS + 10 MC RUNS\n');
fprintf('============================================================\n');
fprintf(' Dataset       : %s\n', database);
fprintf(' Image         : %d x %d x %d\n', rows, cols, bands);
fprintf(' num_PC        : %d\n', num_PC);
fprintf(' num_Pixel     : %d\n', num_Pixel);
fprintf(' spatial k     : %d\n', k);
fprintf(' train ratio   : %.2f%% per class\n', train_ratio*100);
fprintf(' MC runs       : %d\n', N_MC);
fprintf('============================================================\n\n');

%% =========================================================================
% 3. ORIGINAL S^3-PCA FEATURE EXTRACTION
%    DO NOT CHANGE THIS PART IF YOU WANT THE ORIGINAL METHOD.
% =========================================================================
fprintf('Running original S^3-PCA feature extraction...\n');

feature_start = tic;

% Fix feature-extraction randomness so all MC runs use the SAME representation.
% Restore RNG afterwards so MC split seeds are unaffected.
rng_before_feature = rng;
rng(50, 'twister');

% Original demo:
labels = cubseg(data3D, num_Pixel);
dataDR = S3_PCA(data3D, num_PC, k, labels);

rng(rng_before_feature);

feature_time = toc(feature_start);

[rows_dr, cols_dr, num_features] = size(dataDR);

if rows_dr ~= rows || cols_dr ~= cols
    error('S3_PCA output dimensions do not match the original image.');
end

fprintf('S^3-PCA feature extraction complete: %.2f s\n', feature_time);
fprintf('Reduced feature dimensions: %d\n\n', num_features);

%% =========================================================================
% 4. LABEL INFORMATION
% =========================================================================
valid_mask = label_gt(:) > 0;
valid_gt   = label_gt(valid_mask);

unique_classes = unique(valid_gt);
num_classes    = numel(unique_classes);

%% =========================================================================
% 5. MONTE-CARLO EVALUATION
% =========================================================================
OA_all       = zeros(N_MC,1);
AA_all       = zeros(N_MC,1);
kappa_all    = zeros(N_MC,1);
time_all     = zeros(N_MC,1);
classAcc_all = zeros(N_MC,num_classes);

fprintf('============================================================\n');
fprintf(' MONTE-CARLO TEST EVALUATION\n');
fprintf('============================================================\n');

for mc = 1:N_MC

    seed = mc*7 + 13;       % Same seed sequence as SuperSDF_New
    rng(seed, 'twister');

    run_start = tic;

    % Stratified random train / validation / test split.
    % Validation samples are NOT used for final test metrics.
    % [X_train, y_train, X_val, y_val, X_test, y_test, ...
    %     sample_map, train_indices, val_indices, test_indices, X_all_raw] = ...
    %     random_split_data(dataDR, label_gt, ...
    %                       train_ratio, MIN_TRAIN, false); %#ok<ASGLU>
    [X_train, y_train, X_val, y_val, X_test, y_test, sample_map, train_indices, val_indices, test_indices, X_all_raw, split_info] = ...
        spatial_block_split_data_guarded(dataDR, label_gt,train_ratio,20,5,5,5,seed,true);
    run_start = tic;

    % Same RBF-SVM parameters as the original working demo
    svm_template = templateSVM( ...
        'KernelFunction', 'rbf', ...
        'KernelScale', 1/sqrt(0.01), ...
        'BoxConstraint', 100000, ...
        'Standardize', true);

    model = fitcecoc(X_train, y_train, ...
        'Learners', svm_template, ...
        'Verbose', 0);

    % ---------------------------------------------------------
    % TEST SET ONLY
    % ---------------------------------------------------------
    y_test_pred = predict(model, X_test);
    time_all(mc) = toc(run_start);
    cm = confusionmat(y_test, y_test_pred, ...
        'Order', unique_classes);

    % OA
    OA_all(mc) = trace(cm) / sum(cm(:));

    % Per-class accuracy
    for c = 1:num_classes
        denom = sum(cm(c,:));

        if denom > 0
            classAcc_all(mc,c) = cm(c,c) / denom;
        else
            classAcc_all(mc,c) = 0;
        end
    end

    % AA
    AA_all(mc) = mean(classAcc_all(mc,:));

    % Kappa
    N = sum(cm(:));

    row_sums = sum(cm,2);
    col_sums = sum(cm,1)';

    po = trace(cm) / N;
    pe = sum(row_sums .* col_sums) / (N^2);

    if abs(1-pe) > eps
        kappa_all(mc) = (po-pe)/(1-pe);
    else
        kappa_all(mc) = 0;
    end

    

    fprintf(['Run %2d/%d | seed=%d | Train=%d | Val=%d | Test=%d | ', ...
             'OA=%.2f%% | AA=%.2f%% | Kappa=%.2f%% | %.2f s\n'], ...
             mc, N_MC, seed, ...
             numel(y_train), numel(y_val), numel(y_test), ...
             OA_all(mc)*100, AA_all(mc)*100, ...
             kappa_all(mc)*100, time_all(mc));
end

%% =========================================================================
% 6. FINAL 10-RUN RESULTS
% =========================================================================
fprintf('\n============================================================\n');
fprintf(' FINAL S^3-PCA RESULTS OVER %d RUNS - %s\n', N_MC, database);
fprintf('============================================================\n');

fprintf('\n-- Per-Class Accuracy --\n');

for c = 1:num_classes
    fprintf('Class %2d: %.4f +/- %.4f %%\n', ...
        unique_classes(c), ...
        mean(classAcc_all(:,c))*100, ...
        std(classAcc_all(:,c))*100);
end

fprintf('\n------------------------------------------------------------\n');
fprintf('Overall Accuracy : %.4f +/- %.4f %%\n', ...
    mean(OA_all)*100, std(OA_all)*100);

fprintf('Average Accuracy : %.4f +/- %.4f %%\n', ...
    mean(AA_all)*100, std(AA_all)*100);

fprintf('Kappa Coefficient: %.4f +/- %.4f %%\n', ...
    mean(kappa_all)*100, std(kappa_all)*100);

fprintf('------------------------------------------------------------\n');
fprintf('Feature extraction time : %.2f s\n', feature_time);
fprintf('MC average time/run     : %.2f +/- %.2f s\n', ...
    mean(time_all), std(time_all));
fprintf('MC total time           : %.2f s\n', sum(time_all));
fprintf('============================================================\n');

%% =========================================================================
% 7. OPTIONAL BOXPLOT
% =========================================================================
figure('Name','S3-PCA Monte Carlo Results');
boxplot([OA_all, AA_all, kappa_all], ...
    'Labels', {'OA','AA','Kappa'});
ylabel('Score');
title(sprintf('S^3-PCA Monte Carlo (%d runs) - %s', N_MC, database));
grid on;

%% =========================================================================
% 8. ONE REFERENCE CLASSIFICATION MAP
%    VISUALIZATION ONLY - NOT USED FOR OA / AA / KAPPA
% =========================================================================
fprintf('\nGenerating one reference classification map...\n');

rng(50, 'twister');

% [X_train_map, y_train_map, ~, ~, ~, ~, ...
%     ~, ~, ~, ~, X_all_raw_map] = ...
%     random_split_data(dataDR, label_gt, ...
%                       train_ratio, MIN_TRAIN, false);
[X_train_map, y_train_map, X_val, y_val, X_test, y_test, sample_map, train_indices, val_indices, test_indices, X_all_raw_map, split_info] = ...
    spatial_block_split_data_guarded(dataDR, label_gt,train_ratio,20,5,5,10,seed,true);

svm_template_map = templateSVM( ...
    'KernelFunction', 'rbf', ...
    'KernelScale', 1/sqrt(0.01), ...
    'BoxConstraint', 100000, ...
    'Standardize', true);

model_map = fitcecoc(X_train_map, y_train_map, ...
    'Learners', svm_template_map, ...
    'Verbose', 0);

y_pred_img = predict(model_map, X_all_raw_map);

assert(numel(y_pred_img) == rows*cols, ...
    'Prediction count must equal rows*cols.');

classification_map = reshape(y_pred_img, rows, cols);
classification_map(label_gt == 0) = 0;

num_classes_map = double(max(label_gt(:)));
full_colormap = [0 0 0; jet(num_classes_map)];

figure('Name','Ground Truth');
imagesc(label_gt);
colormap(full_colormap);
colorbar;
axis image off;
title(sprintf('Ground Truth - %s', database));

figure('Name','S3-PCA Classification Map');
imagesc(classification_map);
colormap(full_colormap);
colorbar;
axis image off;
title(sprintf('S^3-PCA - %s', database));

%% =========================================================================
% 9. SAVE RESULTS
% =========================================================================
result_file = sprintf('S3PCA_MC10_%s.mat', database);

save(result_file, ...
    'database', 'num_PC', 'num_Pixel', 'k', 'train_ratio', ...
    'OA_all', 'AA_all', 'kappa_all', 'classAcc_all', ...
    'time_all', 'feature_time', 'classification_map');

fprintf('\nResults saved to: %s\n', result_file);
fprintf('Done.\n');
