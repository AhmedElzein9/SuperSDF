% =========================================================================
% demo_S3ULDA_MC10.m
%
% Modified from the original S3ULDA.m supplied by the user.
%
% PURPOSE
%   Keep the ORIGINAL S3-ULDA feature-extraction parameters unchanged,
%   but evaluate the method with the same 10-run Monte-Carlo protocol
%   used by SuperSDF_New.
%
% IMPORTANT
%   - No parameter search is performed.
%   - Original S3-ULDA feature extraction is preserved:
%         labels = cubseg(data3D,num_Pixel);
%         dataDR = SuperULDA(data3D,num_PC,k,labels);
%   - Fixed active parameters are copied from the original S3ULDA.m:
%
%       Indian_pines : num_Pixel=35, k=17, num_PC=10
%       PaviaUF      : num_Pixel=30, k=13, num_PC=35
%       Salinas      : num_Pixel=30, k=15, num_PC=15
%
%   - Same RBF-SVM settings as the original S3ULDA.m:
%         KernelScale = 1/sqrt(0.01)
%         BoxConstraint = 100000
%         Standardize = true
%
%   - Same split ratios currently used by SuperSDF_New:
%         Indian_pines : 5%
%         PaviaUF      : 2%
%         HanChuan      : 1%
%
%   - 10 MC runs with the same seed sequence as SuperSDF_New:
%         rng(mc*7 + 13)
%
%   - OA, AA, Kappa and class accuracy are calculated on TEST pixels only.
%   - The classification map is visualization only.
% =========================================================================

clc;
clear;
close all;

fprintf('Load...\n');

% Keep the original dependency setup.
addpath('.\libsvm-3.21\matlab');
addpath(genpath(cd));

%% =========================================================================
% 1. USER SELECTION
% =========================================================================
database = 'HanChuan';   % 'Indian_pines' | 'PaviaUF' | 'Salinas' | 'HanChuan'

N_MC      = 10;
MIN_TRAIN = 5;

%% =========================================================================
% 2. ORIGINAL DATASET-SPECIFIC S3-ULDA PARAMETERS
% =========================================================================
if strcmp(database,'Indian_pines')

    load Indian_pines_corrected;
    load Indian_pines_gt;
    load Indian_pines_randp;

    data3D   = indian_pines_corrected;
    label_gt = indian_pines_gt;

    % ORIGINAL active parameters from S3ULDA.m
    num_Pixel = 35;
    k         = 17;
    num_PC    = 10;

    % Same label budget as SuperSDF_New
    train_ratio = 0.05;

elseif strcmp(database,'PaviaUF')

    load PaviaU;
    load PaviaU_gt;
    load PaviaU_randp;

    data3D   = paviaU;
    label_gt = paviaU_gt;

    % ORIGINAL active parameters from S3ULDA.m
    num_Pixel = 30;
    k         = 13;
    num_PC    = 35;

    % Same label budget as SuperSDF_New
    train_ratio = 0.02;

elseif strcmp(database,'Salinas')

    S = load('Salinas.mat');

    if ~isfield(S,'img') || ~isfield(S,'gt')
        error('Salinas.mat must contain variables ''img'' and ''gt''.');
    end

    data3D   = S.img;
    label_gt = S.gt;

    % ORIGINAL active parameters from S3ULDA.m
    num_Pixel = 30;
    k         = 15;
    num_PC    = 15;

    % Same label budget as SuperSDF_New
    train_ratio = 0.01;
elseif strcmp(database,'LongKou')

    S = load('LongKou.mat');

    if ~isfield(S,'img') || ~isfield(S,'gt')
        error('LongKou.mat must contain variables ''img'' and ''gt''.');
    end

    data3D   = S.img;
    label_gt = S.gt;

    num_Pixel = 15;
    k         = 15;
    num_PC    = 35;

    train_ratio = 0.01;
elseif strcmp(database,'HanChuan')

    S = load('HanChuan.mat');

    if ~isfield(S,'img') || ~isfield(S,'gt')
        error('HanChuan.mat must contain variables ''img'' and ''gt''.');
    end

    data3D   = S.img;
    label_gt = S.gt;

    num_Pixel = 150;
    k         = 15;
    num_PC    = 35;

    train_ratio = 0.01;
else
    error('Unsupported database: %s', database);
end

data3D   = double(data3D);
label_gt = double(label_gt);

[rows, cols, bands] = size(data3D);

%% =========================================================================
% 3. ORIGINAL NORMALIZATION
% =========================================================================
mx = max(data3D(:));

if mx > 0
    data3D = data3D ./ mx;
end

fprintf('\n============================================================\n');
fprintf(' S3-ULDA - ORIGINAL PARAMETERS + 10 MC RUNS\n');
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
% 4. ORIGINAL S3-ULDA FEATURE EXTRACTION
% =========================================================================
fprintf('Running original S3-ULDA feature extraction...\n');

feature_start = tic;

% Keep the extracted representation fixed across all MC runs.
% Restore RNG afterwards so MC split seeds remain exactly controlled.
rng_before_feature = rng;
rng(50,'twister');

labels = cubseg(data3D,num_Pixel);
dataDR = SuperULDA(data3D,num_PC,k,labels);

rng(rng_before_feature);

feature_time = toc(feature_start);

[rows_dr, cols_dr, num_features] = size(dataDR);

if rows_dr ~= rows || cols_dr ~= cols
    error('SuperULDA output dimensions do not match the original image.');
end

fprintf('S3-ULDA feature extraction complete: %.2f s\n', feature_time);
fprintf('Reduced feature dimensions: %d\n\n', num_features);

%% =========================================================================
% 5. LABEL INFORMATION
% =========================================================================
valid_mask = label_gt(:) > 0;
valid_gt   = label_gt(valid_mask);

unique_classes = unique(valid_gt);
num_classes    = numel(unique_classes);

%% =========================================================================
% 6. MONTE-CARLO EVALUATION
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
    rng(seed,'twister');


    % Same stratified random split protocol as SuperSDF_New.
    % random_split_data returns train, validation and held-out test subsets.
    % [X_train, y_train, X_val, y_val, X_test, y_test, ...
    %     sample_map, train_indices, val_indices, test_indices, X_all_raw] = ...
    %     random_split_data(dataDR, label_gt, ...
    %                       train_ratio, MIN_TRAIN, false); %#ok<ASGLU>
    [X_train, y_train, X_val, y_val, X_test, y_test, sample_map, train_indices, val_indices, test_indices, X_all_raw, split_info] = ...
        spatial_block_split_data_guarded(dataDR, label_gt,train_ratio,20,5,5,5,seed,true);


    run_start = tic;

    % Same SVM settings as original S3ULDA.m
    svm_template = templateSVM( ...
        'KernelFunction','rbf', ...
        'KernelScale',1/sqrt(0.01), ...
        'BoxConstraint',100000, ...
        'Standardize',true);

    model = fitcecoc(X_train, y_train, ...
        'Learners',svm_template, ...
        'Verbose',0);

    % ---------------------------------------------------------------------
    % TEST SET ONLY
    % ---------------------------------------------------------------------
    y_test_pred = predict(model,X_test);
    time_all(mc) = toc(run_start);

    cm = confusionmat(y_test,y_test_pred, ...
        'Order',unique_classes);

    % Overall Accuracy
    OA_all(mc) = trace(cm) / sum(cm(:));

    % Per-class Accuracy
    for c = 1:num_classes
        class_total = sum(cm(c,:));

        if class_total > 0
            classAcc_all(mc,c) = cm(c,c) / class_total;
        else
            classAcc_all(mc,c) = 0;
        end
    end

    % Average Accuracy
    AA_all(mc) = mean(classAcc_all(mc,:));

    % Kappa
    N = sum(cm(:));

    row_sums = sum(cm,2);
    col_sums = sum(cm,1)';

    po = trace(cm)/N;
    pe = sum(row_sums .* col_sums)/(N^2);

    if abs(1-pe) > eps
        kappa_all(mc) = (po-pe)/(1-pe);
    else
        kappa_all(mc) = 0;
    end

    % time_all(mc) = toc(run_start);

    fprintf(['Run %2d/%d | seed=%d | Train=%d | Val=%d | Test=%d | ', ...
             'OA=%.2f%% | AA=%.2f%% | Kappa=%.2f%% | %.2f s\n'], ...
             mc, N_MC, seed, ...
             numel(y_train), numel(y_val), numel(y_test), ...
             OA_all(mc)*100, AA_all(mc)*100, ...
             kappa_all(mc)*100, time_all(mc));
end

%% =========================================================================
% 7. FINAL RESULTS
% =========================================================================
fprintf('\n============================================================\n');
fprintf(' FINAL S3-ULDA RESULTS OVER %d RUNS - %s\n', N_MC, database);
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
% 8. OPTIONAL BOXPLOT
% =========================================================================
figure('Name','S3-ULDA Monte Carlo Results');

boxplot([OA_all,AA_all,kappa_all], ...
    'Labels',{'OA','AA','Kappa'});

ylabel('Score');
title(sprintf('S3-ULDA Monte Carlo (%d runs) - %s', ...
    N_MC,database));
grid on;

%% =========================================================================
% 9. ONE REFERENCE CLASSIFICATION MAP
%    VISUALIZATION ONLY - NOT USED FOR METRICS
% =========================================================================
fprintf('\nGenerating one reference classification map...\n');

rng(50,'twister');

% [X_train_map, y_train_map, ~, ~, ~, ~, ...
%     ~, ~, ~, ~, X_all_raw_map] = ...
%     random_split_data(dataDR, label_gt, ...
%                       train_ratio, MIN_TRAIN, false);
[X_train, y_train, X_val, y_val, X_test, y_test, sample_map, train_indices, val_indices, test_indices, X_all_raw, split_info] = ...
    spatial_block_split_data_guarded(dataDR, label_gt,train_ratio,5,2,5,2,seed,true);

svm_template_map = templateSVM( ...
    'KernelFunction','rbf', ...
    'KernelScale',1/sqrt(0.01), ...
    'BoxConstraint',100000, ...
    'Standardize',true);

model_map = fitcecoc(X_train_map,y_train_map, ...
    'Learners',svm_template_map, ...
    'Verbose',0);

y_pred_img = predict(model_map,X_all_raw_map);

assert(numel(y_pred_img) == rows*cols, ...
    'Prediction count must equal rows*cols.');

classification_map = reshape(y_pred_img,rows,cols);
classification_map(label_gt == 0) = 0;

num_classes_map = double(max(label_gt(:)));
full_colormap = [0 0 0; jet(num_classes_map)];

figure('Name','Ground Truth');
imagesc(label_gt);
colormap(full_colormap);
colorbar;
axis image off;
title(sprintf('Ground Truth - %s',database));

figure('Name','S3-ULDA Classification Map');
imagesc(classification_map);
colormap(full_colormap);
colorbar;
axis image off;
title(sprintf('S3-ULDA - %s',database));

%% =========================================================================
% 10. SAVE RESULTS
% =========================================================================
result_file = sprintf('S3ULDA_MC10_%s.mat',database);

save(result_file, ...
    'database','num_PC','num_Pixel','k','train_ratio', ...
    'OA_all','AA_all','kappa_all','classAcc_all', ...
    'time_all','feature_time','classification_map');

fprintf('\nResults saved to: %s\n',result_file);
fprintf('Done.\n');
reported_time = feature_time + mean(time_all);

fprintf('Feature extraction : %.2f s\n', feature_time);
fprintf('SVM train+test     : %.2f +/- %.2f s\n', ...
        mean(time_all), std(time_all));
fprintf('Reported time      : %.2f s\n', reported_time);