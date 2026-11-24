function results = FS3PCA_New(database, train_ratio)
% =========================================================================
% FS3PCA_MC10
% S^3-PCA baseline using the SAME fixed parameters as the original FS3PCA.m,
% evaluated with 10 Monte-Carlo random train/validation/test splits.
%
% No parameter search is performed.
%
% Fixed S^3-PCA parameters copied from FS3PCA.m:
%   Indian : num_PC = 30, num_Pixel = 75, k = 15
%   PaviaU : num_PC = 30, num_Pixel = 30, k = 13
%   Salinas: num_PC = 30, num_Pixel = 30, k = 15
%
% Classifier parameters are kept the same as FS3PCA.m:
%   RBF-SVM, KernelScale = 1/sqrt(0.01), BoxConstraint = 100000,
%   Standardize = true.
%
% Evaluation:
%   - 10 Monte-Carlo runs
%   - rng(mc*7 + 13), matching SuperSDF_New
%   - random_split_data
%   - OA, AA, Kappa and per-class accuracy computed on TEST set only
%   - full classification map is visualization only
%
% Example:
%   results = FS3PCA_MC10('Indian', 0.05);
%   results = FS3PCA_MC10('PaviaU', 0.02);
%   results = FS3PCA_MC10('Salinas', 0.01);
% =========================================================================

clc;
close all;

if nargin < 1 || isempty(database)
    database = 'Indian';
end

if nargin < 2 || isempty(train_ratio)
    switch database
        case 'Indian'
            train_ratio = 0.05;
        case 'PaviaU'
            train_ratio = 0.02;
        case 'Salinas'
            train_ratio = 0.01;
        otherwise
            error('Unsupported database: %s', database);
    end
end

%% ------------------------------------------------------------------------
% Fixed parameters from the original FS3PCA.m
% -------------------------------------------------------------------------
num_PC = 30;

switch database
    case 'Indian'
        num_Pixel = 75;
        k = 15;

    case 'PaviaU'
        num_Pixel = 30;
        k = 13;

    case 'Salinas'
        num_Pixel = 30;
        k = 15;

    otherwise
        error('Unsupported database: %s', database);
end

% Same SVM parameters as the original FS3PCA.m
SVM_GAMMA = 0.01;
SVM_C     = 100000;

% Same minimum-per-class rule used in SuperSDF_New/random_split_data
MIN_TRAIN = 5;

% Monte-Carlo protocol
N_MC = 10;

fprintf('============================================================\n');
fprintf(' S^3-PCA FIXED-PARAMETER MONTE-CARLO EVALUATION\n');
fprintf('============================================================\n');
fprintf(' Dataset       : %s\n', database);
fprintf(' num_PC        : %d\n', num_PC);
fprintf(' num_Pixel     : %d\n', num_Pixel);
fprintf(' spatial k     : %d\n', k);
fprintf(' train ratio   : %.2f%% per class\n', train_ratio*100);
fprintf(' MC runs       : %d\n', N_MC);
fprintf(' SVM C         : %.0f\n', SVM_C);
fprintf(' SVM gamma     : %.4f\n', SVM_GAMMA);
fprintf('============================================================\n\n');

%% ------------------------------------------------------------------------
% Dependency check
% -------------------------------------------------------------------------
if isempty(which('cubseg'))
    error(['cubseg.m was not found. Add the original S^3-PCA / ERS ', ...
           'folders to the MATLAB path.']);
end

if isempty(which('S3_PCA'))
    error('S3_PCA.m was not found. Add the original S^3-PCA folder.');
end

fprintf('Using cubseg : %s\n', which('cubseg'));
fprintf('Using S3_PCA : %s\n', which('S3_PCA'));

if isempty(which('mex_ers'))
    fprintf(['WARNING: mex_ers was not found. If cubseg uses ERS, ', ...
             'add the folder containing mex_ers.%s.\n\n'], mexext);
else
    fprintf('Using mex_ers: %s\n\n', which('mex_ers'));
end

%% ------------------------------------------------------------------------
% Load dataset
% -------------------------------------------------------------------------
switch database
    case 'Indian'
        if isfile('Indian_pines.mat')
            S = load('Indian_pines.mat');
            if ~isfield(S,'img') || ~isfield(S,'gt')
                error('Indian_pines.mat must contain img and gt.');
            end
            data3D   = double(S.img);
            label_gt = S.gt;
        else
            % Compatibility with the original S^3-PCA files
            A = load('Indian_pines_corrected.mat');
            B = load('Indian_pines_gt.mat');
            data3D   = double(A.indian_pines_corrected);
            label_gt = B.indian_pines_gt;
        end

    case 'PaviaU'
        S = load('PaviaUF.mat');
        if ~isfield(S,'img') || ~isfield(S,'gt')
            error('PaviaUF.mat must contain img and gt.');
        end
        data3D   = double(S.img);
        label_gt = S.gt;

    case 'Salinas'
        S = load('Salinas.mat');
        if ~isfield(S,'img') || ~isfield(S,'gt')
            error('Salinas.mat must contain img and gt.');
        end
        data3D   = double(S.img);
        label_gt = S.gt;
end

[rows, cols, bands] = size(data3D);

if num_PC > bands
    error('num_PC=%d exceeds the number of bands=%d.', num_PC, bands);
end

fprintf('Loaded image: %d x %d x %d bands\n', rows, cols, bands);

% Same normalization as original FS3PCA.m
mx = max(data3D(:));
if mx > 0
    data3D = data3D ./ mx;
end

%% ------------------------------------------------------------------------
% Extract S^3-PCA features ONCE using fixed parameters
% -------------------------------------------------------------------------
fprintf('\nExtracting S^3-PCA features with fixed paper parameters...\n');

feature_start = tic;

% Fix the feature-extraction randomness, but restore the RNG state
% afterwards so the MC split seeds are not affected.
old_rng = rng;
rng(50, 'twister');

labels = cubseg(data3D, num_Pixel);
dataDR = S3_PCA(data3D, num_PC, k, labels);

rng(old_rng);

feature_time = toc(feature_start);

[fr, fc, num_features] = size(dataDR);

if fr ~= rows || fc ~= cols
    error('S3_PCA returned spatial dimensions inconsistent with the image.');
end

fprintf('Feature extraction complete: %.2f s\n', feature_time);
fprintf('Output features: %d\n', num_features);

%% ------------------------------------------------------------------------
% Prepare full-scene and labeled features
% -------------------------------------------------------------------------
X_all_raw = reshape(dataDR, rows*cols, num_features);

valid_mask    = label_gt(:) > 0;
valid_indices = find(valid_mask);
valid_gt      = label_gt(valid_mask);
valid_feat    = X_all_raw(valid_mask, :);

unique_cls = unique(valid_gt);
num_cls    = numel(unique_cls);

%% ------------------------------------------------------------------------
% Monte-Carlo evaluation
% -------------------------------------------------------------------------
OA_mc    = zeros(N_MC,1);
AA_mc    = zeros(N_MC,1);
KAP_mc   = zeros(N_MC,1);
CA_mc    = zeros(N_MC,num_cls);
TIME_mc  = zeros(N_MC,1);

fprintf('\n============================================================\n');
fprintf(' 10-RUN MONTE-CARLO TEST EVALUATION\n');
fprintf('============================================================\n');

for mc = 1:N_MC
    mc_start = tic;

    % Same seed sequence as SuperSDF_New
    rng(mc*7 + 13, 'twister');

    [~,~,~,~,~,~,~, train_indices, ~, test_indices, ~] = ...
        random_split_data(dataDR, label_gt, ...
                          train_ratio, MIN_TRAIN, false);
    % seed = (5)*7 + 13;
    % [X_train, y_train, X_val, y_val, X_test, y_test,sample_map, train_indices, val_indices, test_indices, X_all_raw] = random_split_data(pca_features, gt, train_ratio, 5 ,true);
    % [~,~,~,~,~,~,~, train_indices, ~, test_indices, ~] = ...
    %     spatial_block_split_data_guarded(dataDR, label_gt, train_ratio, 10,5,5,10,seed,true);

    X_train = valid_feat(train_indices,:);
    y_train = valid_gt(train_indices);

    X_test = valid_feat(test_indices,:);
    y_test = valid_gt(test_indices);

    % Same classifier parameters as original FS3PCA.m
    svm_template = templateSVM( ...
        'KernelFunction','rbf', ...
        'KernelScale',1/sqrt(SVM_GAMMA), ...
        'BoxConstraint',SVM_C, ...
        'Standardize',true);

    model = fitcecoc(X_train, y_train, ...
        'Learners',svm_template, ...
        'Verbose',0);

    y_pred = predict(model, X_test);

    % Confusion matrix with fixed class order
    cm = confusionmat(y_test, y_pred, 'Order', unique_cls);

    % OA
    OA_mc(mc) = trace(cm) / sum(cm(:)) * 100;

    % Per-class accuracy
    for c = 1:num_cls
        denom = sum(cm(c,:));
        if denom > 0
            CA_mc(mc,c) = cm(c,c) / denom * 100;
        else
            CA_mc(mc,c) = 0;
        end
    end

    % AA
    AA_mc(mc) = mean(CA_mc(mc,:));

    % Kappa
    N = sum(cm(:));
    row_sums = sum(cm,2);
    col_sums = sum(cm,1)';

    po = trace(cm) / N;
    pe = sum(row_sums .* col_sums) / N^2;

    if abs(1-pe) > eps
        KAP_mc(mc) = (po-pe)/(1-pe) * 100;
    else
        KAP_mc(mc) = 0;
    end

    TIME_mc(mc) = toc(mc_start);

    fprintf(['Run %2d/%d | seed=%d | OA=%.2f%% | AA=%.2f%% | ', ...
             'Kappa=%.2f%% | %.2f s\n'], ...
             mc, N_MC, mc*7+13, OA_mc(mc), AA_mc(mc), ...
             KAP_mc(mc), TIME_mc(mc));
end

%% ------------------------------------------------------------------------
% Results
% -------------------------------------------------------------------------
fprintf('\n============================================================\n');
fprintf(' FINAL S^3-PCA RESULTS - %s\n', database);
fprintf(' Fixed parameters: PC=%d, N_SP=%d, k=%d\n', ...
    num_PC, num_Pixel, k);
fprintf('============================================================\n');

for c = 1:num_cls
    fprintf('Class %2d : %.2f +/- %.2f %%\n', ...
        unique_cls(c), mean(CA_mc(:,c)), std(CA_mc(:,c)));
end

fprintf('------------------------------------------------------------\n');
fprintf('OA    : %.2f +/- %.2f %%\n', mean(OA_mc), std(OA_mc));
fprintf('AA    : %.2f +/- %.2f %%\n', mean(AA_mc), std(AA_mc));
fprintf('Kappa : %.2f +/- %.2f %%\n', mean(KAP_mc), std(KAP_mc));
fprintf('------------------------------------------------------------\n');
fprintf('Feature extraction time : %.2f s\n', feature_time);
fprintf('MC average time/run     : %.2f +/- %.2f s\n', ...
    mean(TIME_mc), std(TIME_mc));
fprintf('MC total time           : %.2f s\n', sum(TIME_mc));
fprintf('============================================================\n');

%% ------------------------------------------------------------------------
% Classification map - visualization only
% -------------------------------------------------------------------------
fprintf('\nGenerating one reference classification map...\n');

rng(50, 'twister');

[~,~,~,~,~,~,~, train_map, ~, ~, ~] = ...
    random_split_data(dataDR, label_gt, ...
                      train_ratio, MIN_TRAIN, false);

svm_template = templateSVM( ...
    'KernelFunction','rbf', ...
    'KernelScale',1/sqrt(SVM_GAMMA), ...
    'BoxConstraint',SVM_C, ...
    'Standardize',true);

model_map = fitcecoc(valid_feat(train_map,:), valid_gt(train_map), ...
    'Learners',svm_template, ...
    'Verbose',0);

y_map = predict(model_map, X_all_raw);

assert(numel(y_map) == rows*cols, ...
    'Prediction count must equal rows*cols.');

classification_map = reshape(y_map, rows, cols);
classification_map(label_gt == 0) = 0;

n_cls_v = double(max(label_gt(:)));
full_colormap = [0 0 0; jet(n_cls_v)];

figure('Name','S3PCA Ground Truth');
imagesc(label_gt);
colormap(full_colormap);
colorbar;
axis image off;
title(sprintf('Ground Truth - %s',database));

figure('Name','S3PCA Classification Map');
imagesc(classification_map);
colormap(full_colormap);
colorbar;
axis image off;
title(sprintf('S^3-PCA - %s',database));

%% ------------------------------------------------------------------------
% Return structured results
% -------------------------------------------------------------------------
results = struct();

results.database       = database;
results.train_ratio    = train_ratio;

results.num_PC         = num_PC;
results.num_Pixel      = num_Pixel;
results.k              = k;

results.OA_runs        = OA_mc;
results.AA_runs        = AA_mc;
results.Kappa_runs     = KAP_mc;
results.CA_runs        = CA_mc;
results.time_runs      = TIME_mc;

results.OA_mean        = mean(OA_mc);
results.OA_std         = std(OA_mc);
results.AA_mean        = mean(AA_mc);
results.AA_std         = std(AA_mc);
results.Kappa_mean     = mean(KAP_mc);
results.Kappa_std      = std(KAP_mc);

results.feature_time   = feature_time;
results.classification_map = classification_map;

end
