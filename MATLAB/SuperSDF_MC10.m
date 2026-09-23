function results = SuperSDF_MC10(DATASET_ID)
% FSuperSDF_MC10
% -------------------------------------------------------------------------
% Monte-Carlo evaluation of SuperSDF using the final paper settings.
%
% Usage:
%   results = FSuperSDF_MC10('Indian');
%   results = FSuperSDF_MC10('Pavia');
%   results = FSuperSDF_MC10('HanChuan');
%
% Dataset .mat files must contain:
%   img : rows x cols x bands hyperspectral cube
%   gt  : rows x cols integer ground-truth map (0 = unlabeled)
%
% Required helper in the same folder:
%   random_split_data_MC.m
%
% Final paper parameters:
%   Indian Pines : k=35, N_SP=60,  D=25,  train=5%, val=5%
%   Pavia Univ.  : k=35, N_SP=240, D=240, train=2%, val=2%
%   HanChuan     : k=35, N_SP=400, D=150, train=1%, val=1%
%
% Evaluation:
%   - SuperSDF features are extracted ONCE.
%   - 10 Monte-Carlo runs are used.
%   - Seeds: 20,27,34,41,48,55,62,69,76,83.
%   - Validation samples remain separate and are not merged into training.
%   - OA, AA, kappa, and per-class accuracies are computed on TEST only.
%
% Timing:
%   feature extraction + mean(SVM training + test prediction)
% -------------------------------------------------------------------------

if nargin < 1 || isempty(DATASET_ID)
    DATASET_ID = 'indian';
end

close all;
clc;

%% 1. FINAL PAPER CONFIGURATION
switch lower(DATASET_ID)
    case {'indian','indianpines','ip'}
        cfg.name        = 'Indian Pines';
        cfg.file        = 'Indian_pines.mat';
        cfg.k           = 35;
        cfg.N_SP        = 60;
        cfg.D           = 25;
        cfg.train_ratio = 0.05;

    case {'pavia','paviau','pu'}
        cfg.name        = 'Pavia University';
        cfg.file        = 'PaviaUF.mat';
        cfg.k           = 35;
        cfg.N_SP        = 240;
        cfg.D           = 240;
        cfg.train_ratio = 0.02;

    case {'hanchuan','hc'}
        cfg.name        = 'WHU-Hi-HanChuan';
        cfg.file        = 'HanChuan.mat';
        cfg.k           = 35;
        cfg.N_SP        = 400;
        cfg.D           = 150;
        cfg.train_ratio = 0.01;

    otherwise
        error('Unknown DATASET_ID. Use Indian, Pavia, or HanChuan.');
end

SIGMA_GAUSS = 0.95;
GAUSS_FSIZE = 55;
COMPACTNESS = 0.1;
LAMBDA_GF   = 1e-4;

SVM_C       = 1e5;
SVM_GAMMA   = 0.01;

MIN_TRAIN_PER_CLASS = 5;
N_MC = 10;
MC_SEEDS = (1:N_MC) * 7 + 13;   % 20,27,...,83

MAKE_CLASSIFICATION_MAP = false;

fprintf('\n============================================================\n');
fprintf(' SuperSDF MC10 - %s\n', cfg.name);
fprintf('============================================================\n');
fprintf(' k              : %d\n', cfg.k);
fprintf(' N_SP           : %d\n', cfg.N_SP);
fprintf(' D              : %d\n', cfg.D);
fprintf(' Train ratio    : %.1f%% per class\n', 100*cfg.train_ratio);
fprintf(' Validation     : same number as training per class\n');
fprintf(' MC runs        : %d\n', N_MC);
fprintf(' Seeds          : %s\n', mat2str(MC_SEEDS));
fprintf('============================================================\n\n');

if exist('random_split_data_MC','file') ~= 2
    error(['random_split_data_MC.m was not found. Put it in the same ' ...
           'folder as FSuperSDF_MC10.m.']);
end

%% 2. LOAD DATA
S = load(cfg.file);
if ~isfield(S,'img') || ~isfield(S,'gt')
    error('%s must contain variables img and gt.', cfg.file);
end

img = double(S.img);
gt  = S.gt;
[rows, cols, bands] = size(img);

fprintf('Image size: %d x %d x %d bands\n', rows, cols, bands);
fprintf('Labeled pixels: %d\n\n', nnz(gt > 0));

%% 3. LABEL-FREE SUPERSDF FEATURE EXTRACTION - ONCE
fprintf('[1/3] Extracting SuperSDF features once...\n');
feature_tic = tic;

X = reshape(img, rows*cols, bands);
R = corrcoef(X);

rng(50,'twister');
[labels,~] = kmeans(R, cfg.k, 'Replicates', 1);

grouped_bands = cell(cfg.k,1);
for i = 1:cfg.k
    grouped_bands{i} = find(labels == i);
end

sdf_images = zeros(rows, cols, cfg.k);

for i = 1:cfg.k
    group_img = mean(img(:,:,grouped_bands{i}), 3);

    lo = min(group_img(:));
    hi = max(group_img(:));
    if hi > lo
        normalized_img = (group_img - lo) / (hi - lo);
    else
        normalized_img = zeros(size(group_img));
    end

    [L,N] = superpixels(normalized_img, cfg.N_SP, ...
        'Method','slic','Compactness',COMPACTNESS);

    pixel_idx_lists = label2idx(L);
    sp_min_img = zeros(size(normalized_img), 'like', normalized_img);

    for j = 1:N
        idx = pixel_idx_lists{j};
        sp_min_img(idx) = min(normalized_img(idx));
    end

    % Same binarization rule as the current FSuperSDF MATLAB implementation.
    threshold = graythresh(sp_min_img);
    binary_img = imbinarize(sp_min_img, 'adaptive', ...
        'ForegroundPolarity','bright', ...
        'Sensitivity',threshold);

    if all(binary_img(:) == 0) || all(binary_img(:) == 1)
        sdf = zeros(size(binary_img));
    else
        sdf = bwdist(~binary_img) - bwdist(binary_img);
    end

    sdf = imguidedfilter(sdf, group_img, ...
        'NeighborhoodSize',[cfg.D cfg.D], ...
        'DegreeOfSmoothing',LAMBDA_GF);

    sdf = imgaussfilt(sdf, SIGMA_GAUSS, ...
        'FilterSize',GAUSS_FSIZE);

    sdf_images(:,:,i) = sdf;
end

feature_time = toc(feature_tic);
fprintf('Feature extraction complete: %.3f s\n\n', feature_time);

%% 4. PREALLOCATE MC RESULTS
valid_gt = gt(gt > 0);
class_order = unique(valid_gt);
n_classes = numel(class_order);

OA_mc    = zeros(N_MC,1);
AA_mc    = zeros(N_MC,1);
Kappa_mc = zeros(N_MC,1);
CA_mc    = zeros(N_MC,n_classes);

classifier_time = zeros(N_MC,1);
run_total_time  = zeros(N_MC,1);

train_counts = zeros(N_MC,n_classes);
val_counts   = zeros(N_MC,n_classes);
test_counts  = zeros(N_MC,n_classes);

last_model = [];
last_X_all_raw = [];

fprintf('[2/3] Running %d Monte-Carlo splits...\n\n', N_MC);

%% 5. MONTE-CARLO LOOP
for mc = 1:N_MC
    run_tic = tic;

    seed = MC_SEEDS(mc);
    rng(seed,'twister');

    fprintf('------------------------------------------------------------\n');
    fprintf('MC %02d/%02d | seed = %d\n', mc, N_MC, seed);

    [X_train, y_train, X_val, y_val, X_test, y_test, ...
        ~, ~, ~, ~, X_all_raw] = ...
        random_split_data_MC(sdf_images, gt, cfg.train_ratio, ...
                             MIN_TRAIN_PER_CLASS, false);

    for c = 1:n_classes
        cls = class_order(c);
        train_counts(mc,c) = sum(y_train == cls);
        val_counts(mc,c)   = sum(y_val   == cls);
        test_counts(mc,c)  = sum(y_test  == cls);
    end

    % Train on TRAIN only; VALIDATION remains unused during final MC testing.
    cls_tic = tic;

    svm_template = templateSVM( ...
        'KernelFunction','rbf', ...
        'KernelScale',1/sqrt(SVM_GAMMA), ...
        'BoxConstraint',SVM_C, ...
        'Standardize',true);

    model = fitcecoc(X_train, y_train, ...
        'Learners',svm_template, ...
        'Verbose',0);

    y_pred = predict(model, X_test);

    classifier_time(mc) = toc(cls_tic);

    cm = confusionmat(y_test, y_pred, 'Order', class_order);

    OA_mc(mc) = 100 * trace(cm) / sum(cm(:));

    row_totals = sum(cm,2);
    ca = 100 * diag(cm) ./ max(row_totals,1);
    CA_mc(mc,:) = ca(:)';
    AA_mc(mc) = mean(ca);

    N = sum(cm(:));
    po = trace(cm) / N;
    pe = sum(sum(cm,2) .* sum(cm,1)') / (N^2);

    if abs(1-pe) < eps
        Kappa_mc(mc) = 0;
    else
        Kappa_mc(mc) = 100 * (po-pe) / (1-pe);
    end

    run_total_time(mc) = toc(run_tic);

    fprintf('OA=%.2f%% | AA=%.2f%% | kappa=%.2f%% | classifier=%.3f s\n', ...
        OA_mc(mc), AA_mc(mc), Kappa_mc(mc), classifier_time(mc));

    last_model = model;
    last_X_all_raw = X_all_raw;
end

%% 6. MC SUMMARY
fprintf('\n============================================================\n');
fprintf(' FINAL MC10 RESULTS - %s\n', cfg.name);
fprintf('============================================================\n');

fprintf('\nPer-class accuracy (%%):\n');
fprintf('%-8s %-18s\n','Class','Mean +/- Std');
fprintf('%s\n', repmat('-',1,32));

for c = 1:n_classes
    fprintf('%-8d %7.2f +/- %-7.2f\n', ...
        class_order(c), mean(CA_mc(:,c)), std(CA_mc(:,c)));
end

fprintf('%s\n', repmat('-',1,32));
fprintf('OA      %7.2f +/- %-7.2f\n', mean(OA_mc),    std(OA_mc));
fprintf('AA      %7.2f +/- %-7.2f\n', mean(AA_mc),    std(AA_mc));
fprintf('kappa   %7.2f +/- %-7.2f\n', mean(Kappa_mc), std(Kappa_mc));

reported_time = feature_time + mean(classifier_time);

fprintf('\nTiming:\n');
fprintf('Feature extraction time          : %.3f s\n', feature_time);
fprintf('Classifier avg/run               : %.3f +/- %.3f s\n', ...
    mean(classifier_time), std(classifier_time));
fprintf('Reported computational time      : %.3f s\n', reported_time);
fprintf('MC avg incl. split + metrics      : %.3f +/- %.3f s\n', ...
    mean(run_total_time), std(run_total_time));

fprintf('\nSplit totals, first MC run:\n');
fprintf('Training   : %d\n', sum(train_counts(1,:)));
fprintf('Validation : %d\n', sum(val_counts(1,:)));
fprintf('Test       : %d\n', sum(test_counts(1,:)));
fprintf('============================================================\n');

%% 7. OPTIONAL CLASSIFICATION MAP
if MAKE_CLASSIFICATION_MAP
    fprintf('\n[3/3] Generating classification map from the final MC model...\n');

    y_full = predict(last_model, last_X_all_raw);
    class_map = reshape(y_full, rows, cols);
    class_map(gt == 0) = 0;

    n_cls_v = double(max(gt(:)));
    cmap = [0 0 0; jet(n_cls_v)];

    figure('Name','Ground Truth');
    imagesc(gt);
    axis image off;
    colormap(cmap);
    colorbar;
    title(sprintf('Ground Truth - %s', cfg.name));

    figure('Name','SuperSDF Classification Map');
    imagesc(class_map);
    axis image off;
    colormap(cmap);
    colorbar;
    title(sprintf('SuperSDF - %s', cfg.name));
end

%% 8. RETURN AND SAVE
results = struct();
results.dataset = cfg.name;
results.config = cfg;
results.mc_seeds = MC_SEEDS;

results.OA_runs = OA_mc;
results.AA_runs = AA_mc;
results.kappa_runs = Kappa_mc;
results.class_accuracy_runs = CA_mc;
results.class_order = class_order;

results.OA_mean = mean(OA_mc);
results.OA_std = std(OA_mc);
results.AA_mean = mean(AA_mc);
results.AA_std = std(AA_mc);
results.kappa_mean = mean(Kappa_mc);
results.kappa_std = std(Kappa_mc);

results.class_accuracy_mean = mean(CA_mc,1);
results.class_accuracy_std = std(CA_mc,0,1);

results.feature_time = feature_time;
results.classifier_time_runs = classifier_time;
results.classifier_time_mean = mean(classifier_time);
results.classifier_time_std = std(classifier_time);
results.reported_time = reported_time;

results.train_counts = train_counts;
results.val_counts = val_counts;
results.test_counts = test_counts;

safe_name = regexprep(cfg.name,'[^A-Za-z0-9]','_');
out_file = sprintf('SuperSDF_MC10_%s.mat', safe_name);
save(out_file,'results');

fprintf('\nResults saved to: %s\n', out_file);
fprintf('Done.\n');

end
