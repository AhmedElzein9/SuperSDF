function [overall_accuracy, avg_accuracy, kappa, class_accuracies] = SoperPCA_New(database, num_PC, train_ratio)
% =========================================================================
% SoperPCA_New
% Fair SuperPCA baseline using the same evaluation protocol as
% SuperSDF_New:
%   - deterministic feature extraction
%   - stratified random train/validation/test split
%   - same RBF-SVM settings as SuperSDF_New
%   - test-set-only OA, AA, Kappa, and per-class accuracy
%   - caller controls the random split seed (e.g., Run_MC)
%
% Requires:
%   cubseg.m
%   SuperPCA.m
%   random_split_data.m
%
% Example:
%   rng(20);
%   [OA,AA,Kappa,CA] = SoperPCA_New('Indian_pines',35,0.05);
%
% Outputs are fractions in [0,1], so the existing Run_MC script can
% multiply them by 100 when printing mean +/- std.
% =========================================================================

%% Fixed classifier/split parameters -- match SuperSDF_New
SVM_C       = 1e5;
SVM_GAMMA   = 0.01;
MIN_TRAIN   = 5;
FEATURE_SEED = 50;

%% Resolve and load dataset
if isstring(database)
    database = char(database);
end

if endsWith(database, '.mat', 'IgnoreCase', true)
    dataFile = database;
    [~, dataset_key, ~] = fileparts(database);
else
    dataFile = [database, '.mat'];
    dataset_key = database;
end

if ~isfile(dataFile)
    error('Dataset file not found: %s', dataFile);
end

S = load(dataFile);
if ~isfield(S,'img') || ~isfield(S,'gt')
    error('File %s must contain variables ''img'' and ''gt''.', dataFile);
end

data3D   = double(S.img);
label_gt = S.gt;
[rows, cols, bands] = size(data3D);

if ~isequal(size(label_gt), [rows, cols])
    error('Ground-truth size %s does not match image spatial size [%d %d].', ...
        mat2str(size(label_gt)), rows, cols);
end
%% ============================================================
% SuperPCA dependencies
% ============================================================

SUPERPCA_ROOT = fullfile(fileparts(mfilename('fullpath')), 'SuperPCA-master');

SUPERPCA_COMMON = fullfile(SUPERPCA_ROOT, 'common');

ERS_DIR = fullfile( ...
    SUPERPCA_ROOT, ...
    'Entropy Rate Superpixel Segmentation');

% Put SuperPCA paths FIRST to avoid conflict with S3-ULDA
addpath(SUPERPCA_ROOT, '-begin');
addpath(SUPERPCA_COMMON, '-begin');
addpath(ERS_DIR, '-begin');

rehash;

fprintf('\nSuperPCA dependency check:\n');
fprintf('cubseg  : %s\n', which('cubseg'));
fprintf('SuperPCA: %s\n', which('SuperPCA'));
fprintf('mex_ers : %s\n', which('mex_ers'));
fprintf('MEX type: %s\n', mexext);

if isempty(which('mex_ers'))
    error(['mex_ers was not found. The folder "%s" must contain ', ...
           'mex_ers.%s'], ERS_DIR, mexext);
end
%% Dataset-specific SuperPCA segmentation parameter
% These preserve the settings from the original SoperPCA.m.
switch lower(dataset_key)
    case {'indian_pines','indian'}
        num_Pixel = 100;
    case {'paviauf','pavia','paviau'}
        num_Pixel = 20;
    case {'salinas'}
        num_Pixel = 100;
    case {'longkou'}
        num_Pixel = 15;
    case {'hanchuan'}
        num_Pixel = 15;   % starting value; tune on validation if needed
    case {'honghu'}
        num_Pixel = 15;   % starting value; tune on validation if needed
    otherwise
        error(['Unsupported database ''%s''. Add its SuperPCA num_Pixel ', ...
               'setting in the switch block.'], dataset_key);
end

fprintf('\n=========================================================\n');
fprintf(' SoperPCA_New - SuperPCA fair evaluation\n');
fprintf('=========================================================\n');
fprintf(' Dataset              : %s\n', dataFile);
fprintf(' Image                 : %d x %d x %d\n', rows, cols, bands);
fprintf(' SuperPCA components   : %d\n', num_PC);
fprintf(' Superpixel parameter  : %d\n', num_Pixel);
fprintf(' Train ratio/class     : %.2f%%\n', train_ratio*100);
fprintf(' SVM C / gamma         : %.0e / %.4f\n', SVM_C, SVM_GAMMA);
fprintf('---------------------------------------------------------\n');

%% Normalize exactly once
mx = max(data3D(:));
if mx > 0
    data3D = data3D ./ mx;
end

%% SuperPCA feature extraction
% Keep feature extraction deterministic without destroying the split seed
% set by the caller (Run_MC).
caller_rng = rng;
rng(FEATURE_SEED, 'twister');

fprintf('Extracting SuperPCA features...\n');
t_feature = tic;
labels = cubseg(data3D, num_Pixel);
dataDR = SuperPCA(data3D, num_PC, labels);
feature_time = toc(t_feature);

% Restore the caller's RNG state so each Monte-Carlo run gets the intended
% split seed rather than FEATURE_SEED.
rng(caller_rng);

if ndims(dataDR) ~= 3 || size(dataDR,1) ~= rows || size(dataDR,2) ~= cols
    error('SuperPCA output must be rows x cols x features. Got size %s.', ...
        mat2str(size(dataDR)));
end

fprintf('Feature extraction complete: %d features, %.2f s\n', ...
    size(dataDR,3), feature_time);

%% Same stratified random split protocol as SuperSDF_New
[X_train, y_train, X_val, y_val, X_test, y_test, ...
 sample_map, train_indices, val_indices, test_indices, X_all_raw] = ...
    random_split_data(dataDR, label_gt, train_ratio, MIN_TRAIN, false); %#ok<ASGLU>

% Sanity checks
assert(size(X_all_raw,1) == rows*cols, ...
    'X_all_raw must have one row per image pixel.');
assert(size(X_train,2) == size(dataDR,3), ...
    'Training feature dimension is inconsistent with SuperPCA output.');

%% Train the SAME RBF-SVM configuration as SuperSDF_New
svm_t = templateSVM('KernelFunction','rbf', ...
    'KernelScale',1/sqrt(SVM_GAMMA), ...
    'BoxConstraint',SVM_C, ...
    'Standardize',true);

fprintf('Training SVM...\n');
t_train = tic;
mdl = fitcecoc(X_train, y_train, 'Learners', svm_t, 'Verbose', 0);
train_time = toc(t_train);

%% TEST-SET-ONLY evaluation
ypred = predict(mdl, X_test);
unique_cls = unique(label_gt(label_gt > 0));
num_cls = numel(unique_cls);

cm = confusionmat(y_test, ypred, 'Order', unique_cls);

% Overall accuracy
overall_accuracy = sum(ypred == y_test) / numel(y_test);

% Per-class accuracy and AA
class_accuracies = zeros(num_cls,1);
for c = 1:num_cls
    denom = sum(cm(c,:));
    if denom > 0
        class_accuracies(c) = cm(c,c) / denom;
    else
        class_accuracies(c) = 0;
    end
end
avg_accuracy = mean(class_accuracies);

% Kappa
N_tot = sum(cm(:));
rs = sum(cm,2);
cs = sum(cm,1)';
po = trace(cm) / N_tot;
pe = sum(rs .* cs) / (N_tot^2);
if abs(1-pe) > eps
    kappa = (po-pe) / (1-pe);
else
    kappa = 0;
end

%% Report
fprintf('\n=========================================================\n');
fprintf(' TEST-SET RESULTS - SuperPCA\n');
fprintf('=========================================================\n');
fprintf(' Train samples : %d\n', numel(y_train));
fprintf(' Val samples   : %d\n', numel(y_val));
fprintf(' Test samples  : %d\n', numel(y_test));
fprintf(' OA            : %.2f%%\n', overall_accuracy*100);
fprintf(' AA            : %.2f%%\n', avg_accuracy*100);
fprintf(' Kappa         : %.2f%%\n', kappa*100);
fprintf(' Feature time  : %.2f s\n', feature_time);
fprintf(' Train time    : %.2f s\n', train_time);
for c = 1:num_cls
    fprintf(' Class %2d      : %.2f%%\n', unique_cls(c), class_accuracies(c)*100);
end
fprintf('=========================================================\n');

% Important: X_all_raw is intentionally NOT used to recompute OA/AA/Kappa.
% It may be used only for a full-scene classification-map visualization.

end
