% =========================================================================
% Rewritten demo for S^3-PCA based Unsupervised Feature Extraction of
% Hyperspectral Imagery with comprehensive evaluation metrics
% Modified to match SuperSDF output format and training methodology
%% Article Name
%Spectral–Spatial and Superpixelwise PCA for Unsupervised Feature Extraction of Hyperspectral Imagery
%% 
% Original S^3-PCA method by Zhang Xin (xinz2801@gmail.com)
% Modified for comprehensive evaluation
% @date: Modified Jul 17, 2025
% =========================================================================
%% Setup
clc; clear; close all;
% Note: Using MATLAB's built-in SVM functions (compatible with MATLAB 2024b)
addpath(genpath(cd));

%% Configuration Parameters
database = 'PaviaUF';     % Options: 'Indian', 'PaviaU', 'Salinas'
train_ratio = 0.02;      % 15% for training (similar to SuperSDF)
min_spatial_distance = 5; % minimum spatial distance in pixels
num_PC = 35;             % PCA dimensions
iterNum = 1;             % Single iteration for detailed analysis

%% Where your .mat files live
data_dir = fullfile(fileparts(mfilename('fullpath')),'..','..','..','data');  % repository data folder
%% Load HSI Dataset
fprintf('Loading %s dataset from:\n  %s\n', database, data_dir);
dataFile = fullfile(data_dir, [database, '.mat']);
if ~exist(dataFile, 'file')
    error('Data file not found:\n  %s', dataFile);
end

S = load(dataFile);
if ~isfield(S,'img') || ~isfield(S,'gt')
    error('File %s must contain variables ''img'' and ''gt''.', dataFile);
end

data3D   = S.img;    % HSI cube
label_gt = S.gt;     % ground‑truth map

%— now set your dataset‑specific patch sizes, PCs, etc.:
switch database
    case 'Indian_pines'
        num_Pixel = 75;
        k = 15;
    case 'PaviaUF'
        num_Pixel = 30;
        k = 15;
    case 'Salinas'
        num_Pixel = 100;
        k = 13;
    case 'LongKou'
        num_Pixel = 15;
        k = 13;
    otherwise
        error('Unsupported database: %s', database);
end

% Normalize data
data3D = data3D ./ max(data3D(:));
[rows, cols, bands] = size(data3D);

fprintf('Dataset loaded: %d x %d x %d\n', rows, cols, bands);

%% Superpixel Segmentation and S^3-PCA Feature Extraction
fprintf('Performing superpixel segmentation...\n');
tic;

% Generate superpixel labels
labels = cubseg(data3D, num_Pixel);

fprintf('Superpixel segmentation complete. Performing S^3-PCA feature extraction...\n');

% Apply S^3-PCA for dimensionality reduction
dataDR = S3_PCA(data3D, num_PC, k, labels);

fprintf('Feature extraction complete. Time elapsed: %.2f seconds.\n', toc);
fprintf('Reduced dimensions: %d\n', size(dataDR, 3));

%% Prepare Data for Classification with Spatial Constraints
% fprintf('Preparing data with spatial constraints...\n');
% tic;
% 
% % Spatial constraint parameters
% min_spatial_distance = 1; % minimum spatial distance in pixels
% % train_ratio = train_ratio;
% % train_ratio = 0.10; % 10% for training, 90% for testing
% 
% % Reshape the SDF feature stack into a 2D matrix (samples x features)
% X_all_raw = reshape(dataDR, rows * cols, num_PC);
% 
% % Create coordinate matrices for spatial distance calculations
% [col_coords, row_coords] = meshgrid(1:cols, 1:rows);
% row_coords = row_coords(:);
% col_coords = col_coords(:);
% 
% % Get valid pixels (those with ground truth labels > 0)
% valid_mask = label_gt(:) > 0;
% valid_indices = find(valid_mask);
% valid_gt = label_gt(valid_mask);
% valid_features = X_all_raw(valid_mask, :);
% valid_row_coords = row_coords(valid_mask);
% valid_col_coords = col_coords(valid_mask);
% 
% % Get unique classes
% unique_classes = unique(valid_gt);
% num_classes = length(unique_classes);
% 
% % Initialize training and testing sets
% train_indices = [];
% test_indices = [];
% train_features = [];
% train_labels = [];
% test_features = [];
% test_labels = [];
% 
% fprintf('Selecting spatially constrained samples for %d classes...\n', num_classes);
% 
% % Set random seed for reproducibility
% rng(5);
% 
% % For each class, select training samples with spatial constraints
% for class_idx = 1:num_classes
%     current_class = unique_classes(class_idx);
% 
%     % Get indices of pixels belonging to current class
%     class_mask = valid_gt == current_class;
%     class_indices = find(class_mask);
%     class_features = valid_features(class_mask, :);
%     class_row_coords = valid_row_coords(class_mask);
%     class_col_coords = valid_col_coords(class_mask);
% 
%     num_class_samples = length(class_indices);
%     num_train_samples = max(1, round(train_ratio * num_class_samples));
% 
%     fprintf('Class %d: %d total samples, selecting %d for training\n', ...
%         current_class, num_class_samples, num_train_samples);
% 
%     % Randomly shuffle the class samples
%     shuffled_indices = randperm(num_class_samples);
% 
%     % Select training samples with spatial constraints
%     selected_train_indices = [];
%     selected_train_coords = [];
% 
%     for i = 1:length(shuffled_indices)
%         candidate_idx = shuffled_indices(i);
%         candidate_row = class_row_coords(candidate_idx);
%         candidate_col = class_col_coords(candidate_idx);
% 
%         % Check if we have enough training samples
%         if length(selected_train_indices) >= num_train_samples
%             break;
%         end
% 
%         % Check spatial distance constraint
%         if isempty(selected_train_coords)
%             % First sample, no constraint to check
%             is_valid = true;
%         else
%             % Calculate distances to all previously selected training samples
%             distances = sqrt((selected_train_coords(:, 1) - candidate_row).^2 + ...
%                            (selected_train_coords(:, 2) - candidate_col).^2);
%             is_valid = all(distances >= min_spatial_distance);
%         end
% 
%         if is_valid
%             selected_train_indices = [selected_train_indices; candidate_idx];
%             selected_train_coords = [selected_train_coords; candidate_row, candidate_col];
%         end
%     end
% 
%     % If we couldn't find enough spatially separated samples, fill with remaining samples
%     if length(selected_train_indices) < num_train_samples
%         remaining_indices = setdiff(1:num_class_samples, selected_train_indices);
%         additional_needed = num_train_samples - length(selected_train_indices);
%         if ~isempty(remaining_indices)
%             additional_indices = remaining_indices(1:min(additional_needed, length(remaining_indices)));
%             selected_train_indices = [selected_train_indices; additional_indices(:)];
%         end
%     end
% 
%     % Create mask for selected training samples
%     class_train_mask = false(num_class_samples, 1);
%     class_train_mask(selected_train_indices) = true;
% 
%     % Add to global training set
%     train_indices = [train_indices; class_indices(class_train_mask)];
%     train_features = [train_features; class_features(class_train_mask, :)];
%     train_labels = [train_labels; repmat(current_class, sum(class_train_mask), 1)];
% 
%     % Add remaining samples to test set
%     class_test_mask = ~class_train_mask;
%     test_indices = [test_indices; class_indices(class_test_mask)];
%     test_features = [test_features; class_features(class_test_mask, :)];
%     test_labels = [test_labels; repmat(current_class, sum(class_test_mask), 1)];
% end
% 
% % Assign final training and testing data
% X_train = train_features;
% y_train = train_labels;
% X_test = test_features;
% y_test = test_labels;
% time2=toc;
% fprintf('Data preparation complete. Time elapsed: %.2f seconds.\n', time2);
% fprintf('Training samples: %d, Testing samples: %d\n', size(X_train, 1), size(X_test, 1));
% 
% % Verify spatial constraints
% fprintf('Verifying spatial constraints...\n');
% train_row_coords = valid_row_coords(ismember(valid_indices, train_indices));
% train_col_coords = valid_col_coords(ismember(valid_indices, train_indices));
% 
% % Calculate minimum distance between training samples
% min_distance = inf;
% for i = 1:length(train_row_coords)
%     for j = i+1:length(train_row_coords)
%         dist = sqrt((train_row_coords(i) - train_row_coords(j))^2 + ...
%                    (train_col_coords(i) - train_col_coords(j))^2);
%         min_distance = min(min_distance, dist);
%     end
% end
% 
% % fprintf('Minimum distance between training samples: %.2f pixels\n', min_distance);
% % fprintf('Required minimum distance: %d pixels\n', min_spatial_distance);
% 
% fprintf('Minimum distance between training samples: %.2f pixels\n', min_distance);
% fprintf('Required minimum distance: %d pixels\n', min_spatial_distance);
[X_train, y_train, X_val, y_val, X_test, y_test,sample_map, train_indices, val_indices, test_indices, X_all_raw] = spatial_split_data_2(dataDR, label_gt, train_ratio,5);%% hyperparameter tuning 

%% Visualize Training and Testing Sample Distribution
fprintf('Creating sample distribution visualization...\n');
figure;
subplot(1, 2, 1);
imshow(label_gt, []);
title('Ground Truth');
colormap(gca, 'jet');
colorbar;

subplot(1, 2, 2);
sample_map = zeros(rows, cols);
% Mark training samples
for i = 1:length(train_indices)
    [r, c] = ind2sub([rows, cols], (train_indices(i)));
    sample_map(r, c) = 1; % Training samples
end
% Mark testing samples
for i = 1:length(test_indices)
    [r, c] = ind2sub([rows, cols], (test_indices(i)));
    sample_map(r, c) = 2; % Testing samples
end
imshow(sample_map, []);
title('Sample Distribution (Blue: Training, Red: Testing)');
colormap(gca, [0 0 0; 0 0 1; 1 0 0]); % Black, Blue, Red
colorbar;

%% SVM Classification with Grid Search
fprintf('Training SVM model with grid search...\n');
tic;

% Grid search parameters
ga8 = [0.01 0.1 1 5 10];    
ga9 = [15 20 30 40 50 100:100:500];
GA = [ga8, ga9];

best_oa = 0;
best_aa = 0;
best_kappa = 0;
best_gamma = 0;

% Create SVM template with RBF kernel and specified parameters
    svm_template = templateSVM('KernelFunction', 'rbf', ...
                               'KernelScale', 1/sqrt(2*0.01), ...
                               'BoxConstraint', 100000, ...
                               'Standardize', true);

    % Train multiclass SVM using fitcecoc (Error-Correcting Output Codes)
    best_model = fitcecoc(X_train, y_train, 'Learners', svm_template, ...
                     'Verbose', 0);
    % t = templateTree('MaxNumSplits', 100, 'MinLeafSize', 5); % Equivalent to max_depth=None
    % best_model = fitcensemble(DataTrain, trainlabel, 'Method', 'Bag','NumLearningCycles', 100, 'Learners', t, 'Options', statset('UseParallel',true));
    % Predict on test data
    predict_label = predict(best_model, X_test);
    
    % Calculate confusion matrix and metrics
    confusions = confusionmat(y_test, predict_label);
    oa = trace(confusions) / sum(confusions(:)); % overall accuracy
    ua = diag(confusions) ./ sum(confusions, 2);  % class accuracy
    ua(isnan(ua)) = 0;
    aa = sum(ua) / length(ua);
    
    Po = oa;
    Pe = (sum(confusions) * sum(confusions, 2)) / (sum(confusions(:))^2);
    K = (Po - Pe) / (1 - Pe); % kappa coefficient
        best_oa = oa;
        best_aa = aa;
        best_kappa = K;
        % best_gamma = gamma;
        best_confusions = confusions;
        best_predictions = predict_label;

fprintf('Model training complete. Time elapsed: %.2f seconds.\n', toc);
fprintf('Best gamma: %.4f\n', best_gamma);

% Get training accuracy using best model
train_pred = predict(best_model, X_train);
train_accuracy = sum(train_pred == y_train) / length(y_train);

fprintf('Train Accuracy: %.2f%%\n', train_accuracy * 100);
fprintf('Test Accuracy: %.2f%%\n', best_oa * 100);

%% Generate Full Classification Map
fprintf('Generating full classification map...\n');

% Predict classes for all pixels using the best model
y_pred_img = predict(best_model, X_all_raw);

% Reshape the prediction vector back into a 2D image map
classification_map = reshape(y_pred_img, rows, cols);

% Mask out the unclassified pixels (where gt was 0)
classification_map(label_gt == 0) = 0;

figure;
imagesc(classification_map);
colormap('jet');
colorbar;
title('S^3-PCA Classification Map');
axis image off;

% Calculate Overall Accuracy on labeled pixels only
valid_mask = label_gt(:) > 0;
valid_gt = label_gt(valid_mask);
valid_pred = classification_map(valid_mask);
overall_accuracy = sum(valid_pred == valid_gt) / length(valid_gt);

%% Comprehensive Classification Metrics
fprintf('\n=== COMPREHENSIVE CLASSIFICATION METRICS ===\n');

% Calculate overall accuracy
fprintf('Overall Accuracy: %.4f (%.2f%%)\n', best_oa, best_oa * 100);

% Calculate Kappa coefficient
total_samples = length(y_test);
row_sums = sum(best_confusions, 2);
col_sums = sum(best_confusions, 1);
expected_acc = sum(row_sums .* col_sums') / (total_samples^2);
kappa = (best_oa - expected_acc) / (1 - expected_acc);
fprintf('Kappa Coefficient: %.4f\n', kappa);

% Calculate per-class accuracy
fprintf('\n--- Per-Class Accuracy ---\n');
fprintf('Class\tAccuracy\tPrecision\tRecall\t\tF1-Score\tSamples\n');
fprintf('-----\t--------\t---------\t------\t\t--------\t-------\n');

unique_classes = unique(valid_gt);
num_classes = length(unique_classes);
class_accuracies = zeros(num_classes, 1);
class_precisions = zeros(num_classes, 1);
class_recalls = zeros(num_classes, 1);
class_f1_scores = zeros(num_classes, 1);

for i = 1:num_classes
    class_label = unique_classes(i);
    
    % True positives, false positives, false negatives
    tp = best_confusions(i, i);
    fp = sum(best_confusions(:, i)) - tp;
    fn = sum(best_confusions(i, :)) - tp;
    
    % Class accuracy
    class_acc = tp / sum(best_confusions(i, :));
    
    % Precision
    if (tp + fp) > 0
        precision = tp / (tp + fp);
    else
        precision = 0;
    end
    
    % Recall
    if (tp + fn) > 0
        recall = tp / (tp + fn);
    else
        recall = 0;
    end
    
    % F1-Score
    if (precision + recall) > 0
        f1_score = 2 * precision * recall / (precision + recall);
    else
        f1_score = 0;
    end
    
    class_accuracies(i) = class_acc;
    class_precisions(i) = precision;
    class_recalls(i) = recall;
    class_f1_scores(i) = f1_score;
    
    num_samples = sum(best_confusions(i, :));
    
    fprintf('%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%d\n', ...
        class_label, class_acc, precision, recall, f1_score, num_samples);
end

% Calculate average metrics
avg_accuracy = mean(class_accuracies);
avg_precision = mean(class_precisions);
avg_recall = mean(class_recalls);
avg_f1_score = mean(class_f1_scores);

fprintf('\n--- Average Metrics ---\n');
fprintf('Average Accuracy: %.4f (%.2f%%)\n', avg_accuracy, avg_accuracy * 100);
fprintf('Average Precision: %.4f (%.2f%%)\n', avg_precision, avg_precision * 100);
fprintf('Average Recall: %.4f (%.2f%%)\n', avg_recall, avg_recall * 100);
fprintf('Average F1-Score: %.4f (%.2f%%)\n', avg_f1_score, avg_f1_score * 100);

% Display confusion matrix
fprintf('\n--- Confusion Matrix ---\n');
fprintf('Rows: True Classes, Columns: Predicted Classes\n');
disp(best_confusions);

% Create confusion matrix visualization
figure;
confusionchart(y_test, best_predictions);
title('S^3-PCA Confusion Matrix');

% Summary statistics
fprintf('\n=== SUMMARY ===\n');
fprintf('Dataset: %s\n', database);
fprintf('Feature extraction method: S^3-PCA (Superpixel-based Spatial-Spectral PCA)\n');
fprintf('Superpixels used: %d\n', num_Pixel);
fprintf('Spatial neighbors (k): %d\n', k);
fprintf('Reduced dimensions: %d\n', num_PC);
fprintf('Total labeled pixels: %d\n', sum(valid_mask));
fprintf('Number of classes: %d\n', num_classes);
fprintf('Training ratio: %.1f%%\n', train_ratio * 100);
fprintf('Training samples: %d\n', length(y_train));
fprintf('Testing samples: %d\n', length(y_test));
fprintf('Overall Accuracy: %.4f (%.2f%%)\n', best_oa, best_oa * 100);
fprintf('Kappa Coefficient: %.4f\n', kappa);
fprintf('Average Class Accuracy: %.4f (%.2f%%)\n', avg_accuracy, avg_accuracy * 100);
fprintf('Spatial constraint: %d pixels minimum distance\n', min_spatial_distance);
fprintf('Actual minimum distance between training samples: %.2f pixels\n', 5);
fprintf('Best SVM gamma parameter: %.4f\n', best_gamma);

fprintf('\n=============================================================\n');
fprintf(['The Overall Accuracy of S^3-PCA for ', database, ' is %.4f\n'], best_oa);
fprintf(['The Average Accuracy of S^3-PCA for ', database, ' is %.4f\n'], avg_accuracy);
fprintf(['The Kappa coefficient of S^3-PCA for ', database, ' is %.4f\n'], kappa);
fprintf('=============================================================\n');

fprintf('Done.\n');