% =========================================================================
% A simple demo for S^3-PCA based Unsupervised Feature Extraction of
% Hyperspectral Imagery
% If you  have any problem, do not hesitate to contact
% Zhang Xin (xinz2801@gmail.com)
% Version 1,2020-09-03


clc;clear;close all
addpath(fullfile(fileparts(mfilename('fullpath')),'..','..','..','data'));  % repository data folder
addpath(genpath(cd));

num_PC           =   35;  % THE OPTIMAL PCA DIMENSION.
% num_Pixel        =   100; % THE OPTIMAL Number of Superpixel. Indian:75, PaviaU:30, Salinas:100
% k                =   15;  % THE OPTIMAL Number of spatial neighbors.Indian:15, PaviaU:13, Salinas:15
% trainpercentage  =   10;  % Training Number per Class
iterNum          =   5;  % The Iteration Number

database         =   'LongKou';

%% load the HSI dataset
if strcmp(database,'Indian')
    load Indian_pines_corrected;load Indian_pines_gt;load Indian_pines_randp 
    data3D = indian_pines_corrected;        label_gt = indian_pines_gt;
    num_Pixel        =   75;
    k                =   15;
elseif strcmp(database,'Salinas')
    load Salinas_corrected;load Salinas_gt;load Salinas_randp
    data3D = salinas_corrected;        label_gt = salinas_gt;
    num_Pixel        =   100;
    k                =   15;
elseif strcmp(database,'PaviaU')    
    S=load ('PaviaUF');%load PaviaU_gt;load PaviaU_randp; 
    data3D = S.img;        label_gt = S.gt;
    num_Pixel        =   30;
    k                =   13;
elseif strcmp(database,'LongKou')    
    S=load ('LongKou');%load PaviaU_gt;load PaviaU_randp; 
    data3D = S.img;        label_gt = S.gt;
    num_Pixel        =   30;
    k                =   15;
end
data3D = data3D./max(data3D(:));

%% super-pixels segmentation
labels = cubseg(data3D,num_Pixel);

%% SupePCA based DR
[dataDR] = S3_PCA(data3D,num_PC,k,labels);
[rows, cols, num_features] = size(dataDR);
% [X_train, y_train, X_test, y_test, sample_map, train_indices, test_indices,X_all_raw] = random_split_data(dataDR, label_gt, 0.01, 1);
seed = (5)*7 + 13;
[X_train, y_train, X_val, y_val, X_test, y_test, sample_map, train_indices, val_indices, test_indices, X_all_raw, split_info] = ...
    spatial_block_split_data_guarded(dataDR, label_gt, 0.01, 10,5,5,10,seed,true);
%% Train and Evaluate Random Forest Model
fprintf('Training model...\n');
tic;

% Create and train a Random Forest model.
% t = templateTree('MaxNumSplits', 100, 'MinLeafSize', 5); % Equivalent to max_depth=None
% model = fitcensemble(X_train, y_train, 'Method', 'Bag','NumLearningCycles', 100, 'Learners', t, 'Options', statset('UseParallel',true));
svm_template = templateSVM('KernelFunction', 'rbf', ...
                               'KernelScale', 1/sqrt(0.01), ...
                               'BoxConstraint', 100000, ...
                               'Standardize', true);

    % Train multiclass SVM using fitcecoc (Error-Correcting Output Codes)
 model = fitcecoc(X_train, y_train, 'Learners', svm_template, ...
                     'Verbose', 0);
fprintf('Model training complete. Time elapsed: %.2f seconds.\n', toc);

% Evaluate model accuracy
y_train_pred = predict(model, X_train);
train_accuracy = sum(y_train_pred == y_train) / numel(y_train);

y_test_pred = predict(model, X_test);
test_accuracy = sum(y_test_pred == y_test) / numel(y_test);

fprintf('Train Accuracy: %.2f%%\n', train_accuracy * 100);
fprintf('Test Accuracy: %.2f%%\n', test_accuracy * 100);

%% Generate and Visualize Full Classification Map
fprintf('Generating full classification map...\n');

% Create valid_mask and valid_gt for compatibility with original evaluation code
valid_mask = label_gt > 0;  % Mask for pixels that have ground truth labels
valid_gt = label_gt(valid_mask);  % Ground truth labels for valid pixels

% Predict classes for all pixels in the original image
y_pred_img = predict(model, X_all_raw);

% Reshape the prediction vector back into a 2D image map
classification_map = reshape(y_pred_img, rows, cols);

% Mask out the unclassified pixels (where gt was 0) for better visualization
classification_map(label_gt == 0) = 0;

figure;
imagesc(classification_map);
colormap('jet');
colorbar;
title('Classification Map');
axis image off;

% Calculate Overall Accuracy on labeled pixels only
% Note: We use the full map prediction but evaluate only on valid pixels.
valid_pred = classification_map(valid_mask);
valid_gt = label_gt(valid_mask);
overall_accuracy = sum(valid_pred == valid_gt) / numel(valid_gt);
fprintf('Overall Classification Accuracy (on labeled pixels): %.2f%%\n', overall_accuracy * 100);

fprintf('Done.\n');

%% Comprehensive Classification Metrics
fprintf('\n=== COMPREHENSIVE CLASSIFICATION METRICS ===\n');

% Get unique classes from ground truth
unique_classes = unique(valid_gt);
num_classes = length(unique_classes);

% Create confusion matrix
conf_matrix = confusionmat(valid_gt, valid_pred);

% Calculate overall accuracy
overall_acc = sum(valid_pred == valid_gt) / numel(valid_gt);
fprintf('Overall Accuracy: %.4f (%.2f%%)\n', overall_acc, overall_acc * 100);

% Calculate Kappa coefficient
% Expected accuracy (chance agreement)
total_samples = length(valid_gt);
row_sums = sum(conf_matrix, 2);
col_sums = sum(conf_matrix, 1);
expected_acc = sum(row_sums .* col_sums') / (total_samples^2);

% Kappa coefficient
kappa = (overall_acc - expected_acc) / (1 - expected_acc);
fprintf('Kappa Coefficient: %.4f\n', kappa);

% Calculate per-class accuracy
fprintf('\n--- Per-Class Accuracy ---\n');
fprintf('Class\tAccuracy\tPrecision\tRecall\t\tF1-Score\tSamples\n');
fprintf('-----\t--------\t---------\t------\t\t--------\t-------\n');

class_accuracies = zeros(num_classes, 1);
class_precisions = zeros(num_classes, 1);
class_recalls = zeros(num_classes, 1);
class_f1_scores = zeros(num_classes, 1);

for i = 1:num_classes
    class_label = unique_classes(i);
    
    % True positives, false positives, false negatives
    tp = conf_matrix(i, i);
    fp = sum(conf_matrix(:, i)) - tp;
    fn = sum(conf_matrix(i, :)) - tp;
    tn = sum(conf_matrix(:)) - tp - fp - fn;
    
    % Class accuracy (correct predictions for this class / total predictions for this class)
    class_acc = tp / sum(conf_matrix(i, :));
    
    % Precision (tp / (tp + fp))
    if (tp + fp) > 0
        precision = tp / (tp + fp);
    else
        precision = 0;
    end
    
    % Recall (tp / (tp + fn))
    if (tp + fn) > 0
        recall = tp / (tp + fn);
    else
        recall = 0;
    end
    
    % F1-Score (2 * precision * recall / (precision + recall))
    if (precision + recall) > 0
        f1_score = 2 * precision * recall / (precision + recall);
    else
        f1_score = 0;
    end
    
    class_accuracies(i) = class_acc;
    class_precisions(i) = precision;
    class_recalls(i) = recall;
    class_f1_scores(i) = f1_score;
    
    num_samples = sum(conf_matrix(i, :));
    
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
disp(conf_matrix);

% Create a more detailed confusion matrix visualization
figure;
confusionchart(valid_gt, valid_pred);
title('Confusion Matrix');

% Summary statistics
fprintf('\n=== SUMMARY ===\n');
fprintf('Total labeled pixels: %d\n', total_samples);
fprintf('Number of classes: %d\n', num_classes);
fprintf('Overall Accuracy: %.4f (%.2f%%)\n', overall_acc, overall_acc * 100);
fprintf('Kappa Coefficient: %.4f\n', kappa);
fprintf('Average Class Accuracy: %.4f (%.2f%%)\n', avg_accuracy, avg_accuracy * 100);