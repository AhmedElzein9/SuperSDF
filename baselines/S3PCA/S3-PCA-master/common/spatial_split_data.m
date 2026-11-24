function [X_train, y_train, X_test, y_test, sample_map, train_indices, test_indices,X_all_raw] = spatial_split_data(sdf_images, gt, train_ratio, min_spatial_distance, visualize)
% SPATIAL_SPLIT_DATA - Split hyperspectral data with spatial constraints
%
% This function splits hyperspectral image data into training and testing sets
% while enforcing spatial constraints to reduce spatial autocorrelation bias
% and improve generalization performance.
%
% Inputs:
%   sdf_images - 3D array (rows x cols x features) of SDF features
%   gt - 2D array (rows x cols) of ground truth labels
%   train_ratio - Proportion of samples to use for training (0 to 1)
%   min_spatial_distance - Minimum spatial distance between training samples (default: 1)
%   visualize - Boolean flag to show visualization (default: true)
%
% Outputs:
%   X_train - Training features matrix (n_train_samples x n_features)
%   y_train - Training labels vector (n_train_samples x 1)
%   X_test - Testing features matrix (n_test_samples x n_features)
%   y_test - Testing labels vector (n_test_samples x 1)
%   sample_map - 2D array showing sample distribution (0=unlabeled, 1=train, 2=test)
%   train_indices - Linear indices of training samples
%   test_indices - Linear indices of testing samples
%
% Example:
%   [X_train, y_train, X_test, y_test, sample_map] = spatial_split_data(sdf_images, gt, 0.1, 2, true);
%
% Author: Based on hyperspectral classification code
% Date: 2025

    % Default parameters
    if nargin < 4 || isempty(min_spatial_distance)
        min_spatial_distance = 1;
    end
    if nargin < 5 || isempty(visualize)
        visualize = true;
    end
    
    fprintf('Preparing data with spatial constraints...\n');
    tic;
    
    % Get dimensions
    [rows, cols, num_features] = size(sdf_images);
    
    % Reshape the SDF feature stack into a 2D matrix (samples x features)
    X_all_raw = reshape(sdf_images, rows * cols, num_features);
    
    % Create coordinate matrices for spatial distance calculations
    [col_coords, row_coords] = meshgrid(1:cols, 1:rows);
    row_coords = row_coords(:);
    col_coords = col_coords(:);
    
    % Get valid pixels (those with ground truth labels > 0)
    valid_mask = gt(:) > 0;
    valid_indices = find(valid_mask);
    valid_gt = gt(valid_mask);
    valid_features = X_all_raw(valid_mask, :);
    valid_row_coords = row_coords(valid_mask);
    valid_col_coords = col_coords(valid_mask);
    
    % Get unique classes
    unique_classes = unique(valid_gt);
    num_classes = length(unique_classes);
    
    % Initialize training and testing sets
    train_indices = [];
    test_indices = [];
    train_features = [];
    train_labels = [];
    test_features = [];
    test_labels = [];
    
    fprintf('Selecting spatially constrained samples for %d classes...\n', num_classes);
    
    % Use current time as seed for true randomness
    rng('shuffle');
    
    % For each class, select training samples with spatial constraints
    for class_idx = 1:num_classes
        current_class = unique_classes(class_idx);
        
        % Get indices of pixels belonging to current class
        class_mask = valid_gt == current_class;
        class_indices = find(class_mask);
        class_features = valid_features(class_mask, :);
        class_row_coords = valid_row_coords(class_mask);
        class_col_coords = valid_col_coords(class_mask);
        
        num_class_samples = length(class_indices);
        num_train_samples = max(1, round(train_ratio * num_class_samples));
        
        fprintf('Class %d: %d total samples, selecting %d for training\n', ...
            current_class, num_class_samples, num_train_samples);
        
        % Enhanced randomization approach
        shuffled_indices = randperm(num_class_samples);
        for shuffle_iter = 1:10  % Multiple shuffles for better randomization
            shuffled_indices = shuffled_indices(randperm(length(shuffled_indices)));
        end
        
        % Add random starting point
        if num_class_samples > 1
            random_start = randi(num_class_samples);
            shuffled_indices = [shuffled_indices(random_start:end), shuffled_indices(1:random_start-1)];
        end
        
        % Select training samples with spatial constraints
        selected_train_indices = [];
        selected_train_coords = [];
        
        % Enhanced selection with multiple attempts
        max_attempts = num_class_samples * 2;
        attempt_count = 0;
        
        for i = 1:length(shuffled_indices)
            candidate_idx = shuffled_indices(i);
            candidate_row = class_row_coords(candidate_idx);
            candidate_col = class_col_coords(candidate_idx);
            
            attempt_count = attempt_count + 1;
            
            % Check if we have enough training samples
            if length(selected_train_indices) >= num_train_samples
                break;
            end
            
            % Stop if too many attempts (avoid infinite loops)
            if attempt_count > max_attempts
                break;
            end
            
            % Check spatial distance constraint
            if isempty(selected_train_coords)
                % First sample, no constraint to check
                is_valid = true;
            else
                % Calculate distances to all previously selected training samples
                distances = sqrt((selected_train_coords(:, 1) - candidate_row).^2 + ...
                               (selected_train_coords(:, 2) - candidate_col).^2);
                is_valid = all(distances >= min_spatial_distance);
            end
            
            if is_valid
                selected_train_indices = [selected_train_indices; candidate_idx];
                selected_train_coords = [selected_train_coords; candidate_row, candidate_col];
            end
        end
        
        % If we couldn't find enough spatially separated samples, use adaptive approach
        if length(selected_train_indices) < num_train_samples
            remaining_indices = setdiff(1:num_class_samples, selected_train_indices);
            additional_needed = num_train_samples - length(selected_train_indices);
            
            if ~isempty(remaining_indices)
                % Randomly shuffle remaining indices
                remaining_indices = remaining_indices(randperm(length(remaining_indices)));
                
                % Try with relaxed spatial constraint if needed
                relaxed_distance = max(1, min_spatial_distance * 0.7); % 70% of original distance
                
                for idx = remaining_indices
                    if additional_needed <= 0
                        break;
                    end
                    
                    candidate_row = class_row_coords(idx);
                    candidate_col = class_col_coords(idx);
                    
                    % Check relaxed spatial constraint
                    if isempty(selected_train_coords)
                        is_valid = true;
                    else
                        distances = sqrt((selected_train_coords(:, 1) - candidate_row).^2 + ...
                                       (selected_train_coords(:, 2) - candidate_col).^2);
                        is_valid = all(distances >= relaxed_distance);
                    end
                    
                    if is_valid
                        selected_train_indices = [selected_train_indices; idx];
                        selected_train_coords = [selected_train_coords; candidate_row, candidate_col];
                        additional_needed = additional_needed - 1;
                    end
                end
                
                % If still not enough, add remaining samples without spatial constraint
                if additional_needed > 0
                    remaining_after_relaxed = setdiff(remaining_indices, selected_train_indices);
                    if ~isempty(remaining_after_relaxed)
                        additional_indices = remaining_after_relaxed(1:min(additional_needed, length(remaining_after_relaxed)));
                        selected_train_indices = [selected_train_indices; additional_indices(:)];
                    end
                end
            end
        end
        
        % Create mask for selected training samples
        class_train_mask = false(num_class_samples, 1);
        class_train_mask(selected_train_indices) = true;
        
        % Add to global training set
        train_indices = [train_indices; class_indices(class_train_mask)];
        train_features = [train_features; class_features(class_train_mask, :)];
        train_labels = [train_labels; repmat(current_class, sum(class_train_mask), 1)];
        
        % Add remaining samples to test set
        class_test_mask = ~class_train_mask;
        test_indices = [test_indices; class_indices(class_test_mask)];
        test_features = [test_features; class_features(class_test_mask, :)];
        test_labels = [test_labels; repmat(current_class, sum(class_test_mask), 1)];
    end
    
    % Final random shuffle of training and testing data for additional randomization
    train_shuffle_idx = randperm(length(train_labels));
    X_train = train_features(train_shuffle_idx, :);
    y_train = train_labels(train_shuffle_idx);
    
    test_shuffle_idx = randperm(length(test_labels));
    X_test = test_features(test_shuffle_idx, :);
    y_test = test_labels(test_shuffle_idx);
    
    fprintf('Data preparation complete. Time elapsed: %.2f seconds.\n', toc);
    fprintf('Training samples: %d, Testing samples: %d\n', size(X_train, 1), size(X_test, 1));
    
    % Verify spatial constraints
    fprintf('Verifying spatial constraints...\n');
    train_row_coords = valid_row_coords(ismember(valid_indices, train_indices));
    train_col_coords = valid_col_coords(ismember(valid_indices, train_indices));
    
    % Calculate minimum distance between training samples
    min_distance = inf;
    if length(train_row_coords) > 1
        for i = 1:length(train_row_coords)
            for j = i+1:length(train_row_coords)
                dist = sqrt((train_row_coords(i) - train_row_coords(j))^2 + ...
                           (train_col_coords(i) - train_col_coords(j))^2);
                min_distance = min(min_distance, dist);
            end
        end
        fprintf('Minimum distance between training samples: %.2f pixels\n', min_distance);
    end
    fprintf('Required minimum distance: %d pixels\n', min_spatial_distance);
    
    % Create sample distribution map
    sample_map = zeros(rows, cols);
    % Mark training samples
    for i = 1:length(train_indices)
        [r, c] = ind2sub([rows, cols], valid_indices(train_indices(i)));
        sample_map(r, c) = 1; % Training samples
    end
    % Mark testing samples  
    for i = 1:length(test_indices)
        [r, c] = ind2sub([rows, cols], valid_indices(test_indices(i)));
        sample_map(r, c) = 2; % Testing samples
    end
    
    % Visualize training and testing sample distribution
    if visualize
        fprintf('Creating sample distribution visualization...\n');
        figure;
        subplot(1, 2, 1);
        imshow(gt, []);
        title('Ground Truth');
        colormap(gca, 'jet');
        colorbar;
        
        subplot(1, 2, 2);
        imshow(sample_map, []);
        title('Sample Distribution (Blue: Training, Red: Testing)');
        colormap(gca, [0 0 0; 0 0 1; 1 0 0]); % Black, Blue, Red
        colorbar;
    end
    
end