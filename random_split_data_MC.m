function [X_train, y_train, X_val, y_val, X_test, y_test, ...
          sample_map, train_indices, val_indices, test_indices, X_all_raw] = ...
          random_split_data_MC(sdf_images, gt, train_ratio, min_train_per_class, visualize)
% RANDOM_SPLIT_DATA_MC
% Exact class-wise random split protocol used for the SuperSDF paper.
%
%   n_train = max(min_train_per_class, round(train_ratio*n_class))
%   n_val   = n_train
%   test    = all remaining labeled samples
%
% No spatial constraint is applied.

if nargin < 4 || isempty(min_train_per_class)
    min_train_per_class = 5;
end

if nargin < 5 || isempty(visualize)
    visualize = false;
end

[rows, cols, n_feat] = size(sdf_images);
X_all_raw = reshape(sdf_images, rows*cols, n_feat);

valid_mask    = gt(:) > 0;
valid_indices = find(valid_mask);
valid_gt      = gt(valid_mask);
valid_feat    = X_all_raw(valid_mask,:);

unique_classes = unique(valid_gt);
n_cls = numel(unique_classes);

TR_idx = cell(n_cls,1);
VL_idx = cell(n_cls,1);
TE_idx = cell(n_cls,1);

for ci = 1:n_cls
    cls = unique_classes(ci);
    c_idx = find(valid_gt == cls);
    n_tot = numel(c_idx);

    n_train_ratio = round(train_ratio * n_tot);
    n_train = max(min_train_per_class, n_train_ratio);
    n_val = n_train;

    while (n_train + n_val) >= n_tot && n_val > 0
        n_val = n_val - 1;
    end

    while (n_train + n_val) >= n_tot && n_train > 1
        n_train = n_train - 1;
    end

    perm = c_idx(randperm(n_tot));

    tr_local = perm(1:n_train);
    vl_local = perm(n_train+1:n_train+n_val);
    te_local = perm(n_train+n_val+1:end);

    TR_idx{ci} = tr_local(:);
    VL_idx{ci} = vl_local(:);
    TE_idx{ci} = te_local(:);
end

train_indices = vertcat(TR_idx{:});
val_indices   = vertcat(VL_idx{:});
test_indices  = vertcat(TE_idx{:});

train_indices = train_indices(randperm(numel(train_indices)));
val_indices   = val_indices(randperm(numel(val_indices)));
test_indices  = test_indices(randperm(numel(test_indices)));

X_train = valid_feat(train_indices,:);
y_train = valid_gt(train_indices);

X_val = valid_feat(val_indices,:);
y_val = valid_gt(val_indices);

X_test = valid_feat(test_indices,:);
y_test = valid_gt(test_indices);

sample_map = zeros(rows,cols);
sample_map(valid_indices(test_indices))  = 3;
sample_map(valid_indices(val_indices))   = 2;
sample_map(valid_indices(train_indices)) = 1;

if visualize
    figure('Name','Random Split');
    imagesc(sample_map);
    axis image off;
    colormap([0 0 0; 0 0.7 0; 0.9 0.7 0; 0.4 0.4 1]);
    colorbar('Ticks',[0 1 2 3], ...
             'TickLabels',{'BG','Train','Val','Test'});
    title(sprintf('Random split - train %.1f%% per class',100*train_ratio));
end

end
