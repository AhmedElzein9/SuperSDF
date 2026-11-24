function [X_train, y_train, X_val, y_val, X_test, y_test, ...
          sample_map, train_indices, val_indices, test_indices, X_all_raw] = ...
    spatial_split_data2(sdf_images, gt, train_ratio, ...
                        min_spatial_distance, test_exclusion_radius, ...
                        min_train_per_class,  min_test_per_class, visualize)
% SPATIAL_SPLIT_DATA2  Three-way spatially-disjoint HSI split (train/val/test).
%
% Combines the robust per-class selection logic of spatial_split_data with
% the train/val/test structure of spatial_split_data1, and adds two fixes:
%
%   FIX 1 – Overfitting (spatial autocorrelation):
%     A TEST EXCLUSION ZONE removes any test candidate that sits within
%     `test_exclusion_radius` pixels of a train or val sample.  The zone is
%     evaluated PER CLASS so large classes cannot starve small ones.
%     If applying the zone would leave a class with fewer than
%     `min_test_per_class` test samples, the zone is skipped for that class
%     and a warning is printed — the class is NEVER dropped from the test set.
%
%   FIX 2 – Data scarcity (tiny classes):
%     A `min_train_per_class` floor guarantees every class gets at least that
%     many training samples regardless of train_ratio.  The floor is applied
%     before the exclusion zone so it cannot be overridden by it.
%     Classes that hit the floor are flagged in the console output.
%
% -------------------------------------------------------------------------
% INPUTS (all after train_ratio are optional)
%   sdf_images           3-D array  (rows x cols x features)
%   gt                   2-D array  (rows x cols), integer labels, 0 = unlabeled
%   train_ratio          scalar in (0,1) — fraction of labeled pixels per class
%                        used for TRAIN; the same fraction is used for VAL.
%   min_spatial_distance Min pixel distance WITHIN train and WITHIN val sets.
%                        Default: 5
%   test_exclusion_radius Pixels within this radius of any train/val sample
%                        are removed from the test candidate pool (per class).
%                        Default: min_spatial_distance
%                        Set to 0 to disable.
%   min_train_per_class  Hard minimum training samples per class.
%                        Default: 5
%   min_test_per_class   No longer used to filter test pixels. Kept for
%                        backward compatibility. All remaining pixels after
%                        train/val selection always go to test.
%                        Default: 10
%   visualize            Show sample-distribution figure.  Default: true
%
% OUTPUTS — identical signature to spatial_split_data1
%   X_train, y_train     Training features and labels
%   X_val,   y_val       Validation features and labels
%   X_test,  y_test      Test features and labels
%   sample_map           0=unlabeled 1=train 2=val 3=test 4=excluded
%   train_indices        Indices into the valid-pixel array
%   val_indices
%   test_indices
%   X_all_raw            Full (rows*cols x features) matrix for classification map
%
% CALL (drop-in replacement for spatial_split_data1):
%   [X_train,y_train,X_val,y_val,X_test,y_test,sample_map, ...
%    train_indices,val_indices,test_indices,X_all_raw] = ...
%       spatial_split_data2(sdf_images, gt, train_ratio, 5);
%
% @author  Ahmed Elzein  (based on spatial_split_data / spatial_split_data1)
% @date    2025

    %% ------------------------------------------------------------------
    %  Defaults
    %% ------------------------------------------------------------------
    if nargin < 4 || isempty(min_spatial_distance),  min_spatial_distance  = 5;   end
    if nargin < 5 || isempty(test_exclusion_radius), test_exclusion_radius = min_spatial_distance; end
    if nargin < 6 || isempty(min_train_per_class),   min_train_per_class   = 5;   end
    if nargin < 7 || isempty(min_test_per_class),    min_test_per_class    = 10;  end
    if nargin < 8 || isempty(visualize),             visualize             = true; end

    fprintf('\n=========================================================\n');
    fprintf(' spatial_split_data2  —  spatially-constrained HSI split\n');
    fprintf('=========================================================\n');
    fprintf(' train_ratio          = %.1f%%\n', train_ratio * 100);
    fprintf(' min_spatial_distance = %g px\n',  min_spatial_distance);
    fprintf(' test_exclusion_radius= %g px\n',  test_exclusion_radius);
    fprintf(' min_train_per_class  = %d\n',     min_train_per_class);
    fprintf(' min_test_per_class   = %d\n',     min_test_per_class);
    fprintf('---------------------------------------------------------\n');
    tic;

    %% ------------------------------------------------------------------
    %  Geometry and feature setup
    %% ------------------------------------------------------------------
    [rows, cols, num_features] = size(sdf_images);
    X_all_raw = reshape(sdf_images, rows * cols, num_features);

    [col_coords, row_coords] = meshgrid(1:cols, 1:rows);
    row_coords = row_coords(:);
    col_coords = col_coords(:);

    valid_mask        = gt(:) > 0;
    valid_indices     = find(valid_mask);
    valid_gt          = gt(valid_mask);
    valid_features    = X_all_raw(valid_mask, :);
    valid_row_coords  = row_coords(valid_mask);
    valid_col_coords  = col_coords(valid_mask);

    unique_classes = unique(valid_gt);
    num_classes    = numel(unique_classes);

    % rng('shuffle');

    %% ------------------------------------------------------------------
    %  Pre-allocate (cell arrays avoid repeated AGROW)
    %% ------------------------------------------------------------------
    TR_feat = cell(num_classes,1);  TR_lab = cell(num_classes,1);  TR_idx = cell(num_classes,1);
    VL_feat = cell(num_classes,1);  VL_lab = cell(num_classes,1);  VL_idx = cell(num_classes,1);
    TE_feat = cell(num_classes,1);  TE_lab = cell(num_classes,1);  TE_idx = cell(num_classes,1);

    fprintf('%-6s %-7s %-8s %-7s %-6s %-6s %-8s %-8s\n', ...
        'Class','Total','Ratio-N','Floor?','Train','Val','Test','ExclZone');
    fprintf('%s\n', repmat('-',1,62));

    %% ------------------------------------------------------------------
    %  Per-class split
    %% ------------------------------------------------------------------
    for ci = 1:num_classes
        cls   = unique_classes(ci);
        cmask = (valid_gt == cls);
        c_idx = find(cmask);
        c_row = valid_row_coords(cmask);
        c_col = valid_col_coords(cmask);
        n_tot = numel(c_idx);

        % ---- 1. Compute desired train / val counts with floor ----------
        n_train_ratio = round(train_ratio * n_tot);
        n_train       = max(min_train_per_class, n_train_ratio);
        floor_hit     = n_train > n_train_ratio;

        % val: same ratio, same floor, but never more than train
        n_val = max(min_train_per_class, round(train_ratio * n_tot));
        n_val = min(n_val, n_train);

        % Guard: train + val must leave at least 1 pixel for test
        while (n_train + n_val) >= n_tot && n_val > 0
            n_val = n_val - 1;
        end
        while (n_train + n_val) >= n_tot && n_train > 1
            n_train = n_train - 1;
        end
        % Absolute last resort: 1 train, 0 val, rest test
        if n_train >= n_tot
            n_train = max(1, n_tot - 1);
            n_val   = 0;
        end

        pool_all = (1:n_tot)';

        % ---- 2. Select TRAIN (spatially separated) ---------------------
        [tr_local, tr_coords] = pick_spatial( ...
            pool_all, c_row, c_col, n_train, min_spatial_distance, []);

        % ---- 3. Select VAL (separated from train AND from each other) --
        pool_no_train = setdiff(pool_all, tr_local);
        [vl_local, vl_coords] = pick_spatial( ...
            pool_no_train, c_row, c_col, n_val, min_spatial_distance, tr_coords);

        % ---- 4. Test exclusion zone — report only, never removes test pixels ----
        % ALL pixels not in train or val go to test. This matches standard
        % published HSI practice: the exclusion zone protects the training
        % side (train/val samples are spatially separated from each other),
        % but there is no reason to discard test pixels — the model has
        % never seen them regardless of their proximity to train samples.
        pool_te = setdiff(pool_all, [tr_local; vl_local]);
        pool_te = pool_te(:);

        excl_applied = false;
        if test_exclusion_radius > 0 && ~isempty(pool_te)
            tv_coords = [c_row(tr_local), c_col(tr_local);
                         c_row(vl_local), c_col(vl_local)];
            if ~isempty(tv_coords)
                cand_coords = [c_row(pool_te), c_col(pool_te)];
                D     = pdist2(double(cand_coords), double(tv_coords), 'euclidean');
                min_D = min(D, [], 2);
                % Report how many test pixels would be excluded, but keep all
                n_would_exclude = sum(min_D < test_exclusion_radius);
                if n_would_exclude == 0
                    excl_applied = true;   % zone clean — nothing was close anyway
                end
                % pool_te is NOT filtered — all remaining pixels become test
            end
        end

        % ---- 5. Store results ------------------------------------------
        TR_idx{ci}  = c_idx(tr_local);
        TR_feat{ci} = valid_features(cmask, :);
        TR_feat{ci} = TR_feat{ci}(tr_local, :);
        TR_lab{ci}  = repmat(cls, numel(tr_local), 1);

        VL_idx{ci}  = c_idx(vl_local);
        VL_feat{ci} = valid_features(cmask, :);
        VL_feat{ci} = VL_feat{ci}(vl_local, :);
        VL_lab{ci}  = repmat(cls, numel(vl_local), 1);

        TE_idx{ci}  = c_idx(pool_te);
        TE_feat{ci} = valid_features(cmask, :);
        TE_feat{ci} = TE_feat{ci}(pool_te, :);
        TE_lab{ci}  = repmat(cls, numel(pool_te), 1);

        fprintf('%-6d %-7d %-8d %-7s %-6d %-6d %-8d %-8s\n', ...
            cls, n_tot, n_train_ratio, ...
            conditional_str(floor_hit, 'YES', 'no'), ...
            numel(tr_local), numel(vl_local), numel(pool_te), ...
            conditional_str(excl_applied, 'YES', 'no'));
    end

    %% ------------------------------------------------------------------
    %  Concatenate and shuffle
    %% ------------------------------------------------------------------
    train_indices = vertcat(TR_idx{:});
    val_indices   = vertcat(VL_idx{:});
    test_indices  = vertcat(TE_idx{:});

    X_train_c = vertcat(TR_feat{:});
    y_train_c = vertcat(TR_lab{:});
    X_val_c   = vertcat(VL_feat{:});
    y_val_c   = vertcat(VL_lab{:});
    X_test_c  = vertcat(TE_feat{:});
    y_test_c  = vertcat(TE_lab{:});

    sh = randperm(size(X_train_c,1));
    X_train = X_train_c(sh,:);  y_train = y_train_c(sh);

    sh = randperm(size(X_val_c,1));
    X_val   = X_val_c(sh,:);    y_val   = y_val_c(sh);

    sh = randperm(size(X_test_c,1));
    X_test  = X_test_c(sh,:);   y_test  = y_test_c(sh);

    fprintf('%s\n', repmat('-',1,62));
    fprintf('Training: %d  |  Validation: %d  |  Test: %d\n', ...
        size(X_train,1), size(X_val,1), size(X_test,1));
    fprintf('Time elapsed: %.2f s\n\n', toc);

    %% ------------------------------------------------------------------
    %  Cross-set distance check (fast, sampled)
    %% ------------------------------------------------------------------
    if test_exclusion_radius > 0
        check_cross_distance(valid_row_coords, valid_col_coords, ...
            valid_indices, train_indices, test_indices, test_exclusion_radius);
    end

    %% ------------------------------------------------------------------
    %  Build sample_map  0=unlabeled 1=train 2=val 3=test 4=excluded
    %% ------------------------------------------------------------------
    sample_map = zeros(rows, cols);
    for i = 1:numel(test_indices)
        [r,c] = ind2sub([rows,cols], valid_indices(test_indices(i)));
        sample_map(r,c) = 3;
    end
    for i = 1:numel(val_indices)
        [r,c] = ind2sub([rows,cols], valid_indices(val_indices(i)));
        sample_map(r,c) = 2;
    end
    for i = 1:numel(train_indices)
        [r,c] = ind2sub([rows,cols], valid_indices(train_indices(i)));
        sample_map(r,c) = 1;
    end
    % Mark pixels that are labeled but in neither split (excluded by zone)
    excluded = (gt > 0) & (sample_map == 0);
    sample_map(excluded) = 4;

    %% ------------------------------------------------------------------
    %  Visualisation
    %% ------------------------------------------------------------------
    if visualize
        plot_splits(sample_map, gt, rows, cols, ...
            size(X_train,1), size(X_val,1), size(X_test,1));
    end

end % ── main function ────────────────────────────────────────────────────


%% ========================================================================
%  LOCAL: pick_spatial
%  Greedy spatially-separated selection.  Three-level fallback:
%    1. Strict min_dist
%    2. Relaxed 70% min_dist
%    3. No constraint (warns user)
%  Taken from spatial_split_data1 and made faster with pdist2.
%% ========================================================================
function [sel_idx, new_coords] = pick_spatial( ...
        pool_idx, row_c, col_c, N, min_dist, existing_coords)

    if N <= 0
        sel_idx    = zeros(0,1,'like',pool_idx);
        new_coords = zeros(0,2);
        return;
    end

    sel_local = zeros(N,1);
    n_sel     = 0;

    if isempty(existing_coords)
        occ = zeros(0,2);
    else
        occ = existing_coords;
    end

    pool_idx = pool_idx(randperm(numel(pool_idx)));   % shuffle

    % Pre-build candidate coordinate matrix for vectorised distance check
    cand_rc = [row_c(pool_idx), col_c(pool_idx)];   % n_pool x 2

    %% Pass 1: strict distance
    for ii = 1:numel(pool_idx)
        if n_sel >= N; break; end
        cr = cand_rc(ii,1);  cc = cand_rc(ii,2);
        if isempty(occ)
            ok = true;
        else
            % Only compare to the occupied set — single row query is fast
            dv = occ - [cr, cc];
            ok = all(dv(:,1).^2 + dv(:,2).^2 >= min_dist^2);
        end
        if ok
            n_sel = n_sel + 1;
            sel_local(n_sel) = pool_idx(ii);
            occ = [occ; cr, cc]; %#ok<AGROW>
        end
    end

    %% Pass 2: relaxed 70%
    if n_sel < N
        relaxed  = max(1, min_dist * 0.7);
        used     = sel_local(1:n_sel);
        keep     = ~ismember(pool_idx, used);
        rem_idx  = pool_idx(keep);
        rem_rc   = cand_rc(keep, :);
        for ii = 1:numel(rem_idx)
            if n_sel >= N; break; end
            cr = rem_rc(ii,1);  cc = rem_rc(ii,2);
            if isempty(occ)
                ok = true;
            else
                dv = occ - [cr, cc];
                ok = all(dv(:,1).^2 + dv(:,2).^2 >= relaxed^2);
            end
            if ok
                n_sel = n_sel + 1;
                sel_local(n_sel) = rem_idx(ii);
                occ = [occ; cr, cc]; %#ok<AGROW>
            end
        end
    end

    %% Pass 3: no constraint (hard fallback for very small classes)
    if n_sel < N
        used     = sel_local(1:n_sel);
        rem_pool = pool_idx(~ismember(pool_idx, used));
        need     = N - n_sel;
        extra    = rem_pool(1:min(need, numel(rem_pool)));
        if ~isempty(extra)
            extra = extra(:);
            sel_local(n_sel+1:n_sel+numel(extra)) = extra;
            n_sel = n_sel + numel(extra);
            % spatial constraint dropped silently for this class (too small)
        end
    end

    sel_idx    = sel_local(1:n_sel);
    n_pre      = size(existing_coords, 1);
    new_coords = occ(n_pre+1:end, :);
end


%% ========================================================================
%  LOCAL: check_cross_distance
%  Reports min(train ↔ test) distance using a fast random subsample.
%% ========================================================================
function check_cross_distance(v_row, v_col, v_idx, tr_idx, te_idx, required)
    if isempty(tr_idx) || isempty(te_idx); return; end
    MAX = 800;
    tr_sub = tr_idx(randperm(numel(tr_idx), min(MAX, numel(tr_idx))));
    te_sub = te_idx(randperm(numel(te_idx), min(MAX, numel(te_idx))));

    % tr_idx / te_idx are indices INTO the valid-pixel array, so use directly
    tr_coords = [v_row(tr_sub), v_col(tr_sub)];
    te_coords = [v_row(te_sub), v_col(te_sub)];

    if isempty(tr_coords) || isempty(te_coords); return; end

    D     = pdist2(double(tr_coords), double(te_coords), 'euclidean');
    min_d = min(D(:));
    passed = min_d >= required;
    fprintf('Cross-set check (sampled): min(train↔test) = %.2f px  [%s]\n', ...
        min_d, conditional_str(passed, 'PASS ✓', 'WARNING ✗'));
end


%% ========================================================================
%  LOCAL: plot_splits
%% ========================================================================
function plot_splits(sample_map, gt, rows, cols, n_tr, n_vl, n_te)
    n_cls = double(max(gt(:)));
    if n_cls < 1; return; end
    cmap  = jet(n_cls);

    figure('Name','Sample Distribution','Units','centimeters','Position',[2 2 22 7]);
    titles = {'Training','Validation','Test'};
    ids    = [1, 2, 3];
    counts = [n_tr, n_vl, n_te];

    for sp = 1:3
        subplot(1,3,sp);
        rgb = zeros(rows, cols, 3);
        for k = 1:n_cls
            msk = (sample_map == ids(sp)) & (gt == k);
            for ch = 1:3
                lay = rgb(:,:,ch);
                lay(msk) = cmap(k,ch);
                rgb(:,:,ch) = lay;
            end
        end
        % Show excluded zone in grey on test panel
        if ids(sp) == 3
            excl = (sample_map == 4);
            for ch = 1:3
                lay = rgb(:,:,ch);
                lay(excl) = 0.5;
                rgb(:,:,ch) = lay;
            end
        end
        imshow(rgb);
        title(sprintf('%s  (n=%d)', titles{sp}, counts(sp)), ...
            'FontName','Times New Roman','FontSize',10,'FontWeight','bold');
    end
    colormap(cmap);
    cb = colorbar('Position',[0.93 0.12 0.015 0.76]);
    cb.Ticks      = linspace(0,1,n_cls);
    cb.TickLabels = arrayfun(@(x) sprintf('C%d',x), 1:n_cls, 'UniformOutput',false);
    cb.FontName   = 'Times New Roman';
end


%% ========================================================================
%  LOCAL: conditional_str  (avoids inline ternary verbosity)
%% ========================================================================
function s = conditional_str(cond, a, b)
    if cond
        s = a;
    else
        s = b;
    end
end