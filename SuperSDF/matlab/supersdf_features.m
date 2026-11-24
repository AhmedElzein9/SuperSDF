function [F, info] = supersdf_features(img, k, N_SP, D, opts, groups)
%SUPERSDF_FEATURES SuperSDF feature maps with module switches (ablation).
%
%   [F, info] = supersdf_features(img, k, N_SP, D, opts, groups)
%
%   opts.useSP    superpixel-minimum map S_k            (default true)
%   opts.useSDF   binarization + signed distance phi    (default true)
%   opts.useGF    guided filter, guide = unnormalized I_k (default true)
%   opts.useGauss Gaussian filter                       (default true)
%
%   groups : 1-by-k cell of band indices from k-means. The caller computes
%            it ONCE per dataset. If empty it is computed here exactly as in
%            SuperSDF_MC10.m (corrcoef + kmeans, rng(50,'twister')).
%
%   Module chain (a disabled module passes its input to the next one):
%     I_k (normalized group mean) -> [SP] -> [SDF] -> [GF] -> [Gauss]
%
%   With all switches on, each channel is computed with exactly the same
%   operations as SuperSDF_MC10.m (lines 133-173). F has exactly k channels;
%   the extra channel of SuperSDF_MC10.m line 175 is NOT added here.
%
%   info.n_constant_masks : channels whose binary mask was constant (SDF = 0)
%   info.groups           : band groups used

if nargin < 6, groups = {}; end
opts = fill_defaults(opts);

COMPACTNESS = 0.1;
LAMBDA_GF   = 1e-4;
SIGMA_GAUSS = 0.95;
GAUSS_FSIZE = 55;

[rows, cols, bands] = size(img);

if isempty(groups)
    R = corrcoef(reshape(img, rows*cols, bands));
    rng(50,'twister');
    labels = kmeans(R, k, 'Replicates', 1);
    groups = arrayfun(@(i) find(labels == i), 1:k, 'UniformOutput', false);
end

F = zeros(rows, cols, k);
n_const = 0;

for i = 1:k
    group_img = mean(img(:,:,groups{i}), 3);

    lo = min(group_img(:));
    hi = max(group_img(:));
    if hi > lo
        normalized_img = (group_img - lo) / (hi - lo);
    else
        normalized_img = zeros(size(group_img));
    end

    Z = normalized_img;

    % --- Superpixel module: superpixel-minimum map S_k
    if opts.useSP
        [L, N] = superpixels(normalized_img, N_SP, ...
            'Method','slic','Compactness',COMPACTNESS);
        pixel_idx_lists = label2idx(L);
        sp_min_img = zeros(size(normalized_img), 'like', normalized_img);
        for j = 1:N
            idx = pixel_idx_lists{j};
            sp_min_img(idx) = min(normalized_img(idx));
        end
        Z = sp_min_img;
    end

    % --- SDF module: binarization (part of SDF) + signed distance
    if opts.useSDF
        threshold = graythresh(Z);
        binary_img = imbinarize(Z, 'adaptive', ...
            'ForegroundPolarity','bright', ...
            'Sensitivity',threshold);
        if all(binary_img(:) == 0) || all(binary_img(:) == 1)
            Z = zeros(size(binary_img));
            n_const = n_const + 1;
        else
            Z = bwdist(~binary_img) - bwdist(binary_img);
        end
    end

    % --- Guided filter, guidance = unnormalized group image I_k
    if opts.useGF
        Z = imguidedfilter(Z, group_img, ...
            'NeighborhoodSize',[D D], ...
            'DegreeOfSmoothing',LAMBDA_GF);
    end

    % --- Gaussian filter
    if opts.useGauss
        Z = imgaussfilt(Z, SIGMA_GAUSS, 'FilterSize', GAUSS_FSIZE);
    end

    F(:,:,i) = Z;   % stored as double, as in SuperSDF_MC10.m
end

info = struct('n_constant_masks', n_const, 'groups', {groups});
end

function opts = fill_defaults(opts)
f = {'useSP','useSDF','useGF','useGauss'};
for i = 1:numel(f)
    if ~isfield(opts, f{i}) || isempty(opts.(f{i}))
        opts.(f{i}) = true;
    end
    opts.(f{i}) = logical(opts.(f{i}));
end
end
