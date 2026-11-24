function [F, info] = p1_features(img, k, N_SP, D, opts, groups)
%P1_FEATURES Extended SuperSDF features for the Phase-1 experiments.
%   With default opts the output is identical to supersdf_features.m with
%   all switches on (paper model V5, k channels).
%
%   opts.useSP, useSDF, useGF, useGauss   module switches (default true)
%   opts.pooling   'min' (default) | 'mean' | 'median' | 'p10' | 'p25'
%   opts.binarize  'adaptive_otsu' (default: adaptive, Sensitivity = Otsu
%                  level) | 'global_otsu' (Z > Otsu level) |
%                  'adaptive_fixed' (adaptive, Sensitivity = opts.sens)
%   opts.sens      sensitivity for 'adaptive_fixed'
%   opts.addMeans  append the k unnormalized group-mean images (2k channels)
%   opts.keepMasks return the binary masks B_k in info.masks
%   opts.keepL     return the SLIC label maps in info.L
%
%   groups: 1-by-k cell of band indices. If empty, k-means is run with
%   rng(opts.kmeansSeed,'twister') (default 50) exactly as SuperSDF_MC10.m.
%   info.t: accumulated time per stage (s).

if nargin < 6, groups = {}; end
opts = defaults(opts);

COMPACTNESS = 0.1; LAMBDA_GF = 1e-4; SIGMA_GAUSS = 0.95; GAUSS_FSIZE = 55;
[rows, cols, bands] = size(img);
t = struct('kmeans',0,'mean',0,'slic',0,'pool',0,'binarize',0,'dist',0,'gf',0,'gauss',0);

if isempty(groups)
    q = tic;
    R = corrcoef(reshape(img, rows*cols, bands));
    rng(opts.kmeansSeed,'twister');
    lab = kmeans(R, k, 'Replicates', 1);
    groups = arrayfun(@(i) find(lab == i), 1:k, 'UniformOutput', false);
    t.kmeans = toc(q);
end

F = zeros(rows, cols, k);
if opts.addMeans, M = zeros(rows, cols, k); end
if opts.keepMasks, masks = false(rows, cols, k); else, masks = []; end
if opts.keepL, Lmaps = zeros(rows, cols, k, 'uint16'); else, Lmaps = []; end
n_const = 0; sens_used = nan(k,1);

for i = 1:k
    q = tic;
    group_img = mean(img(:,:,groups{i}), 3);
    lo = min(group_img(:)); hi = max(group_img(:));
    if hi > lo
        normalized_img = (group_img - lo) / (hi - lo);
    else
        normalized_img = zeros(size(group_img));
    end
    t.mean = t.mean + toc(q);
    if opts.addMeans, M(:,:,i) = group_img; end
    Z = normalized_img;

    if opts.useSP
        q = tic;
        [L, N] = superpixels(normalized_img, N_SP, 'Method','slic','Compactness',COMPACTNESS);
        t.slic = t.slic + toc(q);
        if opts.keepL, Lmaps(:,:,i) = uint16(L); end
        q = tic;
        pix = label2idx(L);
        sp = zeros(size(normalized_img), 'like', normalized_img);
        for j = 1:N
            idx = pix{j};
            v = normalized_img(idx);
            switch opts.pooling
                case 'min',    sp(idx) = min(v);
                case 'mean',   sp(idx) = mean(v);
                case 'median', sp(idx) = median(v);
                case 'p10',    sp(idx) = prctile(v, 10);
                case 'p25',    sp(idx) = prctile(v, 25);
                otherwise, error('Unknown pooling %s', opts.pooling);
            end
        end
        Z = sp;
        t.pool = t.pool + toc(q);
    end

    if opts.useSDF
        q = tic;
        lev = graythresh(Z);
        switch opts.binarize
            case 'adaptive_otsu'
                B = imbinarize(Z, 'adaptive', 'ForegroundPolarity','bright', 'Sensitivity', lev);
                sens_used(i) = lev;
            case 'global_otsu'
                B = imbinarize(Z, lev);
            case 'adaptive_fixed'
                B = imbinarize(Z, 'adaptive', 'ForegroundPolarity','bright', 'Sensitivity', opts.sens);
                sens_used(i) = opts.sens;
            otherwise, error('Unknown binarize %s', opts.binarize);
        end
        t.binarize = t.binarize + toc(q);
        if opts.keepMasks, masks(:,:,i) = B; end
        q = tic;
        if all(B(:) == 0) || all(B(:) == 1)
            Z = zeros(size(B)); n_const = n_const + 1;
        else
            Z = bwdist(~B) - bwdist(B);
        end
        t.dist = t.dist + toc(q);
    end

    if opts.useGF
        q = tic;
        Z = imguidedfilter(Z, group_img, 'NeighborhoodSize',[D D], 'DegreeOfSmoothing',LAMBDA_GF);
        t.gf = t.gf + toc(q);
    end
    if opts.useGauss
        q = tic;
        Z = imgaussfilt(Z, SIGMA_GAUSS, 'FilterSize', GAUSS_FSIZE);
        t.gauss = t.gauss + toc(q);
    end
    F(:,:,i) = Z;
end
if opts.addMeans, F = cat(3, F, M); end

info = struct('n_constant_masks', n_const, 'groups', {groups}, 't', t, ...
    'masks', masks, 'L', Lmaps, 'sens_used', sens_used);
end

function o = defaults(o)
d = struct('useSP',true,'useSDF',true,'useGF',true,'useGauss',true, ...
    'pooling','min','binarize','adaptive_otsu','sens',NaN,'addMeans',false, ...
    'keepMasks',false,'keepL',false,'kmeansSeed',50);
f = fieldnames(d);
for i = 1:numel(f)
    if ~isfield(o, f{i}) || isempty(o.(f{i})), o.(f{i}) = d.(f{i}); end
end
end
