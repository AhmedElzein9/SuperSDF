function SuperSDF_Phase1(part, ids, N_MC)
%SUPERSDF_PHASE1 Phase-1 reviewer experiments. Paper parameters throughout.
%   SuperSDF_Phase1(part, ids, N_MC)
%   part: 'variants' (E1 pooling, E2 binarization, E6a spectral complement;
%                     random protocol, paired with the paper model V5)
%         'kseed'    (E3)   'sep' (E6b)   'timing' (E7)
%         'block'    (E10: V1-V8 under the guarded block split)
%         'write'    (Excel)   'all'
%   ids : cell of 'Indian','Pavia','HanChuan' (default all three)
if nargin < 1 || isempty(part), part = 'all'; end
if nargin < 2 || isempty(ids), ids = {'Indian','Pavia','HanChuan'}; end
if ischar(ids), ids = {ids}; end
if nargin < 3 || isempty(N_MC), N_MC = 10; end
root = fileparts(mfilename('fullpath')); addpath(root);
t0 = tic;
doit = @(p) strcmpi(part,'all') || strcmpi(part,p);

if doit('variants')
    V = mk('V5','Paper model (min pooling, adaptive Otsu-sensitivity)', struct());
    V(end+1) = mk('P-mean','Pooling: mean', struct('pooling','mean'));
    V(end+1) = mk('P-median','Pooling: median', struct('pooling','median'));
    V(end+1) = mk('P-p10','Pooling: 10th percentile', struct('pooling','p10'));
    V(end+1) = mk('P-p25','Pooling: 25th percentile', struct('pooling','p25'));
    V(end+1) = mk('B-global','Binarization: global Otsu threshold', struct('binarize','global_otsu'));
    V(end+1) = mk('B-s0.4','Binarization: adaptive, sensitivity 0.4', struct('binarize','adaptive_fixed','sens',0.4));
    V(end+1) = mk('B-s0.5','Binarization: adaptive, sensitivity 0.5', struct('binarize','adaptive_fixed','sens',0.5));
    V(end+1) = mk('B-s0.6','Binarization: adaptive, sensitivity 0.6', struct('binarize','adaptive_fixed','sens',0.6));
    V(end+1) = mk('C-means','SuperSDF + k group-mean images (2k ch)', struct('addMeans',true));
    for i = 1:numel(ids), p1_run_paired(ids{i}, 'random', 'VAR', V, N_MC); end
end
if doit('kseed'),  for i = 1:numel(ids), p1_kmeans_seed(ids{i}, N_MC); end, end
if doit('sep'),    for i = 1:numel(ids), p1_separability(ids{i}, strcmpi(p1_cfg(ids{i}).short,'PU')); end, end
if doit('timing'), for i = 1:numel(ids), p1_timing(ids{i}, 5); end, end
if doit('block')
    sw = [1 1 1 1; 0 0 0 0; 1 0 0 0; 1 1 0 0; 1 1 1 0; 1 0 1 1; 0 1 1 1; 0 0 1 1];
    nm = {'V5','V1','V2','V3','V4','V6','V7','V8'};
    for j = 1:8
        o = struct('useSP',sw(j,1)==1,'useSDF',sw(j,2)==1,'useGF',sw(j,3)==1,'useGauss',sw(j,4)==1);
        if j == 1, B = mk(nm{j}, nm{j}, o); else, B(end+1) = mk(nm{j}, nm{j}, o); end %#ok<AGROW>
    end
    for i = 1:numel(ids), p1_run_paired(ids{i}, 'block', 'ABL', B, N_MC); end
end
if doit('write'), p1_write_outputs(root, N_MC); end
fprintf('Phase1 part=%s done in %.1f s\n', part, toc(t0));
end

function v = mk(name, label, opts)
v = struct('name', name, 'label', label, 'opts', opts);
end
