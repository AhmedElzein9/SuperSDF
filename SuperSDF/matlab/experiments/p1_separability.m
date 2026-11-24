function S = p1_separability(DATASET_ID, doTSNE)
%P1_SEPARABILITY E6b: class separability of the feature stages and the
%   multi-class code formed by the k binary partitions B_k.
%   Uses all labelled pixels and no classifier; descriptive statistics only.
%   J = trace(Sb)/trace(Sw) on z-scored features (larger = more separable).
%   Binary code: pixel -> k-bit word [B_1..B_k]. Reports the number of
%   distinct words, the normalized conditional entropy H(class|word)/H(class),
%   and, for every pair of classes, the number of partitions in which the
%   two class-majority bits differ (min / median over class pairs).
if nargin < 2, doTSNE = false; end
root = fileparts(mfilename('fullpath'));
cfg = p1_cfg(DATASET_ID);
L = load(fullfile(root, cfg.file), 'img', 'gt');
img = double(L.img); gt = double(L.gt); clear L
[rows, cols, ~] = size(img);
lab = find(gt(:) > 0); y = gt(lab); classes = unique(y); nc = numel(classes);

defs = {'V1 group means',      struct('useSP',false,'useSDF',false,'useGF',false,'useGauss',false);
        'V2 SP-min maps',      struct('useSDF',false,'useGF',false,'useGauss',false);
        'V3 SDF',              struct('useGF',false,'useGauss',false);
        'V5 SuperSDF',         struct('keepMasks',true);
        'V5 + group means',    struct('addMeans',true)};
S = struct('dataset',cfg.name,'short',cfg.short,'tag',cfg.tag);
S.names = defs(:,1); S.J = nan(size(defs,1),1); S.fisher_mean = nan(size(defs,1),1);
X5 = []; X2 = []; masks = [];
for v = 1:size(defs,1)
    [F, info] = p1_features(img, cfg.k, cfg.N_SP, cfg.D, defs{v,2});
    X = reshape(F, rows*cols, []); X = X(lab,:);
    X = (X - mean(X,1)) ./ max(std(X,0,1), eps);
    mu = mean(X,1); Sw = 0; Sb = 0; fr = zeros(nc, size(X,2));
    num = zeros(1,size(X,2)); den = zeros(1,size(X,2));
    for c = 1:nc
        Xc = X(y == classes(c),:); nc_ = size(Xc,1); muc = mean(Xc,1);
        Sw = Sw + sum(sum((Xc - muc).^2));
        Sb = Sb + nc_ * sum((muc - mu).^2);
        num = num + nc_ * (muc - mu).^2; den = den + sum((Xc - muc).^2, 1);
    end
    S.J(v) = Sb / Sw;
    S.fisher_mean(v) = mean(num ./ max(den, eps));
    % Leave-one-out 1-NN accuracy on a class-balanced subsample (<=500/class)
    rng(50,'twister'); sub = [];
    for c = 1:nc
        ic = find(y == classes(c)); ic = ic(randperm(numel(ic), min(500, numel(ic))));
        sub = [sub; ic]; %#ok<AGROW>
    end
    nn = knnsearch(X(sub,:), X(sub,:), 'K', 2);
    S.nn1_loo(v,1) = 100 * mean(y(sub(nn(:,2))) == y(sub));
    if v == 2, X2 = X; end
    if v == 4, X5 = X; masks = info.masks; end
    fprintf('%s E6b %-18s J=%.3f meanFisher=%.3f 1NN-LOO=%.2f\n', cfg.short, defs{v,1}, S.J(v), ...
        S.fisher_mean(v), S.nn1_loo(v));
end

% Binary partition code
Bm = reshape(masks, rows*cols, []); Bm = Bm(lab,:);
[words, ~, wid] = unique(Bm, 'rows');
S.n_words = size(words,1);
pc = accumarray(y - min(classes) + 1, 1); pc = pc(pc > 0) / numel(y);
Hc = -sum(pc .* log2(pc));
J2 = accumarray([wid, double(y)], 1);
pw = sum(J2,2) / numel(y);
Hcw = 0;
for w = 1:size(J2,1)
    p = J2(w,:) / sum(J2(w,:)); p = p(p > 0);
    Hcw = Hcw - pw(w) * sum(p .* log2(p));
end
S.H_class = Hc; S.H_class_given_word = Hcw; S.H_ratio = Hcw / Hc;
maj = zeros(nc, size(Bm,2));
for c = 1:nc, maj(c,:) = mean(Bm(y == classes(c),:), 1) >= 0.5; end
d = []; for a = 1:nc, for b = a+1:nc, d(end+1) = sum(maj(a,:) ~= maj(b,:)); end, end %#ok<AGROW>
S.pair_hamming_min = min(d); S.pair_hamming_median = median(d);
S.n_pairs_identical = sum(d == 0);
fprintf('%s E6b code: words=%d H(c|w)/H(c)=%.3f pairHamming min=%d median=%.1f identicalPairs=%d\n', ...
    cfg.short, S.n_words, S.H_ratio, S.pair_hamming_min, S.pair_hamming_median, S.n_pairs_identical);

if doTSNE
    rng(50,'twister');
    pick = [];
    for c = 1:nc
        ic = find(y == classes(c)); ic = ic(randperm(numel(ic), min(200, numel(ic))));
        pick = [pick; ic]; %#ok<AGROW>
    end
    rng(50,'twister'); Y2 = tsne(X2(pick,:));
    rng(50,'twister'); Y5 = tsne(X5(pick,:));
    S.tsne_V2 = Y2; S.tsne_V5 = Y5; S.tsne_y = y(pick);
    fig = figure('Visible','off','Units','centimeters','Position',[2 2 18 8],'Color','w');
    cm = jet(nc);
    subplot(1,2,1); gscatter(Y2(:,1),Y2(:,2),y(pick),cm,'.',6,'off'); axis off; title('(a) V2: superpixel-min maps','FontName','Times New Roman');
    subplot(1,2,2); gscatter(Y5(:,1),Y5(:,2),y(pick),cm,'.',6,'off'); axis off; title('(b) V5: SuperSDF','FontName','Times New Roman');
    exportgraphics(fig, fullfile(root, sprintf('P1_tSNE_%s.png', cfg.tag)), 'Resolution', 300);
    close(fig);
end
save(fullfile(root, sprintf('SuperSDF_P1_E6b_%s.mat', cfg.tag)), '-struct', 'S');
end
