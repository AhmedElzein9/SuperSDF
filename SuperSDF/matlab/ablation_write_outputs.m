function ablation_write_outputs(root, N_MC)
%ABLATION_WRITE_OUTPUTS Build the xlsx workbook and IEEEtran table from the
%   saved SuperSDF_Ablation_<Dataset>_MC<N>.mat files (complete files only).
%   N_MC = 10 writes SuperSDF_Ablation_Results.xlsx / SuperSDF_Ablation_table.tex;
%   any other N_MC adds the suffix _MC<N>.

if nargin < 1 || isempty(root), root = fileparts(mfilename('fullpath')); end
if nargin < 2 || isempty(N_MC), N_MC = 10; end

tags = {'IndianPines','PaviaUniversity','HanChuan'};
DS = {};
for i = 1:numel(tags)
    f = fullfile(root, sprintf('SuperSDF_Ablation_%s_MC%d.mat', tags{i}, N_MC));
    if isfile(f)
        R = load(f);
        if all(R.done), DS{end+1} = R; end %#ok<AGROW>
    end
end
if isempty(DS)
    fprintf('No complete result files for N_MC=%d; nothing written.\n', N_MC);
    return
end

if N_MC == 10, suffix = ''; else, suffix = sprintf('_MC%d', N_MC); end
xfile = fullfile(root, ['SuperSDF_Ablation_Results' suffix '.xlsx']);
tfile = fullfile(root, ['SuperSDF_Ablation_table' suffix '.tex']);

PM = char(177); CHK = char(10003); DASH = char(8211);
names = DS{1}.variant_names; labels = DS{1}.variant_labels; sw = DS{1}.switches;
nV = numel(names); main = 1:8; iV5 = 5;
nD = numel(DS);
fmt = @(x) sprintf('%.2f %s %.2f', mean(x), PM, std(x));

% ---------------- Ablation_TGRS
hdr = {'Variant','Description','K-means','Superpixel','SDF','Guided filter','Gaussian filter'};
for d = 1:nD
    s = DS{d}.short;
    hdr = [hdr, {[s ' OA'], [s ' AA'], [s ' Kappa'], [s ' dOA vs V5']}]; %#ok<AGROW>
end
C = cell(nV, numel(hdr));
for v = 1:nV
    mk = repmat({DASH},1,4); mk(sw(v,:)==1) = {CHK};
    C(v,1:7) = [names(v), labels(v), {CHK}, mk];
    for d = 1:nD
        R = DS{d}; c0 = 7 + 4*(d-1);
        M = {R.OA, R.AA, R.kappa};
        for j = 1:3
            txt = fmt(M{j}(:,v));
            if v <= 8 && isbest(M{j}, v, main), txt = [txt ' *']; end %#ok<AGROW>
            C{v,c0+j} = txt;
        end
        C{v,c0+4} = sprintf('%+.2f', mean(R.OA(:,v)) - mean(R.OA(:,iV5)));
    end
end
note = cell(2, numel(hdr)); note(:) = {''};
note{1,1} = sprintf('* = best mean among V1-V8 in that column (ties at 2 decimals all marked). Values are mean %s std over %d Monte Carlo runs on the test set.', PM, N_MC);
note{2,1} = 'V5* is a reference row (not ranked): V5 plus the extra last-group channel from SuperSDF_MC10.m line 175 (36 channels).';
writecell([hdr; C; note], xfile, 'Sheet','Ablation_TGRS', 'WriteMode','replacefile');

% ---------------- Numeric
rows = {};
for d = 1:nD
    R = DS{d};
    for v = 1:nV
        rows(end+1,:) = {R.dataset, names{v}, mean(R.OA(:,v)), std(R.OA(:,v)), ...
            mean(R.AA(:,v)), std(R.AA(:,v)), mean(R.kappa(:,v)), std(R.kappa(:,v)), ...
            mean(R.OA(:,v)) - mean(R.OA(:,iV5))}; %#ok<AGROW>
    end
end
T = cell2table(rows, 'VariableNames', {'Dataset','Variant','OA_mean','OA_std', ...
    'AA_mean','AA_std','Kappa_mean','Kappa_std','dOA_vs_V5'});
writetable(T, xfile, 'Sheet','Numeric', 'WriteMode','overwritesheet');

% ---------------- PerClass_<short>
for d = 1:nD
    R = DS{d}; nc = numel(R.class_order);
    H = [{'Class','Train_run1','Val_run1','Test_run1'}, names];
    P = cell(nc, numel(H));
    for ci = 1:nc
        P(ci,1:4) = {R.class_order(ci), R.train_counts(1,ci), R.val_counts(1,ci), R.test_counts(1,ci)};
        for v = 1:nV
            P{ci,4+v} = fmt(R.CA(:,ci,v));
        end
    end
    writecell([H; P], xfile, 'Sheet',['PerClass_' R.short], 'WriteMode','overwritesheet');
end

% ---------------- Runs
rows = {};
for d = 1:nD
    R = DS{d};
    for v = 1:nV
        for m = 1:R.N_MC
            rows(end+1,:) = {R.dataset, names{v}, m, R.seeds(m), R.OA(m,v), R.AA(m,v), R.kappa(m,v)}; %#ok<AGROW>
        end
    end
end
T = cell2table(rows, 'VariableNames', {'Dataset','Variant','Run','Seed','OA','AA','Kappa'});
writetable(T, xfile, 'Sheet','Runs', 'WriteMode','overwritesheet');

% ---------------- Timing
rows = {};
for d = 1:nD
    R = DS{d};
    for v = 1:nV
        rows(end+1,:) = {R.dataset, names{v}, R.kmeans_time, R.feat_time(v), ...
            mean(R.clf_time(:,v)), std(R.clf_time(:,v)), ...
            R.kmeans_time + R.feat_time(v) + mean(R.clf_time(:,v))}; %#ok<AGROW>
    end
end
T = cell2table(rows, 'VariableNames', {'Dataset','Variant','KMeans_s','Feature_s', ...
    'SVM_train_predict_mean_s','SVM_train_predict_std_s','KMeans_plus_Feature_plus_SVMmean_s'});
writetable(T, xfile, 'Sheet','Timing', 'WriteMode','overwritesheet');

% ---------------- Significance
rows = {};
for d = 1:nD
    R = DS{d};
    for v = 1:nV
        if v == iV5, continue; end
        rows(end+1,:) = {R.dataset, names{v}, mean(R.OA(:,v)) - mean(R.OA(:,iV5)), ...
            R.wilcoxon_p(v), R.mcnemar_p(v), R.mcnemar_bc(v,1), R.mcnemar_bc(v,2)}; %#ok<AGROW>
    end
end
T = cell2table(rows, 'VariableNames', {'Dataset','Variant_vs_V5','dOA_mean', ...
    'Wilcoxon_signrank_p_OA','McNemar_exact_p_run1', ...
    'McNemar_b_variant_right_V5_wrong','McNemar_c_variant_wrong_V5_right'});
writetable(T, xfile, 'Sheet','Significance', 'WriteMode','overwritesheet');

% ---------------- Config
K = {'Item','Value'};
K(end+1,:) = {'Protocol', DS{1}.protocol};
K(end+1,:) = {'Monte Carlo runs', N_MC};
K(end+1,:) = {'Seeds (rng twister)', mat2str(DS{1}.seeds)};
K(end+1,:) = {'K-means', 'kmeans on band correlation matrix, Replicates 1, rng(50,''twister''), run once per dataset'};
K(end+1,:) = {'SLIC compactness', DS{1}.fixed.compactness};
K(end+1,:) = {'Binarization', 'imbinarize adaptive, bright, Sensitivity = graythresh (part of SDF module)'};
K(end+1,:) = {'Guided filter DegreeOfSmoothing', DS{1}.fixed.gf_lambda};
K(end+1,:) = {'Guided filter guidance', 'unnormalized group mean image I_k'};
K(end+1,:) = {'Gaussian sigma / FilterSize', sprintf('%.2f / %d', DS{1}.fixed.gauss_sigma, DS{1}.fixed.gauss_fsize)};
K(end+1,:) = {'SVM', sprintf('fitcecoc + templateSVM rbf, BoxConstraint %g, KernelScale 1/sqrt(%g), Standardize true', DS{1}.svm.C, DS{1}.svm.gamma)};
K(end+1,:) = {'Min train per class', DS{1}.min_train};
K(end+1,:) = {'Metrics', 'OA, AA, kappa, per-class accuracy on TEST only (same code as SuperSDF_MC10.m)'};
K(end+1,:) = {'Significance', 'Wilcoxon signed-rank on the per-run OA values vs V5; exact McNemar on run-1 test predictions vs V5'};
K(end+1,:) = {'V5*', 'V5 with the extra channel of SuperSDF_MC10.m line 175 (unnormalized mean image of the last band group)'};
for d = 1:nD
    R = DS{d}; p = [R.short ' '];
    K(end+1,:) = {[p 'dataset'], sprintf('%s (%s)', R.dataset, R.cfg.file)}; %#ok<AGROW>
    K(end+1,:) = {[p 'size rows x cols x bands'], mat2str(R.size)}; %#ok<AGROW>
    K(end+1,:) = {[p 'classes / labeled pixels'], sprintf('%d / %d', numel(R.class_order), R.n_labeled)}; %#ok<AGROW>
    K(end+1,:) = {[p 'k, N_SP, D'], sprintf('%d, %d, %d', R.cfg.k, R.cfg.N_SP, R.cfg.D)}; %#ok<AGROW>
    K(end+1,:) = {[p 'train ratio = val ratio'], R.cfg.train_ratio}; %#ok<AGROW>
    tr = sum(R.train_counts,2); vl = sum(R.val_counts,2); te = sum(R.test_counts,2);
    K(end+1,:) = {[p 'train / val / test (run 1)'], sprintf('%d / %d / %d', tr(1), vl(1), te(1))}; %#ok<AGROW>
    K(end+1,:) = {[p 'train / val / test (min-max over runs)'], sprintf('%d-%d / %d-%d / %d-%d', ...
        min(tr), max(tr), min(vl), max(vl), min(te), max(te))}; %#ok<AGROW>
    K(end+1,:) = {[p 'split indices identical across variants (all runs)'], yesno(all(R.idx_identical))}; %#ok<AGROW>
    K(end+1,:) = {[p 'V5* X_train identical to MC10 split output (all runs)'], yesno(all(R.mc10_equivalent))}; %#ok<AGROW>
    K(end+1,:) = {[p 'constant binary masks per variant'], mat2str(R.n_const(:)')}; %#ok<AGROW>
    K(end+1,:) = {[p 'channels per variant'], mat2str(R.n_channels(:)')}; %#ok<AGROW>
    K(end+1,:) = {[p 'MATLAB version'], R.matlab_version}; %#ok<AGROW>
    K(end+1,:) = {[p 'run date'], R.date}; %#ok<AGROW>
end
writecell(K, xfile, 'Sheet','Config', 'WriteMode','overwritesheet');

% ---------------- LaTeX table (V1-V8)
fid = fopen(tfile, 'w', 'n', 'UTF-8');
fprintf(fid, '%% Requires \\usepackage{booktabs,amssymb,graphicx}\n');
fprintf(fid, '\\begin{table*}[!t]\n\\centering\n');
fprintf(fid, '\\caption{Ablation study of SuperSDF components (OA, AA, $\\kappa$ in \\%%, mean $\\pm$ std over %d Monte Carlo runs, evaluated on the held-out test set)}\n', N_MC);
fprintf(fid, '\\label{tab:ablation}\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\textwidth}{!}{%%\n');
fprintf(fid, '\\begin{tabular}{l ccccc%s}\n\\toprule\n', repmat(' ccc',1,nD));
fprintf(fid, ' & \\multicolumn{5}{c}{Modules}');
for d = 1:nD, fprintf(fid, ' & \\multicolumn{3}{c}{%s}', DS{d}.dataset); end
fprintf(fid, ' \\\\\n\\cmidrule(lr){2-6}');
for d = 1:nD, fprintf(fid, '\\cmidrule(lr){%d-%d}', 7+3*(d-1), 9+3*(d-1)); end
fprintf(fid, '\nVariant & K-means & SP & SDF & GF & GS');
for d = 1:nD, fprintf(fid, ' & OA & AA & $\\kappa$'); end
fprintf(fid, ' \\\\\n\\midrule\n');
for v = main
    mk = repmat({'--'},1,4); mk(sw(v,:)==1) = {'\checkmark'};
    fprintf(fid, '%s & \\checkmark & %s & %s & %s & %s', names{v}, mk{:});
    for d = 1:nD
        R = DS{d}; M = {R.OA, R.AA, R.kappa};
        for j = 1:3
            x = M{j}(:,v);
            if isbest(M{j}, v, main)
                fprintf(fid, ' & $\\mathbf{%.2f \\pm %.2f}$', mean(x), std(x));
            else
                fprintf(fid, ' & $%.2f \\pm %.2f$', mean(x), std(x));
            end
        end
    end
    fprintf(fid, ' \\\\\n');
end
fprintf(fid, '\\bottomrule\n\\end{tabular}}\n\\end{table*}\n');
fclose(fid);

fprintf('Wrote %s and %s (%d dataset(s))\n', xfile, tfile, nD);
end

function t = isbest(M, v, main)
% Best mean among V1-V8 after rounding to 2 decimals (ties all marked)
mu = round(mean(M(:,main),1), 2);
t = round(mean(M(:,v)), 2) == max(mu);
end

function s = yesno(t)
if t, s = 'YES'; else, s = 'NO'; end
end
