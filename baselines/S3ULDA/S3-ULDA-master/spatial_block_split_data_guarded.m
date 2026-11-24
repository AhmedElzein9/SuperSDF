function [X_train,y_train,X_val,y_val,X_test,y_test, ...
    sample_map,train_indices,val_indices,test_indices,X_all_raw,split_info] = ...
    spatial_block_split_data_guarded(sdf_images,gt,train_ratio,block_size,buffer, ...
    min_train_per_class,min_test_per_class,split_seed,visualize,opts)
%SPATIAL_BLOCK_SPLIT_DATA_GUARDED Spatially separated labelled HSI evaluation.
%   Drop-in argument order for the earlier spatial_block_split_data:
%   [Xtr,ytr,Xv,yv,Xte,yte,map,tr,v,te,Xall,info] = ...
%       spatial_block_split_data_guarded(features,gt,ratio,B,gap,5,10,seed,false,opts)
%
%   REQUIRED: features H-by-W-by-F, gt H-by-W nonnegative integer labels
%   (0 means unlabelled), ratio in (0,1), B >= 2 tile side, gap >= 2
%   minimum EUCLIDEAN distance in pixels between retained label centres.
%   Pass an explicit integer seed. Defaults: min_train_per_class=5,
%   min_test_per_class=10, visualize=false, opts=struct(). A different
%   minimum test count is permitted only when supplied explicitly.
%
%   The first 11 outputs match random_split_data: tr/v/te are 1-based
%   POSITIONS into find(gt(:)>0), NOT full-image indices. Full-image indices
%   are info.fullTrainIndices / fullValIndices / fullTestIndices. Thus:
%       fullTe = find(gt(:)>0); fullTe = fullTe(test_indices);
%   Xall is H*W-by-F (MATLAB column-major). map: 0=unlabelled,
%   1=train, 2=val, 3=test, 4=excluded labelled pixels.
%
%   The split is determined from gt and seed ONLY, never from features,
%   predictions, or validation/test accuracy. Each B-by-B tile is assigned
%   exclusively to one retained set. Class-aware rare-first placement
%   spreads training and validation across coarse cells of each class's
%   labelled footprint (using gt only), reserves test witnesses, then
%   chooses exact per-class train and val budgets. Every other eligible
%   labelled pixel in held-out tiles is test.
%   Pixels that are too close to train/val (or lie in a train/val tile but
%   were not sampled) are excluded. The core ASSERTS exclusive tiles,
%   nonoverlapping indices, val-to-train and test-to-train+val bwdist
%   distances. A bounded search can fail without proving infeasibility.
%   The buffer measures RETAINED LABEL CENTRES. It does not guarantee a
%   buffer around every pixel in each selected tile or independent feature
%   context. Test at B=10, gap=5 over ALL seeds before using that geometry
%   on Pavia, HanChuan or LongKou; do not choose it using test accuracy.
%   This block protocol retains different test pixels from the point split.
%
%   Per-class train and validation count =
%       min(max(min_train_per_class,round(ratio*Nc)),
%           floor((Nc-min_test_per_class)/2)).
%   On the supplied Indian Pines and Pavia ground truths, the cap leaves
%   the original random-split train/val budgets unchanged. Check this
%   separately for the actual HanChuan and LongKou ground truths.
%   If a class has too few available pixels/tiles the routine ERRORS:
%   no implicit reduction in B, gap, budget, or minimum test count.
%
%   Optional opts: patchRadius=0, maxAttempts=128, slicLabelMaps=[];
%   quiet=false; exportDir='' (when nonempty, export both full-image
%   MATLAB 1-based indices and Fortran-order Python 0-based indices).
%   allowRelaxation=false, minTestFloor=1 (only changes ACTUAL test minimum
%   when explicitly enabled; emits a warning and records info.relaxed).
%   Setting opts.allowGeometryRelaxation=true is prohibited in this
%   wrapper: B/gap must be fixed across Monte Carlo repetitions.
%   For patchRadius>0, the core checks that each square patch lies within
%   its own tile and patches of different sets do not overlap.
%
%   For Indian Pines, 20-pixel Oats occupies one 10-by-2 connected patch.
%   5 train + 5 val + 10 test with a gap >=2 is mathematically impossible;
%   the class needs explicitly fewer test pixels or a different dataset.
%   With 5/5/1 and B=2,gap=2, spatial LABEL centres can be separated,
%   but this is weak spatial separation and Oats test OA is unstable.
%   SuperSDF's whole-scene SLIC and guided filtering may share UNLABELLED
%   image context even when the label-centre assertions pass. Report the
%   test-distance and optional mean-over-groups SLIC-sharing diagnostics.
%   Do not claim unseen-scene generalization or independent feature inputs.
%   This splitter alone does NOT perform Reviewer 1's nested parameter
%   tuning: within each seed, select N_SP, D, and SVM parameters using only
%   that seed's train/validation labels; evaluate test labels only once.
%
%   Requires Image Processing Toolbox and Statistics and Machine Learning
%   Toolbox. Test with test_spatial_block_split_data_guarded before running SVM.

if nargin<4 || isempty(block_size), error('spatial_block_split_data_guarded:Block','Supply block_size explicitly.'); end
if nargin<5 || isempty(buffer), error('spatial_block_split_data_guarded:Buffer','Supply buffer explicitly.'); end
if nargin<6 || isempty(min_train_per_class), min_train_per_class=5; end
if nargin<7 || isempty(min_test_per_class), min_test_per_class=10; end
if nargin<8 || isempty(split_seed), error('spatial_block_split_data_guarded:Seed','Supply an explicit split_seed.'); end
if nargin<9 || isempty(visualize), visualize=false; end
if nargin<10 || isempty(opts), opts=struct(); end
if ~isstruct(opts) || ~isscalar(opts), error('spatial_block_split_data_guarded:Options','opts must be a scalar struct.'); end
if ~isnumeric(sdf_images) || isempty(sdf_images) || ndims(sdf_images)>3
    error('spatial_block_split_data_guarded:Features','features must be H-by-W-by-F numeric.');
end
if ~isnumeric(train_ratio) || ~isscalar(train_ratio) || ~isfinite(train_ratio) || ...
        train_ratio<=0 || train_ratio>=1
    error('spatial_block_split_data_guarded:Ratio','train_ratio must be in (0,1).');
end
[rows,cols,nFeat]=size(sdf_images);
if ~isequal(size(gt),[rows cols]), error('spatial_block_split_data_guarded:Dimensions','gt must match features spatial dimensions.'); end
if ~isscalar(visualize) || ~ismember(visualize,[0 1])
    error('spatial_block_split_data_guarded:Visualize','visualize must be true or false.');
end
if ~isscalar(block_size) || ~isnumeric(block_size) || ~isfinite(block_size) || block_size~=fix(block_size) || block_size<2
    error('spatial_block_split_data_guarded:WeakGeometry','block_size must be an integer >=2; 1-pixel tiles are point sampling.');
end
if ~isscalar(buffer) || ~isnumeric(buffer) || ~isfinite(buffer) || buffer~=fix(buffer) || buffer<2
    error('spatial_block_split_data_guarded:WeakGeometry','buffer must be an integer >=2; 1-pixel distances permit immediately adjacent centres.');
end
if isfield(opts,'allowGeometryRelaxation') && opts.allowGeometryRelaxation
    error('spatial_block_split_data_guarded:Geometry','Keep B and gap fixed across all seeds; choose a feasible fixed pair from gt first.');
end
if isfield(opts,'mode') && ~strcmpi(char(opts.mode),'blocks')
    error('spatial_block_split_data_guarded:Mode','Only opts.mode=blocks is supported.');
end
if isfield(opts,'minPerClass') && ~isequal(opts.minPerClass,min_train_per_class)
    error('spatial_block_split_data_guarded:Options','opts.minPerClass must equal min_train_per_class.');
end
if isfield(opts,'minTestPerClass') && ~isequal(opts.minTestPerClass,min_test_per_class)
    error('spatial_block_split_data_guarded:Options','opts.minTestPerClass must equal min_test_per_class.');
end
if isfield(opts,'blockSize') && ~isequal(opts.blockSize,block_size)
    error('spatial_block_split_data_guarded:Options','opts.blockSize must equal block_size.');
end
if isfield(opts,'bufferPx') && ~isequal(opts.bufferPx,buffer)
    error('spatial_block_split_data_guarded:Options','opts.bufferPx must equal buffer.');
end
opts.minPerClass=min_train_per_class;
opts.minTestPerClass=min_test_per_class;
opts.blockSize=block_size;
opts.bufferPx=buffer;
opts.mode='blocks';
opts.allowGeometryRelaxation=false;
if ~isfield(opts,'maxAttempts') || isempty(opts.maxAttempts), opts.maxAttempts=128; end
if ~isfield(opts,'patchRadius') || isempty(opts.patchRadius), opts.patchRadius=0; end
if ~isfield(opts,'allowRelaxation') || isempty(opts.allowRelaxation), opts.allowRelaxation=false; end
if ~isfield(opts,'minTestFloor') || isempty(opts.minTestFloor), opts.minTestFloor=1; end
if ~isfield(opts,'quiet') || isempty(opts.quiet), opts.quiet=false; end
if ~isscalar(opts.quiet) || ~ismember(opts.quiet,[0 1])
    error('spatial_block_split_data_guarded:Quiet','opts.quiet must be true or false.');
end
if opts.patchRadius>0 && buffer<opts.patchRadius
    error('spatial_block_split_data_guarded:Patch','buffer must be >= patchRadius.');
end

% The local core returns FULL-IMAGE, column-major, 1-based indices.
[fullTr,fullVa,fullTe,split_info]=spatial_block_split(gt,train_ratio,opts,split_seed);
labelled=find(gt(:)>0);
rank=zeros(numel(gt),1,'uint32');
rank(labelled)=uint32(1:numel(labelled));
train_indices=double(rank(fullTr));
val_indices=double(rank(fullVa));
test_indices=double(rank(fullTe));
assert(all(train_indices>0)&all(val_indices>0)&all(test_indices>0));
X_all_raw=reshape(sdf_images,rows*cols,nFeat);
X_train=X_all_raw(fullTr,:); y_train=gt(fullTr);
X_val=X_all_raw(fullVa,:); y_val=gt(fullVa);
X_test=X_all_raw(fullTe,:); y_test=gt(fullTe);
sample_map=zeros(rows,cols);
sample_map(labelled)=4;
sample_map(fullTe)=3; sample_map(fullVa)=2; sample_map(fullTr)=1;
assert(nnz(sample_map==4)==sum(split_info.classCounts.Excluded));
assert(isequal(labelled(train_indices),fullTr) && ...
       isequal(labelled(val_indices),fullVa) && ...
       isequal(labelled(test_indices),fullTe));
split_info.fullTrainIndices=fullTr;
split_info.fullValIndices=fullVa;
split_info.fullTestIndices=fullTe;
split_info.split_seed=split_seed;
split_info.train_ratio=train_ratio;
split_info.block_size=block_size;
split_info.buffer=buffer;
split_info.n_train=numel(fullTr);
split_info.n_val=numel(fullVa);
split_info.n_test=numel(fullTe);
split_info.n_excluded=nnz(sample_map==4);
split_info.indexConvention='train/val/test_indices address find(gt(:)>0); full*Indices address gt(:).';
split_info.exportFile='';
if isfield(opts,'exportDir') && ~isempty(opts.exportDir)
    if ~(ischar(opts.exportDir) || (isstring(opts.exportDir) && isscalar(opts.exportDir)))
        error('spatial_block_split_data_guarded:Export','opts.exportDir must be a folder name.');
    end
    folder=char(opts.exportDir);
    if ~isfolder(folder), mkdir(folder); end
    train_idx_matlab=fullTr; val_idx_matlab=fullVa; test_idx_matlab=fullTe;
    train_idx_fortran0=int64(fullTr)-1; val_idx_fortran0=int64(fullVa)-1;
    test_idx_fortran0=int64(fullTe)-1;
    split_info.exportFile=fullfile(folder,sprintf( ...
        'split_%dx%d_B%d_g%d_seed%d.mat',rows,cols,block_size,buffer,split_seed));
    save(split_info.exportFile,'train_idx_matlab','val_idx_matlab', ...
        'test_idx_matlab','train_idx_fortran0','val_idx_fortran0', ...
        'test_idx_fortran0','split_info','-v7.3');
end
if ~opts.quiet
    fprintf('\nSpatial split: seed %d; tile %d x %d; minimum centre distance %d px\n', ...
        split_seed,block_size,block_size,buffer);
    fprintf('Train %d; val %d; test %d; excluded labelled %d (%.2f%%)\n', ...
        numel(fullTr),numel(fullVa),numel(fullTe),split_info.n_excluded, ...
        100*split_info.excludedFraction);
    fprintf('Min val-to-train %.2f; min test-to-train+val %.2f; mean test distance %.2f px\n', ...
        split_info.validationTrainMinDistance,split_info.testDistance.minimum, ...
        split_info.testDistance.mean);
    fprintf('Selection: %s\n',split_info.samplingStrategy);
    disp(split_info.classCounts);
    disp(split_info.classTileCoverage);
end
if min(split_info.classCounts.Test)<10 && ~opts.quiet
    warning('spatial_block_split_data_guarded:SmallTestClass', ...
        'At least one class has fewer than 10 retained test pixels; per-class OA is unstable.');
end
if visualize
    figure('Name',sprintf('Spatial split seed %d',split_seed),'Color','w');
    imagesc(sample_map); axis image off; caxis([0 4]);
    colormap([0 0 0;0 .7 0;.85 .65 0;.45 .45 1;.65 .65 .65]);
    colorbar('Ticks',0:4,'TickLabels',{'BG','Train','Val','Test','Excluded'});
    title(sprintf('Spatial tiles %d px; centre distance >=%d px; seed %d', ...
        block_size,buffer,split_seed));
end
end

function [tr, va, te, info] = spatial_block_split(gt, ratio, opts, seed)
%SPATIAL_BLOCK_SPLIT Class-aware, spatially separated HSI label partition.
%   [TR,VA,TE,INFO] = SPATIAL_BLOCK_SPLIT(GT,RATIO,OPTS,SEED) returns
%   1-based MATLAB column-major linear indices into the entire GT image.
%   GT is a nonnegative integer-valued 2-D label map; zero is unlabelled.
%   The feature cube and test predictions NEVER enter the assignment.
%
%   Documented OPTS defaults (all may be specified explicitly):
%     minPerClass=5       Minimum train count before the test-count cap.
%     minTestPerClass=10  Required test count for EACH present class.
%     blockSize=8         Square tile side in pixels, anchored at (1,1).
%     bufferPx=2          Minimum Euclidean centre distance between sets.
%     patchRadius=0       Radius of square input patches, in pixels.
%     mode='blocks'       Fixed; any other method raises an error.
%     maxAttempts=64      Independent seeded class-aware assignments.
%     allowRelaxation=false  Explicit opt-in to the logged ladder below.
%     allowGeometryRelaxation=false  Also allow smaller tiles and buffer.
%     minTestFloor=1      Smallest per-class test count in the opt-in ladder.
%     minBufferPx=1       Smallest buffer in the opt-in geometry ladder.
%     slicLabelMaps=[]    Optional H-by-W-by-G SLIC maps, DIAGNOSTICS ONLY.
%
%   Budgets: nTrain(c)=min(max(minPerClass,round(ratio*Nc)),
%   floor((Nc-minTestPerClass)/2)); nVal(c)=nTrain(c). MATLAB round is used.
%   The budget is fixed even if relaxation is enabled. Other labelled pixels
%   in train/validation tiles and those failing a guard are EXCLUDED. All
%   eligible labelled pixels in remaining test tiles enter TE.
%
%   TILES: a tile has exactly one role (train, validation, or test).
%   Assignment handles classes with few occupied tiles first. Within each
%   class it selects compact groups of points from several coarse spatial
%   cells of the labelled footprint, using only gt and the seed. This
%   avoids concentrating all training at one edge of a class. It protects
%   test witnesses and checks each returned split with BWDIST. If a class
%   has fewer than about 40 train labels, it uses one coarse cell; this
%   keeps tiny classes from claiming too many tiles.
%   A bounded search cannot prove that an unlocated assignment does not
%   exist: a failure message says 'search exhausted', not 'impossible',
%   except for necessary count/tile prechecks.
%
%   RELAXATION LADDER, ONLY WHEN allowRelaxation=true:
%     1. Keep tile size and distance; lower the ACTUAL minimum test count
%        only for classes where the returned test count is below the target.
%     2. If allowGeometryRelaxation=true and no assignment was found,
%        successively halve tile size (down to 1), then lower the buffer
%        (down to minBufferPx, never below patchRadius), repeating step 1.
%   Every actual change is saved in INFO.relaxed and emits a warning. A
%   1-pixel buffer permits immediately adjacent centres: it is not an
%   empty-pixel gap and must be described as such in a paper.
%
%   PATCHES: bufferPx >= patchRadius is necessary, but not sufficient for
%   disjoint square input patches. For patchRadius=r>0 we ALSO require
%   Chebyshev centre distance >2*r between every pair of sets. If that
%   makes a split impossible, this function will not pretend otherwise.
%
%   Requires Image Processing Toolbox (BWDIST) and Statistics and Machine
%   Learning Toolbox (PRCTILE). Tested syntactically by inspection only if
%   MATLAB is unavailable; run test_guarded_block_split_data in MATLAB.

if nargin < 3 || isempty(opts), opts = struct(); end
if nargin < 4, error('spatial_block_split:Seed','Supply an explicit seed.'); end
if ~isstruct(opts) || ~isscalar(opts)
    error('spatial_block_split:Options','opts must be a scalar struct.');
end
defaults = struct('minPerClass',5,'minTestPerClass',10,'blockSize',8, ...
    'bufferPx',2,'patchRadius',0,'mode','blocks','maxAttempts',64, ...
    'allowRelaxation',false,'allowGeometryRelaxation',false, ...
    'minTestFloor',1,'minBufferPx',1,'slicLabelMaps',[]);
names = fieldnames(defaults);
for k = 1:numel(names)
    name = names{k};
    if ~isfield(opts,name) || isempty(opts.(name))
        opts.(name) = defaults.(name);
    end
end
if ~isnumeric(gt) || ~ismatrix(gt) || isempty(gt) || ...
        any(~isfinite(gt(:))) || any(gt(:)<0) || any(gt(:)~=fix(gt(:)))
    error('spatial_block_split:GT','gt must be a finite 2-D nonnegative integer label map.');
end
check_scalar(ratio,0,1,'ratio',false);
check_scalar(seed,0,Inf,'seed',true);
check_scalar(opts.minPerClass,1,Inf,'minPerClass',true);
check_scalar(opts.minTestPerClass,1,Inf,'minTestPerClass',true);
check_scalar(opts.blockSize,1,Inf,'blockSize',true);
check_scalar(opts.bufferPx,0,Inf,'bufferPx',true);
check_scalar(opts.patchRadius,0,Inf,'patchRadius',true);
check_scalar(opts.maxAttempts,1,Inf,'maxAttempts',true);
check_scalar(opts.minTestFloor,1,Inf,'minTestFloor',true);
check_scalar(opts.minBufferPx,0,Inf,'minBufferPx',true);
if ~isscalar(opts.allowRelaxation) || ~islogical_or_binary(opts.allowRelaxation) || ...
        ~isscalar(opts.allowGeometryRelaxation) || ~islogical_or_binary(opts.allowGeometryRelaxation)
    error('spatial_block_split:Options','Relaxation switches must be logical scalars.');
end
if ~ischar(opts.mode) && ~(isstring(opts.mode) && isscalar(opts.mode))
    error('spatial_block_split:Mode','mode must be blocks.');
end
opts.mode = lower(char(opts.mode));
if ~strcmp(opts.mode,'blocks')
    error('spatial_block_split:Mode','mode must be blocks.');
end
if opts.bufferPx < opts.patchRadius
    error('spatial_block_split:Patch','bufferPx must be >= patchRadius.');
end
if opts.minTestFloor > opts.minTestPerClass
    error('spatial_block_split:Options','minTestFloor cannot exceed minTestPerClass.');
end
[h,w] = size(gt);
if 2*opts.patchRadius+1>min(h,w)
    error('spatial_block_split:Patch', ...
        'The square patch exceeds the smaller image dimension. Reduce patchRadius.');
end
if ~isempty(opts.slicLabelMaps) && ...
        (ndims(opts.slicLabelMaps) > 3 || size(opts.slicLabelMaps,1) ~= h || ...
         size(opts.slicLabelMaps,2) ~= w)
    error('spatial_block_split:SLIC','slicLabelMaps must be H-by-W-by-G.');
end

classes = unique(double(gt(gt>0)));
if isempty(classes), error('spatial_block_split:GT','gt has no labelled pixels.'); end
counts = arrayfun(@(c)nnz(gt==c),classes);
originalBudget = max(opts.minPerClass,round(ratio.*counts));
budget = min(originalBudget,floor((counts-opts.minTestPerClass)/2));
bad = find(budget < 1 | 2*budget+opts.minTestPerClass > counts);
if ~isempty(bad)
    error('spatial_block_split:Budget', ...
        'Classes %s lack train/val/test counts; change minTestPerClass or minPerClass.', ...
        mat2str(classes(bad)'));
end

% Keep the caller's RNG untouched while making all random choices explicit.
oldRng = rng;
restoreRng = onCleanup(@()rng(oldRng)); %#ok<NASGU>
rng(seed,'twister');
initialBlock = opts.blockSize;
blocks = initialBlock;
buffers = opts.bufferPx;
if opts.allowRelaxation && opts.allowGeometryRelaxation
    blocks = unique([initialBlock, max(1,floor(initialBlock/2)), 1],'stable');
    bottom = max(opts.patchRadius,opts.minBufferPx);
    buffers = unique([opts.bufferPx,max(bottom,floor(opts.bufferPx/2)),bottom],'stable');
end

best = [];
lastFailure = struct('class',classes(1),'type','train','have',0,'need',budget(1));
usedB = NaN; usedBuffer = NaN;
for bi = 1:numel(buffers)
    currentBuffer = buffers(bi);
    for si = 1:numel(blocks)
        currentBlock = blocks(si);
        if opts.patchRadius>0 && currentBlock<=2*opts.patchRadius
            lastFailure=struct('class',classes(1), ...
                'type','train (no tile interior for patchRadius)', ...
                'have',0,'need',budget(1));
            continue
        end
        if opts.patchRadius>0
            [rGrid,cGrid]=ndgrid(1:h,1:w);
            eligibleMask=mod(rGrid-1,currentBlock)>=opts.patchRadius & ...
                mod(rGrid-1,currentBlock)<currentBlock-opts.patchRadius & ...
                mod(cGrid-1,currentBlock)>=opts.patchRadius & ...
                mod(cGrid-1,currentBlock)<currentBlock-opts.patchRadius;
            eligibleMask(1:opts.patchRadius,:)=false;
            eligibleMask(end-opts.patchRadius+1:end,:)=false;
            eligibleMask(:,1:opts.patchRadius)=false;
            eligibleMask(:,end-opts.patchRadius+1:end)=false;
        else
            eligibleMask=true(h,w);
        end
        [tileMap,tileClassCounts,tileCount] = make_tiles(gt,classes,currentBlock);
        occupied = full(sum(tileClassCounts>0,1))';
        insufficient = find(occupied < 3);
        if ~isempty(insufficient)
            error('spatial_block_split:TileShortage', ...
                ['Classes %s occupy only %s distinct B=%d tiles; at least ' ...
                 'three are needed per class for train/val/test. ' ...
                 'Reduce block_size or select another dataset.'], ...
                mat2str(classes(insufficient)'),mat2str(occupied(insufficient)'),currentBlock);
        end
        % Necessary impossibility certificate for a connected class whose
        % strict budget consumes every labelled class pixel: a separation
        % of two pixels cannot partition that connected component.
        impossible = [];
        if currentBuffer>sqrt(2) || opts.patchRadius>0
            for q = find(counts==2*budget+opts.minTestPerClass)'
                cc = bwconncomp(gt==classes(q),8);
                if cc.NumObjects==1, impossible(end+1)=q; end %#ok<AGROW>
            end
        end
        if ~isempty(impossible)
            q=impossible(1);
            lastFailure=struct('class',classes(q),'type','test (connected fully allocated class)', ...
                'have',0,'need',opts.minTestPerClass);
        end
        % Strict attempt first; relaxed attempt retains train and val counts.
        phases = opts.minTestPerClass;
        if opts.allowRelaxation && opts.minTestFloor < opts.minTestPerClass
            phases = [phases opts.minTestFloor];
        end
        for phase = 1:numel(phases)
            if phase==1 && ~isempty(impossible), continue; end
            [candidate,failure] = search_assignment(gt,classes,budget, ...
                tileMap,tileClassCounts,tileCount,currentBuffer, ...
                opts.patchRadius,phases(phase),opts.maxAttempts,eligibleMask, ...
                opts.minTestPerClass);
            if ~isempty(candidate)
                best = candidate;
                usedB = currentBlock;
                usedBuffer = currentBuffer;
                break
            end
            lastFailure = failure;
        end
        if ~isempty(best), break; end
    end
    if ~isempty(best), break; end
end
if isempty(best)
    if ~opts.allowRelaxation && strcmp(lastFailure.type, ...
            'test (connected fully allocated class)')
        error('spatial_block_split:Impossible', ...
            ['Class %g has test shortage: its connected labelled patch is ' ...
             'fully used by train/val/test budgets at bufferPx=%d. ' ...
             'Lower min_test_per_class explicitly, change the class budget, ' ...
             'or use a different dataset; distance >=2 is impossible ' ...
             'at these counts.'], ...
             lastFailure.class,opts.bufferPx);
    end
    error('spatial_block_split:SearchExhausted', ...
        ['Search exhausted at seed %d: class %g has %s shortage ' ...
         '(found %d, need %d). Change blockSize, bufferPx, patchRadius, ' ...
         'maxAttempts or enable an explicit relaxation. No infeasibility proof is claimed.'], ...
        seed,lastFailure.class,lastFailure.type,lastFailure.have,lastFailure.need);
end

tr = find(best.trMask);
va = find(best.vaMask);
te = find(best.teMask);
owner = best.owner;
tileMap = best.tileMap;
classTr = arrayfun(@(c)nnz(gt(tr)==c),classes);
classVa = arrayfun(@(c)nnz(gt(va)==c),classes);
classTe = arrayfun(@(c)nnz(gt(te)==c),classes);
assert(isequal(classTr,budget) && isequal(classVa,budget), ...
    'spatial_block_split:InternalBudget','The exact train/val budget was not met.');
assert(all(classTe>=best.requiredTest), ...
    'spatial_block_split:InternalTest','Test witnesses were lost.');
assert(isempty(intersect(tr,va)) && isempty(intersect(tr,te)) && isempty(intersect(va,te)), ...
    'spatial_block_split:Overlap','Pixel subsets overlap.');
assert(isempty(intersect(unique(tileMap(tr)),unique(tileMap(va)))) && ...
       isempty(intersect(unique(tileMap(tr)),unique(tileMap(te)))) && ...
       isempty(intersect(unique(tileMap(va)),unique(tileMap(te)))), ...
       'spatial_block_split:Tiles','A tile contains different subsets.');
dv = bwdist(best.trMask);
dt = bwdist(best.trMask | best.vaMask);
assert(all(dv(va)>=usedBuffer) && all(dt(te)>=usedBuffer), ...
    'spatial_block_split:Distance','Euclidean buffer violation.');
if opts.patchRadius > 0
    assert(all(eligibleMask([tr;va;te])), ...
        'spatial_block_split:PatchTiles', ...
        'A retained patch crosses its tile or image boundary.');
    cv = bwdist(best.trMask,'chessboard');
    ct = bwdist(best.trMask | best.vaMask,'chessboard');
    assert(all(cv(va)>2*opts.patchRadius) && all(ct(te)>2*opts.patchRadius), ...
        'spatial_block_split:PatchOverlap','Square input patches overlap.');
end

relaxed = struct('parameter',{},'class',{},'from',{},'to',{},'reason',{});
if usedB ~= initialBlock
    relaxed(end+1) = change('blockSize',NaN,initialBlock,usedB,'search fallback'); %#ok<AGROW>
end
if usedBuffer ~= opts.bufferPx
    relaxed(end+1) = change('bufferPx',NaN,opts.bufferPx,usedBuffer,'search fallback'); %#ok<AGROW>
end
short = find(classTe < opts.minTestPerClass);
for k = short(:)'
    relaxed(end+1) = change('minTestPerClass',classes(k), ...
        opts.minTestPerClass,classTe(k),'insufficient separated test pixels'); %#ok<AGROW>
end
for k = 1:numel(relaxed)
    warning('spatial_block_split:Relaxed', ...
        'Seed %d: %s, class %g, %g -> %g (%s).',seed,relaxed(k).parameter, ...
        relaxed(k).class,relaxed(k).from,relaxed(k).to,relaxed(k).reason);
end

excluded = counts-classTr-classVa-classTe;
assert(all(excluded>=0),'spatial_block_split:InternalCounts','Negative excluded count.');
info = struct;
info.seed = seed;
info.imageSize = [h w];
info.classes = classes;
info.ratio = ratio;
info.mode = opts.mode;
info.samplingStrategy = 'GT-only spatially distributed class tiles';
info.blockSize = usedB;
info.bufferPx = usedBuffer;
info.patchRadius = opts.patchRadius;
info.targetTrain = budget;
info.targetVal = budget;
info.targetMinTest = repmat(opts.minTestPerClass,size(classes));
info.effectiveMinTest = min(classTe,opts.minTestPerClass);
info.classCounts = table(classes,counts,classTr,classVa,classTe,excluded, ...
    'VariableNames',{'Class','Total','Train','Validation','Test','Excluded'});
info.tilesPerClass = full(sum(best.tileClassCounts>0,1))';
trainLabels=double(gt(tr)); valLabels=double(gt(va)); testLabels=double(gt(te));
trainTiles=arrayfun(@(c)numel(unique(tileMap(tr(trainLabels==c)))),classes);
valTiles=arrayfun(@(c)numel(unique(tileMap(va(valLabels==c)))),classes);
testTiles=arrayfun(@(c)numel(unique(tileMap(te(testLabels==c)))),classes);
info.classTileCoverage=table(classes,info.tilesPerClass,trainTiles,valTiles,testTiles, ...
    'VariableNames',{'Class','Occupied','TrainTiles','ValTiles','TestTiles'});
info.tileCount = numel(owner);
info.tileGridSize = best.tileCount;
info.tileOwner = owner;  % 0=default test, 1=train, 2=val, 3=reserved test.
info.pixelAreaFraction = [numel(tr),numel(va),numel(te)]/numel(gt);
info.tileAreaFraction = [nnz(owner(tileMap)==1), ...
    nnz(owner(tileMap)==2),nnz(owner(tileMap)==0 | owner(tileMap)==3)]/numel(gt);
info.excludedFraction = sum(excluded)/sum(counts);
info.effectiveTrainingFraction = numel(tr)/sum(counts);
distance = dt(te);
info.testDistance = struct('minimum',min(distance), ...
    'p05',prctile(distance,5),'mean',mean(distance));
info.validationTrainMinDistance = min(dv(va));
info.slicSharingMean = NaN;
info.slicSharingPerGroup = [];
if ~isempty(opts.slicLabelMaps)
    maps = opts.slicLabelMaps;
    groupShare = zeros(size(maps,3),1);
    for g = 1:size(maps,3)
        L = maps(:,:,g);
        groupShare(g) = mean(ismember(L(te),unique(L([tr;va]))));
    end
    info.slicSharingMean = mean(groupShare);
    info.slicSharingPerGroup = groupShare;
end
info.relaxed = relaxed;
info.attempt = best.attempt;
info.searchGuarantee = 'Every output is verified; bounded search failure is inconclusive.';
end

function [result,failure] = search_assignment(gt,classes,budget,tileMap, ...
    tileCounts,gridSize,buffer,r,required,maxAttempts,eligibleMask,originalTest)
[h,w] = size(gt);
nc = numel(classes);
ids = cell(nc,1);
row = cell(nc,1);
col = cell(nc,1);
for k = 1:nc
    ids{k} = find(gt==classes(k));
    [row{k},col{k}] = ind2sub([h,w],ids{k});
end
tileOccupancy = full(sum(tileCounts>0,1))';
[~,classOrder] = sortrows([tileOccupancy,cellfun(@numel,ids),rand(nc,1)], [1 2 3]);
result = [];
bestDeficit = Inf;
bestTotal = -Inf;
failure = struct('class',classes(classOrder(1)), ...
    'type','train','have',0,'need',budget(classOrder(1)));
for attempt = 1:maxAttempts
    owner = zeros(prod(gridSize),1,'uint8');
    t = false(h,w); v = false(h,w); witness = false(h,w);
    failed = false;
    for pos = 1:nc
        k = classOrder(pos);
        ix = ids{k};
        rr = double(row{k}); cc = double(col{k});
        % GT-only coverage: order class pixels in short compact batches
        % across several parts of the class footprint. All three roles
        % consume this SAME order after the role/distance exclusions.
        % The older directional sort took the first budget points at a
        % class edge, leaving most distant regions exclusively for test.
        ix = spatial_coverage_order(ix,rr,cc,tileMap(ix),budget(k));
        roles = owner(tileMap(ix));
        forbidden = forbidden_by(v | witness,buffer,r);
        possible = ix((roles==0 | roles==1) & eligibleMask(ix) & ~forbidden(ix));
        if numel(possible)<budget(k)
            failure = struct('class',classes(k),'type','train', ...
                'have',numel(possible),'need',budget(k)); failed = true; break
        end
        chosen=possible(1:budget(k));
        t(chosen) = true;
        owner(tileMap(chosen)) = 1;

        roles = owner(tileMap(ix));
        forbidden = forbidden_by(t | witness,buffer,r);
        possible = ix((roles==0 | roles==2) & eligibleMask(ix) & ~forbidden(ix));
        if numel(possible)<budget(k)
            failure = struct('class',classes(k),'type','validation', ...
                'have',numel(possible),'need',budget(k)); failed = true; break
        end
        chosen=possible(1:budget(k));
        v(chosen) = true;
        owner(tileMap(chosen)) = 2;

        roles = owner(tileMap(ix));
        forbidden = forbidden_by(t | v,buffer,r);
        possible = ix((roles==0 | roles==3) & eligibleMask(ix) & ~forbidden(ix));
        if numel(possible)<required
            failure = struct('class',classes(k),'type','test', ...
                'have',numel(possible),'need',required); failed = true; break
        end
        % Reserve a compact cluster of test witnesses; all other eligible
        % labelled pixels in test tiles are added after every class is placed.
        distance = bwdist(t | v);
        [~,farthest] = max(distance(possible));
        [ar,ac] = ind2sub([h,w],possible(farthest));
        [pr,pc] = ind2sub([h,w],possible);
        [~,compact] = sort((double(pr)-ar).^2+(double(pc)-ac).^2);
        chosen = possible(compact(1:required));
        witness(chosen) = true;
        owner(tileMap(chosen)) = 3;
    end
    if failed, continue; end
    testMask = gt>0 & eligibleMask & ...
        (owner(tileMap)==0 | owner(tileMap)==3) & ...
        ~forbidden_by(t | v,buffer,r);
    testLabels = gt(testMask);
    testCount = arrayfun(@(c)nnz(testLabels==c),classes);
    deficit = sum(max(0,originalTest-testCount));
    total = nnz(testMask);
    if all(testCount>=required) && ...
            (deficit<bestDeficit || (deficit==bestDeficit && total>bestTotal))
        bestDeficit=deficit; bestTotal=total;
        result = struct('trMask',t,'vaMask',v,'teMask',testMask, ...
            'owner',owner,'tileMap',tileMap,'tileClassCounts',tileCounts, ...
            'tileCount',gridSize,'requiredTest',required,'attempt',attempt);
        if deficit==0, return; end
    end
end
end

function ordered = spatial_coverage_order(ix,rr,cc,tileIDs,budget)
% Coarse cells improve coverage while compact pixel batches limit the
% number of occupied tiles and the size of the excluded buffer area.
% The fixed rule is determined by class label coordinates and budget only.
n = numel(ix);
q = min(3,max(1,floor(sqrt(double(budget)/10))));
rr = double(rr(:)); cc = double(cc(:));
rowCell = min(q-1,floor(q*(rr-min(rr))/max(1,max(rr)-min(rr)+1)));
colCell = min(q-1,floor(q*(cc-min(cc))/max(1,max(cc)-min(cc)+1)));
cellID = 1 + rowCell + q*colCell;
cellPriority = randperm(q*q);
withinCellRank = zeros(n,1);
occupied = unique(cellID);
for cell = occupied(:)'
    local = find(cellID==cell);
    [~,~,tileGroup] = unique(tileIDs(local));
    counts = accumarray(tileGroup,1);
    % Prefer tiles containing more pixels of this class; random ties and
    % random order within a tile provide variation across seeds/attempts.
    [~,tileOrder] = sortrows([-counts,rand(numel(counts),1)],[1 2]);
    tilePriority = zeros(numel(counts),1);
    tilePriority(tileOrder) = 1:numel(counts);
    [~,localOrder] = sortrows([tilePriority(tileGroup), ...
        rand(numel(local),1)],[1 2]);
    withinCellRank(local(localOrder)) = 1:numel(local);
end
batchSize = max(1,ceil(double(budget)/numel(occupied)));
[~,order] = sortrows([floor((withinCellRank-1)/batchSize), ...
    reshape(cellPriority(cellID(:)),[],1),withinCellRank],[1 2 3]);
ordered = ix(order);
end

function forbidden = forbidden_by(mask,buffer,r)
forbidden = bwdist(mask) < buffer;
if r>0
    forbidden = forbidden | bwdist(mask,'chessboard') <= 2*r;
end
end

function [tileMap,counts,grid] = make_tiles(gt,classes,B)
[h,w] = size(gt);
grid = [ceil(h/B) ceil(w/B)];
tileMap = ceil((1:h)'/B) + (ceil((1:w)/B)-1)*grid(1);
labelled = find(gt>0);
[~,classNo] = ismember(double(gt(labelled)),classes);
counts = sparse(double(tileMap(labelled)),double(classNo),1,prod(grid),numel(classes));
end

function result = change(parameter,classValue,before,after,reason)
result = struct('parameter',parameter,'class',classValue, ...
    'from',before,'to',after,'reason',reason);
end

function check_scalar(x,lo,hi,name,integer)
if ~isnumeric(x) || ~isscalar(x) || ~isfinite(x) || x<lo || x>hi || ...
        (integer && x~=fix(x)) || (~integer && x==lo)
    error('spatial_block_split:Input','%s has an invalid value.',name);
end
end

function yes = islogical_or_binary(x)
yes = islogical(x) || (isnumeric(x) && isfinite(x) && ismember(x,[0 1]));
end

