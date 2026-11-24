function cfg = p1_cfg(id)
%P1_CFG Paper parameters (SuperSDF_MC10.m) and block-split geometry.
%   Block geometry: B=20, gap=5, min 5 test/class is feasible for PU and HC
%   on all 10 seeds. For IP it is infeasible (classes 1,7,8,9,16 occupy <3
%   tiles); the only feasible tested geometry is B=5, gap=2, min 2 test/class.
switch lower(id)
    case {'indian','indianpines','ip'}
        cfg = struct('name','Indian Pines','short','IP','tag','IndianPines', ...
            'file','Indian_pines.mat','k',35,'N_SP',60,'D',25,'train_ratio',0.05);
        cfg.block = struct('B',5,'gap',2,'minTest',2);
    case {'pavia','paviau','pu'}
        cfg = struct('name','Pavia University','short','PU','tag','PaviaUniversity', ...
            'file','PaviaUF.mat','k',35,'N_SP',240,'D',240,'train_ratio',0.02);
        cfg.block = struct('B',20,'gap',5,'minTest',5);
    case {'hanchuan','hc'}
        cfg = struct('name','WHU-Hi-HanChuan','short','HC','tag','HanChuan', ...
            'file','HanChuan.mat','k',35,'N_SP',400,'D',150,'train_ratio',0.01);
        cfg.block = struct('B',20,'gap',5,'minTest',5);
    otherwise
        error('Unknown dataset %s', id);
end
end
