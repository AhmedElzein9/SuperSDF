% Monte-Carlo evaluation for SoperPCA_New
clc; clear; close all;

N_RUNS = 10;
OA_all = zeros(N_RUNS,1);
AA_all = zeros(N_RUNS,1);
kappa_all = zeros(N_RUNS,1);
time_all = zeros(N_RUNS,1);
classAcc_all = [];

% Change these three values per dataset.
DATABASE    = 'Indian_pines';
NUM_PC      = 35;
TRAIN_RATIO = 0.05;

fprintf('==== Running SoperPCA_New %d times with random splits ====\n', N_RUNS);

for run = 1:N_RUNS
    fprintf('\n=== RUN %d/%d ===\n', run, N_RUNS);

    % Same Monte-Carlo seed sequence used in SuperSDF_New.
    rng(run*7 + 13, 'twister');
    t_run = tic;

    [overall_accuracy, avg_accuracy, kappa, class_accuracies] = ...
        SoperPCA_New(DATABASE, NUM_PC, TRAIN_RATIO);

    time_all(run) = toc(t_run);
    OA_all(run) = overall_accuracy;
    AA_all(run) = avg_accuracy;
    kappa_all(run) = kappa;

    if isempty(classAcc_all)
        classAcc_all = zeros(N_RUNS, numel(class_accuracies));
    end
    classAcc_all(run,:) = class_accuracies(:)';

    fprintf('Run time: %.2f s\n', time_all(run));
end

fprintf('\n===== SuperPCA Summary Over %d Runs =====\n', N_RUNS);
fprintf('Overall Accuracy : %.4f +/- %.4f %%\n', mean(OA_all*100), std(OA_all*100));
fprintf('Average Accuracy : %.4f +/- %.4f %%\n', mean(AA_all*100), std(AA_all*100));
fprintf('Kappa            : %.4f +/- %.4f %%\n', mean(kappa_all*100), std(kappa_all*100));
fprintf('Time per run     : %.2f +/- %.2f s (total %.2f s)\n', ...
    mean(time_all), std(time_all), sum(time_all));

fprintf('\n-- Per-Class Accuracy --\n');
meanClassAA = mean(classAcc_all*100,1);
stdClassAA = std(classAcc_all*100,0,1);
for c = 1:numel(meanClassAA)
    fprintf('Class %2d: %.4f +/- %.4f %%\n', c, meanClassAA(c), stdClassAA(c));
end

figure;
boxplot([OA_all, AA_all, kappa_all], 'Labels', {'OA','AA','Kappa'});
title(sprintf('SuperPCA Monte Carlo (%d runs)', N_RUNS));
ylabel('Score');
grid on;

% Optional save:
% save('SuperPCA_Results.mat', 'OA_all', 'AA_all', 'kappa_all', ...
%     'classAcc_all', 'time_all', 'DATABASE', 'NUM_PC', 'TRAIN_RATIO');
