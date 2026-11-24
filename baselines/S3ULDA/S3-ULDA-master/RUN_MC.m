% Monte Carlo Evaluation for SuperSDF
clc;clear;close all;
N_RUNS = 10;
OA_all = zeros(N_RUNS, 1);
AA_all = zeros(N_RUNS, 1);
kappa_all = zeros(N_RUNS, 1);
time_all  = zeros(N_RUNS, 1);          
classAcc_all = [];

fprintf('==== Running SuperSDF %d times with random splits ====\n', N_RUNS);

for run = 1:N_RUNS
    fprintf('\n=== RUN %d/%d ===\n', run, N_RUNS);
    rng(run*7 + 13); % Set different random seed for each run
    t_run = tic; 
    % Call SuperSDF core function (you should refactor the body of your current script into a function)
    
    % [overall_accuracy, avg_accuracy, kappa, class_accuracies] = RAW('Salinas.mat', 0.05);
    % [overall_accuracy, avg_accuracy, kappa, class_accuracies] = PCA('Salinas.mat', 15, 0.05);
    % [overall_accuracy, avg_accuracy, kappa, class_accuracies] = EMAP('Salinas.mat', 0.05);
    % [overall_accuracy, avg_accuracy, kappa, class_accuracies] = ID_CNN('PaviaUF', 0.10);
    % [overall_accuracy, avg_accuracy, kappa, class_accuracies] = DBSSAN ('Salinas',15,0.05);
    % [overall_accuracy, avg_accuracy, kappa, class_accuracies] = SuperNWFE('Salinas.mat', 15, 0.05);
    % [overall_accuracy, avg_accuracy, kappa, class_accuracies] = SoperPCA ('Salinas',15,0.05);
    % [overall_accuracy, avg_accuracy, kappa, class_accuracies] = FS3PCA('PaviaUF',30, 0.10);
    % % [overall_accuracy, avg_accuracy, kappa,class_accuracies] = M3FN(database, 0.10);
    % [overall_accuracy, avg_accuracy, kappa, class_accuracies] = FSuperSDF('PaviaUF.mat', 35, 500, 0.02);
    % [overall_accuracy, avg_accuracy, kappa, class_accuracies] = FSuperSDF('Indian_pines.mat', 30, 75, 0.05);
    [overall_accuracy, avg_accuracy, kappa,class_accuracies] = S3ULDA('Indian_pines',0.05);
    time_all(run) = toc(t_run);
    OA_all(run) = overall_accuracy;
    AA_all(run) = avg_accuracy;
    % classAcc_all(run, :) = class_accuracies(:)';  % Store all class-wise AA
    kappa_all(run) = kappa;
    if isempty(classAcc_all)
        classAcc_all = zeros(N_RUNS, numel(class_accuracies));
    end
    classAcc_all(run, :) = class_accuracies(:)';

    fprintf('Run time: %.2f s\n', time_all(run));
end

% Final summary
fprintf('\n===== Summary Over %d Runs =====\n', N_RUNS);
fprintf('Overall Accuracy: %.4f ± %.4f\n', mean(OA_all*100), std(OA_all*100));
fprintf('Average Accuracy: %.4f ± %.4f\n', mean(AA_all*100), std(AA_all*100));
fprintf('Kappa Coefficient: %.4f ± %.4f\n', mean(kappa_all*100), std(kappa_all*100));
fprintf('Time per run     : %.2f +- %.2f s   (total %.2f s)\n', mean(time_all), std(time_all), sum(time_all));

% Per-class accuracy
fprintf('\n-- Per-Class AA --\n');
meanClassAA = mean(classAcc_all*100, 1);
stdClassAA  = std(classAcc_all*100, 0, 1);
for c = 1:length(meanClassAA)
    fprintf('Class %2d: %.4f ± %.4f\n', c, meanClassAA(c), stdClassAA(c));
end

% Optional: plot boxplots
figure;
boxplot([OA_all, AA_all, kappa_all], 'Labels', {'OA', 'AA', 'Kappa'});
title(sprintf('SuperSDF Monte Carlo (%d runs)', N_RUNS));
ylabel('Score');
grid on;

% % Optional: save
% save('SuperSDF_Results.mat', 'OA_all', 'AA_all', 'Kappa_all', 'classAcc_all');