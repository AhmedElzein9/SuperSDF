% Monte Carlo Evaluation for SuperSDF
N_RUNS = 10;
OA_all = zeros(N_RUNS, 1);
AA_all = zeros(N_RUNS, 1);
kappa_all = zeros(N_RUNS, 1);

fprintf('==== Running SuperSDF %d times with random splits ====\n', N_RUNS);

for run = 1:N_RUNS
    fprintf('\n=== RUN %d/%d ===\n', run, N_RUNS);
    rng(run); % Set different random seed for each run

    % Call SuperSDF core function (you should refactor the body of your current script into a function)
    % [overall_accuracy, avg_accuracy, kappa, class_accuracies] = FSuperSDF('LongKou.mat', 15, 15, 0.10); % Change to your dataset
    [overall_accuracy, avg_accuracy, kappa, class_accuracies] = FS3PCA('Salinas',30, 0.01);
    OA_all(run) = overall_accuracy;
    AA_all(run) = avg_accuracy;
    classAcc_all(run, :) = class_accuracies(:)';  % Store all class-wise AA
    kappa_all(run) = kappa;
end

% Final summary
fprintf('\n===== Summary Over %d Runs =====\n', N_RUNS);
fprintf('Overall Accuracy: %.4f ± %.4f\n', mean(OA_all), std(OA_all));
fprintf('Average Accuracy: %.4f ± %.4f\n', mean(AA_all), std(AA_all));
fprintf('Kappa Coefficient: %.4f ± %.4f\n', mean(kappa_all), std(kappa_all));

% Per-class accuracy
fprintf('\n-- Per-Class AA --\n');
meanClassAA = mean(classAcc_all, 1);
stdClassAA  = std(classAcc_all, 0, 1);
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