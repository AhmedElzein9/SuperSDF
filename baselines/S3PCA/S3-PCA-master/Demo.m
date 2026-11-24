% --- loop_KSSDF.m ---

% (1) Load or define your Data variable here:
% load('yourDataFile.mat');   % e.g. loads variable Data
% % or
% Data ='Salinas';                 % however you obtain Data
tic;
% (2) Define the range of cluster sizes and the fixed N_SP
cluster_range = 30;
N_SP = 30;
train_ratio = 0.01;
D = 20;
% (3) Pre-allocate the result vector
% (3) Pre-allocate the result vector
overall_accuracy = zeros(numel(train_ratio), 1);

% (4) Loop over each cluster size
for i = 1:size(train_ratio)
    % train_ratio = train_ratio(i);
    % overall_accuracy(i) = RAW('Indian_pines.mat', train_ratio(i));
    % overall_accuracy(i) = PCA('Indian_pines.mat', cluster_range, train_ratio(i));
    % overall_accuracy(i)= EMAP('Indian_pines.mat', train_ratio(i));
    % overall_accuracy(i) = ID_CNN('Indian_pines', train_ratio(i));
    % overall_accuracy(i) = DBSSAN ('Indian_pines',cluster_range,train_ratio(i));
    % overall_accuracy(i) = SoperPCA ('Indian_pines',cluster_range,train_ratio(i));
    % overall_accuracy(i) = SuperNWFE('Indian_pines.mat', cluster_range,  train_ratio(i));
    % overall_accuracy(i) = KSSDF('Indian_pines.mat', cluster_range, N_SP,train_ratio(i),D);
    overall_accuracy(i) = FS3PCA('Salinas',cluster_range, train_ratio(i));
    fprintf('Finished train_ratio = %d, O accuracy = %.4f\n', ...
            train_ratio(i), overall_accuracy(i));
end
time =  toc;
% (5) Save the results to a .mat file
% This will create 'overall_accuracy.mat' containing:
%   - cluster_range   (the vector of tested cluster sizes)
%   - overall_accuracy (the corresponding accuracies)
save('overall_accuracy.mat', 'train_ratio', 'overall_accuracy');

