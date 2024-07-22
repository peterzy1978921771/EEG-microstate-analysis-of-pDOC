%% Clear Environment
clc
clear
close all
addpath(genpath('libsvm-3.31'));

%% Import Data
% Load data
load data.mat
load data_labels.mat

rng(1) % Random number seed

% Initialize cross-validation parameters
k = 5; % 5-fold cross-validation
indices = crossvalind('Kfold', data_labels, k);

% Initialize result arrays
results_acc_svm = zeros(k, 1);
results_acc_pso = zeros(k, 1);
results_acc_qpso = zeros(k, 1);

results_prec_svm = zeros(k, 1);
results_prec_pso = zeros(k, 1);
results_prec_qpso = zeros(k, 1);

results_reca_svm = zeros(k, 1);
results_reca_pso = zeros(k, 1);
results_reca_qpso = zeros(k, 1);

results_f1_svm = zeros(k, 1);
results_f1_pso = zeros(k, 1);
results_f1_qpso = zeros(k, 1);

%% Perform 5-fold cross-validation
for fold = 1:k
    % Split training and testing sets
    test_indices = (indices == fold);
    train_indices = ~test_indices;
    p_train = data(train_indices, :);
    t_train = data_labels(train_indices, :);
    p_test = data(test_indices, :);
    t_test = data_labels(test_indices, :);

    %% Data normalization
    % Training set
    [pn_train, inputps] = mapminmax(p_train');
    pn_train = pn_train';
    pn_test = mapminmax('apply', p_test', inputps);
    pn_test = pn_test';

    %% QPSO Algorithm
    % Particle swarm parameters
    popsize = 20; % Population size
    MAXITER = 300; % Maximum iterations
    dimension = 2; % Problem dimension
    irange_l = [-50, -50]; % Lower bound for position initialization
    irange_r = [50, 50]; % Upper bound for position initialization
    xmax = 50; % Upper bound of search range
    xmin = -50; % Lower bound of search range
    M = (xmax - xmin) / 2; % Midpoint of search range
    sum1 = 0;
    st = 0;
    runno = 1; % Number of algorithm runs
    yy2 = zeros(runno, MAXITER); % Record the best fitness value at each iteration
    % SVR
    v1 = 5;
    bestnum = 2;
    a2 = 1;
    a0 = 0.5;
    v1 = 5;
    %% Initialization
    T = cputime; % Record CPU time
    x = (irange_r - irange_l) .* rand(popsize, dimension, 1) + irange_l; % Initialize particle positions
    pbest = x; % Initialize personal best positions
    gbest = zeros(1, dimension); % Initialize global best position variable
    for i = 1:popsize % Calculate fitness of current and personal best positions
        f_x(i) = SVMcgForClass(t_train, pn_train, x(i, 1), x(i, 2), v1, bestnum);
        f_pbest(i) = f_x(i);
    end
    g = min(find(f_pbest == max(f_pbest(1:popsize)))); % Find index of global best position
    gbest = pbest(g, :); % Find global best position
    f_gbest = f_pbest(g); % Record fitness of global best position
    MINIMUM = f_pbest(g);
    % Record the best fitness value at each iteration
    al = [];

    %% Iteration
    for t = 1:MAXITER % Iterative process of the algorithm
        alpha = (a2 - a0) * (MAXITER - t) / MAXITER + a0; % Shrinkage-expansion coefficient calculation a1=1 a0=0.5
        al = [al alpha];

        % New QPSO term representing the average value of pbest
        mbest = sum(pbest) / popsize; % Calculate mbest

        for i = 1:popsize
            % Particle update
            fi = rand(1, dimension);
            p = fi .* pbest(i, :) + (1 - fi) .* gbest; % Calculate random point p
            u = rand(1, dimension);
            b = alpha * abs(mbest - x(i, :));
            v = -log(u);
            x(i, :) = p + ((-1) .^ ceil(0.5 + rand(1, dimension))) .* b .* v; % Update particle positions

            % Constrain particle positions within search range
            z = x(i, :) - (xmax + xmin) / 2;
            z = sign(z) .* min(abs(z), M);
            x(i, :) = z + (xmax + xmin) / 2;

            f_x(i) = SVMcgForClass(t_train, pn_train, x(i, 1), x(i, 2), v1, bestnum); % Calculate fitness of current positions
            if (f_x(i) > f_pbest(i)) % Update personal best positions
                pbest(i, :) = x(i, :);
                f_pbest(i) = f_x(i);
            end
            if f_pbest(i) > f_gbest % Update global best position
                gbest = pbest(i, :);
                f_gbest = f_pbest(i);
            end
            MINIMUM = f_gbest;
        end
        yy2(t) = MINIMUM; % Record the best fitness value at each iteration
    end
    %% Result Analysis
    fitnesszbest_qpso = MINIMUM;
    zbest_qpso = gbest;

    %% Use regression prediction to analyze the best parameters for SVM training
    cmd = [' -t 2', ' -c ', num2str(bestnum^zbest_qpso(1)), ' -g ', num2str(bestnum^zbest_qpso(2))];
    model = svmtrain(t_train, pn_train, cmd);

    [pre1_qpso] = svmpredict(t_train, pn_train, model);
    [pre2_qpso] = svmpredict(t_test, pn_test, model);

    %% Standalone SVM
    % Create/train SVM model
    % Best c and g parameters
    c = 20;
    g = 2;

    cmd = ['-v ', num2str(v1), ' -t 2', ' -c ', num2str(bestnum^c), ' -g ', num2str(bestnum^g)];
    cg = svmtrain(t_train, pn_train, cmd);

    acc = cg;
    bestc = 2^c;
    bestg = 2^g;

    % Create/train SVM
    cmd = [' -t 2', ' -c ', num2str(bestc), ' -g ', num2str(bestg)];
    model2 = svmtrain(t_train, pn_train, cmd);

    %% SVM Simulation Prediction
    [pre1_] = svmpredict(t_train, pn_train, model2);   % Training set prediction results
    [pre2_] = svmpredict(t_test, pn_test, model2);  % Testing set prediction results

    %% PSO-SVM
    % Parameter initialization
    % Two parameters in PSO
    c1 = 1.49445;
    c2 = 1.49445;
    % Linearly decreasing inertia weight
    ws = 0.9;
    we = 0.4;
    maxgen = 300;   % Evolution times
    sizepop = 20;   % Population size
    % Velocity limits
    Vmax = .5;
    Vmin = -.5;
    % Position limits (position variables g, c values)
    popmin = [-50, -50];
    popmax = [50, 50];
    % Linearly decreasing inertia weight
    wmax = 0.9;
    wmin = 0.4;
    % SVR
    v = 5;
    bestnum = 2;

    %% Generate initial particles and velocities
    for i = 1:sizepop
        % Randomly generate a population
        pop(i, :) = popmax(1) * rands(1, 2);    % Initial population
        V(i, :) = Vmax * rands(1, 2);  % Initial velocities
        % Calculate fitness
        fitness(i) = SVMcgForClass(t_train, pn_train, pop(i, 1), pop(i, 2), v, bestnum);   % Chromosome fitness
    end

    %% Personal best and global best
    [bestfitness, bestindex] = max(fitness);
    zbest = pop(bestindex, :);   % Global best
    gbest = pop;    % Personal best
    fitnessgbest = fitness;   % Personal best fitness values
    fitnesszbest = bestfitness;   % Global best fitness value

    %% Iterative Optimization
    for i = 1:maxgen
        for j = 1:sizepop
            % Linear decreasing
            w = ws - (ws - we) * i / maxgen;
            % Velocity update
            V(j, :) = w * V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
            V(j, find(V(j, :) > Vmax)) = Vmax;
            V(j, find(V(j, :) < Vmin)) = Vmin;
            % Position update
            pop(j, :) = pop(j, :) + V(j, :);
            if isempty(find(pop(j, :) > popmax))
                ;
            else
                loc = find(pop(j, :) > popmax);
                pop(j, find(pop(j, :) > popmax)) = popmax(find(pop(j, :) > popmax));
            end
            if isempty(find(pop(j, :) < popmin))
                ;
            else
                loc = find(pop(j, :) < popmin);
                pop(j, find(pop(j, :) < popmin)) = popmin(find(pop(j, :) < popmin));
            end
            % Fitness value
            fitness(j) = SVMcgForClass(t_train, pn_train, pop(j, 1), pop(j, 2), v, bestnum);
        end

        for j = 1:sizepop
            % Personal best update
            if fitness(j) > fitnessgbest(j)
                gbest(j, :) = pop(j, :);
                fitnessgbest(j) = fitness(j);
            end

            % Global best update
            if fitness(j) > fitnesszbest
                zbest = pop(j, :);
                fitnesszbest = fitness(j);
            end
        end
        yy(i) = fitnesszbest;
    end
    %% Result Analysis
    fitnesszbest_pso = fitnesszbest;
    zbest_pso = zbest;

    %% Use regression prediction to analyze the best parameters for SVM training
    cmd = [' -t 2', ' -c ', num2str(bestnum^zbest_pso(1)), ' -g ', num2str(bestnum^zbest_pso(2))];
    model3 = svmtrain(t_train, pn_train, cmd);

    [pre1_pso] = svmpredict(t_train, pn_train, model3);
    [pre2_pso] = svmpredict(t_test, pn_test, model3);

    %% Print selection results
    clc
    disp('Print QPSO selection results');
    str = sprintf('Best Cross Validation MSE = %g Best c = %g Best g = %g', ...
        fitnesszbest_qpso, bestnum^zbest_qpso(1), bestnum^zbest_qpso(2));
    disp(str);

    disp('Print grid search selection results');
    str = sprintf('Best Cross Validation MSE = %g Best c = %g Best g = %g', ...
        acc, bestc, bestg);
    disp(str);

    disp('Print PSO selection results');
    str = sprintf('Best Cross Validation MSE = %g Best c = %g Best g = %g', ...
        fitnesszbest_pso, bestnum^zbest_pso(1), bestnum^zbest_pso(2));
    disp(str);

    %% Testing set error
    cl_num = max(max(t_test)); % Maximum number of categories
    YTest_ = zeros(cl_num, size(t_test, 1));
    YPre_ = zeros(cl_num, size(t_test, 1));
    YPre_pso = zeros(cl_num, size(t_test, 1));
    YPre_qpso = zeros(cl_num, size(t_test, 1));
    for i = 1:size(t_test, 1)
        YTest_(t_test(i), i) = 1;
        YPre_(pre2_(i), i) = 1;
        YPre_pso(pre2_pso(i), i) = 1;
        YPre_qpso(pre2_qpso(i), i) = 1;
    end

    % Confusion matrix
    m_ = confusionmat(t_test, pre2_);
    m_pso = confusionmat(t_test, pre2_pso);
    m_qpso = confusionmat(t_test, pre2_qpso);

    % Accuracy
    acc_ = sum(diag(m_)) / sum(m_(:));
    acc_pso = sum(diag(m_pso)) / sum(m_pso(:));
    acc_qpso = sum(diag(m_qpso)) / sum(m_qpso(:));

    % Precision
    prec_ = mean(diag(m_) ./ sum(m_, 1)');
    prec_pso = mean(diag(m_pso) ./ sum(m_pso, 1)');
    prec_qpso = mean(diag(m_qpso) ./ sum(m_qpso, 1)');

    % Recall
    reca_ = mean(diag(m_) ./ sum(m_, 2));
    reca_pso = mean(diag(m_pso) ./ sum(m_pso, 2));
    reca_qpso = mean(diag(m_qpso) ./ sum(m_qpso, 2));

    % F1
    f1_ = 2 * prec_ .* reca_ ./ (prec_ + reca_);
    f1_pso = 2 * prec_pso .* reca_pso ./ (prec_pso + reca_);
    f1_qpso = 2 * prec_qpso .* reca_qpso ./ (prec_qpso + reca_);

    % Record results
    results_acc_svm(fold) = acc_;
    results_acc_pso(fold) = acc_pso;
    results_acc_qpso(fold) = acc_qpso;

    results_prec_svm(fold) = prec_;
    results_prec_pso(fold) = prec_pso;
    results_prec_qpso(fold) = prec_qpso;

    results_reca_svm(fold) = reca_;
    results_reca_pso(fold) = reca_pso;
    results_reca_qpso(fold) = reca_qpso;

    results_f1_svm(fold) = f1_;
    results_f1_pso(fold) = f1_pso;
    results_f1_qpso(fold) = f1_qpso;
end

%% Calculate averages
mean_acc_svm = mean(results_acc_svm);
mean_acc_pso = mean(results_acc_pso);
mean_acc_qpso = mean(results_acc_qpso);

mean_prec_svm = mean(results_prec_svm);
mean_prec_pso = mean(results_prec_pso);
mean_prec_qpso = mean(results_prec_qpso);

mean_reca_svm = mean(results_reca_svm);
mean_reca_pso = mean(results_reca_pso);
mean_reca_qpso = mean(results_reca_qpso);

mean_f1_svm = mean(results_f1_svm);
mean_f1_pso = mean(results_f1_pso);
mean_f1_qpso = mean(results_f1_qpso);

%% Print average results
fprintf('Average Accuracy SVM: %f, PSO-SVM: %f, QPSO-SVM: %f\n', mean_acc_svm, mean_acc_pso, mean_acc_qpso);
fprintf('Average Precision SVM: %f, PSO-SVM: %f, QPSO-SVM: %f\n', mean_prec_svm, mean_prec_pso, mean_prec_qpso);
fprintf('Average Recall SVM: %f, PSO-SVM: %f, QPSO-SVM: %f\n', mean_reca_svm, mean_reca_pso, mean_reca_qpso);
fprintf('Average F1 SVM: %f, PSO-SVM: %f, QPSO-SVM: %f\n', mean_f1_svm, mean_f1_pso, mean_f1_qpso);

%% Plotting
% Iteration results
figure
plot(1:length(yy), yy, 'Color', [255 096 107] / 255, 'LineWidth', 3); hold on;
plot(1:length(yy), yy2, 'Color', [021 151 165] / 255, 'LineWidth', 3);

legend('PSO-SVM', 'QPSO-SVM', 'FontName', 'Times New Roman', 'FontSize', 11);
xlabel('Iterations', 'FontName', 'Times New Roman', 'FontSize', 12);
ylabel('Fitness value', 'FontName', 'Times New Roman', 'FontSize', 12);
title('Fitness curve', 'FontName', 'Times New Roman', 'FontSize', 12);

% Confusion matrix
[t_test, index_] = sort(t_test);
pre2_ = pre2_(index_);
pre2_pso = pre2_pso(index_);
pre2_qpso = pre2_qpso(index_);

figure
plot(t_test, '-+', 'LineWidth', 2, 'MarkerIndices', 1:5:length(t_test))
hold on
plot(pre2_, '->', 'LineWidth', 2, 'MarkerIndices', 1:5:length(pre2_))
hold on
plot(pre2_pso, '-s', 'LineWidth', 2, 'MarkerIndices', 1:5:length(pre2_pso))
hold on
plot(pre2_qpso, '-o', 'LineWidth', 2, 'MarkerIndices', 1:5:length(pre2_qpso))
legend('Actual value', 'SVM Predictive value', 'PSO-SVM Predictive value', 'QPSO-SVM Predictive value', 'FontName', 'Times New Roman', 'FontSize', 11)
xlabel('Sample', 'FontName', 'Times New Roman', 'FontSize', 12)
ylabel('Predictive value', 'FontName', 'Times New Roman', 'FontSize', 12)
title('Comparison of test set prediction results', 'FontName', 'Times New Roman', 'FontSize', 12)

figure
cm = confusionchart(t_test, pre2_);
cm.Title = 'SVM';
cm.FontName = 'Times New Roman';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.YLabel = 'Actual category';
cm.XLabel = 'Prediction category';

figure
cm = confusionchart(t_test, pre2_pso);
cm.Title = 'PSO-SVM';
cm.FontName = 'Times New Roman';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.YLabel = 'Actual category';
cm.XLabel = 'Prediction category';

figure
cm = confusionchart(t_test, pre2_qpso);
cm.Title = 'QPSO-SVM';
cm.FontName = 'Times New Roman';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.YLabel = 'Actual category';
cm.XLabel = 'Prediction category';

figure
AA = [mean_acc_svm, mean_acc_pso, mean_acc_qpso; ...
    mean_prec_svm, mean_prec_pso, mean_prec_qpso; ...
    mean_reca_svm, mean_reca_pso, mean_reca_qpso; ...
    mean_f1_svm, mean_f1_pso, mean_f1_qpso];
B = bar(AA);
xticklabels({'Accuracy', 'Precision', 'Recall', 'F1'})
legend('SVM', 'PSO-SVM', 'QPSO-SVM', 'FontName', 'Times New Roman', 'FontSize', 11)
B(1).FaceColor = [255 157 140] / 255;
B(2).FaceColor = [230 111 081] / 255;
B(3).FaceColor = [233 196 107] / 255;
title('Evaluation of classification results', 'FontName', 'Times New Roman', 'FontSize', 12)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);