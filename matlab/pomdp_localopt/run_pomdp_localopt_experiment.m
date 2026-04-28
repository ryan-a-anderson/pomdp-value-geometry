function results = run_pomdp_localopt_experiment()
% RUN_POMDP_LOCALOPT_EXPERIMENT
% 
% Main script for illustrating local optima in infinite-horizon discounted
% POMDPs with memoryless stochastic policies.
%
% The script runs two experiments:
%   (1) Batch experiment over many random instances and many random restarts:
%       compares a partially observable setting against a fully observable
%       baseline built from the same transition/reward data.
%   (2) A low-dimensional landscape experiment (2 observations, 2 actions)
%       where the objective J = mu' * V^pi is visualized over the policy
%       parameter space, together with gradient-ascent trajectories.
%
% The optimized variables are policy logits theta(a,o), with
%   pi(a|o) = softmax(theta(:,o)).
% The value function V^pi is always evaluated exactly by solving the Bellman
% system induced by the current stochastic memoryless policy.
%
% Output:
%   results : structure containing the main data and generated figures.
%
% Tested with standard MATLAB functionality only.

    clearvars -except results;
    close all;
    clc;

    rng(7);

    outdir = fullfile(fileparts(mfilename('fullpath')), 'output');
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    %% ================================================================
    %  Experiment 1: many random instances, many random restarts
    %  ================================================================
    batch_cfg = struct();
    batch_cfg.S_list = [4 8 12];
    batch_cfg.A_list = [2 3 4];
    batch_cfg.O_partial_list = [2 3];
    batch_cfg.num_instances = 100;
    batch_cfg.num_restarts = 50;
    batch_cfg.max_iters = 350;
    batch_cfg.step_size = 0.05;
    batch_cfg.grad_clip = 10.0;
    %batch_cfg.gamma_range = [0.80, 0.96];
    batch_cfg.gamma_range = [0.95, 0.98];
    batch_cfg.reward_scale = 1.0;
    batch_cfg.verbose = true;
    batch_cfg.stationary_bonus = 0.15; % try 0.3 ? 
    batch_cfg.subopt_threshold = 0.01;
    %batch_cfg.num_instances_per_combo = 10;

    batch_results = run_batch_experiment(batch_cfg);

    % produce summary table 
    summary_table = summarize_by_configuration(batch_results);
    disp(summary_table);
    writetable(summary_table, fullfile(outdir, 'summary_table.csv'));
    


    fig1 = figure('Color', 'w', 'Position', [100 100 1250 780]);
    tiledlayout(3,2, 'Padding', 'compact', 'TileSpacing', 'compact');

    % (a) Histogram of normalized gaps pooled over instances
    nexttile;
    bins = linspace(0, 1, 26);
    histogram(batch_results.partial.normalized_gap, bins, ...
        'Normalization', 'probability', 'FaceAlpha', 0.65);
    hold on;
    histogram(batch_results.full.normalized_gap, bins, ...
        'Normalization', 'probability', 'FaceAlpha', 0.65);
    xlabel('Normalized suboptimality gap within instance');
    ylabel('Probability');
    title('Pooling all restarts across random instances');
    legend({'Partially observable', 'Fully observable baseline'}, 'Location', 'northeast');
    grid on;

    % (b) Boxplot by observability regime
    nexttile;
    boxplot([batch_results.partial.final_J_all(:); batch_results.full.final_J_all(:)], ...
        [ones(numel(batch_results.partial.final_J_all),1); 2*ones(numel(batch_results.full.final_J_all),1)], ...
        'Labels', {'Partial', 'Full'});
    ylabel('Final objective J = \mu^T V^\pi');
    title('Distribution of final values over all restarts');
    grid on;

    % (c) Instance-wise fraction of suboptimal restarts
    nexttile;
    plot(batch_results.partial.suboptimal_fraction_by_instance, '-o', 'LineWidth', 1.5); hold on;
    plot(batch_results.full.suboptimal_fraction_by_instance, '-s', 'LineWidth', 1.5);
    xlabel('Random instance');
    ylabel('Fraction of suboptimal restarts');
    title('Suboptimal convergence by instance');
    legend({'Partial', 'Full'}, 'Location', 'best');
    ylim([0 1]);
    grid on;

    % (d) Instance-wise spread of final values
    nexttile;
    plot(batch_results.partial.spread_by_instance, '-o', 'LineWidth', 1.5); hold on;
    plot(batch_results.full.spread_by_instance, '-s', 'LineWidth', 1.5);
    xlabel('Random instance');
    ylabel('max(J) - min(J) over restarts');
    title('Spread of converged values by instance');
    legend({'Partial', 'Full'}, 'Location', 'best');
    grid on;

    % (e) Instance-wise spread of final policies 
    nexttile;
    plot(batch_results.partial.policy_spread_by_instance, '-o', 'LineWidth', 1.5); hold on;
    plot(batch_results.full.policy_spread_by_instance, '-s', 'LineWidth', 1.5);
    xlabel('Random instance');
    ylabel('Policy spread');
    title('Spread of converged policies by instance');
    legend({'Partial', 'Full'}, 'Location', 'best');
    grid on;

    exportgraphics(fig1, fullfile(outdir, 'batch_summary.png'), 'Resolution', 200);

    %% ================================================================
    %  Experiment 2: low-dimensional objective landscape
    %  ================================================================
    land_cfg = struct();
    land_cfg.S = 4;
    land_cfg.A = 2;
    land_cfg.O = 2;
    land_cfg.gamma = 0.95;
    land_cfg.num_restarts = 20;
    land_cfg.max_iters = 300;
    land_cfg.step_size = 0.05;
    land_cfg.grad_clip = 10.0;
    land_cfg.grid_n = 151;
    land_cfg.seed = 19;

    landscape_results = run_landscape_experiment(land_cfg);

    fig2 = figure('Color', 'w', 'Position', [100 100 1100 780]);
    tiledlayout(2,2, 'Padding', 'compact', 'TileSpacing', 'compact');

    % Heatmap of J over (p1, p2)
    nexttile([2 1]);
    imagesc(landscape_results.p_grid, landscape_results.q_grid, landscape_results.J_grid);
    set(gca, 'YDir', 'normal');
    hold on;
    contour(landscape_results.p_grid, landscape_results.q_grid, landscape_results.J_grid, 18, 'k');
    colormap(parula);
    colorbar;
    xlabel('p = \pi(a=1 | o=1)');
    ylabel('q = \pi(a=1 | o=2)');
    title('Objective landscape for a 2-observation, 2-action POMDP');
    for k = 1:numel(landscape_results.trajs)
        tr = landscape_results.trajs{k};
        plot(tr.p, tr.q, '-', 'LineWidth', 1.2);
        plot(tr.p(1), tr.q(1), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'w');
        plot(tr.p(end), tr.q(end), 'k.', 'MarkerSize', 16);
    end

    % Histogram of final J values
    nexttile;
    histogram(landscape_results.final_J, 12, 'FaceAlpha', 0.8);
    xlabel('Final objective value');
    ylabel('Count');
    title('Multiple limiting values from different restarts');
    grid on;

    % Scatter of final policies
    nexttile;
    scatter(landscape_results.final_p, landscape_results.final_q, 60, landscape_results.final_J, 'filled');
    xlabel('Final p = \pi(a=1|o=1)');
    ylabel('Final q = \pi(a=1|o=2)');
    title('Converged policies');
    colorbar;
    grid on;
    xlim([0 1]);
    ylim([0 1]);

    exportgraphics(fig2, fullfile(outdir, 'landscape_summary.png'), 'Resolution', 220);

    %% ================================================================
    %  Save results
    %  ================================================================
    results = struct();
    results.summary_table = summary_table;
    results.batch = batch_results;
    results.landscape = landscape_results;
    
    save(fullfile(outdir, 'results.mat'), 'results');

    fprintf('\nSaved outputs to:\n  %s\n', outdir);
    fprintf('Generated files:\n');
    fprintf('  - batch_summary.png\n');
    fprintf('  - landscape_summary.png\n');
    fprintf('  - results.mat\n');

end
%%


%% =====================================================================
function batch_results = run_batch_experiment(cfg)

    partial_by_instance = zeros(cfg.num_instances, cfg.num_restarts);
    full_by_instance    = zeros(cfg.num_instances, cfg.num_restarts);

    partial_gap_by_instance = zeros(cfg.num_instances, cfg.num_restarts);
    full_gap_by_instance    = zeros(cfg.num_instances, cfg.num_restarts);

    partial_subfrac = zeros(cfg.num_instances,1);
    full_subfrac    = zeros(cfg.num_instances,1);

    partial_spread = zeros(cfg.num_instances,1);
    full_spread    = zeros(cfg.num_instances,1);

    partial_policy_spread = zeros(cfg.num_instances,1);
    full_policy_spread    = zeros(cfg.num_instances,1);

    meta = cell(cfg.num_instances,1);

    inst = 0;
    for S = cfg.S_list
        for A = cfg.A_list
            for O = cfg.O_partial_list
                if O > S
                    continue;
                end
                for rep = 1:num_instances_per_combo(cfg, S, A, O)
                %for rep = cfg.num_instances_per_combo
                    inst = inst + 1;

                    seed_here = 1000 + 37*inst + 11*S + 7*A + 5*O;
                    pomdp_partial = random_pomdp(S, O, A, cfg.gamma_range, cfg.reward_scale, seed_here, cfg.stationary_bonus);
                    pomdp_full    = convert_to_fully_observable_baseline(pomdp_partial);

                    if cfg.verbose
                        fprintf('Instance %d/%d: S=%d, O_partial=%d, A=%d, gamma=%.3f\n', ...
                            inst, cfg.num_instances, S, O, A, pomdp_partial.gamma);
                    end

                    vals_partial = run_many_restarts(pomdp_partial, cfg.num_restarts, cfg.max_iters, cfg.step_size, cfg.grad_clip);
                    vals_full    = run_many_restarts(pomdp_full,    cfg.num_restarts, cfg.max_iters, cfg.step_size, cfg.grad_clip);

                    partial_by_instance(inst,:) = vals_partial.final_J(:)';
                    full_by_instance(inst,:)    = vals_full.final_J(:)';

                    partial_best = max(vals_partial.final_J);
                    full_best    = max(vals_full.final_J);

                    %eps_tol = 1e-3;
                    partial_gap_by_instance(inst,:) = max(0, (partial_best - vals_partial.final_J(:)') ./ max(1e-12, abs(partial_best)));
                    full_gap_by_instance(inst,:)    = max(0, (full_best    - vals_full.final_J(:)')    ./ max(1e-12, abs(full_best)));

                    % partial_subfrac(inst) = mean(partial_gap_by_instance(inst,:) > eps_tol);
                    % full_subfrac(inst)    = mean(full_gap_by_instance(inst,:)    > eps_tol);

                    tau = cfg.subopt_threshold;
                    partial_subfrac(inst) = mean(partial_gap_by_instance(inst,:) > tau);
                    full_subfrac(inst)    = mean(full_gap_by_instance(inst,:)    > tau);

                    partial_spread(inst) = max(vals_partial.final_J) - min(vals_partial.final_J);
                    full_spread(inst)    = max(vals_full.final_J)    - min(vals_full.final_J);

                    % partial_policy_spread(inst) = policy_spread(vals_partial.final_pi);
                    % full_policy_spread(inst)    = policy_spread(vals_full.final_pi);
                    partial_policy_spread(inst) = policy_spread_mean(vals_partial.final_pi);
                    full_policy_spread(inst)    = policy_spread_mean(vals_full.final_pi);

                    meta{inst} = struct('S', S, 'O', O, 'A', A, 'gamma', pomdp_partial.gamma);
                end
            end
        end
    end

    batch_results = struct();
    batch_results.meta = meta;

    batch_results.partial.final_J_by_instance = partial_by_instance;
    batch_results.full.final_J_by_instance    = full_by_instance;

    batch_results.partial.final_J_all = partial_by_instance(:);
    batch_results.full.final_J_all    = full_by_instance(:);

    batch_results.partial.normalized_gap = partial_gap_by_instance(:);
    batch_results.full.normalized_gap    = full_gap_by_instance(:);

    batch_results.partial.suboptimal_fraction_by_instance = partial_subfrac;
    batch_results.full.suboptimal_fraction_by_instance    = full_subfrac;

    batch_results.partial.spread_by_instance = partial_spread;
    batch_results.full.spread_by_instance    = full_spread;

    batch_results.partial.policy_spread_by_instance = partial_policy_spread;
    batch_results.full.policy_spread_by_instance    = full_policy_spread;


end

function n = num_instances_per_combo(cfg, S, A, O)
    combos = 0;
    for s = cfg.S_list
        for a = cfg.A_list
            for o = cfg.O_partial_list
                if o <= s
                    combos = combos + 1;
                end
            end
        end
    end
    n = max(1, round(cfg.num_instances / combos));
    if cfg.num_instances < combos
        n = 1;
    end
    
end

% function vals = run_many_restarts(pomdp, num_restarts, max_iters, step_size, grad_clip)
% 
%     final_J = zeros(num_restarts, 1);
%     final_theta = cell(num_restarts, 1);
%     histories = cell(num_restarts, 1);
% 
%     for r = 1:num_restarts
%         theta0 = 0.25 * randn(pomdp.A, pomdp.O);
% 
%         [theta, hist] = policy_gradient_ascent(pomdp, theta0, max_iters, step_size, grad_clip);
%         [J, ~, V, pi] = objective_and_grad(pomdp, theta);
% 
%         final_J(r) = J;
%         final_theta{r} = theta;
%         histories{r} = struct('J', hist.J, 'grad_norm', hist.grad_norm, 'V', V, 'pi', pi);
%     end
% 
%     vals = struct();
%     vals.final_J = final_J;
%     vals.final_theta = final_theta;
%     vals.histories = histories;
% end

function vals = run_many_restarts(pomdp, num_restarts, max_iters, step_size, grad_clip)

    final_J = zeros(num_restarts, 1);
    final_theta = cell(num_restarts, 1);
    final_pi = cell(num_restarts, 1);
    histories = cell(num_restarts, 1);

    for r = 1:num_restarts
        theta0 = 0.25 * randn(pomdp.A, pomdp.O);

        [theta, hist] = policy_gradient_ascent(pomdp, theta0, max_iters, step_size, grad_clip);
        [J, ~, V, pi] = objective_and_grad(pomdp, theta);

        final_J(r) = J;
        final_theta{r} = theta;
        final_pi{r} = pi;
        histories{r} = struct('J', hist.J, 'grad_norm', hist.grad_norm, 'V', V, 'pi', pi);
    end

    vals = struct();
    vals.final_J = final_J;
    vals.final_theta = final_theta;
    vals.final_pi = final_pi;
    vals.histories = histories;
end

function [theta, hist] = policy_gradient_ascent(pomdp, theta0, max_iters, step_size, grad_clip)

    theta = theta0;
    hist.J = zeros(max_iters,1);
    hist.grad_norm = zeros(max_iters,1);

    for t = 1:max_iters
        [J, grad] = objective_and_grad(pomdp, theta);

        gnorm = norm(grad(:));
        if gnorm > grad_clip
            grad = grad * (grad_clip / gnorm);
            gnorm = grad_clip;
        end

        theta = theta + step_size * grad;

        hist.J(t) = J;
        hist.grad_norm(t) = gnorm;

        if gnorm < 1e-7
            hist.J = hist.J(1:t);
            hist.grad_norm = hist.grad_norm(1:t);
            break;
        end
    end
end

function [J, grad, V, pi] = objective_and_grad(pomdp, theta)
% Compute J(theta) = mu' * V^pi and its exact gradient w.r.t. theta(a,o).
%
% Policy:
%   pi(:,o) = softmax(theta(:,o))
%
% Value function:
%   V = r_pi + gamma * P_pi * V
%     = (I - gamma P_pi)^{-1} r_pi

    pi = softmax_columns(theta);

    S = pomdp.S;
    O = pomdp.O;
    A = pomdp.A;
    T = pomdp.T;   % S x S x A
    Z = pomdp.Z;   % O x S
    R = pomdp.R;   % S x A
    mu = pomdp.mu; % S x 1
    gamma = pomdp.gamma;

    [Ppi, rpi] = induced_markov_chain_and_reward(T, Z, R, pi);

    M = eye(S) - gamma * Ppi;
    V = M \ rpi;
    J = mu' * V;

    % Adjoint variable: w = (I - gamma Ppi)^(-T) mu
    w = M' \ mu;

    grad = zeros(A, O);

    for o = 1:O
        pio = pi(:,o);  % A x 1
        for a0 = 1:A

            dr = zeros(S,1);
            dP = zeros(S,S);

            for s = 1:S
                z = Z(o, s);
                if z == 0
                    continue;
                end

                % dq(a|s)/d theta(a0,o)
                dq = z * pio .* ((1:A)' == a0) - z * pio * pio(a0);
                % Equivalent to z * d softmax / d theta_a0

                % reward derivative at state s
                dr(s) = sum(dq .* R(s,:)');

                % transition derivative for row s
                row = zeros(1,S);
                for a = 1:A
                    row = row + dq(a) * squeeze(T(s,:,a));
                end
                dP(s,:) = row;
            end

            grad(a0,o) = w' * (dr + gamma * dP * V);
        end
    end

end

function [Ppi, rpi] = induced_markov_chain_and_reward(T, Z, R, pi)
% T  : S x S x A
% Z  : O x S
% R  : S x A
% pi : A x O

    [S, ~, A] = size(T);
    O = size(Z,1);

    Ppi = zeros(S,S);
    rpi = zeros(S,1);

    for s = 1:S
        q = zeros(A,1); % q(a|s)
        for o = 1:O
            q = q + Z(o,s) * pi(:,o);
        end

        for a = 1:A
            Ppi(s,:) = Ppi(s,:) + q(a) * squeeze(T(s,:,a));
            rpi(s)   = rpi(s)   + q(a) * R(s,a);
        end
    end
end

function pomdp = random_pomdp(S, O, A, gamma_range, reward_scale, seed_here, stationary_bonus)

    rng(seed_here);

    T = zeros(S,S,A);

% default creation of transitions 
% for a = 1:A
%     for s = 1:S
%         % alpha = 0.5 + 3*rand(1,S);
%         alpha = 0.1 + rand(1,S);
%         alpha = alpha + (a==1)*5*((1:S) == s); 
%         alpha(s) = alpha(s) + stationary_bonus * S;
%         T(s,:,a) = dirichlet_sample(alpha);
%     end
% end

% make transitions more action dependent     
% good = 1:floor(S/2);
% bad  = floor(S/2)+1:S;
% 
% T = zeros(S,S,A);
% 
% for a = 1:A
%     for s = 1:S
%         alpha = 0.2 + rand(1,S);
% 
%         % self-transition bias
%         alpha(s) = alpha(s) + stationary_bonus * S;
% 
%         % action-dependent basin bias
%         if a == 1
%             alpha(good) = alpha(good) + 4.0;
%         elseif a == 2
%             alpha(bad) = alpha(bad) + 4.0;
%         else
%             % for A > 2, assign each action to a rotating preferred subset
%             idx = mod((1:S) + (a-1) - 1, S) + 1;
%             alpha(idx(1:ceil(S/2))) = alpha(idx(1:ceil(S/2))) + 3.0;
%         end
% 
%         T(s,:,a) = dirichlet_sample(alpha);
%     end
% end

% create hidden basins / traps 
good = 1:floor(S/2);
bad  = floor(S/2)+1:S;

T = zeros(S,S,A);

for a = 1:A
    for s = 1:S
        alpha = 0.05 + rand(1,S);

        if ismember(s, good)
            % baseline persistence in the good basin
            alpha(good) = alpha(good) + 5.0;
        else
            % baseline persistence in the bad basin
            alpha(bad) = alpha(bad) + 5.0;
        end

        % action-dependent pushes between basins
        if a == 1
            alpha(good) = alpha(good) + 4.0;
        elseif a == 2
            alpha(bad) = alpha(bad) + 4.0;
        end

        % extra self-loop
        alpha(s) = alpha(s) + stationary_bonus * S;

        T(s,:,a) = dirichlet_sample(alpha);
    end
end



    % Z = zeros(O,S);
    % for s = 1:S
    %     alpha = 0.35 + 2.4*rand(1,O);
    %     Z(:,s) = dirichlet_sample(alpha)';
    % end

    % make observations ambiguous in a structured way 
    if S == 4 && O == 2
    Z = [0.85 0.15 0.85 0.15;
         0.15 0.85 0.15 0.85];
    else
    Z = zeros(O,S);
    for s = 1:S
        alpha = 0.05 + rand(1,O);
        if mod(s,2) == 1
            alpha(1) = alpha(1) + 5.0;
        else
            alpha(min(O,2)) = alpha(min(O,2)) + 5.0;
        end
        Z(:,s) = dirichlet_sample(alpha)';
    end
    end



%    R = reward_scale * (2*rand(S,A) - 1);

% make rewards structured 
R = zeros(S,A);

for s = 1:S
    if ismember(s, good)
        R(s,1) =  3 + 0.5*randn();   % action 1 good in good basin
        R(s,2) =  0 + 0.5*randn();
    else
        R(s,1) = -2 + 0.5*randn();
        R(s,2) =  2 + 0.5*randn();   % action 2 good in bad basin
    end
end

if A > 2
    R(:,3:end) = 0.3 * randn(S, A-2);
end


    mu = dirichlet_sample(0.7 + 2*rand(1,S))';
    gamma = gamma_range(1) + (gamma_range(2)-gamma_range(1)) * rand();

    pomdp = struct();
    pomdp.S = S;
    pomdp.O = O;
    pomdp.A = A;
    pomdp.T = T;
    pomdp.Z = Z;
    pomdp.R = R;
    pomdp.mu = mu;
    pomdp.gamma = gamma;
end

function mdp = convert_to_fully_observable_baseline(pomdp_partial)
% Use the same states/actions/transitions/rewards/discount/initial
% distribution, but replace the observation kernel by identity:
% each state emits its own unique observation.

    S = pomdp_partial.S;

    mdp = pomdp_partial;
    mdp.O = S;
    mdp.Z = eye(S);
end

function results = run_landscape_experiment(cfg)

    rng(cfg.seed);
    pomdp = random_pomdp(cfg.S, cfg.O, cfg.A, [cfg.gamma cfg.gamma], 1.0, cfg.seed, 0.2);

    p_grid = linspace(1e-3, 1-1e-3, cfg.grid_n);
    q_grid = linspace(1e-3, 1-1e-3, cfg.grid_n);
    J_grid = zeros(cfg.grid_n, cfg.grid_n);

    for i = 1:cfg.grid_n
        p = p_grid(i);
        for j = 1:cfg.grid_n
            q = q_grid(j);
            pi = [p, q; 1-p, 1-q];
            theta = log(max(pi, 1e-12)); % softmax invariant up to columnwise shifts
            [J, ~] = objective_and_grad(pomdp, theta);
            J_grid(j,i) = J;
        end
    end

    trajs = cell(cfg.num_restarts,1);
    final_J = zeros(cfg.num_restarts,1);
    final_p = zeros(cfg.num_restarts,1);
    final_q = zeros(cfg.num_restarts,1);

    for r = 1:cfg.num_restarts
        theta0 = 0.5 * randn(cfg.A, cfg.O);

        [theta, hist] = policy_gradient_ascent(pomdp, theta0, cfg.max_iters, cfg.step_size, cfg.grad_clip);
        [J, ~, ~, pi] = objective_and_grad(pomdp, theta);

        [p_path, q_path] = theta_history_to_policy_path(pomdp, theta0, cfg.max_iters, cfg.step_size, cfg.grad_clip);

        trajs{r} = struct('p', p_path, 'q', q_path, 'histJ', hist.J);
        final_J(r) = J;
        final_p(r) = pi(1,1);
        final_q(r) = pi(1,2);
    end

    results = struct();
    results.pomdp = pomdp;
    results.p_grid = p_grid;
    results.q_grid = q_grid;
    results.J_grid = J_grid;
    results.trajs = trajs;
    results.final_J = final_J;
    results.final_p = final_p;
    results.final_q = final_q;

end

function [p_path, q_path] = theta_history_to_policy_path(pomdp, theta0, max_iters, step_size, grad_clip)
% Re-run gradient ascent while recording the 2D policy coordinates.

    theta = theta0;
    p_path = zeros(max_iters+1,1);
    q_path = zeros(max_iters+1,1);

    pi = softmax_columns(theta);
    p_path(1) = pi(1,1);
    q_path(1) = pi(1,2);

    actual_len = max_iters + 1;

    for t = 1:max_iters
        [~, grad] = objective_and_grad(pomdp, theta);

        gnorm = norm(grad(:));
        if gnorm > grad_clip
            grad = grad * (grad_clip / gnorm);
            gnorm = grad_clip;
        end

        theta = theta + step_size * grad;
        pi = softmax_columns(theta);
        p_path(t+1) = pi(1,1);
        q_path(t+1) = pi(1,2);

        if gnorm < 1e-7
            actual_len = t + 1;
            break;
        end
    end

    p_path = p_path(1:actual_len);
    q_path = q_path(1:actual_len);
end

function P = softmax_columns(X)
    X = X - max(X, [], 1);
    E = exp(X);
    P = E ./ sum(E, 1);
end

function x = dirichlet_sample(alpha)
    y = gamrnd(alpha, 1);
    x = y / sum(y);
end


function d = policy_spread(pi_list)
    R = numel(pi_list);
    d = 0;

    for i = 1:R
        p1 = pi_list{i}(:);
        for j = i+1:R
            p2 = pi_list{j}(:);
            d = max(d, norm(p1 - p2) / sqrt(numel(p1)));
        end
    end
end

function d = policy_spread_mean(pi_list)
    R = numel(pi_list);
    acc = 0;
    count = 0;

    for i = 1:R
        p1 = pi_list{i}(:);
        for j = i+1:R
            p2 = pi_list{j}(:);
            acc = acc + norm(p1 - p2) / sqrt(numel(p1));
            count = count + 1;
        end
    end

    d = acc / max(count,1);
end


function summary = summarize_by_configuration(batch_results)
% Aggregate per-instance metrics by unique (S,A,O) configuration.

    n = numel(batch_results.meta);

    S_all = zeros(n,1);
    A_all = zeros(n,1);
    O_all = zeros(n,1);

    for i = 1:n
        S_all(i) = batch_results.meta{i}.S;
        A_all(i) = batch_results.meta{i}.A;
        O_all(i) = batch_results.meta{i}.O;
    end

    configs = unique([S_all, A_all, O_all], 'rows');

    m = size(configs,1);

    S = zeros(m,1);
    A = zeros(m,1);
    O = zeros(m,1);

    partial_value_spread_mean = zeros(m,1);
    partial_value_spread_std  = zeros(m,1);
    partial_subopt_mean       = zeros(m,1);
    partial_subopt_std        = zeros(m,1);
    partial_policy_spread_mean = zeros(m,1);
    partial_policy_spread_std  = zeros(m,1);

    full_value_spread_mean = zeros(m,1);
    full_value_spread_std  = zeros(m,1);
    full_subopt_mean       = zeros(m,1);
    full_subopt_std        = zeros(m,1);
    full_policy_spread_mean = zeros(m,1);
    full_policy_spread_std  = zeros(m,1);

    for k = 1:m
        S(k) = configs(k,1);
        A(k) = configs(k,2);
        O(k) = configs(k,3);

        idx = (S_all == S(k)) & (A_all == A(k)) & (O_all == O(k));

        pv = batch_results.partial.spread_by_instance(idx);
        ps = batch_results.partial.suboptimal_fraction_by_instance(idx);
        pp = batch_results.partial.policy_spread_by_instance(idx);

        fv = batch_results.full.spread_by_instance(idx);
        fs = batch_results.full.suboptimal_fraction_by_instance(idx);
        fp = batch_results.full.policy_spread_by_instance(idx);

        partial_value_spread_mean(k)  = mean(pv);
        partial_value_spread_std(k)   = std(pv);
        partial_subopt_mean(k)        = mean(ps);
        partial_subopt_std(k)         = std(ps);
        partial_policy_spread_mean(k) = mean(pp);
        partial_policy_spread_std(k)  = std(pp);

        full_value_spread_mean(k)  = mean(fv);
        full_value_spread_std(k)   = std(fv);
        full_subopt_mean(k)        = mean(fs);
        full_subopt_std(k)         = std(fs);
        full_policy_spread_mean(k) = mean(fp);
        full_policy_spread_std(k)  = std(fp);
    end

    summary = table( ...
        S, A, O, ...
        partial_value_spread_mean, partial_value_spread_std, ...
        partial_subopt_mean, partial_subopt_std, ...
        partial_policy_spread_mean, partial_policy_spread_std, ...
        full_value_spread_mean, full_value_spread_std, ...
        full_subopt_mean, full_subopt_std, ...
        full_policy_spread_mean, full_policy_spread_std);
end


%%


