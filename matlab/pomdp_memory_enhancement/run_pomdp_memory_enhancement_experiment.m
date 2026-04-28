function results = run_pomdp_memory_enhancement_experiment()
    clearvars -except results; close all; clc; rng(11);
    outdir = fullfile(fileparts(mfilename('fullpath')), 'output');
    if ~exist(outdir, 'dir'), mkdir(outdir); end

    batch_cfg = struct();
    batch_cfg.S_list = [4 8];
    batch_cfg.A_list = [2 3];
    batch_cfg.O_partial_list = [2];
    batch_cfg.num_instances_per_combo = 5;
    batch_cfg.num_restarts = 40;
    batch_cfg.max_iters = 350;
    batch_cfg.step_size = 0.05;
    batch_cfg.grad_clip = 10.0;
    batch_cfg.gamma_range = [0.95, 0.98];
    batch_cfg.reward_scale = 1.0;
    batch_cfg.stationary_bonus = 0.15;
    batch_cfg.verbose = true;
    batch_cfg.memory_list = [0 1 2];
    batch_cfg.subopt_threshold = 0.01;

    batch_results = run_batch_experiment_memory(batch_cfg);
    summary_table = summarize_memory_experiment_by_configuration(batch_results);
    writetable(summary_table, fullfile(outdir, 'summary_table.csv'));
    export_summary_table_markdown_memory(summary_table, fullfile(outdir, 'summary_table.md'));

    fig = figure('Color', 'w', 'Position', [100 100 1400 900]);
    tiledlayout(2,2, 'Padding', 'compact', 'TileSpacing', 'compact');
    regimes = batch_results.regimes; nr = numel(regimes);

    nexttile;
    m = zeros(1,nr); s = zeros(1,nr);
    for r = 1:nr
        vals = batch_results.metrics.(regimes{r}).value_spread_by_instance;
        m(r) = mean(vals); s(r) = std(vals);
    end
    errorbar(1:nr, m, s, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
    xticks(1:nr); xticklabels(regimes); xtickangle(30);
    ylabel('Value spread'); title('Value spread across restarts by regime'); grid on;

    nexttile;
    m = zeros(1,nr); s = zeros(1,nr);
    for r = 1:nr
        vals = batch_results.metrics.(regimes{r}).suboptimal_fraction_by_instance;
        m(r) = mean(vals); s(r) = std(vals);
    end
    errorbar(1:nr, m, s, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
    xticks(1:nr); xticklabels(regimes); xtickangle(30);
    ylabel(sprintf('Fraction with gap > %.3f', batch_cfg.subopt_threshold));
    title('Materially suboptimal restart fraction by regime'); grid on;

    nexttile;
    m = zeros(1,nr); s = zeros(1,nr);
    for r = 1:nr
        vals = batch_results.metrics.(regimes{r}).policy_spread_by_instance;
        m(r) = mean(vals); s(r) = std(vals);
    end
    errorbar(1:nr, m, s, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
    xticks(1:nr); xticklabels(regimes); xtickangle(30);
    ylabel('Policy spread'); title('Policy spread across restarts by regime'); grid on;

    nexttile; hold on; data = []; group = [];
    for r = 1:nr
        vals = batch_results.metrics.(regimes{r}).final_J_all(:);
        data = [data; vals];
        group = [group; r*ones(numel(vals),1)];
    end
    boxchart(group, data);
    xticks(1:nr); xticklabels(regimes); xtickangle(30);
    ylabel('Final objective J'); title('Distribution of converged objective values'); grid on;

    exportgraphics(fig, fullfile(outdir, 'memory_enhancement_summary.png'), 'Resolution', 220);

    results = struct();
    results.batch = batch_results;
    results.summary_table = summary_table;
    results.output_dir = outdir;
    save(fullfile(outdir, 'results.mat'), 'results');

    fprintf('\nSaved outputs to:\n  %s\n', outdir);
end

function batch_results = run_batch_experiment_memory(cfg)
    valid_configs = count_valid_configs(cfg);
    num_instances = valid_configs * cfg.num_instances_per_combo;
    partial_regimes = cell(1, numel(cfg.memory_list));
    for i = 1:numel(cfg.memory_list), partial_regimes{i} = regime_name_partial(cfg.memory_list(i)); end
    regimes = [partial_regimes, {'full'}];
    nr = numel(regimes);

    metrics = struct();
    for r = 1:nr
        reg = regimes{r};
        metrics.(reg).final_J_by_instance = zeros(num_instances, cfg.num_restarts);
        metrics.(reg).normalized_gap_by_instance = zeros(num_instances, cfg.num_restarts);
        metrics.(reg).value_spread_by_instance = zeros(num_instances,1);
        metrics.(reg).suboptimal_fraction_by_instance = zeros(num_instances,1);
        metrics.(reg).policy_spread_by_instance = zeros(num_instances,1);
    end
    meta = cell(num_instances,1);

    inst = 0;
    for S = cfg.S_list
        for A = cfg.A_list
            for O = cfg.O_partial_list
                if O > S, continue; end
                for rep = 1:cfg.num_instances_per_combo
                    inst = inst + 1;
                    seed_here = 5000 + 97*inst + 11*S + 7*A + 5*O + rep;
                    pomdp_partial = random_pomdp(S, O, A, cfg.gamma_range, cfg.reward_scale, seed_here, cfg.stationary_bonus);
                    pomdp_full = convert_to_fully_observable_baseline(pomdp_partial);
                    if cfg.verbose
                        fprintf('Instance %d/%d: S=%d, O=%d, A=%d, gamma=%.3f\n', inst, num_instances, S, O, A, pomdp_partial.gamma);
                    end

                    for kk = 1:numel(cfg.memory_list)
                        k = cfg.memory_list(kk); reg = regime_name_partial(k);
                        model = build_observation_memory_model(pomdp_partial, k);
                        vals = run_many_restarts_generic(model, cfg.num_restarts, cfg.max_iters*(1+k), cfg.step_size, cfg.grad_clip);
                        metrics.(reg).final_J_by_instance(inst,:) = vals.final_J(:)';
                        bestJ = max(vals.final_J);
                        gaps = max(0, (bestJ - vals.final_J(:)') ./ max(1e-12, abs(bestJ)));
                        metrics.(reg).normalized_gap_by_instance(inst,:) = gaps;
                        metrics.(reg).value_spread_by_instance(inst) = max(vals.final_J) - min(vals.final_J);
                        metrics.(reg).suboptimal_fraction_by_instance(inst) = mean(gaps > cfg.subopt_threshold);
                        metrics.(reg).policy_spread_by_instance(inst) = policy_spread_mean(vals.final_pi);
                    end

                    reg = 'full';
                    model_full = build_observation_memory_model(pomdp_full, 0);
                    vals_full = run_many_restarts_generic(model_full, cfg.num_restarts, cfg.max_iters, cfg.step_size, cfg.grad_clip);
                    metrics.(reg).final_J_by_instance(inst,:) = vals_full.final_J(:)';
                    bestJ = max(vals_full.final_J);
                    gaps = max(0, (bestJ - vals_full.final_J(:)') ./ max(1e-12, abs(bestJ)));
                    metrics.(reg).normalized_gap_by_instance(inst,:) = gaps;
                    metrics.(reg).value_spread_by_instance(inst) = max(vals_full.final_J) - min(vals_full.final_J);
                    metrics.(reg).suboptimal_fraction_by_instance(inst) = mean(gaps > cfg.subopt_threshold);
                    metrics.(reg).policy_spread_by_instance(inst) = policy_spread_mean(vals_full.final_pi);

                    meta{inst} = struct('S', S, 'A', A, 'O', O, 'gamma', pomdp_partial.gamma);
                end
            end
        end
    end

    for r = 1:nr
        reg = regimes{r};
        metrics.(reg).final_J_all = metrics.(reg).final_J_by_instance(:);
        metrics.(reg).normalized_gap_all = metrics.(reg).normalized_gap_by_instance(:);
    end
    batch_results = struct('metrics', metrics, 'regimes', {regimes}, 'meta', {meta}, 'config', cfg);
end

function vals = run_many_restarts_generic(model, num_restarts, max_iters, step_size, grad_clip)
    final_J = zeros(num_restarts,1); final_theta = cell(num_restarts,1); final_pi = cell(num_restarts,1); histories = cell(num_restarts,1);
    for r = 1:num_restarts
        theta0 = 0.25 * randn(model.A, model.Y);
        [theta, hist] = policy_gradient_ascent_generic(model, theta0, max_iters, step_size, grad_clip);
        [J, ~, V, pi] = objective_and_grad_generic(model, theta);
        final_J(r) = J; final_theta{r} = theta; final_pi{r} = pi;
        histories{r} = struct('J', hist.J, 'grad_norm', hist.grad_norm, 'V', V, 'pi', pi);
    end
    vals = struct('final_J', final_J, 'final_theta', {final_theta}, 'final_pi', {final_pi}, 'histories', {histories});
end

function [theta, hist] = policy_gradient_ascent_generic(model, theta0, max_iters, step_size, grad_clip)
    theta = theta0; hist.J = zeros(max_iters,1); hist.grad_norm = zeros(max_iters,1);
    for t = 1:max_iters
        [J, grad] = objective_and_grad_generic(model, theta);
        gnorm = norm(grad(:));
        if gnorm > grad_clip, grad = grad * (grad_clip / gnorm); gnorm = grad_clip; end
        theta = theta + step_size * grad;
        hist.J(t) = J; hist.grad_norm(t) = gnorm;
        if gnorm < 1e-7, hist.J = hist.J(1:t); hist.grad_norm = hist.grad_norm(1:t); break; end
    end
end

function [J, grad, V, pi] = objective_and_grad_generic(model, theta)
    pi = softmax_columns(theta);
    X = model.X; A = model.A; gamma = model.gamma;
    Ppi = zeros(X, X); rpi = zeros(X,1);

    for x = 1:X
        s = model.state_s_of_x(x); h = model.mem_of_x(x);
        for o = 1:model.O
            z = model.Z(o, s); y = model.y_of_oh(o, h); piy = pi(:, y);
            rpi(x) = rpi(x) + z * sum(piy .* model.R(s,:)');
            next_h = model.next_mem_of_oh(o, h); cols = model.x_of_s_and_mem(:, next_h);
            row_add = zeros(1, X);
            for a = 1:A, row_add(cols) = row_add(cols) + piy(a) * squeeze(model.T(s,:,a)); end
            Ppi(x,:) = Ppi(x,:) + z * row_add;
        end
    end

    M = eye(X) - gamma * Ppi; V = M \ rpi; J = model.mu_aug' * V; w = M' \ model.mu_aug;
    grad = zeros(A, model.Y);

    for y0 = 1:model.Y
        h0 = model.mem_of_y(y0); o0 = model.obs_of_y(y0);
        piy = pi(:, y0);
        for a0 = 1:A
            dr = zeros(X,1); dP = zeros(X,X);
            e = zeros(A,1); e(a0) = 1; dpi = piy .* (e - piy(a0));
            x_list = model.x_list_by_mem{h0};
            for idx = 1:numel(x_list)
                x = x_list(idx); s = model.state_s_of_x(x); z = model.Z(o0, s);
                if z == 0, continue; end
                dr(x) = dr(x) + z * sum(dpi .* model.R(s,:)');
                next_h = model.next_mem_of_oh(o0, h0); cols = model.x_of_s_and_mem(:, next_h);
                row_add = zeros(1, X);
                for a = 1:A, row_add(cols) = row_add(cols) + dpi(a) * squeeze(model.T(s,:,a)); end
                dP(x,:) = dP(x,:) + z * row_add;
            end
            grad(a0, y0) = w' * (dr + gamma * dP * V);
        end
    end
end

function model = build_observation_memory_model(pomdp, k)
    S = pomdp.S; O = pomdp.O; A = pomdp.A;
    if k == 0
        H = 1; mem_table = zeros(1,0); y_table = (1:O)'; obs_of_y = (1:O)'; mem_of_y = ones(O,1);
        y_of_oh = @(o,h) o; next_mem_of_oh = @(o,h) 1;
    else
        H = O^k; mem_table = all_tuples(O, k);
        y_table = zeros(O*H, k+1); obs_of_y = zeros(O*H,1); mem_of_y = zeros(O*H,1); idx = 0;
        for h = 1:H
            for o = 1:O
                idx = idx + 1; y_table(idx,:) = [o, mem_table(h,:)]; obs_of_y(idx) = o; mem_of_y(idx) = h;
            end
        end
        y_of_oh = @(o,h) o + (h-1)*O;
        next_mem_lookup = zeros(O, H);
        for h = 1:H
            for o = 1:O
                new_mem = [o, mem_table(h,1:k-1)];
                next_mem_lookup(o,h) = tuple_to_index(new_mem, O);
            end
        end
        next_mem_of_oh = @(o,h) next_mem_lookup(o,h);
    end

    X = S * H; x_of_s_and_mem = zeros(S, H); state_s_of_x = zeros(X,1); mem_of_x = zeros(X,1); x_list_by_mem = cell(H,1);
    idx = 0;
    for h = 1:H
        x_list_by_mem{h} = zeros(S,1);
        for s = 1:S
            idx = idx + 1; x_of_s_and_mem(s,h) = idx; state_s_of_x(idx) = s; mem_of_x(idx) = h; x_list_by_mem{h}(s) = idx;
        end
    end

    mu_aug = zeros(X,1);
    if H == 1
        for s = 1:S, mu_aug(x_of_s_and_mem(s,1)) = pomdp.mu(s); end
    else
        p_obs = pomdp.Z * pomdp.mu; p_obs = p_obs / sum(p_obs);
        mem_prob = zeros(H,1);
        for h = 1:H
            prob = 1;
            for j = 1:k, prob = prob * p_obs(mem_table(h,j)); end
            mem_prob(h) = prob;
        end
        mem_prob = mem_prob / sum(mem_prob);
        for h = 1:H
            for s = 1:S, mu_aug(x_of_s_and_mem(s,h)) = pomdp.mu(s) * mem_prob(h); end
        end
    end

    model = struct('S', S, 'O', O, 'A', A, 'k', k, 'T', pomdp.T, 'Z', pomdp.Z, 'R', pomdp.R, 'gamma', pomdp.gamma, ...
                   'H', H, 'X', X, 'Y', O*H, 'mem_table', mem_table, 'y_table', y_table, ...
                   'obs_of_y', obs_of_y, 'mem_of_y', mem_of_y, 'x_of_s_and_mem', x_of_s_and_mem, ...
                   'state_s_of_x', state_s_of_x, 'mem_of_x', mem_of_x, 'x_list_by_mem', {x_list_by_mem}, ...
                   'y_of_oh', y_of_oh, 'next_mem_of_oh', next_mem_of_oh, 'mu_aug', mu_aug);
end

function summary = summarize_memory_experiment_by_configuration(batch_results)
    n = numel(batch_results.meta); S_all = zeros(n,1); A_all = zeros(n,1); O_all = zeros(n,1);
    for i = 1:n, S_all(i) = batch_results.meta{i}.S; A_all(i) = batch_results.meta{i}.A; O_all(i) = batch_results.meta{i}.O; end
    configs = unique([S_all, A_all, O_all], 'rows'); m = size(configs,1);

    vars = {'S', 'A', 'O'}; data = cell(1, 3 + 6*numel(batch_results.regimes)); data{1} = configs(:,1); data{2} = configs(:,2); data{3} = configs(:,3); col = 4;
    for r = 1:numel(batch_results.regimes)
        reg = batch_results.regimes{r}; met = batch_results.metrics.(reg);
        value_mean = zeros(m,1); value_std = zeros(m,1); sub_mean = zeros(m,1); sub_std = zeros(m,1); pol_mean = zeros(m,1); pol_std = zeros(m,1);
        for k = 1:m
            idx = (S_all == configs(k,1)) & (A_all == configs(k,2)) & (O_all == configs(k,3));
            vv = met.value_spread_by_instance(idx); ss = met.suboptimal_fraction_by_instance(idx); pp = met.policy_spread_by_instance(idx);
            value_mean(k) = mean(vv); value_std(k) = std(vv); sub_mean(k) = mean(ss); sub_std(k) = std(ss); pol_mean(k) = mean(pp); pol_std(k) = std(pp);
        end
        reg_clean = matlab.lang.makeValidName(reg);
        vars = [vars, {sprintf('%s_value_spread_mean', reg_clean), sprintf('%s_value_spread_std', reg_clean), sprintf('%s_subopt_mean', reg_clean), sprintf('%s_subopt_std', reg_clean), sprintf('%s_policy_spread_mean', reg_clean), sprintf('%s_policy_spread_std', reg_clean)}];
        data{col} = value_mean; col = col + 1; data{col} = value_std; col = col + 1; data{col} = sub_mean; col = col + 1; data{col} = sub_std; col = col + 1; data{col} = pol_mean; col = col + 1; data{col} = pol_std; col = col + 1;
    end
    summary = table(data{:}, 'VariableNames', vars);
end

function export_summary_table_markdown_memory(summary_table, filename)
    fid = fopen(filename, 'w'); if fid == -1, error('Could not open file: %s', filename); end
    cleanup = onCleanup(@() fclose(fid));
    regs = {};
    for i = 1:numel(summary_table.Properties.VariableNames)
        name = summary_table.Properties.VariableNames{i};
        tok = regexp(name, '^(.*)_value_spread_mean$', 'tokens', 'once');
        if ~isempty(tok), regs{end+1} = tok{1}; end
    end

    fprintf(fid, '| Config (S,A,O) |');
    for r = 1:numel(regs), reg = regs{r}; fprintf(fid, ' Value Spread (%s) | Subopt. Fraction (%s) | Policy Spread (%s) |', reg, reg, reg); end
    fprintf(fid, '\n|---|'); for r = 1:numel(regs), fprintf(fid, '---:|---:|---:|'); end; fprintf(fid, '\n');

    for i = 1:height(summary_table)
        fprintf(fid, '| (%d,%d,%d) |', summary_table.S(i), summary_table.A(i), summary_table.O(i));
        for r = 1:numel(regs)
            reg = regs{r};
            fprintf(fid, ' %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f |', ...
                summary_table.(sprintf('%s_value_spread_mean', reg))(i), summary_table.(sprintf('%s_value_spread_std', reg))(i), ...
                summary_table.(sprintf('%s_subopt_mean', reg))(i), summary_table.(sprintf('%s_subopt_std', reg))(i), ...
                summary_table.(sprintf('%s_policy_spread_mean', reg))(i), summary_table.(sprintf('%s_policy_spread_std', reg))(i));
        end
        fprintf(fid, '\n');
    end
end

function name = regime_name_partial(k), name = sprintf('partial_k%d', k); end

function n = count_valid_configs(cfg)
    n = 0;
    for S = cfg.S_list, for A = cfg.A_list, for O = cfg.O_partial_list, if O <= S, n = n + 1; end, end, end, end
end

function d = policy_spread_mean(pi_list)
    R = numel(pi_list); acc = 0; count = 0;
    for i = 1:R
        p1 = pi_list{i}(:);
        for j = i+1:R
            p2 = pi_list{j}(:);
            acc = acc + norm(p1 - p2) / sqrt(numel(p1)); count = count + 1;
        end
    end
    d = acc / max(count,1);
end

function pomdp = random_pomdp(S, O, A, gamma_range, reward_scale, seed_here, stationary_bonus)
    rng(seed_here); 
    T = zeros(S,S,A);
    
    % for a = 1:A
    %     for s = 1:S
    %         alpha = 0.5 + 3*rand(1,S); alpha(s) = alpha(s) + stationary_bonus * S; T(s,:,a) = dirichlet_sample(alpha);
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
    % for s = 1:S, 
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


    R = reward_scale * (2*rand(S,A) - 1); mu = dirichlet_sample(0.7 + 2*rand(1,S))'; gamma = gamma_range(1) + (gamma_range(2)-gamma_range(1)) * rand();
    pomdp = struct('S', S, 'O', O, 'A', A, 'T', T, 'Z', Z, 'R', R, 'mu', mu, 'gamma', gamma);
end

function mdp = convert_to_fully_observable_baseline(pomdp_partial), S = pomdp_partial.S; mdp = pomdp_partial; mdp.O = S; mdp.Z = eye(S); end
function P = softmax_columns(X), X = X - max(X, [], 1); E = exp(X); P = E ./ sum(E, 1); end
function x = dirichlet_sample(alpha), y = gamrnd(alpha, 1); x = y / sum(y); end
function T = all_tuples(base, len)
    n = base^len; T = zeros(n, len);
    for idx = 0:n-1
        v = zeros(1, len); z = idx;
        for j = len:-1:1, v(j) = mod(z, base) + 1; z = floor(z / base); end
        T(idx+1,:) = v;
    end
end
function idx = tuple_to_index(v, base)
    idx0 = 0;
    for j = 1:numel(v), idx0 = idx0 * base + (v(j) - 1); end
    idx = idx0 + 1;
end
