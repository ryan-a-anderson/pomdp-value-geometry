function export_summary_table_markdown(summary_table, filename)
% Export summary_table to a markdown table.
%
% Example: 
%   export_summary_table_markdown(results.summary_table, 'summary_table.md')

    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open file: %s', filename);
    end

    cleanup = onCleanup(@() fclose(fid));

    % Header
    fprintf(fid, '| Config (S,A,O) | Value Spread (Partial) | Value Spread (Full) | Subopt. Fraction (Partial) | Subopt. Fraction (Full) | Policy Spread (Partial) | Policy Spread (Full) |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|\n');

    for i = 1:height(summary_table)
        config_str = sprintf('(%d,%d,%d)', ...
            summary_table.S(i), ...
            summary_table.A(i), ...
            summary_table.O(i));

        val_partial = pm_string( ...
            summary_table.partial_value_spread_mean(i), ...
            summary_table.partial_value_spread_std(i));

        val_full = pm_string( ...
            summary_table.full_value_spread_mean(i), ...
            summary_table.full_value_spread_std(i));

        sub_partial = pm_string( ...
            summary_table.partial_subopt_mean(i), ...
            summary_table.partial_subopt_std(i));

        sub_full = pm_string( ...
            summary_table.full_subopt_mean(i), ...
            summary_table.full_subopt_std(i));

        pol_partial = pm_string( ...
            summary_table.partial_policy_spread_mean(i), ...
            summary_table.partial_policy_spread_std(i));

        pol_full = pm_string( ...
            summary_table.full_policy_spread_mean(i), ...
            summary_table.full_policy_spread_std(i));

        fprintf(fid, '| %s | %s | %s | %s | %s | %s | %s |\n', ...
            config_str, val_partial, val_full, sub_partial, sub_full, pol_partial, pol_full);
    end

    fprintf('Markdown table written to %s\n', filename);
end

function s = pm_string(mu, sigma)
    s = sprintf('%.3f ± %.3f', mu, sigma);
end