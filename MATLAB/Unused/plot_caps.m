function [h] = plot_caps(t, y1, y2, clust_timestamps, plotinds, durl, durr)
    % plot the detected CAPs superimposed on the signal
    % see go_shape2
    
    plotmode = 2;
    
    if nargin < 5
        plotinds = true(1, length(y1));
    end
    
    if ~exist('durl', 'var') || isempty(durl)
        durl = 0.002;
    end
    
    if ~exist('durr', 'var') || isempty(durr)
        durr = 0.002;
    end
    
    clust_colors = [0, 0.45, 0.74; ...
                    0.85, 0.33, 0.1; ...
                    0.93, 0.69, 0.13; ...
                    0.49, 0.18, 0.56; ...
                    0.47, 0.67, 0.19; ...
                    0.3, 0.75, 0.93; ...
                    0.64, 0.08, 0.18];
    % clust_colors = hsv(length(clust_timestamps)+1);clust_colors = clust_colors([2:end, 1], :);

    sr = 1 / (t(2) - t(1));
    % plots the entire signal 
    h = figure('Visible', 'off');
    hcleanup = onCleanup(@() set(h, 'Visible', 'on'));  % Make the figure visible if the code crashes too
    if ~isempty(y2)
        reduce_plot(t(plotinds), y1(plotinds), 'k', t(plotinds), y2(plotinds), 'y');
    else
%        reduce_plot(t(plotinds), y1(plotinds), 'k');
    end
    hold on;    
    ideal_clusters = length(clust_timestamps);
    for ii = 1:ideal_clusters
        index = mod(ii - 1, size(clust_colors, 1)) + 1;
        % creates sig_to_plot with same dimensions as y
        if plotmode == 1
            sig_to_plot = nan(1, length(y1));
        elseif plotmode == 2
            sig_to_plot = [];
            t_to_plot = [];
        end
        % takes timestamps of each cluster and transforms y data to 
        % highlight each event
        for jj = 1:length(clust_timestamps{ii})
            ind1 = max(1, round((clust_timestamps{ii}(jj)-t(1)) * sr + 1 - durl * sr));                        
            ind2 = min(length(y1), round((clust_timestamps{ii}(jj)-t(1)) * sr + 1 + durr * sr));
            if plotmode == 1
                sig_to_plot(ind1:ind2) = y1(ind1:ind2);
            elseif plotmode == 2
                sig_to_plot = [sig_to_plot; y1(ind1:ind2, 1); NaN];
                t_to_plot = [t_to_plot; t(ind1:ind2); NaN];
            end
        end
        if plotmode == 2
            sig_to_plot = sig_to_plot(1:end-1);
            t_to_plot = t_to_plot(1:end-1);
        end
        
        if plotmode == 1
            reduce_plot(t(plotinds), sig_to_plot(plotinds), 'Color', clust_colors(index, :));
        elseif plotmode == 2
            plot(t_to_plot, sig_to_plot, 'Color', clust_colors(index, :));
        end
        xlabel('Time (seconds)');
    end
    axis tight;axis normal;
    delete(hcleanup);