function [clust_timestamps, yf, tf, real_clusters, hf1, hf2] = main_cap_sort_2(X, idx, timestamps, maxtime, t_local, plotflag1, plotflag2)
% IN:
%   X: Waveforms (#waveforms, dim_of_each_wf)
%   idx: "labels" -- cluster indices 
%   timestamps: obj.timestamps
%   maxtime: obj.timestamps(end)
% OUT:
%   clust_timestamps: cell-obj containing timestamps elements in-
%                     different clusters
%   yf: Stores eventrates for each cluster. it containes 
%   tf: Stores time for each bin edge
%
%         for k2=1:num_clusters(k)
%             err(k2) = mean(sqrt(sum(X(idx==k2,:)-(ones(size(X(idx==k2,:),1),1)*mean(X(idx==k2,:)))).^2));
%         end
% Average of the  point-to-centroid distances for Neural 1
% err2= mean(sumd); 

if ~exist('plotflag1', 'var') || isempty(plotflag1)
    plotflag1 = true;
end
if ~exist('plotflag2', 'var') || isempty(plotflag2)
    plotflag2 = true;
end

% handles to figures
hf1 = [];
hf2 = [];

% treat no labels as a single group
if ~any(idx)
    idx = idx + 1;
end

Twin = 1;
bin_edges = 0:Twin:ceil(maxtime);
tf = bin_edges(1:end - 1); % stores time for each bin-edge
real_clusters = [];
num_clusters = max(idx);
clust_timestamps = cell(1, num_clusters);
isi_val = cell(1, num_clusters);
yf = [];
median_waveforms = [];
for k2 = 1:num_clusters


%             figure;
%             plot(t_local{k}, X(idx==k2,:)', colors(k2))

    %xlabel('Time (sec)')

    %event_rate = histc(peak_timestamps{k}(idx==k2), bin_edges) / Twin;
    % ############# Gabriel-comment: ###################
    % Each waveform has a corresponding cluster index in idx. 
    % This could also be "No cluster"
    % event_rate = histcounts(timestamps(idx == k2), bin_edges) / Twin;

    % ------------------------------------------------------
    
    event_rate = histcounts(timestamps(idx == k2), bin_edges) / Twin;

    if mean(event_rate) > 0.1
        real_clusters = [real_clusters, k2];  %#ok
    end
        isi_val{k2} = diff(timestamps(idx == k2)); % Derivative-ish
        %counts number of occurances for isi
        histc(isi_val{k2}, 50);  % TODO Why not histcounts? And returns?
        %filters out isi values greater than .1 to remove noise/outliers
        indices = ((isi_val{k2})>.1);
        %removes value we do not want
        isi_val{k2}(indices) = 0;       
        % build feature array
        yf = [yf  event_rate'];  %#ok 
        %builds array of waveforms for each cluster
        median_waveforms = [median_waveforms median(X(idx==k2,:), 1)'];  %#ok
        %builds array of timestamps for each cluster
        clust_timestamps{k2} = timestamps(idx == k2);
end

color_values = [0 0.45 0.74 ; 0.85 0.33 0.1 ; 0.93 0.69 0.13 ; 0.49 0.18 0.56 ; 0.47 0.67 0.19; 0.3 0.75 0.93; 0.64 0.08 0.18];
color_values = repmat(color_values, [5, 1]);
if size(median_waveforms, 2) > 8
    color_values = rand(size(median_waveforms,2),3);
end
if plotflag1    
    % color_values = hsv(10);color_values = color_values([2:end, 1], :);

    hf1 = figure;hold on;
    for ii = 1:size(median_waveforms, 2)
        plot(t_local, median_waveforms(:, ii), 'LineWidth', 3, 'Color', color_values(ii, :));
    end
    hold off;
    xlabel('Time (sec)')
    ylabel('Voltage (mV)')
end
% err1=0;

%figure below plots histogram of our clusters as well as it's waveforms
real_clusters = 1:num_clusters;
if plotflag2
    hf2 = figure();
    hold on
    % real_clusters_old = real_clusters;
    % real_clusters = 1:num_clusters;
    maxK = length(real_clusters);
    for i=1:maxK
        index = mod(real_clusters(i) - 1, size(color_values, 1)) + 1;
        subplot(maxK, 4, (i - 1) * 4 + 1:(i - 1) * 4 + 3);
        hst = histogram(nonzeros(isi_val{i}), 0:0.001:.05);
    %     if nnz(hst.Values) < 0.5 * length(hst.Values)        
    %         real_clusters_old = setdiff(real_clusters_old, real_clusters(i));
    %         real_clusters(i) = 0;
    %     end
        h = findobj(gca, 'Type', 'Histogram');
    %     if ismember(real_clusters(i), real_clusters_old)
            mycol = color_values(index,:);
    %     else
    %         mycol = [0.8, 0.8, 0.8];
    %     end
        h.FaceColor = mycol;
        xlim([-0.002, 0.022]);
        ylabel('CAP Counts');
        if i == maxK
            xlabel('Time (s)');
        end
        subplot(maxK, 4, i * 4);
        plot(t_local*1000, median_waveforms(:,i), ...
             'Color', mycol, 'LineWidth', 3);
         if i == maxK
            xlabel('Time (ms)');
        end
    end
    % real_clusters = real_clusters_old;
    % real_clusters = nonzeros(real_clusters);
    boldify(hf2);
    hold off
end