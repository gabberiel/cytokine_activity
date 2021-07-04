classdef Waveforms < handle
    properties
        X  % N x K array of waveforms where N is the number of waveforms and K is the number of samples per waveform
        Y  % N x D array of dimensionality reduced waveforms wher N is the number of waveforms and D is the lower dimensionality
        timestamps
        sr
        clust_timestamps
        labels     % formerly called T, cluster labels -- gg: i.e which cluster each waveform is assosiated to..?
        labels2    % placeholder for temporary labels
        respthresh
        traininds  % t-SNE and DBSCAN are both O(N^2) in memory so we need a training set
        k
        
        epsilon
        minpts
        
        tf
        yf
        real_clusters
        powspreadthresh
    end
    
    properties (Dependent)
        % gg: Only "pointers" to existing data..?
        t_local
        testinds
    end
    
    methods
        function obj = Waveforms(timestamps, X, sr)
            obj.timestamps = timestamps;
            obj.X = X;
            obj.sr = sr;
            obj.labels = zeros(size(X, 1), 1);
        end
        
        function value = get.t_local(obj)
            value = (0:size(obj.X, 2)-1) / obj.sr;
        end
        
        function value = get.testinds(obj)
            value = setdiff(1:size(obj.X, 1), obj.traininds);            
        end
        
        function plot_amps(obj)
            figure;ax = gca;
            plot(ax, obj.timestamps, max(abs(obj.X), [], 2), '.');
        end
        
        %-----------------------------------------------------------------
        %------------------AMPLITUDE THRES. -----------------------------------
        %-----------------------------------------------------------------
        function amp_thresh(obj, ampthresh)
            % TODO would it be better to store keepinds, or to remove the
            % indices here?
            traininds_ = zeros(size(obj.X, 1), 1, 'logical');  % before modifying obj.X
            
            inds = max(abs(obj.X), [], 2) > ampthresh;
            obj.timestamps = obj.timestamps(inds);
            obj.X = obj.X(inds, :);
            
            if ~isempty(obj.Y)
                obj.Y = obj.Y(inds, :);
            end
            if ~isempty(obj.labels)
                obj.labels = obj.labels(inds);
            end
            if ~isempty(obj.traininds)
                traininds_(obj.traininds) = true;
                obj.traininds = reshape(find(traininds_(inds)), 1, []);
            end
            if ~isempty(obj.clust_timestamps)
                obj.event_rates();
            end            
        end
        %-----------------------------------------------------------------
        %------------------ separate_resp_amps -----------------------------------
        %-----------------------------------------------------------------
        
        function [threshi] = separate_resp_amps(obj, T, plotflag)
            if ~exist('plotflag', 'var') || isempty(plotflag)
                plotflag = false;
            end
            
            amps = max(abs(obj.X), [], 2);
            bands = zeros(size(amps), 'logical');
            bimodal = @(x, a, mu1, sig1, mu2, sig2) a * normpdf(x, mu1, sig1) + (1 - a) * normpdf(x, mu2, sig2);
            ct = 1;            
            tvect = 0:T:max(obj.timestamps) - T;
            thresh = NaN(1, length(tvect));
            for t = tvect
                inds = (obj.timestamps > t & obj.timestamps <= t + T);
                x = amps(inds);
                medx = median(x);
                x1 = x(x < medx);
                x2 = x(x >= medx);
                mu1 = mean(x1);
                mu2 = mean(x2);
                sig1 = std(x1);
                sig2 = std(x2);
                options = statset;
                options.MaxIter = 5e4;
                [lambdaHat,lambdaCI] = mle(x, 'pdf', bimodal, 'start', [0.5, mu1, sig1, mu2, sig2], ...
                    'LowerBound', zeros(1, 5), 'upperBound', [1 repmat(max(amps), [1 4])], 'options', options);
                lambdaHatc = mat2cell(lambdaHat, 1, ones(1, 5));
                if abs(lambdaHat(4) - lambdaHat(2)) > max(lambdaHat(3), lambdaHat(5)) && abs(lambdaHat(1) - 0.5) < 0.45
                    thresh(ct) = fminsearch(@(x) bimodal(x, lambdaHatc{:}), mean([lambdaHat(4) lambdaHat(2)]));
                    if thresh(ct) < min([lambdaHat(4) lambdaHat(2)]) || thresh(ct) > max([lambdaHat(4) lambdaHat(2)])
                        thresh(ct) = NaN;                        
                    end
                    % bands(inds) = x > thresh;
                end
                ct = ct + 1;
            end
            
            threshi = interp1(tvect(~isnan(thresh)), thresh(~isnan(thresh)), obj.timestamps, 'cubic', 'extrap');
            obj.respthresh = threshi;
            
            if plotflag
                figure;plot(obj.timestamps, amps, '.');
                hold on;plot(obj.timestamps, threshi);hold off;
            end
        end
        %-----------------------------------------------------------------
        %------------------DIMENSIONALITY REDUCTION -----------------------------------
        %-----------------------------------------------------------------
        function [] = dimensionality_reduction(obj, method, varargin)
            if ~exist('method', 'var') || isempty(method)
                method = 'tsne';
            end
            if ~exist('compute_mapping', 'file')
                fprintf('addpath(genpath(''C:\\Users\\NDDA1\\Documents\\Todd code\\drtoolbox''))\n');
                addpath(genpath('C:\Users\NDDA1\Documents\Todd code\drtoolbox'));
            end
            
            switch method
                case 'tsne'
                    % assign defaults                    
                    perpl = 30;                                        
                    no_dims = 2;  % couls also be a Ytr result
                    nTrain = min(10e3, size(obj.X, 1));
                    obj.traininds = round(linspace(1, size(obj.X, 1), nTrain));
                    Xtr = obj.X(obj.traininds, :);
                    mycols = 10*log10(sum(Xtr.^2, 2));
                    maxiter = 200;
                    videofile = 'temp.avi';
                    
                    % modify defaults if specified
                    for iparams = 1:2:length(varargin)
                        name = varargin{iparams};
                        val = varargin{iparams+1};
                        switch name                            
                            case 'perplexity'
                                perpl = val;
                            case 'dims'
                                no_dims = val;
                            case 'videofile'
                                videofile = val;
                            case 'maxiter'
                                maxiter = val;
                            case 'color'
                                mycols = val;
                            case 'traininginds'
                                obj.traininds = val;
                            otherwise
                                fprintf('Invalid Name/Value pair %s\n', name);
                        end
                    end
                    
                    % check that parameters are consistent
                    Xtr = obj.X(obj.traininds, :);
                    assert(length(mycols) == size(Xtr, 1), ...
                        'color labels must be the same size as the number of training waveforms');                    
                    
                    % perform t-SNE
                    %[Ytr, ~] = compute_mapping(Xtr, 'tSNE', no_dims, size(Xtr, 2), perpl, mycols, maxiter, videofile);
                    %[Ytr, ~] = tsne(Xtr, 'NumDimensions', no_dims, 'Perplexity', perpl, 'MaxIter', maxiter)
                    %[Ytr, ~] = tsne(Xtr, 'NumDimensions', no_dims, 'Perplexity', perpl);
                    % qqq:---- Gabe modifiet to one output arg and commented out inputs...----- 
                    %Ytr = tsne(Xtr) %, 'NumDimensions', no_dims, 'Perplexity', perpl);
                    Ytr = tsne(Xtr)%, [], no_dims, initial_dims, perplexity)
                    %---------------------------------------------------------------
                    obj.Y = zeros(size(obj.X, 1), no_dims);
                    obj.Y(obj.traininds, :) = Ytr;
                otherwise
                    fprintf('%s not supported\n', method);
            end
        end
        %-----------------------------------------------------------------
        %------------------ kt-SNE -----------------------------------
        %-----------------------------------------------------------------

        function ktsne(obj, k)
            % perform kernel t-SNE
            obj.k = k;
            [~, Ytest, ~, ~] = ktsne(obj.X, obj.traininds, [], false, k, obj.Y(obj.traininds, :), []);                    
            obj.Y(obj.testinds, :) = Ytest;
        end
        %-----------------------------------------------------------------
        %------------------  DBSCAN -----------------------------------
        %-----------------------------------------------------------------
        function dbscan(obj, epsilon, minpts)
            % DBSCAN (use GUI if epsilon and minpts are omitted)
            if (~exist('epsilon', 'var') || isempty(epsilon)) && ...
                    (~exist('minpts', 'var') || isempty(minpts))                
                mypow = 10*log10(sum(obj.X(obj.traininds, :).^2, 2));
                dbscangui(obj.Y(obj.traininds, :), mypow, obj.timestamps(obj.traininds));
            else
                Ytr = obj.Y(obj.traininds, :);                
                Ttr = DBSCAN(Ytr, epsilon, minpts);
                
                obj.epsilon = epsilon;
                obj.minpts = minpts;
                obj.labels = -ones(size(obj.X, 1), 1);
                obj.labels(obj.traininds) = Ttr; 
                
                testinds_ = sort([obj.testinds, obj.traininds(obj.labels(obj.traininds) == 0)]);
                Ytest = obj.Y(testinds_, :);
                
                idx = knnsearch(Ytr(Ttr > 0, :), Ytest, 'K', 600);
                Ttr2 = Ttr(Ttr > 0);            
                Ttest = mode(Ttr2(idx), 2);
                obj.labels(testinds_) = Ttest; 
            end
        end
        %-----------------------------------------------------------------
        %------------------ -----------------------------------
        %-----------------------------------------------------------------
        
        function merge_clusters(obj, inds, plotflag1, plotflag2)
            if ~exist('plotflag1', 'var') || isempty(plotflag1)
                plotflag1 = true;
            end
            if ~exist('plotflag2', 'var') || isempty(plotflag2)
                plotflag2 = true;
            end
            % general purpose cluster merging
%             if ~isempty(obj.cluster_timestamps)
%                 obj.cluster_timestamps{inds(1)} = cell2mat(obj.cluster_timestamps(inds)')';
%                 obj.cluster_timestamps(inds(2:end)) = [];
%             end
            if ~isempty(obj.labels)
                obj.labels(ismember(obj.labels, inds(2:end))) = inds(1);
            end
%             if ~isempty(obj.yf)
%                 obj.yf(:, inds(1)) = sum(obj.yf(:, inds), 2);
%             end
%             if ~isempty(obj.real_clusters)
%                 obj.real_clusters(ismember(obj.real_clusters, inds(2:end))) = [];
%             end
            obj.event_rates(plotflag1, plotflag2);  % sets clust_timestamps, yf, tf, and real_clusters based on labels and timestamps
        end
        %-----------------------------------------------------------------
        %------------------ -----------------------------------
        %-----------------------------------------------------------------
        
        function split_clusters_2(obj, inds)
            % split clusters was the original method that split based on
            % amplitude but I want a general purpose splitter that will
            % word based on indices
            obj.labels(inds) = length(obj.clust_timestamps) + 1;
            ulabels = unique(obj.labels);
            for ilabel = 1:length(ulabels)
                assert(ulabels(ilabel) >= ilabel);
                if ulabels(ilabel) > ilabel
                    obj.labels(obj.labels == ulabels(ilabel)) = ilabel;
                end
            end
%             emptyclusters = [];
%             for cind = 1:length(obj.clust_timestamps)                
%                 inds2 = ~ismember(obj.clust_timestamps{cind}, obj.timestamps(inds));
%                 obj.clust_timestamps{cind} = obj.clust_timestamps{cind}(inds2);
%                 if isempty(obj.clust_timestamps{cind})
%                     emptyclusters = [emptyclusters; cind];
%                 end
%             end
%             obj.clust_timestamps = [obj.clust_timestamps {obj.timestamps(inds)}];
%             prevclusters = unique(obj.labels(inds));
%             obj.labels(inds) = length(obj.clust_timestamps);
%             obj.clust_timestamps(emptyclusters) = [];
%             for iec = 1:length(emptyclusters)
%                 assert(~any(obj.labels == emptyclusters(iec)), 'should not be labels for empty cluster');
%                 obj.labels(obj.labels > emptyclusters(iec)) = obj.labels(obj.labels > emptyclusters(iec)) - 1;
%             end            
%             if ~isempty(intersect(prevclusters, obj.real_clusters))
%                 obj.real_clusters = [obj.real_clusters; length(obj.clust_timestamps)];
%             end
            obj.event_rates()  % sets clust_timestamps, yf, tf, and real_clusters based on labels and timestamps
        end
        %-----------------------------------------------------------------
        %------------------ -----------------------------------
        %-----------------------------------------------------------------
        function [h] = manual_clustering(obj)
            % manual clustering GUI
            h = manual_clustering(obj.Y(obj.traininds, 1), obj.Y(obj.traininds, 2), obj.timestamps(obj.traininds), obj.X(obj.traininds, :));
            % TODO create a function to assign the results
        end
        %-----------------------------------------------------------------
        %------------------ -----------------------------------
        %-----------------------------------------------------------------
        
        function [] = split_custers(obj, powspreadthresh)
            % Apply amplitude filtering (split clusters by amplitude)            
            if ~exist('powspreadthresh', 'var') || isempty(powspreadthresh)
                powspreadthresh = 15;
            end
            obj.powspreadthresh = powspreadthresh;
            ampthresh = 0.015;  % shouldn't have any effect if amp_thresh is called first
            mypow = 10*log10(sum(obj.X.^2, 2));
            maxabs = max(abs(obj.X), [], 2);
            keepinds = true(size(obj.X, 1), 1);
            keepinds(maxabs < ampthresh) = 0;
            uT = reshape(unique(obj.labels), 1, []);

            for it = uT
                inds = find(obj.labels == it & keepinds);
                powspread = quantile(mypow(inds), 0.95) - quantile(mypow(inds), 0.05);
                disp(powspread);
                if powspread > powspreadthresh
                    thresh = quantile(mypow(inds), 0.85);
                    inds2 = inds(mypow(inds) > thresh);
                    obj.labels(inds2) = max(obj.labels) + 1;
                end
            end            
            obj.labels(~keepinds) = [];
            obj.X(~keepinds, :) = [];
            obj.timestamps(~keepinds) = [];
            obj.Y(~keepinds, :) = [];            
        end
        %-----------------------------------------------------------------
        %------------------  -----------------------------------
        %-----------------------------------------------------------------
        
        function h = scatter(obj, colormode, inds, cmap)
            % PLOT t-SNE result. i.e low dimensional representation of
            % data...
            if ~exist('inds', 'var') || isempty(inds)
                inds = obj.traininds;
            end
            if ~exist('cmap', 'var') || isempty(cmap)
                cmap = 'parula';
            end
            
            h = figure;
            ax = gca;
            
            switch colormode
                case 'power'
                    mycol = 10*log10(sum(obj.X.^2, 2));
                    title(ax, 't-SNE Power (dB)');
                case 'label'
                    mycol = obj.labels;
                    title(ax, 'CAP Clusters');
                otherwise
                    fprintf('unsupported colormode %s\n', colormode);
            end
            markersize = 5;
            
            scatter3(ax, obj.Y(inds, 1), obj.Y(inds, 2), obj.timestamps(inds)/60, markersize, mycol(inds), 'filled');
            colormap(cmap);
            colorbar(ax);
            xlabel(ax, 't-SNE Dimension 1');
            ylabel(ax, 't-SNE Dimension 2');
            title(ax, 't-SNE Clusters');
        end
        %-----------------------------------------------------------------
        %---------------- PLOT-ALL-WAVEFOORMS -----------------------------------
        %-----------------------------------------------------------------
        function plot_all_waveforms(obj, N, real_clusters)
            if ~exist('N', 'var') || isempty(N)
                N = 100;
            end
            if ~exist('real_clusters', 'var') || isempty(real_clusters)
                real_clusters = unique(obj.labels);
            end
            
            hf = figure;
            ax = gca;
            clust_colors = get(ax, 'ColorOrder');
            hold on;
            ct = 1;
            for ii = 1:max(obj.labels)  % :-1:1
                % ii2 = mod(ii-1, size(clust_colors, 1))+1;
                if ismember(ii, real_clusters)
                    % mycol = clust_colors(ii2, :);
                    mycol = clust_colors(ct, :);
                    ct = ct + 1;
                else
                    continue
                    mycol = [0.8, 0.8, 0.8];
                end
                [h, ~, v] = rgb2hsv(mycol);
                [r, g, b] = hsv2rgb([h, 0.25, v]);
                inds = find(obj.labels==ii & isfinite(obj.Y(:, 1)));
                if isempty(inds)
                    continue
                end
                disp('plot nr 1... ')
                plot(obj.t_local * 1000, obj.X(inds(1:N:end), :)', 'Color', [r, g, b]);
            end
            ct = 1;
            for ii = 1:max(obj.labels)
                % ii2 = mod(ii-1, size(clust_colors, 1))+1;
                if ismember(ii, real_clusters)
                    % mycol = clust_colors(ii2, :);
                    mycol = clust_colors(ct, :);
                    ct = ct + 1;
                else
                    continue
                    mycol = [0.8, 0.8, 0.8];
                end
                disp('plot nr 2...')
                plot(obj.t_local * 1000, median(obj.X(obj.labels==ii, :), 1), 'Color', mycol, 'LineWidth', 2);
            end
            xlabel('Time (ms)');
            ylabel('Voltage (\muV)');
            title('Waveforms by Cluster');
            boldify(hf);
        end
        %-----------------------------------------------------------------
        %------------------ EVENT-RATE -----------------------------------
        %-----------------------------------------------------------------
        function [hf1, hf2] = event_rates(obj, plotflag1, plotflag2)
            % obj.yf stores event-rates
            % obj.tf stores time for each bin-edge
            if ~exist('plotflag1', 'var') || isempty(plotflag1)
                plotflag1 = true;
            end
            if ~exist('plotflag2', 'var') || isempty(plotflag2)
                plotflag2 = true;
            end
            %[obj.clust_timestamps, obj.yf, obj.tf, obj.real_clusters, hf1, hf2] = main_cap_sort_2(obj.X, ...
            %    obj.labels,[0:(60*60*1.5)/136259:60*60*1.5], obj.timestamps(end), obj.t_local, plotflag1, plotflag2);            
            
            % ORIGINALLY:
            [obj.clust_timestamps, obj.yf, obj.tf, obj.real_clusters, hf1, hf2] = main_cap_sort_2(obj.X, ...
                obj.labels, obj.timestamps, obj.timestamps(end), obj.t_local, plotflag1, plotflag2);            
        end
        %-----------------------------------------------------------------
        %-----------------------------------------------------------------
        function X = fit_model(obj)
            x = linspace(-3, 3, size(obj.X, 2));
            yhat = @(X) (X(1)*normpdf(x, X(2), X(3)) - X(4)*normpdf(x, X(5), X(6)));
            % yhat = @(X) (X(1)*poisspdf(x, X(2)) - X(3)*poisspdf(x, X(4)));
            X0 = [250;0;1;250;0;0.5];
            % X0 = [250;10;250;5];
            options = optimset;
            options.MaxFunEvals = 1e5;
            options.MaxIter = 1e5;
            
            X = zeros(size(obj.X, 1), 6);
            for ii = 1:size(obj.X, 1)
                if max(abs(obj.X(ii, :))) < 20
                    continue
                end
                y = obj.X(ii, :);
                myfun = @(X) norm(y - yhat(X), 2);
                X(ii, :) = fminsearch(myfun, X0, options);                
                figure(4);plot(x, y, '-b', x, yhat(X(ii, :)), '-r');title(myfun(X(ii, :)));
                if mod(ii, 10) == 0
                    fprintf('%d\n', ii);
                end
                
            end
        end
        %-----------------------------------------------------------------
        %------------------ PLOT EVENT-RATE -----------------------------------
        %-----------------------------------------------------------------
        
        function h = plot_event_rates(obj, real_clusters_, N)
            if ~exist('real_clusters_', 'var') || isempty(real_clusters_)
                real_clusters_ = obj.real_clusters;
            end
            if ~exist('N', 'var') || isempty(N)
                N = 1;
            end
                
            h = figure;
            ax = gca;
            defaultcolors = get(ax, 'ColorOrder');
            hold(ax, 'on');
            for ii = 1:size(obj.yf, 2)                
                if ismember(ii, real_clusters_)
                    %mycol = defaultcolors(mod(ii - 1, size(defaultcolors, 1)) + 1, :);
                    mycol = rand(1, 3);
                else
                    mycol = [0.8, 0.8, 0.8];
                end
                %OBS: movmean(X,N) = "moving average" -- smoothing kernel of
                % size N
                plot(ax, obj.tf / 60, movmean(obj.yf(:, ii), N), 'Color', mycol);
            end
            hold(ax, 'off');
            xlabel('Time (min)');
            ylabel('Event Rate (CAPs / second)');
            title('Event Rate');
            legend(arrayfun(@(x) sprintf('CAP Cluster %d', x), 1:size(obj.yf, 2), 'UniformOutput', false));
        end
    end
end