classdef RespiratoryBurst < handle
    
    properties        
        nwind
        minpeakwidth
        maxpeakwidth
        minpeakprom
        minpeakdist
        k
        t
        y
        sr
        
        pks
        locs
        w
        p        
        nonrespinds
        snr_
        movstd
        mu
    end
    
    methods
        function obj = RespiratoryBurst(y, nwind, minpeakdist, minpeakprom, minpeakwidth, maxpeakwidth, k)
            if ~exist('nwind', 'var') || isempty(nwind)
                nwind = 0.1;           % number of seconds in the moving window
            end
            if ~exist('minpeakwidth', 'var') || isempty(minpeakwidth)
                minpeakwidth = 0.105;  % minpeakwidth should be slightly larger than nwind to avoid detecting
            end
            if ~exist('maxpeakwidth', 'var') || isempty(maxpeakwidth)
                maxpeakwidth = 1.2;    % avoid very wide peaks like that which would occur in batch 1 sid 7
            end
            if ~exist('minpeakprom', 'var') || isempty(minpeakprom)
                minpeakprom = 0.1;         % sensitivity / specificity trade-off
            end
            if ~exist('minpeakdist', 'var') || isempty(minpeakdist)
                minpeakdist = 0.35;     % based on respiratory rate
            end
            if ~exist('k', 'var') || isempty(k)
                k = 1;     % scaling of width
            end

            assert(isa(y, 'timeseries'))
            obj.y = y.Data(:, 1);
            obj.t = y.Time;
            obj.sr = 1 / (y.Time(2) - y.Time(1));
            
            obj.nwind = nwind;
            obj.minpeakdist = minpeakdist;
            obj.minpeakprom = minpeakprom;
            obj.minpeakwidth = minpeakwidth;
            obj.maxpeakwidth = maxpeakwidth;
            
            fprintf('std_peaks...\n');
            obj.std_peaks();
            fprintf('form_nonrespinds...\n');
            obj.form_nonrespinds(k);            
        end
        
        function std_peaks(obj)
            % filter out outliers
            movstd1 = movstd(obj.y, obj.nwind * obj.sr);  %#ok
            mu1 = movmean(obj.y, obj.nwind * obj.sr);
            inds = abs(obj.y - mu1) > 3 * movstd1;
            y_ = obj.y;
            y_(inds) = NaN;
            % compute moving standard deviation without outliers
            obj.movstd = movstd(y_, obj.nwind * obj.sr, 'omitnan');  %#ok
            obj.mu = movmean(y_, obj.nwind * obj.sr, 'omitnan');
            % find the respiratory peaks
            [obj.pks, obj.locs, obj.w, obj.p] = findpeaks(obj.movstd, 'MinPeakDistance', obj.minpeakdist * obj.sr, ...
                                                          'MinPeakProminence', obj.minpeakprom, ...
                                                          'MinPeakWidth', obj.minpeakwidth * obj.sr, ...
                                                          'MaxPeakWidth', obj.maxpeakwidth * obj.sr);
        end
        
        function form_nonrespinds(obj, k)
            obj.k = k;
            obj.nonrespinds = ones(size(obj.y), 'logical');
            y_ = NaN(size(obj.y));
            obj.snr_ = zeros(length(obj.locs), 1);
            for ii = 1:length(obj.locs)
                inds = (obj.locs(ii) - ceil(k*obj.w(ii)/2)):(obj.locs(ii) + ceil(k*obj.w(ii)/2));
                y_(inds) = obj.y(inds);
                
                temp2 = obj.y(inds);temp2 = temp2(abs(temp2) < 10*std(temp2));
                sigpow = var(temp2, 0);
                x0 = max(1, inds(end)-round(3*obj.sr));
                inds2 = x0 - 1 + find(obj.nonrespinds(x0:inds(end)), round(obj.sr), 'last');
                temp2 = obj.y(inds2);temp2 = temp2(abs(temp2) < 10*std(temp2));
                obj.snr_(ii) = sigpow / var(temp2, 0);
                obj.nonrespinds(inds) = false;
            end
        end
        
        function plot(obj)
            y_ = obj.y;
            y_(obj.nonrespinds) = NaN;
            if exist('reduce_plot', 'file')
                figure;reduce_plot(obj.t, [obj.y y_ obj.movstd]);
            else
                fprintf('It is recommended to download https://www.mathworks.com/matlabcentral/fileexchange/40790-plot--big-\n');
                figure;plot(obj.t, [obj.y y_ obj.movstd]);
            end
            hold on;plot(obj.t(obj.locs), obj.pks, 'g.');hold off;
        end
        
    end
    
end