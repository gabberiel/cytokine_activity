function [indspos, indsneg, thresh] = so_cfar(y, w, g, windowmode, ...
                                              plotflag, nstd, plotinds, t)
    % Apply an adaptive threshold.
    % Compute a rolling std in each window, and choose the side that has
    % the smallest value. w are the windows, g are the guard regions, and
    % CUT is the cell under test.
    %
    % |----w-----|--g--|CUT|--g--|----w-----|
    %
    % y           - neural signal
    % w           - width of the CFAR window on either side in samples
    % g           - width of the guard region on either side in samples
    % windowmode  - 'rollingstd' (single window) or 'socfar'
    % plotflag    - display a plot
    % nstd        - number of standard deviations on each side of the mean
    %               to apply the threshold
    % plotinds    - indices to display in the plot
    % 
    % indspos     - indices that exceed the positive threshold
    % indsneg     - indices that exceed the negative threshold
                      
    % the window length must be odd
    assert(mod(w, 2) == 1, 'w must be odd');
    
    % ensure the signal is a row vector
    y = reshape(y, [length(y), 1]);
    
    if nargin < 4
        windowmode = 'rollingstd';
    end
    
    % default value for plotting
    if nargin < 5
        plotflag = false;
    end
    
    % default number of standard deviations from the mean
    if nargin < 6
        nstd = [5, 5];
    end
    if isscalar(nstd)
        nstd = repmat(nstd, [1, 2]);
    end
    
    % default number of samples to plot
    if nargin < 7
        plotinds = 1:1e6;
    end
    
    % rolling mean and rolling std
    rollingstd = movstd(y, w);
    rollingmean = movmean(y, w);
    
    % remove outliers from movstd
    y_ = y;
    y_(abs((y - rollingmean) ./ rollingstd) > 4) = NaN;
    rollingstd = movstd(y_, w, 'omitnan');    
    
    if isequal(windowmode, 'socfar')
        % pair leading and lagging windows
        cfarwindowsstd = [rollingstd(1:end - (w + 2 * g)), ...
                          rollingstd(w + 2 * g + 1:end)];
        cfarwindowsmean = [rollingmean(1:end - (w + 2 * g)), ...
                           rollingmean(w + 2 * g + 1:end)];
        
        % choose the window with the smaller std
        [mv, mi] = min(cfarwindowsstd, [], 2);
        clear('cfarwindowsstd');

        % select corresponding mean
        mean2 = zeros(size(y));
        for x = 1:length(mi)
            mean2((w - 1) / 2 + g + 1 + x) = cfarwindowsmean(x, mi(x));
        end
        clear('cfarwindowsmean');
        
        % selected std with zero padding
        yout = [zeros((w - 1) / 2 + g + 1, 1); ...
                mv; ...
                zeros((w - 1) / 2 + g, 1)];
    else
        yout = rollingstd;
        mean2 = rollingmean;
    end
    clear('rollingstd');
    clear('rollingmean');
    
    % value of the cell under test
    % cut = y((w - 1) / 2 + g + 1:end - ((w - 1) / 2 + g + 1));
    
    % plot
    %{
    if plotflag        
        figure;
        reduce_plot(t(plotinds), [y(plotinds) - mean2(plotinds), ...
                      nstd(1) * yout(plotinds), ...
                      -nstd(2) * yout(plotinds)]);
        xlabel('Time (seconds)');
    end
    %}
    
    % find threshold crossings
    thresh = [nstd(1) * yout + mean2, -nstd(2) * yout + mean2];
    % thresh = 10 * (t <= 1474.6) + 20 * (t > 1474.6 & t <= 1610) + 15 * (t > 1610 & t <= 2000) + 12 * (t > 2000);
    % thresh = [thresh -thresh];
    indspos = find(y > thresh(:, 1));
    indsneg = find(y < thresh(:, 2));
    