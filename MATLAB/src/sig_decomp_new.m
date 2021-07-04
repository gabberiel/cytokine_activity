function [timestamps, waveforms, sig, return_cardiac, thresh, threshc] = ...
         sig_decomp_new(t, y, cardiac, thresh_type, num_std_dev, ...
                        useWavelets, wdur, gdur, alignmentmode)
% 
% T. Zanos (adapted from C. Bouton Version)
%
% Signal Decomposition Algorithm to identify CAP occurence based on Wavelet
% decomposition
% Decomposes nerve signal (extracting ECG and neural events such as compound action potentials/CAPs)
%
% Syntax:
% OLD??? [peak_timestamps, ywm, thresh, Npeak] = sig_decomp(t, y, thresh_type,num_std_dev);
% [timestamps, waveforms, sig, thresh] = sig_decomp_new(t, y, cardiac, thresh_type, 
%                                   num_std_dev,useWavelets, wdur, gdur, alignmentmode)
%
% OUT:
% timestamps : occurence times of CAPs (in sec)
% waveforms : waveforms of detected CAPs from neural-enhanced signal (cwt-coefs)
% sig : wavelet coefficients. (neurally enhanced signal)
% thresh : actual value of threshold applied to neurally-enhanced signal
% threshc : actual value of threshold applied to cardiac-enhanced signal
%
% IN:
% t : time vector of signal
% y : signal
% thresh_type : type of threshold (1=positive (default), 2=negative, 3=both)
% num_std_dev : number of standard deviations for the value of the
%               threshold (default = 3). If thresh_type = 3 (both negative and positive),
%               num_std_dev will correspond to the same threshold for both
%               signs if it has one value, or the two values will
%               correspond to the standard deviations for the negative and
%               positive respectively (for example [3,2])
% usewavelets -
% wdur        - SO-CFAR window length in seconds
% gdur        - SO-CFAR gaurd length in seconds

if ~exist('cardiac', 'var') || isempty(cardiac)
    cardiac = [];
end
if ~exist('thresh_type', 'var') || isempty(thresh_type)
    thresh_type = 4;
end
if ~exist('num_std_dev', 'var') || isempty(num_std_dev)
    num_std_dev = 5;
end
if ~exist('useWavelets', 'var') || isempty(useWavelets)
    useWavelets = false;
end
if ~exist('wdur', 'var') || isempty(wdur)
    wdur = 1501 / 8000;
end
if ~exist('wdur', 'var') || isempty(wdur)
    gdur = 10 / 8000;
end

Twave = 0.001;  % width of wavelet (in sec);  Neural 1 (neural event / compound action potential) 
Tcardiac = 0.005; % assumed cardiac duration. 
% NL = 1; % Number of loops for thresholding

if thresh_type == 1
    thresh_sign = 1; % triggering threshold can be positive or negative for ECG/neural event;-
                     % -triggering on wavelet transformed signal
elseif thresh_type == 2
    thresh_sign = -1;
else
    thresh_sign(1) = -1;
    thresh_sign(2) = +1;
    % NL = 2; %Two Thresholds
end

if length(num_std_dev)==1
    std_dev(1) = num_std_dev;
    std_dev(2) = num_std_dev;
else
    std_dev = num_std_dev;
end

% 
Tpeak = 1.75*(Twave);  % 3.5 ms waveforms
% Tpeak = 5*(Twave);       % 10 ms waveforms

dt = t(2) - t(1);  % in sec
fs = (1 / dt);  % Hz

Npeak = ceil(Tpeak/dt);

tic
cardiacmethod = 1;
% Wavelet coefficients (wavelet transformed signal):
if useWavelets
    % (Bouton) switched to db3 as slightly better than db4 for isolating 
    % waveforms of interest; 
    wavelet = 'db3';
    scales = Twave/dt;
    sig = cwt(y, scales, wavelet)'; % Stored  Wavelet coefficients -- (wavelet transformed signal) 

    % TODO I could use the cardiac channel, but I would have to scale it so it was on the same time axis
    if isempty(cardiac)
        if cardiacmethod == 1
            cardiac = cwt(y, Tcardiac/dt, wavelet)'; % Stored cardiac Wavelet coefficients
            return_cardiac = cardiac;
        elseif cardiacmethod == 2
            cardiac = y;  % The wavedec is performed in detect_cardiac
        end
    end
    disp('Wavelet Transform completed');
else
    disp('Original signal used, not wavelet...')
    sig = y;
end


toc


if thresh_type == 1 || thresh_type == 2
   % Threshold:

   std_noise = median(abs(sig)/0.6745); % Estimate of the standard deviation of the background noise (Donoho & Johnstone, 1994), less sensitive to high event rate periods
   % thresh = num_std_dev(1) * std_noise;
   assert(isequal(std_dev(1), num_std_dev(1)), ...
          'replaced num_std_dev(1) with std_dev(1)');
   thresh = std_dev(1) * std_noise;
   
   trig = thresh_sign*sig > thresh;   %  flip sign for triggering purposes and trigger
   trig = [0 diff(trig)']';  % 1st (+) and 2nd (-) edge of crossings
      
%    TODO index_first_edge = so_cfar(sig, 1501, 10, 'socfar', false, true, 5, N);
   index_first_edge = reshape(find(trig > 0), 1, []);
%    first_edge=(trig>0);
%    
%    index_sig=1:length(first_edge);
%    index_first_edge=index_sig(first_edge==1);
elseif thresh_type == 3
   trig1 = zeros(2, length(sig));
   for i=1:2       
       std_noise = median(abs(sig)/0.6745); % Estimate of the standard deviation of the background noise (Donoho & Johnstone, 1994), less sensitive to high event rate periods
       thresh = std_dev(i) * std_noise;
       trig1(i,:) = sig*thresh_sign(i) > thresh;   % flip sign for triggering purposes and trigger  
   end
   
   trig = trig1(1,:)|trig1(2,:);
   trig = [0 diff(trig)]';  % 1st (+) and 2nd (-) edge of crossings
   
   first_edge=(trig>0);
  
   index_sig=1:length(first_edge);
   index_first_edge=index_sig(first_edge==1);
elseif thresh_type == 4
    % make parameters a function of the sampling rate and spike duration
    w = round(wdur * fs);
    if mod(w, 2) == 0
        w = w + 1;
    end
    g = round(gdur * fs);
    if mod(g, 2) == 0
        g = g + 1;
    end    
    [indspos, indsneg, thresh] = so_cfar(sig, w, g, 'socfar', ...
                                         true, std_dev(1), uint32(1:length(t)), t);        % fs*250*1
    
    % combine threshold crossings
    w2 = ceil(0.001 / dt);  % number of samples in 1 ms
    index_first_edge_ = combine_threshold_crossings(indspos, indsneg, 'union', w2);
    % detect cardiac events
    if ~isempty(cardiac)
        if cardiacmethod == 1
            [indsposc, indsnegc, threshc] = so_cfar(cardiac, w, g, 'socfar', ...
                                                    true, std_dev(end), uint32(1:length(t)), t);        % fs*250*1

            index_first_edgec_ = combine_threshold_crossings(indsposc, indsnegc, 'union', w2);
            temp = zeros(size(y), 'logical');
            for ii = 1:length(index_first_edgec_)
                % Set all values +- ~6.25ms of the cardiac event-peak to
                % true. These are the inidicies that will be removed from
                % the found CAP-indicies.
                temp(max(1, index_first_edgec_(ii) - round(50 / 8000 * fs)): ...
                     min(index_first_edgec_(ii) + round(50 / 8000 * fs), length(y))) = true;
            end
            excludesamples = find(temp);
        elseif cardiacmethod == 2
            [~, excludesamples] = detect_cardiac(cardiac, fs);        
        end
        % excludesamples = [];
        % TODO raise threshold for cardiac use sig
        % When is there a cardiac channel present?
    else
        % excludesamples = find(detect_respiratory(1/16, fs, y, 4, 0.2));
        excludesamples = [];        
    end
    % gg: This returns the indicies where we had a neural-enhanced signal
    % and NOT a cardiac enhanced..
    index_first_edge = setdiff(index_first_edge_, excludesamples);
elseif thresh_type == 5
    % Was experimenting with removing respiratory
    % make parameters a function of the sampling rate and spike duration
    w = round(wdur * fs);
    if mod(w, 2) == 0
        w = w + 1;
    end
    g = round(gdur * fs);
    if mod(g, 2) == 0
        g = g + 1;
    end    
    [indspos, indsneg, thresh] = so_cfar(sig, w, g, 'socfar', ...
                                         true, std_dev, uint32(1:min(length(t), fs*250*1)), t);        
    
    % TODO the breathing rate changes over time. This is not a robust method for blanking out the respiratory modulation.
    [pks, locs, w, p] = findpeaks(thresh(:, 1), 'MinPeakDistance', round(fs * 0.5));    
    for ii = 1:length(locs)
        inds = max(1, min(size(thresh, 1), locs(ii) + (-round(w(ii) * 3):round(w(ii) * 3))));
        thresh(inds, 1) = 1000;
        thresh(inds, 2) = -1000;
    end    
    % hold on;plot((1:size(thresh, 1)) / fs, thresh(:, 1));hold off;
    indspos = find(sig > thresh(:, 1));
    indsneg = find(sig < thresh(:, 2));
    
    % combine threshold crossings
    w2 = ceil(0.001 / dt);  % number of samples in 1 ms
    index_first_edge_ = combine_threshold_crossings(indspos, indsneg, 'union', w2);
    % detect cardiac events
    if ~isempty(cardiac)        
        [~, excludesamples] = detect_cardiac(cardiac, fs);        
        % excludesamples = [];
        % TODO raise threshold for cardiac use sig
        % When is there a cardiac channel present?
    else
        % excludesamples = find(detect_respiratory(1/16, fs, y, 4, 0.2));
        excludesamples = [];        
    end
    index_first_edge = setdiff(index_first_edge_, excludesamples);
elseif thresh_type == 6
    % experimenting with constant threshold on perineal nerve
    b = fir1(100, 1/8);
    y1f = filtfilt(b, 1, sig);
    thresh = 0.5;
    [~, index_first_edge] = findpeaks(y1f .* (y1f > thresh), 'MinPeakDistance', round(fs/160));
elseif thresh_type == 7
    std_noise = median(abs(sig)/0.6745);
    thresh = std_noise * num_std_dev;
    threshside = 'both';
    switch threshside
        case 'pos'
            index_first_edge_ = find(sig(2:end) >= thresh & sig(1:end-1) < thresh) + 1;
            index_first_edge_c = find(cardiac(2:end) >= thresh & cardiac(1:end-1) < thresh) + 1;
        case 'neg'
            index_first_edge_ = find(sig(2:end) <= -thresh & sig(1:end-1) > -thresh) + 1;
            index_first_edge_c = find(cardiac(2:end) <= -thresh & cardiac(1:end-1) > -thresh) + 1;
        case 'both'
            index_first_edge_ = find(sig(2:end) >= thresh & sig(1:end-1) < thresh | ...
                sig(2:end) <= -thresh & sig(1:end-1) > -thresh) + 1;
            index_first_edge_c = find(cardiac(2:end) >= thresh & cardiac(1:end-1) < thresh | ...
                cardiac(2:end) <= -thresh & cardiac(1:end-1) > -thresh) + 1;
    end
    
    
    if ~isempty(cardiac)
        sr = 1 / (t(2) - t(1));
        nsamp2 = round(5 / 800 * sr);
        inds = max(0, mi - nsamp2):min(mi + nsamp2, length(y2));
        
        excludesamples = zeros(size(sig));
        excludesamples(index_first_edge_c) = 1;
        imdilate(excludesamples, [zeros(1, 2*nsamp2) ones(1, 2*nsamp2)])  % TODO
        % [~, excludesamples] = detect_cardiac(cardiac, fs);        
        % excludesamples = [];
        % TODO raise threshold for cardiac use sig
        % When is there a cardiac channel present?
    else
        % excludesamples = find(detect_respiratory(1/16, fs, y, 4, 0.2));
        excludesamples = [];        
    end
    index_first_edge = setdiff(index_first_edge_, excludesamples);
elseif thresh_type == 8
    % combine threshold crossings
    w2 = ceil(0.001 / dt);  % number of samples in 1 ms
    sr = 1 / (t(2) - t(1));
    mu = movmean(sig, round(sr*5));
    sigma_ = movstd(sig, round(sr*5));
    sig_2 = (sig - mu) ./ sigma_;
    indspos = find(sig_2 > 3);
    indsneg = find(sig_2 < -3);
    thresh = bsxfun(@plus, bsxfun(@times, sigma_, [3 -3]), mu);
    index_first_edge_ = combine_threshold_crossings(indspos, indsneg, 'union', w2);
    % detect cardiac events
    if ~isempty(cardiac)
        if cardiacmethod == 1
            w = round(wdur * fs);
            if mod(w, 2) == 0
                w = w + 1;
            end
            g = round(gdur * fs);
            if mod(g, 2) == 0
                g = g + 1;
            end
            
            [indsposc, indsnegc, threshc] = so_cfar(cardiac, w, g, 'socfar', ...
                                                    true, std_dev(end), uint32(1:length(t)), t);        % fs*250*1
            index_first_edgec_ = combine_threshold_crossings(indsposc, indsnegc, 'union', w2);
            temp = zeros(size(y), 'logical');
            for ii = 1:length(index_first_edgec_)
                % if ii <= 1 || index_first_edgec_(ii) - index_first_edgec_(ii-1) > 400
                    temp(max(1, index_first_edgec_(ii) - round(50 / 8000 * fs)): ...
                         min(index_first_edgec_(ii) + round(50 / 8000 * fs), length(y))) = true;
                % end
            end
            excludesamples = find(temp);
        elseif cardiacmethod == 2
            [~, excludesamples] = detect_cardiac(cardiac, fs);
        end
        % excludesamples = [];
        % TODO raise threshold for cardiac use sig
        % When is there a cardiac channel present?
    else
        % excludesamples = find(detect_respiratory(1/16, fs, y, 4, 0.2));
        excludesamples = [];        
    end
    index_first_edge = setdiff(index_first_edge_, excludesamples);
end

% Get peaks:
% TODO align on max, min, or median of the zero crossings between max and min
%      raw or wavelet
% TODO Why would I assume first two and last two indices are near the start/end?
% don't use peaks near start/end of data (too close to edge)
index_first_edge = index_first_edge(3:end - 2); 
waveforms = zeros(length(index_first_edge), 2 * Npeak + 1);
timestamps = zeros(length(index_first_edge), 1);
% alignmentmode = 'maxabs';
badinds = [];

if strcmp(alignmentmode, 'centerpeak')
    tic;
    [pks1, locs1] = findpeaks_(sig, 'MinPeakDistance', round(.0005 * fs), 'MinPeakHeight', 0);
    toc;
    [pks2, locs2] = findpeaks_(-sig, 'MinPeakDistance', round(.0005 * fs), 'MinPeakHeight', 0);
    toc;
    pks = [pks1; pks2];
    locs = [locs1; locs2];
    [locs, si] = sort(locs, 'ascend');
    pks = pks(si);
    pkarray = zeros(size(sig));
    pkarray(locs) = pks;
end

% figure(89);hold on;
for k2 = 1:length(index_first_edge)
    % make the index_range twice as wide as I expect a CAP to be (TODO can
    % this capture two CAPs?)
    index_range1 = floor(index_first_edge(k2) - Npeak): ...
                  floor(index_first_edge(k2) + Npeak);  % waveforms is returned and only used for size
    index_range = index_range1; 
                  % floor(index_first_edge(k2) - 2 * Npeak): ...
                  % floor(index_first_edge(k2) + 2 * Npeak);
    % use sig (cwt-convolved) (not y) because sig cleaner for peak alignment     
    sig_local1 = sig(index_range1);
    sig_local = sig(index_range);
    if max(index_range) > length(t)
        continue
    end
    time_local = t(index_range);       
    waveforms(k2,:) = sig_local1;
    % find peak (positive or negative depending on threshold sign)
    % mask = NaN(size(sig_local));
    % mask(Npeak + 1 + (-5:5)) = 1;    
    mask = ones(size(sig_local));
    switch alignmentmode
        case 'max'
            [~, peak_index] = max(sig_local.*mask);
        case 'min'
            [~, peak_index] = min(sig_local.*mask);
        case 'maxabs'
            [~, peak_index] = max(abs(sig_local.*mask));
        case 'midzc'
            [~, minind] = min(sig_local);
            [~, maxind] = max(sig_local);
            
            % verify that the signal crosses 0
            if sig_local(minind) > 0 || sig_local(maxind) < 0
                badinds = [badinds, k2];  %#ok
                continue
            end
            
            % sort the min and max indices
            ind1 = min(minind, maxind);
            ind2 = max(minind, maxind);
            
            % find zero crossings
            zcinds = find((sign(sig_local(ind1:ind2 - 1)) > 0 & ...
                           sign(sig_local(ind1 + 1:ind2)) < 0) | ...
                          (sign(sig_local(ind1:ind2 - 1)) < 0 & ...
                           sign(sig_local(ind1 + 1:ind2)) > 0));
            %interpolate
            for ii = 1:length(zcinds)
                y1 = sig_local(ind1 + zcinds(ii) - 1);
                y2 = sig_local(ind1 + zcinds(ii));
                zcinds(ii) = zcinds(ii) + y1 / (y1 - y2);
            end
            % find locations that are exactly zero
            zinds = find(sig_local == 0);
            % do not expect multiple exactly zeros in a row
            assert(~ismember(1, diff(zinds)), 'two zeros in a row')
            % round the median of the zero crossings
            peak_index = round(median([zcinds, zinds]));
        case 'centerpeak'
            % get location of all peaks            
            % [pks1, locs1] = findpeaks(sig_local, 'MinPeakDistance', round(.0005 * fs), 'MinPeakHeight', 0);
            % [pks2, locs2] = findpeaks(-sig_local, 'MinPeakDistance', round(.0005 * fs), 'MinPeakHeight', 0);
            % concatenate and sort based on location
            % pks = [pks1; pks2];
            % locs = [locs1; locs2];
            % [locs, si] = sort(locs, 'ascend');
            % pks = pks(si);
            % find the peak with the largest magnitude
            locs = find(pkarray(index_range) ~= 0);
            pks = pkarray(locs + index_range(1) - 1);
            [mv, mi] = max(pks);
            % which set of at most three consecutive peaks that includes
            % the max peak has the largest sum
            [mv2, mi2] = max([sum(pks(max(1, mi-2):min(length(locs), max(1, mi-2)+2))), ...
                              sum(pks(max(1, mi-1):min(length(locs), max(1, mi-1)+2))), ...
                              sum(pks(max(1, mi-0):min(length(locs), max(1, mi-0)+2)))]);
            temp = min(length(locs), [max(1, mi-2)+1, max(1, mi-1)+1, max(1, mi-0)+1]);
            % choose the center peak from the largest three consecutive
            peak_index = locs(temp(mi2));
        case 'odd_center_2_min'
            minpeakdist = 0.0015;
            minpeakprom = 1.0;
            minpeakwidth = 0;
            maxpeakwidth = 0.01;
            [pks, locs, w, p] = findpeaks(sig_local, 'MinPeakDistance', minpeakdist * fs, ...
                              'MinPeakProminence', minpeakprom, ...
                              'MinPeakWidth', minpeakwidth * fs, ...
                              'MaxPeakWidth', maxpeakwidth * fs);
            [pks2, locs2, w2, p2] = findpeaks(-sig_local, 'MinPeakDistance', minpeakdist * fs, ...
                              'MinPeakProminence', minpeakprom, ...
                              'MinPeakWidth', minpeakwidth * fs, ...
                              'MaxPeakWidth', maxpeakwidth * fs);
            alllocs = [locs; locs2];
            allpeaks = [pks; pks2];
            if length(alllocs) > 3 || length(alllocs) < 1
%                 plot(time_local, sig_local, 'b-', time_local(locs), pks, 'g.', time_local(locs2), -pks2, 'r.');
                peak_index = [];
                % disp('shouldn''t happen');
            elseif length(alllocs) == 2
%                 plot(time_local, sig_local, 'c-', time_local(locs), pks, 'g.', time_local(locs2), -pks2, 'r.');
                [~, mi] = min(allpeaks);
                peak_index = alllocs(mi);
            elseif length(alllocs) == 1
%                 plot(time_local, sig_local, 'm-', time_local(locs), pks, 'g.', time_local(locs2), -pks2, 'r.');
                temp = sort(alllocs);
                peak_index = temp(ceil(length(temp)/2));
            elseif length(alllocs) == 3
%                 plot(time_local, sig_local, 'y-', time_local(locs), pks, 'g.', time_local(locs2), -pks2, 'r.');
                temp = sort(alllocs);
                peak_index = temp(ceil(length(temp)/2));
            end
            
    end
    if ~isempty(peak_index)
        timestamps(k2) = time_local(peak_index);
        waveforms(k2, :) = sig(index_range1 + peak_index - 1 - Npeak);
        
        % waveforms(k2, :) = y(index_range1 + peak_index - 1 - Npeak);

    end
end
timestamps(badinds) = [];
waveforms(badinds, :) = [];

[timestamps, ia] = unique(timestamps);
waveforms = waveforms(ia, :); % OUTPUTS WAVELET CONVOLVED SIGNAL ????
   
% **** Uncomment this plot to check thresholds: ****   
% % figure
% % plot(t, sig(:,k2)) % ywm(:,k))
% % hold on
% % plot(t, y, 'k')
% % %plot(t(index_first_edge), peak{k}, 'r.')
% % plot(t, thresh_sign(k2)*thresh(k)*ones(size(t)), 'g')
% % %plot(t, repmat(cap_peak_thresh, length(t), 1));
% % title(strcat('Event T=',num2str(Twave(k))));
 
toc



end
 