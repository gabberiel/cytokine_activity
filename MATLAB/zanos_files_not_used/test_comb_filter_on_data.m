function [yf] = test_comb_filter_on_data(sr, y, upto, wl, fnotch, Q)
% Wrapper function that applies comb_band_stop to a signal
% A narrow notch filter is applied adaptively to mitigate harmonics from 
% power line interference, harmonics generated from the recording device,
% and bursts of energy at specific frequencies localized in time.
%
% sr      - sampling rate in Hz
% y       - input signal
% upto    - maximum frequency for checking harmonics of fnotch Hz (aliased 
%           harmonics will be mitigated if set larger than sr / 2)
% wl      - window length for the spectrogram in seconds
% fnotch  - fundamental frequency of power line interference (60 Hz in US)
% Q       - notch center frequency / bandwidth
%
% yf      - filtered signal
%
% Example:
% load('mouse 5\baseline\channel_1_.mat', 't', 'y');
% sr = 1 / (t(2) - t(1));
% yf = test_comb_filter_on_data(sr, y, 1.5*sr, 5, 60, 1000);
% figure;plot(t.', [y(1:1e6).', yf(1:1e6).']);

%% assign default values
N = length(y);
if ~exist('wl', 'var')
    wl = 5;
end
if ~exist('fnotch', 'var')
    fnotch = 60;
end
if ~exist('Q', 'var')
    Q = 1000;
end

%% compute the spectrogram
[s, w, t_] = spectrogram(y, wl * sr, 0);
tmin = (1:wl * sr:N - wl * sr) / sr / 60; % time in minutes
f = sr * w / (2 * pi);                    % frequency in Hz
df = f(2) - f(1);

% Repeat the spectrogram if there are aliased harmonics
f2 = f;
s2 = s;
ct = 0;
while f2(end) < upto - df / 2  % need to handle precision
    f2 = [f2(1:end-1);f2(end) + f];
    % every other repeat of s is the mirror image
    if mod(ct, 2) == 0
        s2 = [s2;s(end-1:-1:1, :)];  %#ok
    else
        s2 = [s2;s(2:end, :)];       %#ok
    end
    ct = ct + 1;
end
s = s2;
f = f2;
clear s2 f2

sdb = 20*log10(abs(s));        % power spectrum in decibels
h1 = figure;imagesc(tmin, f, sdb);

%% Estimate of the frequency of the power line interference with jitter
maxord = floor(upto / fnotch);
stdmaxind = zeros(1, maxord);
flinehat = zeros(length(t_), maxord);
% loop over all harmonics
for k = 1:maxord
    % place a window around the harmonic (+/- 4 Hz is arbitrary)
    inds = find(f > fnotch * k - 4 & f < fnotch * k + 4);
    % estimate the peak frequency within the window at each time
    [~, mi] = max(abs(s(inds, :)), [], 1);
    % divide by the order to get an estimate of the fundamental frequency
    fline1 = f(inds(1) + mi - 1) / k;
    % when the SNR is low the std of the max indices should be high
    stdmaxind(k) = std(mi);        
    flinehat(1:length(mi), k) = fline1;
end
% use a median filter to smooth the line noise frequency estimate
medflinehat = movmedian(flinehat, 5, 1);
% average several harmonics with high SNR
fharmonics = (stdmaxind < 12 & stdmaxind > 0);
fline = median(medflinehat(:, fharmonics), 2);

% check the time varying frequency for energy in case there is overlap
% refine fharmonics
% arbitrary window size, could adjust based on sr and wl
smoothingwindow = ones(100, 5) / 500; 
hpf = sdb - conv2(sdb, smoothingwindow, 'same');
meanpow = zeros(1, maxord);
for k = 1:maxord
    % get indices at the frequency where power is expected
    inds = min(length(f), max(1, interp1(f, 1:length(f), k*fline, ...
                                         'nearest', 'extrap')));
    val = zeros(length(inds), 1);
    for ii = 1:length(inds)
        val(ii) = hpf(inds(ii), ii);
    end
    meanpow(k) = mean(val);
end
fharmonics = find(meanpow > 5 & stdmaxind > 0);

%% Apply the comb band stop filter
% get corresponding spectrogram window start locations in the signal
startinds = 1:wl * sr:N - wl * sr;
assert(length(fline) == length(startinds));

% Second order IIR comb band stop filter
yf = zeros(size(y));
% loop over each time window
zisout = zeros(length(fharmonics), 2);
for ii = 1:length(startinds)    
    curinds = (ii - 1) * wl * sr + 1:ii * wl * sr;    
    % only apply the notch filter to harmonics with high SNR
    [yf(curinds), zisout] = comb_band_stop(y(curinds), sr, Q, ...
        fline(ii), upto, fharmonics, zisout);
end

%% Recompute the spectrogram of the filtered signal
% spectrogram magnitude in dB
[s, w, ~] = spectrogram(yf, wl * sr, 0);
sdb = 20*log10(abs(s));
tmin = (1:wl * sr:N - wl * sr) / sr / 60; % time in minutes
f = sr * w / (2 * pi);                    % frequency in Hz
% highpass filtered version of the spectrogram magnitude response in dB 
smoothingwindow = ones(100, 5) / 500; % could adjust based on sr and wl
hpf = sdb - conv2(sdb, smoothingwindow, 'same');
% figure;imagesc(tmin, f, hpf);
% h = figure;imagesc(tmin, f, sdb);

%% Remove constant frequency interference
% compute the mean to find constant frequency interference across time
mn = mean(hpf, 2);
df = f(2) - f(1);

% threshold the mean over time of the highpass filtered spectrogram
% 0.5 Hz is an arbitrary choice for combining threshold crossings
CC = bwconncomp(imclose(mn > 2.5, ones(ceil(0.5 / df), 1)));
% get the magnitude spectrum of the signal to fine tune the frequencies
magf = abs(fft(yf));
f_ = linspace(0, sr, N + 1);
f_ = f_(1:end - 1);
df_ = f_(2) - f_(1);

% verify that the detected bandwidth of the constant frequency interference
% is small so that it isn't confused with biological signals
ind1 = zeros(1, length(CC.PixelIdxList));
ind2 = zeros(1, length(CC.PixelIdxList));
for ii = 1:length(CC.PixelIdxList)
    [~, mi] = max(mn(CC.PixelIdxList{ii}));
    temp = find(mn(1:CC.PixelIdxList{ii}(mi)) <= 0.5 * mn(CC.PixelIdxList{ii}(mi)), 1, 'last');
    if isempty(temp)
        ind1(ii) = mi;
    else
        ind1(ii) = CC.PixelIdxList{ii}(mi) - temp;
    end
    temp = find(mn(CC.PixelIdxList{ii}(mi):end) <= 0.5 * mn(CC.PixelIdxList{ii}(mi)), 1, 'first');
    if isempty(temp)
        ind2(ii) = length(CC.PixelIdxList{ii}) - mi + 1;
    else        
        ind2(ii) = temp;
    end
end
keepinds = df * (ind1 + ind2 - 1) <= 0.75;

% fine tune each frequency that passed the thresold by finding the peak in
% the PSD
constfreq = cellfun(@(x) median(f(x)), CC.PixelIdxList);
constfreq = constfreq(keepinds);
for ii = 1:length(constfreq)
    ind = find(f_ <= constfreq(ii), 1, 'last');
    inds = ind + (-ceil(0.1 / df_):ceil(0.1 / df_));
    [~, mi] = max(magf(inds));
    constfreq(ii) = f_(inds(mi));
end
% apply the filter to a single frequency (use a very narrow filter)
for ii = 1:length(constfreq)
    [yf, ~] = band_stop(yf, constfreq(ii), 2 * Q, sr, zeros(1, 2));
end

%% Remove bursts in time and frequency
% TODO not working in general
inds = find(mean(abs(s), 1) > 3);
% figure;plot(y(wl*sr*inds(1) +(-wl*sr:wl*sr)));

if false
    % A signal with a burst will have a low value for the variable burst on the
    % interval [0, 1]
    % I expect Gaussian random noise to have a value of pi / 4
    burst = mean(abs(s), 2).^2 ./ mean(abs(s).^2, 2);
    % best to choose a low threshold like 0.4 to avoid false detections
    CC = bwconncomp(imclose(burst < 0.4, ones(ceil(0.5 / df), 1)));
    burstfreq = cellfun(@(x) median(f(x)), CC.PixelIdxList);
    burstind = cellfun(@(x) round(median(x)), CC.PixelIdxList);
    intervals = zeros(2, length(burstind));
    % determine the interval
    badinds = [];
    for ii = 1:length(burstind)
        burstdb = 20*log10(abs(s(burstind(ii), :)));
        % 3 std's is arbitrary
        CC = bwconncomp(imclose(zscore(burstdb) > 3, ones(1, 3)));
        % TODO There can be multiple time intervals per burst
        if ~isempty(CC.PixelIdxList)
            minind = cellfun(@(x) min(x), CC.PixelIdxList);
            maxind = cellfun(@(x) max(x), CC.PixelIdxList);
            intervals(:, ii) = [max(1, minind - 1); min(size(s, 2), maxind + 1)];
        else
            badinds = [badinds, ii];  %#ok
        end
    end
    intervals(:, badinds) = [];
    burstfreq(badinds) = [];
    % refine the burst frequency
    for ii = 1:length(burstfreq)
        ind = find(f_ <= burstfreq(ii), 1, 'last');
        inds = ind + (-ceil(1.0 / df_):ceil(1.0 / df_));
        inds = inds(inds > 0 & inds <= length(magf));
        [~, mi] = max(magf(inds));
        burstfreq(ii) = f_(inds(mi));
    end
    % apply the filter
    for ii = 1:size(intervals, 2)
        inds = wl * sr * intervals(1, ii):wl * sr * intervals(2, ii);
        [yf(inds), ~] = band_stop(yf(inds), burstfreq(ii), Q / 10, sr, zeros(1, 2));
    end
end

%% Plots
% h1 = figure;plot(f, mean(hpf, 2));
% h2 = figure;plot(f, ace);
% linkaxes([get(h1, 'CurrentAxes'), get(h2, 'CurrentAxes')], 'x');

% plot PSD
mag = abs(fft(y));
magf = abs(fft(yf));
f_ = linspace(0, sr, N + 1);
f_ = f_(1:end - 1);
figure;plot(f_(1:end/2), 20*log10(reshape([mag(1:end/2), magf(1:end/2)], [], 2)));

[s, ~, ~] = spectrogram(yf, wl * sr, 0);
h2 = figure;imagesc(tmin, f, 20*log10(abs(s)));

linkaxes([get(h1, 'CurrentAxes'), get(h2, 'CurrentAxes')], 'xy');