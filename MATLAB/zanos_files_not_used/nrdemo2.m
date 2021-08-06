% ARAMS MODIFIERADE FIL

%% Instantiate the NeuralRecording object
datafile = '';
nr = NeuralRecording(datafile);
nr.apply_gain(20);  % gain of 50 in mV so multiply by 20 to convert to uV

% apply Butterworth high-pass filter
%nr.hpf(10);

% downsample
nr.downsample(5);

%% Adaptive threshold
% addpath(REDUCE_PLOT_PATH);
% testa ändra std till 2,1 och kör TNF -> IL1B igen
% if usewavelets then nr.waveforms_ will be based on cwt
usewavelets = true;  % threshold the wavelet transform of the signal
num_std = 1;         % number of standard deviations for the threshold
wdur = 1501 / 8000; % 1501 / 8000;  % SO-CFAR window duration in seconds
gdur = 10 / 8000; % 10 / 8000;    % SO-CFAR guard duration in seconds
usedownsampled = false;    % use the downsampled data to perform adaptive thresholding
alignmentmode = 'max'; % align the waveforms
threshType = 4;
nr.adaptive_threshold(usewavelets, num_std, wdur, gdur, usedownsampled, alignmentmode, threshType);

%% Exportera/importera efter adaptive threshold
import = true;
export = false;
if export
    save('nr_TNF_IL1B_adaptive-threshtype-4.mat', 'nr', '-v7.3')
end
if import
    load('nr_IL1B-TNF_adaptive-threshtype-4.mat')
end

%% Testa threshold baserat på amplitud
ampthresh = 10000;
nr.waveforms.amp_thresh(ampthresh);

%% Exportera waveforms
waveforms = nr.waveforms.X;
save('waveforms_IL1B-TNF-03_adaptive-thresh-4.mat', 'waveforms', '-double')

%% Full t-SNE, skippa kernel t-SNE
tic
[nr.waveforms.Y, ~] = tsne(nr.waveforms.X, 'NumDimensions', 2, 'Perplexity', 100);
toc

%% Exportera/importera efter t-SNE + kernel t-SNE
import = true;
export = false;
if export
    save('nr_IL1B_TNF_amp-10000_t-SNE-perpl-30_adaptive-threshtype-4.mat', 'nr', '-v7.3')
end
if import
    load('nr_IL1B-TNF_t-SNE-perpl-100_adaptive-threshtype-4.mat')
end

%% Använd deras dbscan
epsilon = 5.5;
minpts = 450;
nr.waveforms.traininds = round(linspace(1, size(nr.waveforms.X, 1), 3*1e4));
traininds = nr.waveforms.traininds;
testinds = setdiff(1:size(nr.waveforms.X,1), nr.waveforms.traininds);
nr.waveforms.dbscan(epsilon, minpts);

% Plot
gscatter(nr.waveforms.Y(:,1), nr.waveforms.Y(:,2), nr.waveforms.labels)
title('DBSCAN epsilon 5.5, minpts 450')
xlabel('t-SNE dimension 1')
ylabel('t-SNE dimension 2')

%% Event Rates
nr.waveforms.event_rates(true, false);

% Visualization
N = 50;
real_clusters = 1:max(nr.waveforms.labels);
nr.waveforms.plot_event_rates(real_clusters, N);
usedownsampled = false;
nr.plot_caps(usedownsampled);

%% Exportera waveforms
waveforms = nr.waveforms.X;
save('waveforms-R10_IL1B_TNF_03.mat', 'waveforms', '-double')

%% Event rates från python klustring
nr = load('nr_IL1B-TNF_adaptive-threshtype-4.mat').nr;
data = load('7clusters_0.9g-0.9e.mat');
nr.waveforms.Y = data.z;
nr.waveforms.labels = data.labels'+1;
%nr.waveforms.event_rates(false, false);

% Visualization
N = 50;
%real_clusters = 1:max(nr.waveforms.labels);
%nr.waveforms.plot_event_rates(real_clusters, N);