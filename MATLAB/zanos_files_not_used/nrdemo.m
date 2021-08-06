%% Instantiate the NeuralRecording object
datafile = 'R10_6.30.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_05';
nr = NeuralRecording(datafile);
nr.apply_gain(20);  % gain of 50 in mV so multiply by 20 to convert to uV

% apply Butterworth high-pass filter
nr.hpf(10);

%% Downsample
Nd = 5;
nr.downsample(Nd);
%% Adaptive threshold
% qqq: Gabe: use downsample = true
% addpath(REDUCE_PLOT_PATH);
% testa ändra std till 2,1 och kör TNF -> IL1B igen
% if usewavelets then nr.waveforms_ will be based on cwt
usewavelets = true;  % threshold the wavelet transform of the signal
num_std = 1;         % number of standard deviations for the threshold
wdur = 1501 / 8000; % 1501 / 8000;  % SO-CFAR window duration in seconds
gdur = 10 / 8000; % 10 / 8000;    % SO-CFAR guard duration in seconds
usedownsampled = true;    % use the downsampled data to perform adaptive thresholding
alignmentmode = 'max'; % align the waveforms
threshType = 4;
nr.adaptive_threshold(usewavelets, num_std, wdur, gdur, usedownsampled, alignmentmode, threshType);

%% Exportera waveforms
waveforms = nr.waveforms.Y;
save('waveforms-R10_IL1B_TNF_03.mat', 'waveforms', '-double')
%% Exportera/importera efter adaptive threshold
import = true;
export = false;
if export
    save('nr_demofile_full-data_adaptive-threshtype-4.mat', 'nr', '-v7.3')
end
if import
    %load('nr_demofile_full-data_adaptive-threshtype-4.mat', 'nr')
    load('R10_6.30.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_05.mat', 'nr')
end

%% Amplitude threshold
nr.waveforms.plot_amps();  % verify that the threshold is appropriate

ampthresh = 10;

% If a constant threshold isn't appropriate try this
% [x, y] = ginput;
% ampthresh = interp1(x, y, nr.waveforms.timestamps, 'linear', 'extrap');

nr.waveforms.amp_thresh(ampthresh);

%% remove nonunique timestamps
[nr.waveforms.timestamps, inds] = unique(nr.waveforms.timestamps);
nr.waveforms.X = nr.waveforms.X(inds, :);

%% Dimensionality Reduction
%addpath(genpath('DRTOOLBOX_PATH'));

ktsnefile = '';
if exist(ktsnefile, 'file')
    nr.load_ktsne(ktsnefile);
    nr.waveforms.amp_thresh(0.3);  % The file may not include this threshold
else
    perpl = 50;
    no_dims = 2;
    videofile = '';  % 'temp.avi';
    maxiter = 500;
    nTrain = min(10e3, size(nr.waveforms.X, 1));
    traininds = round(linspace(1, size(nr.waveforms.X, 1), nTrain));
    Xtr = nr.waveforms.X(traininds, :);
    mycols = 10*log10(sum(Xtr.^2, 2));
    k = 5.0;
    nr.waveforms.dimensionality_reduction('tsne', 'perplexity', perpl, ...
        'dims', no_dims, 'videofile', videofile, 'maxiter', maxiter, ...
        'color', mycols, 'traininginds', traininds);  % t-SNE
    %nr.waveforms.ktsne(k);                          % kernel t-SNE
    %nr.save_tsne_file(ktsnefile);
end
%% Visualization
nr.waveforms.scatter('power', nr.waveforms.traininds);
%nr.waveforms.scatter('power', 1:size(nr.waveforms.X, 1));

%% Full t-SNE, skippa kernel t-SNE
tic
[nr.waveforms.Y, ~] = tsne(nr.waveforms.X, 'NumDimensions', 2, 'Perplexity', 100);
toc
%% qqq: Gabe variant 
tic
nr.waveforms.Y = tsne(nr.waveforms.X) %, 'NumDimensions', 2, 'Perplexity', 100);
toc
%% Exportera/importera efter t-SNE + kernel t-SNE
import = true;
export = false;
if export
    save('nr_demofile_full-data_t-SNE-perpl-100_adaptive-threshtype-4.mat', 'nr', '-v7.3')
end
if import
    load('nr_demofile_full-data_t-SNE-perpl-100_adaptive-threshtype-4.mat', 'nr')
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
title('DBSCAN epsilon 5.5, minpts 450, t-SNE on all points')
xlabel('t-SNE dimension 1')
ylabel('t-SNE dimension 2')


%% Clustering
nr.waveforms.dbscan();  % DBSCAN GUI
% nr.waveforms.manual_clustering();
epsilon = 6.6;
minpts = 100;
nr.waveforms.dbscan(epsilon, minpts);

% split the clusters based on power
powspreadthresh = 15;
nr.waveforms.split_custers(powspreadthresh);

% Visualization
if ~exist('real_clusters', 'var') || isempty(real_clusters)
    real_clusters = 1:max(nr.waveforms.labels);
end
h = figure;ax = gca;mycolormap = [get(ax, 'ColorOrder');0.8, 0.8, 0.8];close(h);
% mycolormap = [jet(15);0.8, 0.8, 0.8];  % In case there are repeated colors 
inds = mod(0:max(nr.waveforms.labels)-1, size(mycolormap, 1)-1) + 1;
inds(~ismember(1:max(nr.waveforms.labels), real_clusters)) = size(mycolormap, 1);
nr.waveforms.scatter('label', nr.waveforms.traininds, mycolormap(inds, :));
nr.waveforms.scatter('label', 1:size(nr.waveforms.X, 1), mycolormap(inds, :));

N = 10;
nr.waveforms.plot_all_waveforms(N);

%% Event Rates
nr.waveforms.event_rates();

% Visualization
N = 100;
nr.waveforms.plot_event_rates(real_clusters, N);
usedownsampled = false;
nr.plot_caps(usedownsampled);

%% Save Outfile
nr.save_outfile();
