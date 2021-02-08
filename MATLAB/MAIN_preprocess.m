%% Initiate the NeuralRecording object
%datafile = 'R10_6.30.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_05';
%datafile = 'test';
clear all; close all; clc;

%path_to_matlab_files = '../../../../"Google Drive/Data/mat_filer/'

datafile='' % R10_6.27.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_01'
nr = NeuralRecording(datafile);
nr.apply_gain(20);  % gain of 50 in mV so multiply by 20 to convert to uV

% % apply Butterworth high-pass filter
nr.hpf(10);

%% Downsample
datafile='R10_Exp2_7.15.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_10'

Nd = 5; % gg: only use every e.g 5th recorded datapoint.. 
% i.e Nd=5 decrease sample_rate from qqq: 40000 to 8000 Hz.
% or dt from 2.5e-5 to 1.25e-4
nr.downsample(Nd);


%% Adaptive threshold
% addpath(REDUCE_PLOT_PATH);

%----------------- Gabriel-comments ----------------------------------------------
%  SO-CFAR <=> "Smallest of constant false-alarm rate" filter
% SO-CFAR "used to determine the threshold that rides on the resperatory
% modulation"
% SO-CFAR applied to decomposed signal (wavelet)
%---------------------------------------------------------------------
% if usewavelets then nr.waveforms_ will be based on cwt
usewavelets = true;  % threshold the wavelet transform of the signal
num_std = 3;         % number of standard deviations for the threshold
wdur = 1501 / 8000; % 1501 / 8000;  % SO-CFAR window duration in seconds
gdur = 10 / 8000; % 10 / 8000;    % SO-CFAR guard duration in seconds
usedownsampled = true;    % use the downsampled data to perform adaptive thresholding
alignmentmode = 'max'; % align the waveforms
threshType = 4;
nr.adaptive_threshold(usewavelets, num_std, wdur, gdur, usedownsampled, alignmentmode, threshType);
%% remove nonunique timestamps
[nr.waveforms.timestamps, inds] = unique(nr.waveforms.timestamps);
nr.waveforms.X = nr.waveforms.X(inds, :);
%% Amplitude threshold
nr.waveforms.plot_amps();  % verify that the threshold is appropriate
%%
ampthresh = 5;

% If a constant threshold isn't appropriate try this
% [x, y] = ginput;
% ampthresh = interp1(x, y, nr.waveforms.timestamps, 'linear', 'extrap');

nr.waveforms.amp_thresh(ampthresh);

%% Exportera waveforms
%name ='amp_thresh_R10_Exp2_71516_BALBC_TNF_05ug_IL1B35ngperkg_10'
%datafile='R10_6.28.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_02'
waveforms = nr.waveforms.X;
save(['preprocessed/','wf',datafile,'.mat'], 'waveforms', '-double')
%% Exportera Timestamps..! -- Crusial for Event_rate calculation..
timestamps = nr.waveforms.timestamps;  
save(['preprocessed/','ts',datafile,'.mat'], 'timestamps', '-double')

%% Exportera downsampled raw_rec
rawrec = nr.datadec.data;
save([datafile,'rawrec.mat'], 'rawrec', '-double')

%% Automating process to loop through and preprocess all plx-converted Recordings:
clear all; clc; close all
files = dir('../../../../Google Drive/Data/mat_filer/*.mat');
for file = files(22:72)'%(22:35)' % 65:72 saline injections
    %close all
    file_path = ['../../Google Drive/Data/mat_filer/',file.name]
    % ******* Sec. 1: ********
    nr = NeuralRecording(file_path);
    nr.apply_gain(20);  % gain of 50 in mV so multiply by 20 to convert to uV
    % % apply Butterworth high-pass filter
    nr.hpf(10);
    % ******* Sec. 2: ********
    Nd = 5; % gg: only use every e.g 5th recorded datapoint.. 
    % i.e Nd=5 decrease sample_rate from qqq: 40000 to 8000 Hz.
    % or dt from 2.5e-5 to 1.25e-4
    nr.downsample(Nd);
    % ******* Sec. 3: ********
    usewavelets = true;  % threshold the wavelet transform of the signal
    num_std = 3;         % number of standard deviations for the threshold
    wdur = 1501 / 8000; % 1501 / 8000;  % SO-CFAR window duration in seconds
    gdur = 10 / 8000; % 10 / 8000;    % SO-CFAR guard duration in seconds
    usedownsampled = true;    % use the downsampled data to perform adaptive thresholding
    alignmentmode = 'max'; % align the waveforms
    threshType = 4;
    nr.adaptive_threshold(usewavelets, num_std, wdur, gdur, usedownsampled, alignmentmode, threshType);
    % ******* Sec. 4: ********
    % remove nonunique timestamps
    [nr.waveforms.timestamps, inds] = unique(nr.waveforms.timestamps);
    nr.waveforms.X = nr.waveforms.X(inds, :);
    % ******* Sec. 5: ********
    % Amplitude threshold
    %nr.waveforms.plot_amps();  % verify that the threshold is appropriate
    %ampthresh = 5;
    %nr.waveforms.amp_thresh(ampthresh);
    % Exportera waveforms
    waveforms = nr.waveforms.X;
    save(['preprocessed/','wf',file.name], 'waveforms', '-double')
    % Exportera Timestamps.
    timestamps = nr.waveforms.timestamps;  
    save(['preprocessed/','ts',file.name], 'timestamps', '-double')
end


%% Save plots of all raw recordings
clear all; clc; close all

%files = dir('../../../../Google Drive/Data/mat_filer/*.mat');
files = dir('converted_plx/*.mat');

%set(gcf,'Visible','off');
for file = files(43:72)'%(22:35)' % 65:72 saline injections %22:72
    %file_path = ['../../Google Drive/Data/mat_filer/',file.name]
    file_path = ['converted_plx/',file.name]
    % ******* Sec. 1: ********
    nr = NeuralRecording(file_path,50,'y1');
    nr.apply_gain(20);  % gain of 50 in mV so multiply by 20 to convert to uV
    % % apply Butterworth high-pass filter
    nr.hpf(10);
    % ******* Sec. 2: ********
    Nd = 5; % gg: only use every e.g 5th recorded datapoint.. 
    % i.e Nd=5 decrease sample_rate from qqq: 40000 to 8000 Hz.
    % or dt from 2.5e-5 to 1.25e-4
    nr.downsample(Nd);
    plot_rawrec(nr,file.name,false)
    
end
%% Convert PLX files to .mat and save them to specified path
clear all; clc; close all
files = dir('../../Google Drive/Data/PLX_filer/*.plx');
save_to_path = 'converted_plx/'
for file = files'%(22:35)' % 65:72 saline injections
    file_path = ['../../Google Drive/Data/PLX_filer/',file.name]
    importplxfiles(file_path,save_to_path)
end

%% ALLT OVAN �R VAD SOM ANV�NDES SOM PREPROCESSING AV GABRIEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Exportera/importera efter adaptive threshold
import = true;
export = false;
if export
    save('nr_demofile_full-data_adaptive-threshtype-4.mat', 'nr', '-v7.3')
end
if import
    load('nr_demofile_full-data_adaptive-threshtype-4.mat', 'nr')
end

%% Amplitude threshold
nr.waveforms.plot_amps();  % verify that the threshold is appropriate

ampthresh = 4;

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
    maxiter = 500; %gg: G�r f�ga skillnad..?
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

%% Visualization
if ~exist('real_clusters', 'var') || isempty(real_clusters)
    real_clusters = 1:max(nr.waveforms.labels);
end
h = figure;ax = gca;mycolormap = [get(ax, 'ColorOrder');0.8, 0.8, 0.8];close(h);
% mycolormap = [jet(15);0.8, 0.8, 0.8];  % In case there are repeated colors 
inds = mod(0:max(nr.waveforms.labels)-1, size(mycolormap, 1)-1) + 1;
inds(~ismember(1:max(nr.waveforms.labels), real_clusters)) = size(mycolormap, 1);
nr.waveforms.scatter('label', nr.waveforms.traininds, mycolormap(inds, :));
nr.waveforms.scatter('label', 1:size(nr.waveforms.X, 1), mycolormap(inds, :));
%% 
N = 10;
nr.waveforms.plot_all_waveforms(N);
%% TEST OF EVENT RATES BY MANUALLY LABEL WAVEFORMS...
% ---------------------------------------------------
pre_injection_idx = nr.waveforms.timestamps < 30*60;
post_first = (30*60 < nr.waveforms.timestamps) & (nr.waveforms.timestamps < 60*60);
post_second = nr.waveforms.timestamps > 60*60;

nr.waveforms.labels(pre_injection_idx) = 1;
nr.waveforms.labels(post_first) = 1;
nr.waveforms.labels(post_second) = 1;


%% Event Rates
%nr.waveforms.labels = 0;

%nr.waveforms.labels(1:end/10) = 3;
%nr.waveforms.labels(1:end) = 1;
nr.waveforms.event_rates(true,true);


%% Visualization
N = 100;
nr.waveforms.plot_event_rates([1,2,3], N);

%nr.waveforms.plot_event_rates(real_clusters, N);
%%
usedownsampled = false;
nr.plot_caps(usedownsampled);

%% Save Outfile
nr.save_outfile();
