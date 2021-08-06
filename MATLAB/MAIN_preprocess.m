% 
% MAIN script to preprocess raw signal in master thesis work by Gabriel.
% ----------------------------------------------------------------------
%
% 1. First cell is used to load and concatinate individual .rhs files. 
% (KI-recordings)
% 2. Second cell is used to convert individual .plx-files to .mat files.
% (Zanos et. al. recordings).
% ---
% Either use first part of script by selecting one specific recording and 
% run one cell at the time, or use last cell and loop through all
% recordings in specified directory. 
% --
% The final two cells are an approach to make use of all Channels of the
% KI-recordings. (Extracting the CAPs from individual recordings and then
% only use CAPs that are found in "x out of all" of the different channels.)

% ----------------------------------------------------------------------

%% Load, concatinate, and save RAW .rhs KI-DATA as .mat files. (using Marys electrode)
% Marys electrode have 16 channels (some can be turned off due to errors..)
% Data is saved as a concatinated .mat file for each channel. 
clear all; clc; close all;

% The specified folder containes files that each containes a short period of the recording, 
% e.g. 1min. (The files becomes to large if full recording..):

% file_dir_path = '../../KI_raw_recordings/Baseline_10min_LPS_10min_KCl_10min_210617_103421/';
% file_dir_path = '../../KI_raw_recordings/Baseline_10min_LPS_10min_KCl_10min_210617_142447/';
% file_dir_path = '../../KI_raw_recordings/Baseline_10min_Saline_10min_KCl_10min_210617_122538/';
% file_dir_path = '../../KI_raw_recordings/210715_30min_baseline_30min_SALINE_injection_30min_KCl_mouse2_210715_131453/';
file_dir_path = '../../KI_raw_recordings/210715_30min_baseline_30min_ZYA_injection_30min_KCl_mouse1_210715_105653/';

saveas_path = 'preprocessed/';
downsample = 1; % 2 for 90 min KI recordings (memory issues). 1 for 30 min KI recordings.
channels = 1;
[channels, t] = convert_rhs_to_mat(file_dir_path, saveas_path, downsample, channels);

%% Convert PLX files to .mat and save them to specified path
clear all; clc; close all
path_to_plx = '../../Google Drive/Data/PLX_filer/';
files = dir([path_to_plx, '*.plx']); 
save_to_path = 'converted_plx/'; 
for file = files' %(22:35)' % 65:72 saline injections
    file_path = [path_to_plx, file.name]
    importplxfiles(file_path, save_to_path);
end


%% Initiate the NeuralRecording object
clear all; clc;

datafile=''
nr = NeuralRecording(datafile);  % KI-data
% nr = NeuralRecording(file_path, 50, 'y1');    % Zanos-data

% Apply gain for zanos data: (Not KI-data, this is already in uV.)
% nr.apply_gain(20);  % gain of 50 in mV so multiply by 20 to convert to uV


% % apply Butterworth high-pass filter. 
% Ex. hpf(10) will remove all frequencies below 10 Hz. 
nr.hpf(10);

%% Downsample
% During analysis of Zanos-data, this was set to Nd=5.
% i.e Nd=5 decrease sample_rate from qqq: 40000 to 8000 Hz.
% or <=> dt from 2.5e-5 to 1.25e-4
% For KI-data, downsampling (if any) is done in "convert_rhs_to_mat()".


Nd = 1;
nr.downsample(Nd);

%% Adaptive threshold
%----------------- Gabriel-comments ----------------------------------------------
%  SO-CFAR <=> "Smallest of constant false-alarm rate" filter
% SO-CFAR "used to determine the threshold that rides on the resperatory
% modulation"
% SO-CFAR applied to decomposed signal (wavelet)
%---------------------------------------------------------------------
% if usewavelets then nr.waveforms_ will be based on cwt
usewavelets = true;  % threshold the wavelet transform of the signal
num_std = 3;         % number of standard deviations for the threshold
wdur = 1501 / 8000; % 1501 / 8000 = 188ms  % SO-CFAR window duration in seconds 
gdur = 10 / 8000; % 10 / 8000 = 13 ms    % SO-CFAR guard duration in seconds
usedownsampled = true;    % use the downsampled data to perform adaptive thresholding
alignmentmode = 'max'; % align the waveforms
threshType = 4;
nr.adaptive_threshold(usewavelets, num_std, wdur, gdur, usedownsampled, alignmentmode, threshType);

%% remove nonunique timestamps
[nr.waveforms.timestamps, inds] = unique(nr.waveforms.timestamps);
nr.waveforms.X = nr.waveforms.X(inds, :);

%% Amplitude threshold
nr.waveforms.plot_amps();  % verify that the threshold is appropriate

%% Exportera waveforms
%name ='amp_thresh_R10_Exp2_71516_BALBC_TNF_05ug_IL1B35ngperkg_10'
%datafile='R10_6.28.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_02'
waveforms = nr.waveforms.X;
save(['preprocessed/','wf',datafile,'.mat'], 'waveforms', '-double')

%% Exportera Timestamps..! 
timestamps = nr.waveforms.timestamps;  
save(['preprocessed/','ts',datafile,'.mat'], 'timestamps', '-double')

%% Exportera downsampled raw_rec
rawrec = nr.datadec.data;
save([datafile,'rawrec.mat'], 'rawrec', '-double')


%% Preprocess and extract CAP-instances for all converted .mat files in specified folder.
% Each preprocessed file is then saved as a waveform (wf)- and one
% timestamps (ts)- file
clear all; clc; close all
zanos = false; % Specify if it is recording by Zanos et. al. or not (false => KI recordings) 
if zanos
    path = '../../Google Drive/Data/mat_filer/'; % Zanos
else
    path = 'preprocessed/'; % KI
end
files = dir([path, '*.mat']); 
% start_string_of_file = 'Baseline';
start_string_of_file = '210715_30min_baseline_30min_ZYA_injection_30min_KCl_mouse1_210715_105653A-000_second';

for file = files' 
    if startsWith(file.name, start_string_of_file)
        file_path = [path, file.name]
        % ******* Sec. 1: ********
        if zanos
            nr = NeuralRecording(file_path, 50, 'y1');
            nr.apply_gain(20);  % gain of 50 in mV so multiply by 20 to convert to uV for Zanos!!
            % ******* Sec. 2: ********
            Nd = 5; % gg: only use every e.g 5th recorded datapoint.. 
            % i.e Nd=5 decrease sample_rate from qqq: 40000 to 8000 Hz.
            % or dt from 2.5e-5 to 1.25e-4
        else
            nr = NeuralRecording(file_path);
            Nd = 1; % Downsampling is done when loading files
        end
        % apply Butterworth high-pass filter
        % nr.hpf(10);
        nr.hpf(60); % 90 min KI recordings. noisy.. 

        nr.downsample(Nd);
        % ******* Sec. 3: ********
        usewavelets = true;  % threshold the wavelet transform of the signal
        num_std = 3;         % number of standard deviations for the threshold
        wdur = 1501 / 8000; % 1501 / 8000 = 188ms  % SO-CFAR window duration in seconds 
        gdur = 10 / 8000; % 10 / 8000 = 13 ms    % SO-CFAR guard duration in seconds
        usedownsampled = true;    % use the downsampled data to perform adaptive thresholding
        alignmentmode = 'max'; % align the waveforms
        threshType = 4;
        nr.adaptive_threshold(usewavelets, num_std, wdur, gdur, usedownsampled, alignmentmode, threshType);
        
        % ******* Sec. 4: ********
        % remove nonunique timestamps
        [nr.waveforms.timestamps, inds] = unique(nr.waveforms.timestamps);
        nr.waveforms.X = nr.waveforms.X(inds, :);
        % Export/save waveforms and timestamps
        waveforms = nr.waveforms.X;
        timestamps = nr.waveforms.timestamps;  

        save(['preprocessed2/','wf',file.name], 'waveforms', '-double')
        save(['preprocessed2/','ts',file.name], 'timestamps', '-double')
        disp(['Size of waveforms : ', num2str(size(waveforms))])

    end
end

%% Load data from multiple channels (KI-recordings).
% This data is used for preprocessing step in next cell.

clear all; clc; close all

path = 'preprocessed/'; % KI

files = dir([path, '*.mat']); 
%start_string_of_file = 'Baseline_10min_LPS_10min_KCl_10min_210617_142447';
%start_string_of_file = 'Baseline_10min_LPS_10min_KCl_10min_210617_103421';
% start_string_of_file = 'Baseline_10min_Saline_10min_KCl_10min_210617_122538';

start_string_of_file = '210715_30min_baseline_30min_ZYA_injection_30min_KCl_mouse1';
%start_string_of_file = '210715_30min_baseline_30min_SALINE';


first10000 = false; % true => Only save first 10000 observations

count = 1;
for file = files'
    if startsWith(file.name, start_string_of_file)
        file_path = [path, file.name];
        % ******* Sec. 1: ********
        nr = NeuralRecording(file_path);
        Nd = 1;
        % apply Butterworth high-pass filter
        
        % nr.hpf(10);
        nr.hpf(60); % KI 90 min recordings
        
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
        % Export/save waveforms and timestamps 
        waveforms = nr.waveforms.X;
        timestamps = nr.waveforms.timestamps;

        if count > 1
            all_data(end+1) = cell2struct(repmat({[]},2,1),['wf'; 'ts']);
        else
            all_data = cell2struct(repmat({[]},2,1),['wf'; 'ts']);
        end
        if first10000
            all_data(count).wf = waveforms(1:200000,:);
            all_data(count).ts = timestamps(1:200000);
        else
            all_data(count).wf = waveforms;
            all_data(count).ts = timestamps;
        end
        count = count + 1;
        disp(['Size of waveforms : ', num2str(size(waveforms))])
    end
end

%% Find final waveform/timestamp results using data from all channels.
% OBS! This cell uses the result from the previous cell. ( "all_data" )
% Only consider CAPs/waveforms that was found in at least
% "n_channel_thresh" out of all channels.


n_channel_thresh = 12; % Out of 15 or 16
[waveforms, timestamps] = multiple_channels_threshold(all_data, n_channel_thresh);

save(['preprocessed2/','ts_final3_',start_string_of_file], 'timestamps', '-double')
save(['preprocessed2/','wf_final3_',start_string_of_file], 'waveforms', '-double')

