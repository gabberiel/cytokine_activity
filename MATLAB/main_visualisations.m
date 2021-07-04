% Visualise and plot different parts of preprocessing

%% Load Individual file.
close all; clear all; clc;
% datafile='preprocessed/Baseline_10min_LPS_10min_KCl_10min_210617_142447A-003.mat'
datafile= 'preprocessed/Baseline_10min_LPS_10min_KCl_10min_210617_103421A-003.mat'

nr = NeuralRecording(datafile);
% nr_2 =  NeuralRecording(datafile);
% Do this for zanos data: (Not KI-data, this is already in uV.)
% nr.apply_gain(20);  % gain of 50 in mV so multiply by 20 to convert to uV
%% apply Butterworth high-pass filter on nr_2
nr.hpf(10);
% Downsample both
Nd = 1; 
nr.downsample(Nd);
% nr_2.downsample(Nd);
%% Adaptive threshold
usewavelets = true;  % threshold the wavelet transform of the signal
num_std = 3;         % number of standard deviations for the threshold
wdur = 1501 / 8000; % 1501 / 8000 = 188ms  % SO-CFAR window duration in seconds 
gdur = 10 / 8000; % 10 / 8000 = 13 ms    % SO-CFAR guard duration in seconds
usedownsampled = true;    % use the downsampled data to perform adaptive thresholding
alignmentmode = 'max'; % align the waveforms
threshType = 4;
nr.adaptive_threshold(usewavelets, num_std, wdur, gdur, usedownsampled, alignmentmode, threshType);
% nr_2.adaptive_threshold(usewavelets, num_std, wdur, gdur, usedownsampled, alignmentmode, threshType);

%% Plot range of Raw-recording
close all;
fig_path = 'Figures/Preprocessing/';
fig_name = 'raw_rec_';
range_to_plot = [21000:23000];

X = nr.data.Data(range_to_plot,1);
T = nr.data.Time(range_to_plot); % 
fontsize = 18;
fig = figure(1);

plot(T,X);
% axis([-1500 1500])
xlabel('Time [sec]','interpret','Latex','FontSize',fontsize);
ylabel('Voltage [$\mu V$]','interpret','Latex','FontSize',fontsize);
title('Raw Recording','FontSize',fontsize)
saveas(gcf,[fig_path, fig_name, datafile(14:end-4), '.png'])

%% Plot Pre/post hpf-filter
close all; 
range_to_plot = [20000:100000];
figure(1)
fig_path = 'Figures/Preprocessing/hpf/';
fig_name = 'hpf_10Hz';
freq = 2.09;
sin_breath = sin(-0.4 + 2 * pi * freq * nr_2.data.Time(range_to_plot) ) * 20 ;
plot(nr_2.data.Time(range_to_plot),nr_2.data.Data(range_to_plot))
hold on
% plot(nr_2.data.Time(range_to_plot),sin_breath, 'linewidth', 4)
% legend('Raw Signal', 'Breathing : 126 bpm ')
%xlabel('Time, [sec]')
%ylabel('\mu V')
%title('Before High Pass Filter.')
%saveas(gcf,[fig_path, fig_name, '_pre_hpf.png'])

hold on
%figure(2)
plot(nr.data.Time(range_to_plot),nr.data.Data(range_to_plot))
legend('Pre hpf', 'Post hpf')
xlabel('Time, [sec]')
ylabel('\mu V')
%title('After High Pass Filter.')

title('Effect of High Pass Filter.')
saveas(gcf,[fig_path, fig_name, '_both.png'])


%% Plot Thresholds using so_cfar:
close all;
range_to_plot = [40000:200000];
range_to_plot = [20000:23000];

fig_path = 'Figures/Preprocessing/thresholds/';
fig_name = 'both_thresholds_small_range_incl_raw';
t_sec = 10;
% range_to_plot = [100000:100:length(nr.sig)-100000];

% Plot wavelet sig vs threshold for this
t = nr.threshold.Time(range_to_plot);
thresh = nr.threshold.Data(range_to_plot);
thresh_cardiac = nr.threshold_cardiac.Data(range_to_plot);
raw = nr.datadec.Data(range_to_plot);
neural_cwt_sig = nr.sig(range_to_plot);
cardiac_cwt_sig = nr.cardiac(range_to_plot);

figure(1)
hold on
plot(t, raw, 'm')
plot(t, neural_cwt_sig, 'b')

plot(t, thresh, 'r')
% figure(2)
plot(t, cardiac_cwt_sig, 'k')
plot(t, -thresh_cardiac, 'g')
plot(t, thresh_cardiac, 'g')
plot(t, -thresh, 'r')


legend('Raw Signal', 'Neurally Enhances signal', 'Neural event threshold', 'Cardiac Enhanced signal', 'Cardiac event Threshold' )
xlabel('time [sec]')
ylabel('Wavlet convolved, No unit.')
title('Example of Adaptive Threshold.')
saveas(gcf,[fig_path, fig_name, '.png'])

%% Get data to compare threshold pre/post injections
close all;
fig_path = 'Figures/Preprocessing/thresholds/';
fig_name = 'pre_vs_post_injection_neural_';
t_sec = 10*60;

% Get data to compare threshold pre/post injections
t_idx = round(t_sec / nr.dt); 
pre_inj_idx = [1:t_idx];
post_inj_idx = [t_idx:t_idx*3-100000];

t_pre = nr.threshold.Time(pre_inj_idx) / 60;
t_post = nr.threshold.Time(post_inj_idx) / 60;

thresh_pre = nr.threshold.Data(pre_inj_idx);
thresh_post = nr.threshold.Data(post_inj_idx);

figure(1)
hold on
plot(t_pre, thresh_pre, 'm')
plot(t_post, thresh_post, 'b')

plot(t_pre, -thresh_pre, 'm')
plot(t_post, -thresh_post, 'b')

legend('Pre Injection', 'Post Injection')
xlabel('time [min]')
ylabel('Threshold Value')
title('Adaptive Threshold Value During Full Recording.')
saveas(gcf,[fig_path, fig_name, datafile(14:end-4), '.png'])


%% Plot Extracted Neural Events "in raw recording".

close all;
fontsize = 18;

i_wf = 4003;
fig_path = 'Figures/Preprocessing/';
fig_name = 'extracted_caps_in_raw_3';
i_pre = 53;
i_post = 53;

event_idx = round(nr.waveforms.timestamps(i_wf) / nr.dt);
region_range = 8000;
region_data = nr.datadec.Data([(event_idx - region_range ) :(event_idx + region_range+1)], 1);
region_t = [(nr.waveforms.timestamps(i_wf) - region_range*nr.dt) ...
    :nr.dt:(nr.waveforms.timestamps(i_wf) + (region_range+1)*nr.dt)];
hold on
plot(region_t, region_data)
for i_wf = 3900:4104
    event_idx = round(nr.waveforms.timestamps(i_wf) / nr.dt);
    wf_t = [(nr.waveforms.timestamps(i_wf) - i_pre*nr.dt) ...
        :nr.dt:(nr.waveforms.timestamps(i_wf) + i_post*nr.dt)];
    %wf_t = [0:nr.dt:((i_pre+i_post)*nr.dt)];
    wf = nr.datadec.Data([(event_idx - i_pre ) :(event_idx + i_post)], 1);
    plot(wf_t, wf, 'r', 'linewidth', 2)
end
xlabel('Time (sec)','interpret','Latex','FontSize',fontsize);
ylabel('Voltage $\mu V$','interpret','Latex','FontSize',fontsize);
% title('Extracted CAP','FontSize',fontsize)

title('Example of Neural Event in Raw Recording','FontSize',fontsize)
saveas(gcf,[fig_path, fig_name, '.png'])

%% Plot of Waveforms using multiple electrodes
close all;
fontsize = 18;

i_wf = 4;
i_electrode = 1;
fig_path = 'Figures/Preprocessing/';
fig_name = 'extracted_caps_in_raw_3';
i_pre = 53;
i_post = 53;

event_idx = round(all_ts(i_wf, i_electrode) / nr.dtdec);
region_range = 2000;
electrode_data = all_wf(i_wf + 10000*(i_electrode-1), :)
region_data = nr.datadec.Data([(event_idx - region_range ) :(event_idx + region_range+1)], 1);
region_t = [(nr.waveforms.timestamps(i_wf) - region_range*nr.dtdec) ...
    :nr.dtdec:(nr.waveforms.timestamps(i_wf) + (region_range+1)*nr.dtdec)];
hold on
% plot(region_t, region_data)
colors = [];
for ii = 1:15
    colors = [colors; [1/15 * ii, 1 - 1/15*ii, 1/15 * ii]];
end
colors = ['b', 'k', 'r', 'g', 'y', 'c', 'm']

for i_wf = 1:10
    %wf_t = [0:nr.dtdec:((i_pre+i_post)*nr.dtdec)];
    color_i = 0
    for i_electrode = i_electrode % [1:2:14]
        color_i = color_i + 1;
        event_idx = round(all_ts(i_wf, i_electrode) / nr.dtdec);
        wf_t = [(all_ts(i_wf, i_electrode) - i_pre*nr.dtdec) ...
        :nr.dtdec:(all_ts(i_wf, i_electrode) + i_post*nr.dtdec)];
        wf = all_wf(i_wf + 10000*(i_electrode-1), :);
        if all_ts(i_wf, i_electrode) < 0.33
            plot(wf_t, wf, 'linewidth', 2, 'Color', colors(color_i))
        end
        %plot([1:length(wf)], wf, 'linewidth', 2)
        

    end
end
legend('1', '3', '5', '7', '9', '11', '13')

xlabel('Time (sec)','interpret','Latex','FontSize',fontsize);
ylabel('Voltage $\mu V$','interpret','Latex','FontSize',fontsize);
% title('Extracted CAP','FontSize',fontsize)

title('Example of Neural Event in Raw Recording','FontSize',fontsize)


%% Compare extracted CAPs for each channel at specific time-point.
% "all_wf" and "all_ts" are loaded from cell in MAIN_preprocess
close all;
fontsize = 18;

fig_path = 'Figures/Preprocessing/compare_channels/';
fig_name = 'extracted_caps_at_specific_time_t_';
i_pre = 53;
i_post = 53;
t_CAP = 1.50; % Seconds..

diff_times = all_ts - t_CAP;
[val_of_closest_CAP, idx_of_closest_CAP] = min ( abs(diff_times), [], 1 );

chan_with_found_CAP = abs(val_of_closest_CAP - min(val_of_closest_CAP)) < 0.005;


% wf_t = [(t_CAP - i_pre*nr.dtdec) : nr.dtdec : (t_CAP + i_post*nr.dtdec)];
hold on
find_t = true;
colors = ['b', 'k', 'r', 'g', 'y', 'c', 'm'];
channels = ['01', '03', '05', '07', '09', '11', '13'];
color_i = 0;
for channel = 1:7
    if chan_with_found_CAP(channel)
        color_i = color_i +1;
        if find_t
           t_CAP_local = all_ts(idx_of_closest_CAP(channel), channel);
           wf_t = [(t_CAP_local - i_pre*nr.dtdec) : nr.dtdec : (t_CAP_local + i_post*nr.dtdec)];

        end
        wf = all_wf(idx_of_closest_CAP(channel) + 10000*(channel-1), :);
        chan_text_idx = channel*2-1;
        txt = ['Chan=', num2str(channels(chan_text_idx:chan_text_idx+1))];
        plot(wf_t, wf, 'linewidth', 2, 'Color', colors(color_i), 'DisplayName',txt)
    end
end
% legend('1', '3', '5', '7', '9', '11', '13')
legend show
xlabel('Time (sec)','interpret','Latex','FontSize',fontsize);
ylabel('Voltage $\mu V$','interpret','Latex','FontSize',fontsize);
title(['CAPs at t=', num2str(t_CAP), ' sec. N_channels=',num2str(sum(chan_with_found_CAP(1:7)) ...
        ) ],'FontSize',fontsize, 'interpret','Latex')
saveas(gcf,[fig_path, fig_name,num2str(t_CAP), '.png'])


%% Save plots of all raw recordings
clear all; clc; close all
zanos = false;
%files = dir('../../Google Drive/Data/mat_filer/*.mat');
if zanos
    disp('Zanos folder...')
    path = 'converted_plx/';
else
    disp('KI Folder...')
    path = 'preprocessed/'; % KI
end
files = dir([path, '*.mat']); % KI

%set(gcf,'Visible','off');
for file = files(end-1:end)'%(22:35)' % Zanos: 65:72 => saline injections %22:72
    %file_path = ['../../Google Drive/Data/mat_filer/',file.name]
    file_path = [path, file.name]
    % ******* Sec. 1: ********
    if zanos
        nr = NeuralRecording(file_path, 50, 'y1');
        nr.apply_gain(20);  % gain of 50 in mV so multiply by 20 to convert to uV for Zanos!!
        save_path = 'Figures/Raw_recordings/';
    else
        nr = NeuralRecording(file_path);
        save_path = 'Figures/Raw_recordings_KI/';
    end
    % % apply Butterworth high-pass filter
    nr.hpf(10);
    Nd = 10; % Speed up plots..
    nr.downsample(Nd);
    fig_saveas = [save_path,file.name,'.png'];
    plot_rawrec(nr, file.name, false, fig_saveas)
end

%% Save plots of all raw neural / cardiac enhanced recordings
clear all; clc; close all
zanos = false; % If we use the zanos recordings or KI

%files = dir('../../Google Drive/Data/mat_filer/*.mat');
if zanos
    path = 'converted_plx/';
else    
    path = 'preprocessed/'; % KI
end
files = dir([path, '*.mat']); % KI

%set(gcf,'Visible','off');
for file = files(6:9)'%(22:35)' % 65:72 saline injections %22:72
    %file_path = ['../../Google Drive/Data/mat_filer/',file.name]
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
        Nd = 1;
    end
    % % apply Butterworth high-pass filter
    nr.hpf(10);
    nr.downsample(Nd);
    one_cap_leangth = round(3.5*10^(-3) / nr.dt);
    range_to_plot = [10000:10000 + one_cap_leangth*10];
    range_to_plot = [10000:20000];

    plot_cwt_transform(nr, range_to_plot, false, file.name);
    
end
