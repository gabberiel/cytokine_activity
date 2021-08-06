function [concatinated_channels, concatinated_t] = convert_rhs_to_mat ... 
                                                        (dir_to_rhs, save_to_path, downsample, channels)
% Opens all .rhs files in the specifed directory "dir_to_rhs" containing files which
% are each e.g. 1 min-parts of full recording.
% The data (containing multible channels) is concatinated to get full recording-time 
% and one file per channel is saved, as well as a timestamp for each.
% The file are saved as "save_to_path" + the channel name in rhs-file.
% 
% MORE INPUTS:
% downsample : Integer. 
%   If 1, all samples is used. If 2, every second sample is used etc.
% channels : 'all' or integer
%   If 'all', then all channels are saved. Else, the specified channel.

if nargin < 4 || isempty(channels)
    channels = 'all';
end
% Get all .rhs files in specified directory:
files = dir([dir_to_rhs, '/*.rhs']);
concatinated_channels = [];
concatinated_t = [];
for file = files'
    file_path = [dir_to_rhs, file.name]
    [amplifier_channels, amplifier_data, t] = read_Intan_RHS2000_file(file_path);
    if channels == 'all'
        concatinated_channels = [concatinated_channels, amplifier_data(:,1:downsample:end)];
    else
        concatinated_channels = [concatinated_channels, amplifier_data(channels,1:downsample:end)];
    end 
    concatinated_t = [concatinated_t, t(1:downsample:end)];
end

t = double(concatinated_t);
data_size = size(concatinated_channels);
for i = 1 : data_size(1)
    y1  = double(concatinated_channels(i,:));
    savefile = [save_to_path, dir_to_rhs(25:end-1) , ...
        amplifier_channels(i).native_channel_name , '_second.mat'];
    save(savefile, 'y1', 't')
    disp(['file saved as, ', savefile])
end
% plot(t, amplifier_data)