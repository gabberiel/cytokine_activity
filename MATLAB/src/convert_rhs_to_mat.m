function [concatinated_channels, concatinated_t] = convert_rhs_to_mat ... 
                                                        (dir_to_rhs, save_to_path, downsample)
% Opens all .rhs files in the specifed directory "dir_to_rhs" containing files which
% are each e.g. 1 min-parts of full recording.
% The data are concatinated to get full recording-time and one file for
% each possible channel is saved, as well as a timestamp for each.


% Get all .rhs files in specified directory:
files = dir([dir_to_rhs, '/*.rhs']);
concatinated_channels = [];
concatinated_t = [];
for file = files'
    file_path = [dir_to_rhs, file.name]
    [amplifier_channels, amplifier_data, t] = read_Intan_RHS2000_file(file_path);
    concatinated_channels = [concatinated_channels, amplifier_data(:,1:downsample:end)];
    concatinated_t = [concatinated_t, t(1:downsample:end)];
end

t = double(concatinated_t);
data_size = size(concatinated_channels);
for i = 1 : data_size(1)
    y1  = double(concatinated_channels(i,:));
    savefile = [save_to_path, dir_to_rhs(25:end-1) , ...
        amplifier_channels(i).native_channel_name , '.mat'];
    save(savefile, 'y1', 't')
    disp(['file saved as, ', savefile])
end
% plot(t, amplifier_data)