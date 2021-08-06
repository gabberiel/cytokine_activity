function [final_waveforms, final_timestamps] = ...
         multiple_channels_threshold(all_data, n_channel_thresh)
     % Used to consider/"make use off" all Channels in KI-data.
     %
     % Uses the preprocessed signals (waveforms and timestamps) from each
     % channel in multi-channel electrode to extract a final list of
     % waveforms and timestamps that was found to be neural events in at
     % least "n_channels_thresh" out of "n_total" number of channels.
     %
     % The following steps are applied:
     % 1. All unique timestamps are extracted using all the channels. --
     %    uniqueness is considered on the 1.75 ms-scale. i.e timestamps that
     %    are closer than two millisecond to each other are considered to be the
     %    same and duplicates are removed.
     % 2. For each unique timestamp,:
     %      - Find the number of channels that has a neural event with a 
     %      timestamp at +- 1ms from the unique timestamp.
     %      - If the number of channels with a neural event extracted at
     %      this time is larger than "n_channel_thresh", then this
     %      waveform-timestamp pair is consider to be a significant neural
     %      event and is saved in "final_waveforms".
     % 
     % args:
     % ----
     % all_data : struct of shape (n_channels, 1)
     %      fields: wf : all waveforms for channel.
     %              ts : all_timestamps for channel
     %
     % n_channel_thresh : integer
     %      Threshold value for number of channels that has to detect an
     %      event. 
     %
     % Most is done in for-loops, causing this code to be very slow. 
     % Different number of extracted CAPs for each channel makes a
     % vectorised approach difficult..
     
     n_channels = length(all_data);
     [~, dim_waveforms] = size(all_data(1).wf);
     all_t = [];
     for i = 1:n_channels
         all_t = [all_t; all_data(i).ts];
     end
     
     all_t = round(all_t, 3); % Round to ms scale.
     all_t = unique(all_t); % All  unique timestamps at ms scale.
     diff = all_t(1:end-1) - all_t(2:end);
     not_unique = abs(diff) < 1.75e-3;   % Timestamps closer than 1.75ms from eachother are to be removed
     all_unique_t = all_t(~not_unique);  % Save only unique timestamps
     clear ('all_t', 'diff', 'not_unique')  % Clear some space.
     
     final_waveforms = zeros(length(all_unique_t), dim_waveforms);
     final_timestamps = zeros(length(all_unique_t), 1);
     tic
     count = 0;
     for t_i = 1: length(all_unique_t)
         t = all_unique_t(t_i);
         chan_count_t = 0;
         prel_wf = [];
         for i = 1:n_channels
            % find number of channels that found a waveform (neural event) at
            % time t +- 1 ms. 
            time_diffs = all_data(i).ts - t;
            [val_of_closest_CAP, idx_of_closest_CAP] = min ( abs(time_diffs));
            if abs(val_of_closest_CAP) < 0.001
                chan_count_t = chan_count_t + 1;
                prel_wf = [prel_wf; all_data(i).wf(idx_of_closest_CAP, :)];
            end 
         end
         % If enought channel observed a CAP at time t, save it.
         if chan_count_t > n_channel_thresh
             count = count + 1;
             final_waveforms(count, :) =  mean(prel_wf, 1);
             final_timestamps(count, 1) = t;
         end

     end
     toc
     non_zero = final_timestamps > 0;
     final_waveforms = final_waveforms(non_zero,:);
     final_timestamps = final_timestamps(non_zero);
     
        