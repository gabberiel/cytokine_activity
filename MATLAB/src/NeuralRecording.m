classdef NeuralRecording < handle
    properties
        datafile    % data file that contains the raw data
        data        % timeseries object
        datadec     % Downsampled timeseries object
        dt          % delta time (s)
        dtdec       % Downsampled dt
        sr          % sampling rate (Hz)
        srdec
        sig         % cwt to emphasize CAPs
        cardiac     % cwt to emphasize cardiac
        nonrespinds % logical mask of nonrespiratory indices
        waveforms   % Waveforms object
        resp
        threshold   % timeseries object that stores thresholds for neurally enhanced signal (using cwt)
        threshold_cardiac   % timeseries object that stores thresholds for cadiac enhanced signal (using cwt)

        Nd          % Downsample parameter
        powerSpectrum
        motionArtifactRemove
        results
        
        % detector parameters
        num_std
        usewavelets
        usedownsampled
        wdur
        gdur
        alignmentmode
        
        userdata
    end
    
    properties (Dependent)
        defaultktsnefile
        defaultoutfile
    end
    
    methods
        function obj = NeuralRecording(datafile, gain, chan)
            % Constructor
            if ~exist('datafile', 'var') || ~exist(datafile, 'file')
                % open file dialog
                filterspec = {'*.mat', 'mat files (*.mat)'; ...
                    '*.plx', 'plexon files (*.plx)'; ...
                    '*.*', 'all files (*.*)'};
                defaultname = 'Z:\R1\Insulin\Mouse2.mat';
                [filename, pathname, filterindex] = uigetfile(filterspec, 'Select a data file', defaultname);
                if isequal(filename, 0)
                    obj.datafile = '';
                    obj.data = timeseries();
                    obj.dt = 0;
                    obj.sr = 0;
                    return
                elseif filterindex == 2
                    importplxfiles
                else
                    datafile = [pathname filename];
                end
            end
            
            if ~exist('gain', 'var') || isempty(gain)
                % The gain is often 50 for plexon files
                gain = 1;
            end
            
            obj.datafile = datafile;
            h = matfile(obj.datafile);            
            if ~exist('chan', 'var') || isempty(chan)
                % uiimport  % This isn't exactly what I want so I'll have
                % to make my own listbox selection gui
                if exist('listboxgui.fig', 'file') && exist('listboxgui.m', 'file')
                    matvars = properties(h);
                    chanNames = matvars(cellfun(@(x) x(1) == 'y' && ~strcmp(x, 'y2'), matvars));
                    % open a listbox GUI to select the channels
                    [hfig, hlistbox] = listboxgui('ChannelNames', chanNames);
                    % wait for the user to click a button which will call uiresume
                    uiwait(hfig);
                    if isvalid(hfig)
                        % The user clicked OK or cancel. Get the channel selection
                        allchans = get(hlistbox, 'String');
                        inds = get(hlistbox, 'Value');
                        chan = allchans(inds);
                        accepted = get(hfig, 'UserData');
                        % close the listbox GUI
                        close(hfig);
                    else
                        % The user closed the listbox GUI
                        accepted = false;
                    end
                    if ~accepted || isempty(inds)
                        % User canceled
                        return
                    end
                else
                    chan = {'y1'};
                end
            end
            
            % check if y1 is in the mat file
            if iscell(chan)
                fprintf(['loading ' repmat('%s, ', [1, length(chan)]) '\b\b...\n'], chan{:});
                temp = load(obj.datafile, chan{:});
                y1 = struct2array(temp);
                if isa(y1, 'single')
                    y1 = double(y1);
                end
            elseif isprop(h, chan)
                fprintf('loading %s...\n', chan);
                temp = load(obj.datafile, chan);
                y1 = temp.(chan);
                if isa(y1, 'single')
                    y1 = double(y1);
                end
            elseif isprop(h, 'y')
                fprintf('loading y...\n');
                load(obj.datafile, 'y');
                y1 = y;
                clear('y');
            end
            
            % ensure y1 contains column vectors
            if size(y1, 1) < size(y1, 2)
                y1 = y1';
            end
            
            % check if there is a cardiac channel y2 in the mat file
            if isprop(h, 'y2')
                fprintf('loading y2...\n');
                load(obj.datafile, 'y2');
                if isa(y2, 'single')
                    y2 = double(y2);
                end
                if length(y1) == length(y2)
                    y = [y1 reshape(y2, [], 1)];
                elseif length(y1) < length(y2)
                    y = [y1 reshape(y2(1:length(y1)), [], 1)];
                elseif ~isempty(y2)
                    y = [reshape(y1(1:length(y2), :), [], size(y1, 2)) reshape(y2, [], 1)];
                else
                    y = y1;
                end
            else
                y = y1;
            end
            % check how time vector is stored in the mat file
            fprintf('loading t...\n');
            if isprop(h, 't')
                load(obj.datafile, 't');
                %                 % adjust the time vector if needed
                %                 if length(t) < size(y, 1)
                %                     t0 = t(1);
                %                     dt = t(2) - t(1);
                %                     t = t0 + (0:length(y1)-1) * dt;
                %                 elseif length(t) > size(y, 1)
                %                     t = t(1:length(y1));
                %                 end
                t0 = t(1);
                dt = t(2) - t(1);
                t = t0 + (0:length(y1)-1) * dt;
            elseif isprop(h, 't0') && isprop(h, 'dt')
                load(obj.datafile, 't0', 'dt');
                t = t0 + (0:size(y, 1)-1) * dt;
            end
            t = reshape(t, [], 1);
            
            % form the time series
            fprintf('forming timeseries...\n');
            obj.data = timeseries(y * gain, t);
            obj.data.DataInfo.Units = '\muV';
            obj.data.TimeInfo.Units = 's';
            obj.data.Name = 'Voltage';
            obj.dt = obj.data.Time(2) - obj.data.Time(1);
            obj.sr = 1 / obj.dt;
        end
        
        function apply_gain(obj, k)
            obj.data.Data = obj.data.Data * k;
            if ~isempty(obj.datadec)
                obj.datadec.Data = obj.datadec.Data * k;
            end
        end
        
        function hpf(obj, fc)
            if ~exist('fc', 'var') || isempty(fc)
                fc = 10;
            end
            [b, a] = butter(2, fc/(obj.sr/2), 'high');
            obj.data.Data = filtfilt(b, a, double(obj.data.Data));
        end
        
        function value = get.defaultktsnefile(obj)
            [pathstr, name, ext] = fileparts(obj.datafile);
            value = [pathstr filesep name '_ktsne' ext];
        end
        
        function value = get.defaultoutfile(obj)
            [pathstr, name, ext] = fileparts(obj.datafile);
            value = [pathstr filesep 'outfile_' name ext];
        end
        
        function detect_respiratory(obj, nwind, minpeakdist, minpeakprom, minpeakwidth, maxpeakwidth, k, usedownsampled)
            % Detect respiratory intervals
            % nwind        - number of samples in the moving window
            % minpeakdist  - based on respiratory rate
            % minpeakprom  - sensitivity / specificity trade-off
            % minpeakwidth - minpeakwidth should be slightly larger than nwind to avoid detecting
            % maxpeakwidth - avoid very wide peaks like that which would occur in batch 1 sid 7
            % k            - scaling of width
            if ~exist('nwind', 'var') || isempty(nwind)
                nwind = 0.1;           % number of seconds in the moving window
            end
            if ~exist('minpeakwidth', 'var') || isempty(minpeakwidth)
                minpeakwidth = 0.105;  % minpeakwidth should be slightly larger than nwind to avoid detecting
            end
            if ~exist('maxpeakwidth', 'var') || isempty(maxpeakwidth)
                maxpeakwidth = 1.2;    % avoid very wide peaks like that which would occur in batch 1 sid 7
            end
            if ~exist('minpeakprom', 'var') || isempty(minpeakprom)
                minpeakprom = 0.35;         % sensitivity / specificity trade-off
            end
            if ~exist('minpeakdist', 'var') || isempty(minpeakdist)
                minpeakdist = 0.35;     % based on respiratory rate
            end
            if ~exist('k', 'var') || isempty(k)
                k = 1;     % scaling of width
            end
            if ~exist('usedownsampled', 'var') || isempty(usedownsampled)
                usedownsampled = true;     % based on respiratory rate
            end
            
            if usedownsampled
                obj.resp = RespiratoryBurst(obj.datadec, nwind, minpeakdist, minpeakprom, minpeakwidth, maxpeakwidth, k);
            else
                obj.resp = RespiratoryBurst(obj.data, nwind, minpeakdist, minpeakprom, minpeakwidth, maxpeakwidth, k);
            end
        end
        
        function downsample(obj, Nd)
            % Downsample the the data
            % some operations should not be performed at the full sampling
            % rate and repeatedly downsampling the data is inefficient
            fprintf('downsampling...\n');
            obj.Nd = Nd;
            t2 = obj.data.time(1:Nd:end);
            ydec = resample(obj.data.Data, 1, Nd);
            obj.datadec = timeseries(ydec, t2);
            obj.dtdec = obj.dt * Nd;
            obj.srdec = obj.sr / Nd;
        end
        
        function adaptive_notch_filter(obj, fHz, Q, windowlength)
            obj.sr = round(obj.sr);
            % Apply the adaptive notch filter to the data
            if ~exist('fHz', 'var') || isempty(fHz)
                fHz = 60;  % fundamental frequency of power line interference
            end
            if ~exist('Q', 'var') || isempty(Q)
                Q = 1000;  % width of notch filter (bandwidth / center frequency)
            end
            if ~exist('windowlength', 'var') || isempty(windowlength)
                windowlength = 5;  % seconds
            end

            for chan = 1  % :size(obj.data.Data, 2)
                obj.data.Data(:, chan) = test_comb_filter_on_data(obj.sr, obj.data.Data(:, chan), ...
                                          obj.sr / 2, windowlength, fHz, Q);
            end

        end
        
        function notch_filter(obj, f0s, qs, plotflag)
            % Fixed notch filters
            if ~exist('plotflag', 'var') || isempty(plotflag)
                plotflag = false;
            end
            
            assert(length(f0s) == length(qs));
            for ii = 1:length(f0s)
                f0 = f0s(ii);
                q = qs(ii);
                w0 = f0 / (obj.sr / 2);
                [b, a] = iirnotch(w0, w0/q);
                obj.data.Data = filter(b, a, obj.data.Data);
            end
            
            if plotflag
                % TODO are these variables defined?
                f = linspace(0, obj.sr, length(y1)+1);f = f(1:end-1);
                figure;reduce_plot(f, 20*log10(abs(fft([y1 yf]))));xlim([0 150])
            end
        end
        
        function detect_cardiac(obj, usewavelets)
            % detect cardiac artifacts and exclude as CAP detections
            if ~exist('usewavelets', 'var') || isempty(usewavelets)
                usewavelets = true;
            end
            
            if usewavelets
                obj.cardiac = obj.cap_wavelet();
            else
                obj.cardiac = obj.data(:, 2);
            end
            % TODO threshold
        end
        
        function [sig] = cap_wavelet(obj)
            % compute a wavelet transform to emphasize CAPs
            sig = cwt(obj.data, 0.001 / obj.dt, 'db3');
        end
        
        function [cardiac] = cardia_wavelet(obj)
            % compute a wavelet transform to emphasize cardiac artifacts
            cardiac = cwt(obj.data, 0.005 / obj.dt, 'db3');       
        end
        %########################################################################
        %------------------------------------------------------------------------
        %----------- ADAPTIVE THRESHOLD -----------------------------------------
        %------------------------------------------------------------------------
        %------------------------------------------------------------------------
        function adaptive_threshold(obj, usewavelets, num_std, wdur, gdur, usedownsampled, alignmentmode, threshType, usetruncated)
            fprintf('Initiating Adaptive threshold...\n');
            % will create an AdaptiveThreshold object
            if ~exist('usewavelets', 'var') || isempty(usewavelets)
                usewavelets = false;  %
                fprintf('OBS: usewavelets set to false..!\n');
            end
            if ~exist('num_std', 'var') || isempty(num_std)
                num_std = 3;  %
            end
            if ~exist('wdur', 'var') || isempty(wdur)
                wdur = 1501 / 8000;  %
            end
            if ~exist('gdur', 'var') || isempty(gdur)
                gdur = 10 / 8000;  %
            end
            if ~exist('alignmentmode', 'var') || isempty(alignmentmode)
                alignmentmode = 'maxabs';
            end
            if ~exist('usetruncated', 'var') || isempty(usetruncated)
                usetruncated = false;
                fprintf('OBS: usetruncated set to false..!\n');
            end
            assert(ismember(alignmentmode, {'maxabs', 'max', 'min', 'midzc'}), ...
                'alignmentmode = %s is not supported\n', alignmentmode);
            
            if ~exist('usedownsampled', 'var') || isempty(usedownsampled)
                usedownsampled = (obj.sr > 8000);
                fprintf(['OBS: usewavelets set to: ',num2str((obj.sr > 8000)), '..! 1=true... \n']);
            end
            
            if ~exist('threshType', 'var') || isempty(threshType)
                threshType = 4;  % adaptive threshold using SO-CFAR
                % threshType = 8;  % slowly adapting threshold
            end
            
            if usedownsampled && isempty(obj.Nd)
                Nd = max(1, round(obj.sr / 8000));  %#ok
                obj.downsample(Nd);         %#ok
            end
            
            if usedownsampled && ~usetruncated
                if size(obj.datadec.Data, 2) == 1
                    y1 = obj.datadec.Data;
                    y2 = [];
                else
                    y1 = obj.datadec.Data(:, 1:end-1);
                    y2 = obj.datadec.Data(:, end);
                end
                t = obj.datadec.Time;
            
            elseif usedownsampled && usetruncated
                y1 = obj.motionArtifactRemove.y;
                y2 = [];
                t = obj.datadec.Time;
                
            else
                disp('OBS, Not using dowsampled data to adaptive_threshold..')
                if size(obj.data.Data, 2) == 1
                    y1 = obj.data.Data;
                    y2 = [];
                else
                    y1 = obj.data.Data(:, 1:end-1);
                    y2 = obj.data.Data(:, end);
                end
                t = obj.data.Time;
            end
            
            fprintf('sig_decomp_new...\n');

            [timestamps, waveforms_, obj.sig, obj.cardiac, thresh, threshc] = sig_decomp_new(t, y1, y2, threshType, num_std, usewavelets, wdur, gdur, alignmentmode);
            % if usewavelets then waveforms_ will be based on cwt.. i.e the
            % "waveforms_" are the cwt-convolved signal waveforms
            obj.threshold = timeseries(thresh, t);
            obj.threshold_cardiac = timeseries(threshc, t);
            if usedownsampled && ~usetruncated
                % upsample waveforms
                Npeak = ceil(1.75*.001/obj.dt);
                waveforms2 = zeros(length(timestamps), 2*Npeak + 1);
                timestamps2 = zeros(size(timestamps));
                inds = find(ismember(obj.data.Time, timestamps));
                y1 = obj.data.Data(:, 1);
                for idet = 1:length(timestamps)
                    inds_ = inds(idet) - obj.Nd:inds(idet) + obj.Nd;
                    if inds_(1) <= 0 || inds_(end) > length(y1)
                        continue
                    end
                    switch alignmentmode
                        case 'maxabs'
                            [~, mi] = max(abs(y1(inds_)));
                        case 'max'
                            [~, mi] = max(y1(inds_));
                        case 'min'
                            [~, mi] = min(y1(inds_));
                        case 'midzc'                            
                            [~, mi] = min(abs(y1(inds_)));
                    end
                    center = inds(idet) - obj.Nd + mi;
                    center = min(max(center, Npeak + 1), length(y1) - Npeak);
                    waveforms2(idet, :) = y1(center - Npeak:center + Npeak);
                    timestamps2(idet) = obj.data.Time(center);
                end
                % The Waveform-class inputs:
                % Waveforms(timestamps, X, sr) where,
                % X: N x K array of waveforms where N is the number of waveforms and K is the number of samples per waveform
                % sr: sampling rate
                
                % The stored waveforms are from the RAW-RECORDING, not
                % cwt-convolved... :
                disp('Setting nr.waveforms to be the neural-event-indices from raw signal..')
                obj.waveforms = Waveforms(timestamps2, waveforms2, obj.sr);
            else
                disp('OBS! OBS! The wavelet convolved signal is used as nr.waveforms..')
                obj.waveforms = Waveforms(timestamps, waveforms_, obj.srdec);
            end
            
            obj.usewavelets = usewavelets;
            obj.num_std = num_std;
            obj.wdur = wdur;
            obj.gdur = gdur;
            obj.usedownsampled = usedownsampled;
            obj.alignmentmode = alignmentmode;
        end
        %------------------------------------------------------------------------
        %------------------------------------------------------------------------
        function getRMS(obj)
            RMSvalue = rms(obj.motionArtifactRemove.y);
            obj.results.RMSvalue = RMSvalue;
            
        end
        %------------------------------------------------------------------------
        %------------------------------------------------------------------------
        function getPowerSpectrum(obj, visualize)
            y = obj.data.Data;
            t = obj.data.Time;
            [W,E] = ezfft(t,y);
            %convert W from phase to frequency 
            W=W/(2*pi);
            
            obj.powerSpectrum.energy = E;
            obj.powerSpectrum.frequency = W;
            
            if visualize
                plot(W,E);
                xlabel('Power Amplitude')
                ylabel('Time (seconds)')
            end
            
        end
        %------------------------------------------------------------------------
        %------------------------------------------------------------------------
        function truncateForMotionArtifact(obj, channelNum, trunc_voltage, savedir, usedownsampled, visualize)
            
            % Firstly, extract either the downsampled or full signal:
            if usedownsampled
                y = obj.datadec.Data;
                t = obj.datadec.Time;
            else
                y = obj.data.Data;
                t = obj.data.Time;
            end
            
                if visualize
                    figure;hold on;
                    plot(t,y);
                    plot(t, ones(1, length(t))*trunc_voltage, 'r')
                    plot(t, ones(1, length(t))*-trunc_voltage, 'r')
                    legend('neural recording', 'voltage cut off for motion artifact')
                    
                    %is this a good choice?
                    userin = input('Is this a good threshold? Y/N : ','s');
                    if strcmp(userin, 'N')
                        trunc_voltage = input('What is your new threshold choice?','s');
                    end
                    
                end
            %remove motion artifact: typically greater than 300 mV
            y(movmax(abs(y), round(30e3*.003)) > trunc_voltage) = 0 ;
            
            obj.motionArtifactRemove.y = y;
            obj.motionArtifactRemove.t = t;
%             
%             filename2=sprintf('channel_%d_trunc.mat',channelNum);
%             
%             pathname = fileparts(savedir);
%             matfile = fullfile(pathname, filename2);
%             save(matfile, 'y', 't');
        end
        %------------------------------------------------------------------------
        %------------------------------------------------------------------------
        function plot(obj, inds)
            % plot the time series
            
            if ~exist('reduce_plot', 'file')
                % add the path for reduce_plot
                foldername = uigetdir('C:\Users\NDDA1\Documents\Todd code\plotbig', 'select path for plotbig');
                if isequal(foldername, 0)
                    return
                else
                    addpath(foldername);
                end
                if ~exist('reduce_plot', 'file')
                    return
                end
            end
            % It's really slow if I pass the timeseries object to reduce_plot
            figure;
            N = size(obj.data.Data, 2);
            
            if ~exist('inds', 'var') || isempty(inds)
                if N == 1
                    inds = {1};
                else
                    inds = {1:N-1, N};
                end
            elseif ~iscell(inds)
                inds = {inds};
            end
            N1 = length(inds);
            
            ax = zeros(1, N1);
            for ii = 1:N1
                ax(ii) = subplot(N1, 1, ii);
                if ii == 1
                    title(sprintf('Time Series Plot: %s', obj.data.Name));
                end
                reduce_plot(obj.data.Time, obj.data.Data(:, inds{ii}));
                ylabel(sprintf('%s (%s)', obj.data.Name, obj.data.DataInfo.Units));
                if ii == N1
                    xlabel(sprintf('Time (%s)', obj.data.TimeInfo.Units));
                end
            end
            linkaxes(ax, 'x');
        end
        %------------------------------------------------------------------------
        %------------------------------------------------------------------------
        function [fig] = plot_waveform(obj, ind, fig)
            % Manually label a waveform as a detection or false alarm
            % Typically called from a loop
            % ind - the index of the waveform
            % fig - optional figure to use for plotting the waveforms
            if ~exist('fig', 'var') || isempty(fig)
                fig = uifigure();
                uibutton(fig, 'push', 'Position', [20 50 100 22], 'Text', 'Det', 'ButtonPushedFcn', @obj.buttoncb);
                uibutton(fig, 'push', 'Position', [20 25 100 22], 'Text', 'FA', 'ButtonPushedFcn', @obj.buttoncb);
                ax = uiaxes(fig, 'Position', [120 75 400 300]);
            else
                assert(isa(fig, 'matlab.ui.Figure'));
                temp = get(fig, 'Children');
                ax = temp(1);
            end
            
            obj.userdata = ind;
            
            Npeak = (size(obj.waveforms.X, 2)-1) / 2;
            tlocal = obj.waveforms.timestamps(ind) + (1/obj.sr)*(-Npeak:Npeak);
            tlocal2 = obj.waveforms.timestamps(ind) + (1/obj.sr)*(-15*Npeak:15*Npeak);
            center = find(ismember(obj.data.Time, obj.waveforms.timestamps(ind)));
            data2 = obj.data.Data(center + (-15*Npeak:15*Npeak), 1);
            threshinds = (obj.threshold.Time >= tlocal2(1) & obj.threshold.Time <= tlocal2(end));
            plot(ax, tlocal2-tlocal2(1), data2, '-', tlocal-tlocal2(1), obj.waveforms.X(ind, :), '-', ...
                obj.threshold.Time(threshinds)-tlocal2(1), obj.threshold.Data(threshinds, :), '-r');
            title(ax, ind);
            uiwait(fig);
        end
        %------------------------------------------------------------------------
        %------------------------------------------------------------------------
        function buttoncb(obj, eventdata, val)
            if isequal(val.Source.Text, 'Det')
                obj.waveforms.labels(obj.userdata) = 1;
            elseif isequal(val.Source.Text, 'FA')
                obj.waveforms.labels(obj.userdata) = -1;
            end
            uiresume(get(val.Source, 'Parent'));
        end
        %------------------------------------------------------------------------
        %------------------------------------------------------------------------
        function h = plot_caps(obj, usedownsampled, inds)
            % plot of colored CAPs by group superimposed on the original time series
            
            if ~exist('inds', 'var') || isempty(inds)
                inds = 1:length(obj.waveforms.timestamps);
            end
            
            if ~exist('usedownsampled', 'var') || isempty(usedownsampled)
                usedownsampled = (obj.sr > 8000);
            end
            
            if usedownsampled
                % downsample if necessary
                Nd_ = round(obj.sr / 8000);
                if isempty(obj.Nd)
                    obj.downsample(Nd_);
                end
                t2 = obj.datadec.Time;
                y1dec = obj.datadec.Data(:, 1);
            end
            
            if isempty(obj.waveforms.clust_timestamps)
                % all timestamps, not clustered
                if usedownsampled
                    % interpolate timestamps to downsampled times
                    clust_timestamps = cellfun(@(x) interp1(t2, t2, x, 'nearest'), {obj.waveforms.timestamps(inds)}, 'UniformOutput', false);
                    h = plot_caps(t2, y1dec, [], clust_timestamps, 1:length(t2));
                else
                    % not downsampled
                    h = plot_caps(obj.data.Time, obj.data.Data, [], ...
                        {obj.waveforms.timestamps}, 1:length(obj.data.Time));
                end
            else
                % clustered timestamps
                if usedownsampled
                    % interpolate timestamps to downsampled times
                    clust_timestamps = cellfun(@(x) interp1(t2, t2, x, 'nearest'), obj.waveforms.clust_timestamps, 'UniformOutput', false);
                    h = plot_caps(t2, y1dec, [], clust_timestamps, 1:length(t2));
                else
                    % not downsampled
                    h = plot_caps(obj.data.Time, obj.data.Data, [], ...
                        obj.waveforms.clust_timestamps, 1:length(obj.data.Time));
                end
            end
            
            if ~isempty(obj.threshold)
                hold on;
                %reduce_plot(obj.threshold.Time, obj.threshold.Data);
                hold off;
            end
        end
        %------------------------------------------------------------------------
        %------------------------------------------------------------------------
        % TODO should I change how I save files? Is there a better way to
        % do this? Most things are associated with the Waveforms class
        % except the detector parameters.
        function save_tsne_file(obj, ktsnefile)
            traininds = obj.waveforms.traininds;               %#ok
            testinds = setdiff(1:size(obj.waveforms.X, 1), obj.waveforms.traininds);
            Ytr = obj.waveforms.Y(obj.waveforms.traininds, :); %#ok
            Ytest = obj.waveforms.Y(testinds, :);              %#ok
            
            timestamps = obj.waveforms.timestamps;         %#ok
            X = obj.waveforms.X;
            t_local = (0:size(X, 2)-1) / obj.waveforms.sr; %#ok
            
            num_std = obj.num_std;             %#ok
            useWavelets = obj.usewavelets;     %#ok
            wdur = obj.wdur;                   %#ok
            gdur = obj.gdur;                   %#ok
            alignmentmode = obj.alignmentmode; %#ok
            
            if ~exist('ktsnefile', 'var') || isempty(ktsnefile)
                uisave({'Ytr', 'Ytest', 'traininds', 'testinds', 'timestamps', ...
                    'X', 't_local', 'num_std', 'useWavelets', 'wdur', 'gdur', 'alignmentmode'}, obj.defaultktsnefile);
            else
                save(ktsnefile, 'Ytr', 'Ytest', 'traininds', 'testinds', 'timestamps', ...
                    'X', 't_local', 'num_std', 'useWavelets', 'wdur', 'gdur', 'alignmentmode');
            end
        end
        %------------------------------------------------------------------------
        %------------------------------------------------------------------------
        
        function save_outfile(obj, outfile)
            clust_timestamps = obj.waveforms.clust_timestamps;  %#ok
            yf = obj.waveforms.yf;                              %#ok
            tf = obj.waveforms.tf;                              %#ok
            real_clusters = obj.waveforms.real_clusters;        %#ok
            epsilon = obj.waveforms.epsilon;                    %#ok
            minpts = obj.waveforms.minpts;                      %#ok
            k = obj.waveforms.k;                                %#ok
            powspreadthresh = obj.waveforms.powspreadthresh;    %#ok
            T = obj.waveforms.labels;                           %#ok
            X = obj.waveforms.X;                                %#ok
            Y = obj.waveforms.Y;                                %#ok
            traininds = obj.waveforms.traininds;                %#ok
            testinds = obj.waveforms.testinds;                  %#ok
            respind = [];  %#ok % no longer used. Used to represent the index of the respiratory cluster
            
            if ~exist('outfile', 'var') || isempty(outfile)
                uisave({'clust_timestamps', 'yf', 'tf', 'real_clusters', ...
                    'epsilon', 'minpts', 'k', 'powspreadthresh', 'T', 'X', 'Y', 'traininds', 'testinds', 'respind'}, obj.defaultoutfile);
            else
                save(outfile, 'clust_timestamps', 'yf', 'tf', 'real_clusters', ...
                    'epsilon', 'minpts', 'k', 'powspreadthresh', 'T', 'X', 'Y', 'traininds', 'testinds', 'respind');
            end
        end
        %------------------------------------------------------------------------
        %------------------------------------------------------------------------
        function ktsnefile = load_ktsne(obj, ktsnefile)
            if ~exist('ktsnefile', 'var') || isempty(ktsnefile)
                % TODO open file dialog
                [filename, pathname, filterindex] = uigetfile( ...
                    {'*tsne.mat', 't-SNE mat file (*tsne.mat)'; ...
                    '*.*', 'All files (*.*)'}, ...
                    'Select t-SNE File', 'Z:\R1\ktsne\');
                if isequal(pathname, 0)
                    ktsnefile = '';
                    return
                end
                ktsnefile = [pathname filename];
            end
            load(ktsnefile, 'X', 'traininds', 'Ytr', 'Ytest', 'timestamps', 't_local');
            if isempty(obj.waveforms)
                sr_ = 1 / (t_local(2) - t_local(1));
                obj.waveforms = Waveforms(timestamps, X, sr_);
            else
                obj.waveforms.timestamps = timestamps;
                obj.waveforms.X = X;
            end
            obj.waveforms.traininds = traininds;
            Y = zeros(size(X, 1), size(Ytr, 2));
            Y(obj.waveforms.traininds, :) = Ytr;
            Y(obj.waveforms.testinds, :) = Ytest;
            obj.waveforms.Y = Y;
            obj.waveforms.labels = zeros(size(X, 1), 1);
        end
        %------------------------------------------------------------------------
        %------------------------------------------------------------------------
        function outfile = load_outfile(obj, outfile)
            if ~exist('outfile', 'var') || isempty(outfile)
                % TODO open file dialog
                [filename, pathname, filterindex] = uigetfile( ...
                    {'outfile*.mat', 'outfile mat file (outfile*.mat)'; ...
                    '*.*', 'All files (*.*)'}, ...
                    'Select outfile', 'Z:\R1\outfiles\');
                if isequal(pathname, 0)
                    outfile = '';
                    return
                end
                outfile = [pathname filename];
            end
            load(outfile, 'clust_timestamps', 'yf', 'tf', 'real_clusters', ...
                    'epsilon', 'minpts', 'k', 'powspreadthresh', 'T', 'X', 'Y', 'traininds', 'testinds', 'respind');
            obj.waveforms.clust_timestamps = clust_timestamps;
            timestamps = sort(cell2mat(obj.waveforms.clust_timestamps'));
            if isempty(obj.waveforms)
                temp = mod((timestamps - timestamps(1)) * obj.srdec, 1);
                if any(temp < 0.99 & temp > 0.01)
                    sr_ = 1 / obj.sr;
                else
                    sr_ = 1 / obj.srdec;
                end
                obj.waveforms = Waveforms(timestamps, X, sr_);
            else
                obj.waveforms.timestamps = timestamps;
                if exist('X', 'var')
                    obj.waveforms.X = X;               
                end
            end
            obj.waveforms.tf = tf;
            obj.waveforms.yf = yf;
            obj.waveforms.real_clusters = real_clusters;
            obj.waveforms.epsilon = epsilon;
            if exist('minpts', 'var')
                obj.waveforms.minpts = minpts;
            end
            obj.waveforms.k = k;
            obj.waveforms.powspreadthresh = powspreadthresh;
            if exist('T', 'var')
                obj.waveforms.labels = T;
            elseif exist('clust_timestamps', 'var')
                obj.waveforms.labels = zeros(length(obj.waveforms.timestamps), 1);
                for ii = 1:length(clust_timestamps)
                    inds = ismember(obj.waveforms.timestamps, clust_timestamps{ii});
                    obj.waveforms.labels(inds) = ii;
                end
            end
            if exist('Y', 'var')
                obj.waveforms.Y = Y;
            end
            if exist('traininds', 'var')
                obj.waveforms.traininds = traininds;
            end
        end
    end
end