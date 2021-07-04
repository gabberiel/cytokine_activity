function importplxfiles(filetoopen,save_to_path)
% import plx 1
% example file to open .plx file and read “WB01”, “WB09” and “KBD1”
% **Note that the Plexon "Matlab Offline Files SDK" must be in the MATLAB path**

% 1. use the MATLAB uigetfile to select one or more .plx files for analysis
% 2. use the Plexon function to read the continuous data from the file
% 3. use the Plexon function to read the event data from the file
% 4. plot the data

%--------------------------------------------------------------
% 1. use MATLAB function uigetfile to select one or more files to analyze
% all of the files selected must be in the same location on the disk
% the names of the files that are selected get stored in filestoopen
% if ~exist('filestoopen', 'var') || any(arrayfun(@(x) ~exist(x, 'file'), filestoopen))
%     [filestoopen, mypath, filterindex] = uigetfile('*.plx','Select a .plx file','MultiSelect', 'on');
%     if isa(filestoopen, 'char')
%         filestoopen = {filestoopen};        
%     end
%     filestoopen = cellfun(@(x) [mypath, x], filestoopen, 'UniformOutput', false);
% end

% make the selected file path the current location
% cd(mypath);
% eval(['cd ''',mypath,'''']);

% check to see if multiple files were selected
% when multiple files are selected uigetfile returns a cell type variable

Nd=1;
%Gabe edit: should be read automatically..? do not see it
% nfiles = 1
% %filetoopen=filestoopen;
% mypath = ''
% % loop through the files
% for filenumber=1:nfiles
%     if nfiles>1
%         filetoopen=[mypath,cell2mat(filestoopen(filenumber))];
%     end;

%--------------------------------------------------------------    
% 2. read the A/D data from the plx file
% Note use plx_ad_v to get the data in mV
% the corresponding function for pl2 files is PL2Ad
% the function plx_ad returns the data in raw ADC units
% that you then have to scale manually

% note: adfreq, etc could potentially be different for different channels
% e.g. for field potential channels vs wideband channels
disp(filetoopen)
disp(class(filetoopen))
[adfreq, n, ts, fn, y1] = plx_ad_v(filetoopen, 'WB01');
haswb09 = true;
if haswb09
    [adfreq, n, ts, fn, y2] = plx_ad_v(filetoopen, 'WB09');
end


%y1=decimate(y1, Nd);
%y2=decimate(y2, Nd);

% A/D data is stored in the file in mV
% Optional: rescale the A/D data into volts instead of millivolts
%WB01_ad = WB01_ad*1e-3;
%WB09_ad = WB09_ad*1e-3;

%--------------------------------------------------------------
% 3. Load the event data from the file
[KBD1_n, KBD1_ts, KBD1_sv] = plx_event_ts(filetoopen, 'KBD1');

%--------------------------------------------------------------
% 4. plot the A/D data from one channel

% the A/D data starts at time ts and contains fn points sampled at
% adfreq
% if ts and fn are vectors, then the recording was paused and resumed
% each pair of ts and fn corresponds to a period of recording
% called a fragment

% time (of fragment 1)
t=ts(1)+[0:fn(1)-1]*(1/adfreq);
%t=decimate(t,Nd);
%plot(t,y1,'g');
%hold on;
%plot(t,y2,'r');

% plot the event data as vertical lines from top to bottom of AD graph
% Note: if KBD1_ts is a vector, this will plot all of the events
%a=axis;
%plot([KBD1_ts,KBD1_ts],[a(3),a(4)],'k:');

% label the axes
%xlabel('Time (s)');
%ylabel('Amplitude (mV)');

%-------------------------------------------------------
%Decimate and filter the signal to save

% TODO do I want to decimate the 40 kHz signal by a factor of 5? What is the passband of the analog filter?
y1dec=decimate(y1, Nd);
if haswb09
    y2dec=decimate(y2, Nd);
end
 % y2dec=decimate(S.y2, Nd);
 % y3dec=decimate(S.y3, Nd);-`
 % y4dec=decimate(S.y4, Nd);
 % y5dec=decimate(S.y5, Nd);
 % y6dec=decimate(S.y6, Nd);
 % y7dec=decimate(S.y7, Nd);
 % y8dec=decimate(S.y8, Nd);
  t = t(1:Nd:end);
%  [b, a]=butter(2,(120/((adfreq/Nd)/2)), 'high');
% y1=filter(b,a,y1dec);
y1 = y1dec;
if haswb09
    % y2=filter(b,a,y2dec);
    y2 = y2dec;
end
%y2=filter(b,a,y2dec);
%y3=filter(b,a,y3dec);
%y4=filter(b,a,y4dec);
%y5=filter(b,a,y5dec);
%y6=filter(b,a,y6dec);
%y7=filter(b,a,y7dec);
%y8=filter(b,a,y8dec);
  

    % [~, filename_, ~] = fileparts(filetoopen);savefile = strcat(filename_, '.mat');
    % savefile = strcat('R10_batch3_',num2str(filenumber),'dec');
    % savefile = 'R10_batch1_4.mat';
    
    % savefile = strcat('R6_IL1RKO__IL1B--TNF_',num2str(filenumber),'');  % not decimated
    % savefile = strcat('R6_IL1RKO__IL1B--TNF_',num2str(filenumber),'dec');
    % savefile = strcat('R6_IL1RKO_TNF--IL1B_',num2str(filenumber),'dec');
    % savefile = strcat('R6_TNFRKO_IL1B--TNF_',num2str(filenumber),'dec');
    % savefile = strcat('R6_TNFRKO_TNF--IL1B_',num2str(filenumber),'dec');
    % savefile = strcat('R6_TNFR1&R2KO_IL1B_',num2str(filenumber),'dec');
    % savefile = 'Cortec_100_RPeroneal_HMGB1_20ugperkg_3_6_2015_dec.mat';
    plx_path = '../../Google Drive/Data/PLX_filer/'
    savefile = [save_to_path,filetoopen(length(plx_path):end-4) '.mat'];
    
    if haswb09
        save(savefile, 'y1', 'y2', 't');
    else        
        save(savefile, 'y1', 't');
    end

%
%save(strcat('MetabolicR1_',num2str(1+(3*(filenumber-1))),'orig'), 'y1', 'y2', 't');
%save(strcat('R7_multi_',num2str(filenumber),'orig'), 'y1', 'y2', 't');
%end; % loop through files (if applicable)



