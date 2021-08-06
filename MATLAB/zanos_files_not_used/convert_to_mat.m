function convert_to_mat(filetoopen)
% viktigt filetoopen ska vara en char array

% H�r om vandlar vi plx filer till mat filer utan bearbetning

% Channesls som har n�mnts �WB01�, �WB09� and �KBD1"

% Note use plx_ad_v to get the data in mV
vagus_channel = 'WB01';
cardiac_channel = 'WB09';
%Zanos hela bunten
[adfreq1, ~, ts1, fn1, vagus_milli_voltage] = plx_ad_v(filetoopen, vagus_channel);
[adfreq2, ~, ts2, fn2, cardiac_milli_voltage] = plx_ad_v(filetoopen, cardiac_channel);

% Det h�r ska vara n�gon form av event data
% men vad betyder det och hur anv�nder man det ?
%[KBD1_n, KBD1_ts, KBD1_sv] = plx_event_ts(filetoopen, 'KBD1');


% rad kopierad fr�m Zanos
% farligt d� vi bara har ett hum om
% vad raden g�r
disp("this is fn1")
disp(fn1)
vagus_times=ts1(1)+[0:fn1(1)-1]*(1/adfreq1);
cardiac_times= ts2(1)+[0:fn2(1)-1]*(1/adfreq2);


% Tensor dimension makar inte riktigt sense att kalla det
filename_dimensions = size(filetoopen);
% plockar ut namnet utan .plx
last_name_char_index = filename_dimensions(2)-4;
% l�gger till r�tt fil�ndelse till char array namnet 
save_file =[filetoopen(1:last_name_char_index) '.mat'];
%cd ..\mat_filer;
%cd ..;
save(save_file, 'vagus_milli_voltage', 'vagus_times','cardiac_milli_voltage','cardiac_times')
%cd ..\PLX_filer;
%cd PLX_filer;
end
