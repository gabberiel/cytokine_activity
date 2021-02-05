function [y, zisout] = comb_band_stop(x, fs, Q, fline, upto, oddonly, zisin)
    % apply a comb band stop filter as described in 
    % http://www.biopac.com/wp-content/uploads/acqknowledge-4-software-guide.pdf
    % x     - input signal
    % fs    - sampling frequency in Hz
    % Q     - notch center frequency / bandwidth
    % fline - fundumental line freequency
    % upto  - maximum frequency of harmonics to notch
    % oddonly - either a vector of the orders of the harmonics to filter or
    %           a boolean indicating whether or not to filter all harmonics 
    %           or only odd harmonics. If a vector upto is not used.
    % zisin - initial state of second order IIR filter for each harmonic
    %
    % y     - filtered signal
    % zisout- final state of second order IIR filter for each harmonic
    
    % set default values
    if nargin < 3
        Q = 5;
    end
    if nargin < 4
        fline = 60;
    end
    if nargin < 5
        upto = fs / 2;
    end
    if nargin < 6
       oddonly = true; 
    end
    % if ~isscalar(oddonly)
        % Here oddonly is a vector of the order of the harmonics to filter
        fc = fline * oddonly;
    % elseif oddonly
    %     fc = fline:2 * fline:upto;
    % else
    %     fc = fline:fline:upto;
    % end
    % modfc = mod(fc, fs);
    % fc = sort([modfc(modfc < fs / 2), fs - modfc(modfc > fs / 2)]);    
    if nargin < 7
        zisin = zeros(length(fc), 2);
    end
    zisout = zeros(size(zisin));
    
    % set the output to the input
    y = x;    
    % for each line frequency harmonic
    for ii = 1:length(fc)
        [y, zisout(ii, :)] = band_stop(y, fc(ii), Q, fs, zisin(ii, :));
%         fnotch = fc(ii);
%         % get the bandwidth
%         bw = fnotch / Q;
%         % set the radius of the poles to achieve the desired bandwidth
%         r = 1 - bw / fs;
%         % compute the normalized notch frequency
%         theta = 2 * pi * fnotch / fs;
%         % filter the data
%         b = [1, -2*cos(theta), 1];
%         a = [1, -2*r*cos(theta), r*r];        
%         [y, zisout(ii, :)] = filter(b, a, y, zisin(ii, :));        
    end

end
