function [y, zi] = band_stop(y, fnotch, Q, fs, zi)
% get the bandwidth
bw = fnotch / Q;
% set the radius of the poles to achieve the desired bandwidth
r = 1 - bw / fs;
% compute the normalized notch frequency
theta = 2 * pi * fnotch / fs;
% filter the data
b = [1, -2*cos(theta), 1];
a = [1, -2*r*cos(theta), r*r];
[y, zi] = filter(b, a, y, zi);