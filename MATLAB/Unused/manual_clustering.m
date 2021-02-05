function [h] = manual_clustering(x, y, t, X)
% x - Nx1 x-coordinate for scatter plot
% y - Nx1 y-coordinate for scatter plot
% t - Nx1 time coordinate for scatter plot
% X - NxM waveform array where N
%
% h = manual_clustering(Ytr(:, 1), Ytr(:, 2), timestamps(traininds)/60, X(traininds, :));

% TODO add event rates (requires test set too)

% create a scatter plot
h = figure;
markersize = 5;
s = scatter3(x, y, t, markersize, zeros(length(x), 1), 'filled');
view([0, 90]);
% store the handle to the scatter plot in the figure's appdata
setappdata(h, 'curscatter', s);
setappdata(h, 'n', 1);
setappdata(h, 'mycolormap', [0, 0, 0]);
setappdata(h, 'X', X);

% set the callback functions
set(h, 'WindowButtonDownFcn', {@wbd});
set(h, 'WindowButtonMotionFcn', {@wbm});
set(h, 'WindowButtonUpFcn', {@wbu});
set(h, 'DeleteFcn', {@wclose});

function wbd(h, evd)
% window button down function callback
% set the down state to true
setappdata(h, 'down', true);
% get the first selection coordinate
ax = get(h, 'CurrentAxes');
cp = get(ax, 'CurrentPoint');
x = cp(1, 1);
y = cp(1, 2);
setappdata(h, 'coords', [x y]);
% create a selection plot
hold on;pl = plot(x, y);hold off;
setappdata(h, 'curplot', pl);


function wbm(h, evd)
% window button motion function callback
if getappdata(h, 'down')
    % if mouse button is clicked get subsequent selection coordinates
    ax = get(h, 'CurrentAxes');
    cp = get(ax, 'CurrentPoint');
    x = cp(1, 1);
    y = cp(1, 2);
    coords = getappdata(h, 'coords');
    setappdata(h, 'coords', [coords;x y]);
    % continue to plot the selection
    curplot = getappdata(h, 'curplot');
    set(curplot, 'XData', coords(:, 1), 'YData', coords(:, 2));
end


function wbu(h, evd)
% window button up function callback
% set the down state to false
setappdata(h, 'down', false);
% close the selection and complete the plot
curplot = getappdata(h, 'curplot');
coords = getappdata(h, 'coords');
set(curplot, 'XData', coords([1:end 1], 1), 'YData', coords([1:end 1], 2));
% reset the selection coordinates
setappdata(h, 'coords', []);
% determine the indices of the points that are inside the selection
s = getappdata(h, 'curscatter');
n = getappdata(h, 'n');
xq = get(s, 'XData');
yq = get(s, 'YData');
xv = coords(:, 1);
yv = coords(:, 2);
in = inpolygon(xq, yq, xv, yv);
% assign the indices to the base workspace where the variable X resides
% assignin('base', 'in', in);
% set the color of the line to the datapoints on the scatter plot
cdata = get(s, 'CData');
col1 = n;
col = get(curplot, 'Color');
setappdata(h, 'n', n + 1);
mycolormap = getappdata(h, 'mycolormap');
mycolormap = [mycolormap; col];
setappdata(h, 'mycolormap', mycolormap);
% assignin('base', 'col', col);
cdata(in) = col1;
set(s, 'CData', cdata);
colormap(mycolormap);
% plot the waveforms of the selected points in the base workspace
disp('manual_clustering using testinds');

X = getappdata(h, 'X');
figure(99);hold on;plot(X(in, :)', 'Color', col);hold off
% evalin('base', 'figure(99);hold on;plot(X(traininds(in), :)'', ''Color'', col);hold off');


function wclose(h, evd)
% disable callbacks prior to deleting the figure
set(h, 'WindowButtonDownFcn', '');
set(h, 'WindowButtonMotionFcn', '');
set(h, 'WindowButtonUpFcn', '');
close(h);
