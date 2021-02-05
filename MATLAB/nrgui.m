function varargout = nrgui(varargin)
% TODO enabled menu items
% TODO menu items for loading ktsne files and outfiles
% TODO implement clustering
% NRGUI MATLAB code for nrgui.fig
%      NRGUI, by itself, creates a new NRGUI or raises the existing
%      singleton*.
%
%      H = NRGUI returns the handle to a new NRGUI or the handle to
%      the existing singleton*.
%
%      NRGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NRGUI.M with the given input arguments.
%
%      NRGUI('Property','Value',...) creates a new NRGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before nrgui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to nrgui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help nrgui

% Last Modified by GUIDE v2.5 13-Dec-2017 11:25:51

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @nrgui_OpeningFcn, ...
                   'gui_OutputFcn',  @nrgui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before nrgui is made visible.
function nrgui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to nrgui (see VARARGIN)

% Choose default command line output for nrgui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes nrgui wait for user response (see UIRESUME)
% uiwait(handles.figure1);
enable_disable_all_controls(handles, 'off');
set(handles.menu_save, 'Enable', 'off');
set(handles.menu_load_tsne, 'Enable', 'off');
set(handles.menu_load_outfile, 'Enable', 'off');
set(handles.menu_close, 'Enable', 'off');


function enable_disable_all_controls(handles, val)
set(handles.pushbutton_reload, 'Enable', val);
set(handles.pushbutton_tsne, 'Enable', val);
set(handles.pushbutton_eventrates, 'Enable', val);
set(handles.pushbutton_ici, 'Enable', val);
set(handles.pushbutton_merge, 'Enable', val);
set(handles.pushbutton_cluster, 'Enable', val);
set(handles.pushbutton_timeseriescaps, 'Enable', val);
set(handles.checkbox_training, 'Enable', val);
set(handles.checkbox_tsnepower, 'Enable', val);
set(handles.popupmenu_cluster, 'Enable', val);
set(handles.listbox_clusters, 'Enable', val);
set(handles.edit_p1, 'Enable', val);
set(handles.text_p1, 'Enable', val);
set(handles.edit_p2, 'Enable', val);
set(handles.text_p2, 'Enable', val);
set(handles.edit_p3, 'Enable', val);
set(handles.text_p3, 'Enable', val);


% --- Outputs from this function are returned to the command line.
function varargout = nrgui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
if isfield(handles, 'nr')
    varargout{2} = handles.nr;
else
    varargout{2} = [];
end


% --------------------------------------------------------------------
function menu_file_Callback(hObject, eventdata, handles)
% hObject    handle to menu_file (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menu_open_Callback(hObject, eventdata, handles)
% hObject    handle to menu_open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% delete and previous NeuralRecording objects
% The old dataset will be gone even if this operation is cancelled
nrgui('menu_close_Callback', hObject, eventdata, handles);
handles = guidata(handles.figure1);  % handles may have been updated inside the callback

handles.nr = NeuralRecording();
if ~isempty(handles.nr.datafile)
    enable_disable_all_controls(handles, 'on');
    set(handles.menu_save, 'Enable', 'on');
    set(handles.menu_load_tsne, 'Enable', 'on');
    set(handles.menu_load_outfile, 'Enable', 'on');
    set(handles.menu_close, 'Enable', 'on');
    guidata(handles.figure1, handles);
    notify(handles.popupmenu_cluster, 'Action');
end

% --------------------------------------------------------------------
function menu_save_Callback(hObject, eventdata, handles)
% hObject    handle to menu_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menu_close_Callback(hObject, eventdata, handles)
% hObject    handle to menu_close (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isfield(handles, 'nr')    
    delete(handles.nr)
    enable_disable_all_controls(handles, 'off');
    
    set(handles.menu_open, 'Enable', 'on');
    set(handles.menu_quit, 'Enable', 'on');
    
    set(handles.menu_save, 'Enable', 'off');
    set(handles.menu_load_tsne, 'Enable', 'off');
    set(handles.menu_load_outfile, 'Enable', 'off');
    set(handles.menu_close, 'Enable', 'off');
end

% --------------------------------------------------------------------
function menu_quit_Callback(hObject, eventdata, handles)
% hObject    handle to menu_quit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(handles.figure1)

% --- Executes on selection change in listbox_clusters.
function listbox_clusters_Callback(hObject, eventdata, handles)
% hObject    handle to listbox_clusters (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox_clusters contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox_clusters
if strcmp(eventdata.EventName, 'Action')
    modify_tsne_scatter(handles)
    
    selectedclusters = get(hObject, 'Value');
    n = length(get(hObject, 'String'));
    if isfield(handles, 'htsne') && isvalid(handles.htsne)
        highlight_selected_clusters(selectedclusters, n, handles.htsne);
    end
    if isfield(handles, 'hici') && isvalid(handles.hici)
        highlight_selected_ici(selectedclusters, n, handles.hici);
    end
    if isfield(handles, 'heventrates') && isvalid(handles.heventrates)
        highlight_selected_event_rates(selectedclusters, n, handles.heventrates);
    end
    
    if length(selectedclusters) >= 1
        set(handles.pushbutton_cluster, 'Enable', 'on');
    else
        set(handles.pushbutton_cluster, 'Enable', 'off');
    end
    
    if length(selectedclusters) >= 2
        set(handles.pushbutton_merge, 'Enable', 'on');
    else
        set(handles.pushbutton_merge, 'Enable', 'off');
    end
end


function highlight_selected_clusters(clusterinds, n, htsne)
cmap = parula(n);
% coloroder = get(get(htsne, 'CurrentAxes'), 'ColorOrder');
% cmap = colororder(mod(0:n-1, size(colororder, 1))+1, :);

cmap(clusterinds, :) = repmat([0.8 1.0 0.0], [length(clusterinds) 1]);
colormap(get(htsne, 'CurrentAxes'), cmap);


function highlight_selected_ici(clusterinds, n, hici)
% TODO will the mapping always be the same? I think so.
% TODO set default colormap to parula
% TODO power isn't working after setting colormap to discrete values
cmap = parula(n);
mylines = findobj(hici, 'Type', 'Line');
myhists = findobj(hici, 'Type', 'Hist');
if length(mylines) ~= n
    return
end
arrayfun(@(x) set(mylines(n-x+1), 'Color', cmap(x, :)), 1:n);
arrayfun(@(x) set(myhists(n-x+1), 'FaceColor', cmap(x, :)), 1:n);
set(mylines(n-clusterinds+1), 'Color', [0.8 1.0 0.0]);
set(myhists(n-clusterinds+1), 'FaceColor', [0.8 1.0 0.0]);


function highlight_selected_event_rates(clusterinds, n, heventrates)
cmap = parula(n);
mylines = findobj(heventrates, 'Type', 'Line');
if length(mylines) ~= n
    return
end
arrayfun(@(x) set(mylines(n-x+1), 'Color', cmap(x, :)), 1:n);
set(mylines(n-clusterinds+1), 'Color', [0.8 1.0 0.0]);

% --- Executes during object creation, after setting all properties.
function listbox_clusters_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox_clusters (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_merge.
function pushbutton_merge_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_merge (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isfield(handles, 'nr') && isvalid(handles.nr) && ...
        isprop(handles.nr, 'waveforms') && ~isempty(handles.nr.waveforms) && isvalid(handles.nr.waveforms)
    clusterinds = get(handles.listbox_clusters, 'Value');
    if length(clusterinds) < 2
        return
    end
    
    plotflags1 = false;
    plotflags2 = false;    
    handles.nr.waveforms.merge_clusters(clusterinds, plotflags1, plotflags2);
    [~, ~, ic] = unique(handles.nr.waveforms.labels);
    handles.nr.waveforms.labels = ic - 1 + min(handles.nr.waveforms.labels);
    
    % update the plots
    modify_tsne_scatter(handles);  % checks for validity within
    handles = guidata(handles.figure1);
    modify_ici(handles);
    handles = guidata(handles.figure1);
    modify_event_rates(handles);
    handles = guidata(handles.figure1);
    
    populate_listbox(handles.listbox_clusters, handles.nr, min(clusterinds));
    
    % TODO clusters names were preserved, colorbar still goes to old max
    % value, several new plots were formed
end

% --- Executes on button press in pushbutton_eventrates.
function pushbutton_eventrates_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_eventrates (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

nrgui('pushbutton_ici_Callback', hObject, eventdata, handles);
handles = guidata(handles.figure1);  % handles may have been updated inside the callback
if ~isfield(handles, 'heventrates') || ~isvalid(handles.heventrates)
    % Visualization
    N = 30;  % moving average window length
    handles.heventrates = handles.nr.waveforms.plot_event_rates([], N);
    guidata(handles.figure1, handles);
end

% set focus to the event rate figure
figure(handles.heventrates);
populate_listbox(handles.listbox_clusters, handles.nr);


% --- Executes on button press in pushbutton_ici.
function pushbutton_ici_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_ici (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
nr = handles.nr;

if ~isprop(nr, 'waveforms') || isempty(nr.waveforms)
    % prompt the user to load the t-SNE file
    % TODO or run t-SNE?
    ktsnefile = nr.load_ktsne();
    if isempty(ktsnefile)
        return
    end
    handles.ktsnefile = ktsnefile;
    guidata(handles.figure1, handles);
    
    set(handles.listbox_clusters, 'Value', []);
    button = questdlg('Load an outfile?', 'Load Outfile', 'Yes', 'No', 'No');
    if strcmp(button, 'Yes')
        outfile = nr.load_outfile();
        if ~isempty(outfile)
            handles.outfile = outfile;
            guidata(handles.figure1, handles);
            if isfield(handles, 'htsne') && isvalid(handles.htsne)
                modify_tsne_scatter(handles)
            end
        end
    end
end
   
% TODO Why does hici say it's deleted?
plotflag2 = ~isfield(handles, 'hici') || isempty(handles.hici) || ~isvalid(handles.hici);
if plotflag2
    [~, handles.hici] = nr.waveforms.event_rates(false, plotflag2);
    guidata(handles.figure1, handles);
else
    nr.waveforms.event_rates(false, plotflag2);
    % set focus to the ICI figure
    figure(handles.hici);
end


% --- Executes on button press in pushbutton_tsne.
function pushbutton_tsne_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_tsne (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% TODO t-SNE plot came up as a single color and checkboxes threw errors
nr = handles.nr;
if ~isfield(handles, 'htsne') || ~isvalid(handles.htsne)
    if ~isprop(nr, 'waveforms') || isempty(nr.waveforms)
        % prompt the user to load the t-SNE file
        % TODO or run t-SNE?
        ktsnefile = nr.load_ktsne();
        if isempty(ktsnefile)
            return
        end
        handles.ktsnefile = ktsnefile;
        set(handles.listbox_clusters, 'Value', []);
    end
    if ~isprop(handles, 'htsne') || ~isvalid(handles.htsne)
        if get(handles.checkbox_tsnepower, 'Value')
            colormode = 'power';
        else
            colormode = 'label';
        end
        
        if get(handles.checkbox_training, 'Value')
            inds = nr.waveforms.traininds;
            
        else
            inds = 1:size(nr.waveforms.X, 1);
        end
        
        handles.htsne = nr.waveforms.scatter(colormode, inds);
        set(findobj(handles.htsne, 'Type', 'Scatter'), ...
            'HitTest', 'on', 'PickableParts', 'visible', ...
            'ButtonDownFcn', {@click_on_scatter, handles});
    end
    guidata(handles.figure1, handles);
else
    % set focus to the t-SNE figure
    figure(handles.htsne);
end
populate_listbox(handles.listbox_clusters, handles.nr);


function populate_listbox(lb, nr, val)
if isprop(nr, 'waveforms') && isvalid(nr.waveforms) && ~isempty(nr.waveforms)
    ulabels = unique(nr.waveforms.labels);
    if isequal(ulabels, 0)
        clusternames = {'no clusters'};
    else
        clusternames = arrayfun(@(x) sprintf('cluster %d', x), ulabels, 'UniformOutput', false);
    end
    set(lb, 'String', clusternames);
    if exist('val', 'var') && nargin >= 3
        set(lb, 'Value', val);
        notify(lb, 'Action');
    end
end


function click_on_scatter(hObject, eventdata, handles)
if strcmp(eventdata.EventName, 'Hit') && eventdata.Button == 1
    % See https://undocumentedmatlab.com/blog/figure-keypress-modifiers
    keyboardmodifier = get(gcf, 'SelectionType');  % shift + click: 'extend', ctrl + click: 'alt', click: 'normal'
    scatterdata = [eventdata.Source.XData;eventdata.Source.YData;eventdata.Source.ZData]';
    [mv, mi] = min(sum(bsxfun(@minus, scatterdata, eventdata.IntersectionPoint).^2, 2));
    if mv < 1e-10
        clusternum = eventdata.Source.CData(mi);
        switch keyboardmodifier
            case 'normal'
                set(handles.listbox_clusters, 'Value', clusternum);
            case {'extend', 'alt'}
                clusternums = get(handles.listbox_clusters, 'Value');
                if ismember(clusternum, clusternums)
                    newclusternums = setdiff(clusternums, clusternum);
%                     if isempty(newclusternums)
%                         newclusternums = 0;
%                     end
                    set(handles.listbox_clusters, 'Value', newclusternums);
                else
                    set(handles.listbox_clusters, 'Value', sort([clusternums clusternum]));
                end
            case 'open'
                % ignore double-clicks
                return
            otherwise
                fprintf('keyboardmodifier: %s\n', keyboardmodifier);
                return
        end
        notify(handles.listbox_clusters, 'Action');
    end
end


% --- Executes on button press in pushbutton_reload.
function pushbutton_reload_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_reload (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isfield(handles, 'ktsnefile')
    handles.nr.load_ktsne(handles.ktsnefile);
end
if isfield(handles, 'outfile')
    handles.nr.load_outfile(handles.outfile);
end
populate_listbox(handles.listbox_clusters, handles.nr);
notify(handles.listbox_clusters, 'Action');
if isfield(handles, 'htsne') && isvalid(handles.htsne)
    modify_tsne_scatter(handles)
    handles = guidata(handles.figure1);
end
if isfield(handles, 'hici') && isvalid(handles.hici)
    modify_ici(handles)
    handles = guidata(handles.figure1);
end
if isfield(handles, 'heventrates') && isvalid(handles.heventrates)
    modify_event_rates(handles)
    handles = guidata(handles.figure1);
end

% --- Executes on selection change in popupmenu_cluster.
function popupmenu_cluster_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_cluster (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu_cluster contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_cluster
clusteringmethods = get(hObject, 'String');
val = get(hObject, 'Value');
clusteringmethod = clusteringmethods{val};
% See http://scikit-learn.org/stable/modules/clustering.html for other clustering methods
switch clusteringmethod
    case 'DBSCAN'
        % epsilon and minpts
        v = {'on', 'on', 'off'};
        labels = {'epsilon', 'minpts'};
    case 'K-Means'
        % K (use default options)
        v = {'on', 'off', 'off'};
        labels = {'K'};
    case 'clusterdata'
        % 0 < cutoff < 2 and int(cutoff) >= 2 + options
        v = {'on', 'off', 'off'};
        labels = {'cutoff'};
    case 'manual'
        % no parameters
        v = {'off', 'off', 'off'};
        labels = {};
    otherwise
        fprintf('clusteringmethod: %s\n', clusteringmethod);
        return
end
for p = 1:3
    set(handles.(['edit_p' num2str(p)]), 'Visible', v{p});
    set(handles.(['text_p' num2str(p)]), 'Visible', v{p});
    if strcmp(v{p}, 'on')
        set(handles.(['text_p' num2str(p)]), 'String', labels{p});
    else
        set(handles.(['text_p' num2str(p)]), 'String', ['Parameter ' num2str(p)]);
    end
end


% --- Executes during object creation, after setting all properties.
function popupmenu_cluster_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu_cluster (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_p1_Callback(hObject, eventdata, handles)
% hObject    handle to edit_p1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_p1 as text
%        str2double(get(hObject,'String')) returns contents of edit_p1 as a double


% --- Executes during object creation, after setting all properties.
function edit_p1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_p1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_p2_Callback(hObject, eventdata, handles)
% hObject    handle to edit_p2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_p2 as text
%        str2double(get(hObject,'String')) returns contents of edit_p2 as a double


% --- Executes during object creation, after setting all properties.
function edit_p2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_p2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_p3_Callback(hObject, eventdata, handles)
% hObject    handle to edit_p3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_p3 as text
%        str2double(get(hObject,'String')) returns contents of edit_p3 as a double


% --- Executes during object creation, after setting all properties.
function edit_p3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_p3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_cluster.
function pushbutton_cluster_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_cluster (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% get parameters from the GUI
selectedclusters = get(handles.listbox_clusters, 'Value');
clusterstrings = get(handles.listbox_clusters, 'String');
if length(selectedclusters) == 1 && strcmp(clusterstrings{selectedclusters}, 'no clusters')
    selectedclusters = 0;
end

clusteralgs = get(handles.popupmenu_cluster, 'String');
val = get(handles.popupmenu_cluster, 'Value');
clusteralg = clusteralgs{val};
        
% get the training points from the selected clusters
traininds = handles.nr.waveforms.traininds;
inds = ismember(handles.nr.waveforms.labels(traininds), selectedclusters);
selectedinds = traininds(inds);

% some clustering algorithms might operate in the dimensionality reduced
% space and other might operate in the original space
% TODO and some might operate using all of the datapoints
Xselected = handles.nr.waveforms.X(selectedinds, :);
Yselected = handles.nr.waveforms.Y(selectedinds, :);

% cluster
switch clusteralg
    case 'DBSCAN'
        % epsilon and minpts
        epsilon = str2double(get(handles.edit_p1, 'String'));
        minpts = str2double(get(handles.edit_p2, 'String'));        
        idx = DBSCAN(Yselected, epsilon, minpts);        
    case 'K-Means'
        % K (use default options)
        K = str2double(get(handles.edit_p1, 'String'));
        idx = kmeans(Xselected, K);
    case 'clusterdata'
        % 0 < cutoff < 2 and int(cutoff) >= 2 + options
        cutoff = str2double(get(handles.edit_p1, 'String'));
        idx = clusterdata(Xselected, cutoff);
    case 'manual'
        % no parameters
        idx = [];
    otherwise
        fprintf('clusteringmethod: %s\n', clusteringmethod);
        return
end

% assign cluster 0 to 0, create new cluster labels at the end
uidx = unique(idx);
maxval = max(handles.nr.waveforms.labels);
% copy labels into labels2
handles.nr.waveforms.labels2 = handles.nr.waveforms.labels;
for newcluster = 1:length(uidx)
    indsnew = ismember(idx, uidx(newcluster));
    if uidx(newcluster) == 0
        handles.nr.waveforms.labels2(selectedinds(indsnew)) = 0;        
    else
        handles.nr.waveforms.labels2(selectedinds(indsnew)) = maxval + newcluster;
    end
end

% get labeled and unlabeled training inds for all datapoints
labeledtraininginds = traininds(handles.nr.waveforms.labels2(traininds) > 0);
unlabeledtraininginds = traininds(handles.nr.waveforms.labels2(traininds) == 0);
testinds = sort([handles.nr.waveforms.testinds unlabeledtraininginds]);
% get training and test set for knnsearch
Ytr = handles.nr.waveforms.Y(labeledtraininginds, :);
Ytest = handles.nr.waveforms.Y(testinds, :);
traininglabels = handles.nr.waveforms.labels2(labeledtraininginds);
% run knnsearch
idxtest = knnsearch(Ytr, Ytest, 'K', 5);
handles.nr.waveforms.labels2(testinds) = mode(traininglabels(idxtest), 2);
                
% ensure that the cluster numbers are consecutive
[~, ~, ic] = unique(handles.nr.waveforms.labels2);
handles.nr.waveforms.labels2 = ic;

% Visualize the potential new clusters (A lot of copy and paste here)
hscattsne = findobj(handles.htsne, 'Type', 'Scatter');
if get(handles.checkbox_training, 'Value')
    dispinds = handles.nr.waveforms.traininds;
else
    dispinds = 1:size(handles.nr.waveforms.X, 1);
end
xdata = handles.nr.waveforms.Y(dispinds, 1);
ydata = handles.nr.waveforms.Y(dispinds, 2);
zdata = handles.nr.waveforms.timestamps(dispinds) / 60;
if get(handles.checkbox_tsnepower, 'Value')
    cdata = 10*log10(sum(handles.nr.waveforms.X(inds, :).^2, 2));
    ax = get(handles.htsne, 'CurrentAxes');
    caxis(ax, [min(cdata) max(cdata)]);
    colormap(ax, 'parula');
else
    cdata = handles.nr.waveforms.labels2(dispinds);
    mincdata = min(cdata);
    maxcdata = max(cdata);
    ax = get(handles.htsne, 'CurrentAxes');
    if mincdata == maxcdata
        caxis(ax, [mincdata-.1 maxcdata+.1]);
    else
        caxis(ax, [mincdata maxcdata]);
    end    
    colormap(ax, 'parula');
end

set(hscattsne, 'XData', xdata, 'YData', ydata, 'ZData', zdata, 'CData', cdata);


% --- Executes on button press in checkbox_training.
function checkbox_training_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_training (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_training
modify_tsne_scatter(handles)


function modify_tsne_scatter(handles)
if isfield(handles, 'nr') && isvalid(handles.nr) && ...
        isprop(handles.nr, 'waveforms') && ~isempty(handles.nr.waveforms) && ...
        isfield(handles, 'htsne') && isvalid(handles.htsne)
    hscattsne = findobj(handles.htsne, 'Type', 'Scatter');
    
    if get(handles.checkbox_training, 'Value')
        inds = handles.nr.waveforms.traininds;
    else
        inds = 1:size(handles.nr.waveforms.X, 1);
    end
    
    xdata = handles.nr.waveforms.Y(inds, 1);
    ydata = handles.nr.waveforms.Y(inds, 2);
    zdata = handles.nr.waveforms.timestamps(inds) / 60;
    if get(handles.checkbox_tsnepower, 'Value')
        cdata = 10*log10(sum(handles.nr.waveforms.X(inds, :).^2, 2));
        ax = get(handles.htsne, 'CurrentAxes');
        caxis(ax, [min(cdata) max(cdata)]);
        colormap(ax, 'parula');
    else
        cdata = handles.nr.waveforms.labels(inds);
        mincdata = min(cdata);
        maxcdata = max(cdata);
        ax = get(handles.htsne, 'CurrentAxes');
        if mincdata == maxcdata
            caxis(ax, [mincdata-.1 maxcdata+.1]);
        else
            caxis(ax, [mincdata maxcdata]);
        end
        colormap(ax, parula(1000));
    end
    
    set(hscattsne, 'XData', xdata, 'YData', ydata, 'ZData', zdata, 'CData', cdata);
    
    if ~get(handles.checkbox_tsnepower, 'Value')
        notify(handles.listbox_clusters, 'Action');
    end
end


function modify_ici(handles)
if isfield(handles, 'nr') && isvalid(handles.nr) && ...
        isprop(handles.nr, 'waveforms') && ~isempty(handles.nr.waveforms) && ...
        isfield(handles, 'hici') && isvalid(handles.hici)
    % recreate ICI figure and restore the figure position
    pos = get(handles.hici, 'Position');
    close(handles.hici);
    nrgui('pushbutton_ici_Callback', handles.pushbutton_ici, [], handles);
    handles = guidata(handles.figure1);  % handles gets modified within
    set(handles.hici, 'Position', pos);
end

function modify_event_rates(handles)
if isfield(handles, 'nr') && isvalid(handles.nr) && ...
        isprop(handles.nr, 'waveforms') && ~isempty(handles.nr.waveforms) && ...
        isfield(handles, 'heventrates') && isvalid(handles.heventrates)
    % mylines = findobj(handles.heventrates, 'Type', 'Line');
    
    % recreate ICI figure and restore the figure position
    pos = get(handles.heventrates, 'Position');
    close(handles.heventrates);
    nrgui('pushbutton_eventrates_Callback', handles.pushbutton_eventrates, [], handles);
    handles = guidata(handles.figure1);  % handles gets modified within
    set(handles.heventrates, 'Position', pos);
end

% --- Executes on button press in pushbutton_timeseriescaps.
function pushbutton_timeseriescaps_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_timeseriescaps (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isfield(handles, 'nr') && isvalid(handles.nr)
    usedownsampled = true;
    handles.nr.plot_caps(usedownsampled);
end

% --- Executes on button press in checkbox_tsnepower.
function checkbox_tsnepower_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_tsnepower (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_tsnepower
modify_tsne_scatter(handles)


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
if isfield(handles, 'nr') && isvalid(handles.nr)
    delete(handles.nr);
end
if isfield(handles, 'ktsnefile')
    handles = rmfield(handles, 'ktsnefile');
end
if isfield(handles, 'outfile')
    handles = rmfield(handles, 'outfile');
end
if isfield(handles, 'htsne') && isvalid(handles.htsne)
    close(handles.htsne);
end
if isfield(handles, 'heventrates') && isvalid(handles.heventrates)
    close(handles.heventrates)
end
if isfield(handles, 'hici') && isvalid(handles.hici)
    close(handles.hici)
end
if isfield(handles, 'hcapstime') && isvalid(handles.hcapstime)
    close(handles.hcapstime)
end

delete(hObject);
guidata(handles.figure1, handles);


% --------------------------------------------------------------------
function menu_load_tsne_Callback(hObject, eventdata, handles)
% hObject    handle to menu_load_tsne (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isfield(handles, 'nr') && isvalid(handles.nr)
    ktsnefile = handles.nr.load_ktsne();
    if isempty(ktsnefile)
        return
    end
    handles.ktsnefile = ktsnefile;
    guidata(handles.figure1, handles);
    if isfield(handles, 'htsne') && isvalid(handles.htsne)
        modify_tsne_scatter(handles)
    end
end

% --------------------------------------------------------------------
function menu_load_outfile_Callback(hObject, eventdata, handles)
% hObject    handle to menu_load_outfile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isfield(handles, 'nr') && isvalid(handles.nr)
    outfile = handles.nr.load_outfile();
    if isempty(outfile)
        return
    end
    handles.outfile = outfile;
    guidata(handles.figure1, handles);
    if isfield(handles, 'htsne') && isvalid(handles.htsne)
        modify_tsne_scatter(handles)
    end
end


% --- Executes on button press in pushbutton_accept.
function pushbutton_accept_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_accept (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isprop(handles.nr.waveforms, 'labels2')
    handles.nr.waveforms.labels = handles.nr.waveforms.labels2;
    handles.nr.waveforms.labels2 = [];
    
    clusterinds = get(handles.listbox_clusters, 'Value');
    populate_listbox(handles.listbox_clusters, handles.nr, min(clusterinds));
    
    % update the plots
    modify_tsne_scatter(handles);  % checks for validity within
    handles = guidata(handles.figure1);
    modify_ici(handles);
    handles = guidata(handles.figure1);
    modify_event_rates(handles);
    handles = guidata(handles.figure1);
    
    % TODO clusterinds is wrong
    % TODO I copied and pasted a lot
    % Some of the clusters don't looked mapped correcly
    % clicking on t-SNE picks wrong clusters
    % histograms and event rates not highlighted
    
end
