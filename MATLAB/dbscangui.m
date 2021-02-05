function varargout = dbscangui(varargin)
% DBSCANGUI MATLAB code for dbscangui.fig
%      DBSCANGUI, by itself, creates a new DBSCANGUI or raises the existing
%      singleton*.
%
%      H = DBSCANGUI returns the handle to a new DBSCANGUI or the handle to
%      the existing singleton*.
%
%      DBSCANGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DBSCANGUI.M with the given input arguments.
%
%      DBSCANGUI('Property','Value',...) creates a new DBSCANGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before dbscangui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to dbscangui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help dbscangui

% Last Modified by GUIDE v2.5 30-Oct-2017 11:40:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @dbscangui_OpeningFcn, ...
                   'gui_OutputFcn',  @dbscangui_OutputFcn, ...
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


% --- Executes just before dbscangui is made visible.
function dbscangui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to dbscangui (see VARARGIN)

% Choose default command line output for dbscangui
handles.output = hObject;
handles.Ytr = varargin{1};
handles.power = varargin{2};
handles.time = varargin{3};
handles.scatter = scatter3(handles.axes1, handles.Ytr(:, 1), handles.Ytr(:, 2), handles.time, 5, handles.power, 'filled');
title('t-SNE Power (dB)');colorbar;
view(handles.axes1, [0, 0, 1]);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes dbscangui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = dbscangui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function edit_epsilon_Callback(hObject, eventdata, handles)
% hObject    handle to edit_epsilon (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_epsilon as text
%        str2double(get(hObject,'String')) returns contents of edit_epsilon as a double
epsilon = str2double(get(handles.edit_epsilon, 'String'));
minpts = str2double(get(handles.edit_minpts, 'String'));
update_axes(handles, epsilon, minpts);


% --- Executes during object creation, after setting all properties.
function edit_epsilon_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_epsilon (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_minpts_Callback(hObject, eventdata, handles)
% hObject    handle to edit_minpts (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_minpts as text
%        str2double(get(hObject,'String')) returns contents of edit_minpts as a double
epsilon = str2double(get(handles.edit_epsilon, 'String'));
minpts = str2double(get(handles.edit_minpts, 'String'));
update_axes(handles, epsilon, minpts);


% --- Executes during object creation, after setting all properties.
function edit_minpts_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_minpts (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_eps_plus.
function pushbutton_eps_plus_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_eps_plus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
epsilon = str2double(get(handles.edit_epsilon, 'String'));
epsilon = epsilon + 0.1;
set(handles.edit_epsilon, 'String', num2str(epsilon));
minpts = str2double(get(handles.edit_minpts, 'String'));
update_axes(handles, epsilon, minpts);


% --- Executes on button press in pushbutton_eps_minus.
function pushbutton_eps_minus_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_eps_minus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
epsilon = str2double(get(handles.edit_epsilon, 'String'));
epsilon = epsilon - 0.1;
set(handles.edit_epsilon, 'String', num2str(epsilon));
minpts = str2double(get(handles.edit_minpts, 'String'));
update_axes(handles, epsilon, minpts);

% --- Executes on button press in pushbutton_minpts_plus.
function pushbutton_minpts_plus_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_minpts_plus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
minpts = str2double(get(handles.edit_minpts, 'String'));
minpts = minpts + 1;
set(handles.edit_minpts, 'String', num2str(minpts));
epsilon = str2double(get(handles.edit_epsilon, 'String'));
update_axes(handles, epsilon, minpts);

% --- Executes on button press in pushbutton_minpts_minus.
function pushbutton_minpts_minus_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_minpts_minus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
minpts = str2double(get(handles.edit_minpts, 'String'));
minpts = minpts - 1;
set(handles.edit_minpts, 'String', num2str(minpts));
epsilon = str2double(get(handles.edit_epsilon, 'String'));
update_axes(handles, epsilon, minpts);


function update_axes(handles, epsilon, minpts)
Ttr = DBSCAN(handles.Ytr, epsilon, minpts);
set(handles.scatter, 'CData', Ttr);
