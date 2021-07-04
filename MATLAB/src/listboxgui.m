function varargout = listboxgui(varargin)
% LISTBOXGUI MATLAB code for listboxgui.fig
%      LISTBOXGUI, by itself, creates a new LISTBOXGUI or raises the existing
%      singleton*.
%
%      H = LISTBOXGUI returns the handle to a new LISTBOXGUI or the handle to
%      the existing singleton*.
%
%      LISTBOXGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LISTBOXGUI.M with the given input arguments.
%
%      LISTBOXGUI('Property','Value',...) creates a new LISTBOXGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before listboxgui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to listboxgui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help listboxgui

% Last Modified by GUIDE v2.5 08-Dec-2017 12:46:38

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @listboxgui_OpeningFcn, ...
                   'gui_OutputFcn',  @listboxgui_OutputFcn, ...
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


% --- Executes just before listboxgui is made visible.
function listboxgui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to listboxgui (see VARARGIN)

% Choose default command line output for listboxgui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes listboxgui wait for user response (see UIRESUME)
% uiwait(handles.figure1);
for ii = 1:2:length(varargin)
    switch varargin{ii}
        case 'ChannelNames'
            set(handles.listbox_chans, 'String', varargin{ii+1});
        otherwise
            disp('invalid Name');
    end
end

% --- Outputs from this function are returned to the command line.
function varargout = listboxgui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
varargout{2} = handles.listbox_chans;


% --- Executes on selection change in listbox_chans.
function listbox_chans_Callback(hObject, eventdata, handles)
% hObject    handle to listbox_chans (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox_chans contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox_chans


% --- Executes during object creation, after setting all properties.
function listbox_chans_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox_chans (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_ok.
function pushbutton_ok_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_ok (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.figure1, 'UserData', true);
uiresume(gcbf);

% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.figure1, 'UserData', false);
uiresume(gcbf);
