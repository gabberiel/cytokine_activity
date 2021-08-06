
clc

% Convert them all!!

% ´Dir returnerar object med info om sökvägarna
% första och andra elementet är . , ..
file_list = dir;
size(file_list)

%Borde lägga till en loadingbar
for info_obj = file_list(3:end)'
    name = info_obj.name;
    disp(name);
    file_name = info_obj.name;
    convert_to_mat(file_name);
    disp('In the loop') 
end


