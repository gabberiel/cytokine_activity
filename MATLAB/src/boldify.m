function [] = boldify(h)
myaxes = findobj(h, 'Type', 'Axes');
for ii = 1:length(myaxes)
    set(myaxes(ii), 'FontSize', 12);set(myaxes(ii), 'FontWeight', 'bold');
    mylines = findobj(myaxes(ii), 'Type', 'Line');
    
    allthin = true;
    for jj = 1:length(mylines)
        if get(mylines(jj), 'LineWidth') > 1
            allthin = false;
            break
        end
    end
    if allthin
        for jj = 1:length(mylines)
            set(mylines(jj), 'LineWidth', 2);
        end
    end
    
end