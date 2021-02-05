function [] = save_rawrec(nr,rec_string,show)
% PLOT downsampled recording. 

X = nr.datadec.Data;
T = nr.datadec.Time/60; % To minutes
fontsize = 18;
fig = figure(1);
if show
    plot(T,X);
    xlabel('Time (min)','interpret','Latex','FontSize',fontsize);
    ylabel('Voltage $\mu V$','interpret','Latex','FontSize',fontsize);
    title('Raw Recording','FontSize',fontsize)
else
    set(fig,'Visible','off');
    %hold on;
    plot(T,X(:,1));
    xlabel('Time (min)','interpret','Latex','FontSize',fontsize);
    ylabel('Voltage $\mu V$','interpret','Latex','FontSize',fontsize);
    title('Raw Recording','FontSize',fontsize)
    saveas(fig,['Figures/y1_raw_rec_',rec_string,'.png'])
    close(fig)
    
    fig=figure(1);
    set(fig,'Visible','off');
    %hold on;
    plot(T,X(:,2));
    xlabel('Time (min)','interpret','Latex','FontSize',fontsize);
    ylabel('Voltage $\mu V$','interpret','Latex','FontSize',fontsize);
    title('Raw Recording','FontSize',fontsize)
    saveas(fig,['Figures/y2_raw_rec_',rec_string,'.png'])
    close(fig)
end