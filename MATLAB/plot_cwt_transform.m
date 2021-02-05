function [] = plot_cwt_transform(nr,range,show)
% PLOT cwt transformed signals (downsampled)


dt = nr.datadec.Time(2) - nr.datadec.Time(1);
T = nr.datadec.Time(range); 
X = nr.datadec.Data(range);
X_neural = cwt(X,0.001*1.75/dt,'db3');
X_cardiac = cwt(X,0.005*1.75/dt,'db3');
fontsize = 18;
fig = figure(1);
if show
    figure(1)
    plot(T,X_neural);
    hold on;
    plot(T,X);
    xlabel('Time (sec)','interpret','Latex','FontSize',fontsize);
    ylabel('Voltage $\mu V$','interpret','Latex','FontSize',fontsize);
    title('Neurally Enhanced Signal (1ms scale)','FontSize',fontsize)
    legend('cwt','raw')
    
    figure(2)
    plot(T,X_cardiac);
    hold on;
    plot(T,X);
    xlabel('Time (sec)','interpret','Latex','FontSize',fontsize);
    ylabel('Voltage $\mu V$','interpret','Latex','FontSize',fontsize);
    title('Cardiac Enhanced Signal (5ms scale)','FontSize',fontsize)
    legend('cwt','raw')
else
    set(fig,'Visible','off');
    %hold on;
    plot(T,X_neural);
    hold on;
    plot(T,X);
    xlabel('Time (sec)','interpret','Latex','FontSize',fontsize);
    ylabel('Voltage $\mu V$','interpret','Latex','FontSize',fontsize);
    title('Neurally Enhanced Signal (1ms scale)','FontSize',fontsize)
    legend('cwt','raw')
    saveas(fig,['Figures/CWT/cwt_transformd_neural.png'])
    close(fig)
    
    fig=figure(1);
    set(fig,'Visible','off');
    %hold on;
    plot(T,X_cardiac);
    hold on;
    plot(T,X);
    xlabel('Time (sec)','interpret','Latex','FontSize',fontsize);
    ylabel('Voltage $\mu V$','interpret','Latex','FontSize',fontsize);
    title('Cardiac Enhanced Signal (5ms scale)','FontSize',fontsize)
    legend('cwt','raw')
    saveas(fig,['Figures/CWT/cwt_transformd_cardiac.png'])
    close(fig)
end
