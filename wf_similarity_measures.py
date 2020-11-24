import numpy as np
import matplotlib.pyplot as plt

def standardise_wf(waveforms):
    '''
    Computes the correlation between the different standardised waveforms

    Parameters
    ----------
        waveforms : (number_of_waveforms, size_of_waveform) array_like
            Original wavefroms
    Retruns
    ----------
        std_waveforms : (number_of_waveforms, size_of_waveform) array_like
            Standardised wavefroms
    '''
    mean = np.mean(waveforms, axis=-1)
    std  = np.std(waveforms, axis=-1)  
    waveforms = waveforms - mean[:,None]
    std_waveforms = waveforms/std[:,None]
    return std_waveforms

def wf_correlation(main_idx,std_waveforms):
    '''
    Waveform similarity measure based on correlation. 
    Computes the correlation between the different standardised std_waveforms and the main_idx-waveform.

    Parameters
    ----------
        std_waveforms : (number_of_std_waveforms, size_of_waveform) array_like
            standardised wavefroms
        
        main_idx : integer_like
            Specifices which waveform to get event rate from. 
    
    Returns
    -------
        correlations : (number_of_std_waveforms, 1) array_like
            Corr(wf, std_waveforms) 
    '''
    correlations = np.matmul(std_waveforms,std_waveforms[main_idx,:].T) / (std_waveforms.shape[-1]-1)
    
    assert np.isnan(np.sum(std_waveforms))==False, 'Nans in "waveforms"'
    assert np.isnan(np.sum(correlations))==False, 'Nans in "correlations"'

    return correlations

def trans_inv_corr_wf():
    '''
    Translation Invariant version of the correlation similarity measure

    Parameters
    ----------
    
    Returns
    -------
    '''
    pass

def label_from_corr(correlations, threshold=0.5,return_boolean=True):
    '''
    Returns labeled waveforms based on correlation above threshold value.
    If return_boolean is True then labels are True/False array.
    If return_boolean is False then labels are 1 if above threshold, otherwise 0.
    Parameters
    ----------
        correlations : (number_of_waveforms, 1) array_like
            Corr(wf, waveforms) 
        threshols : float
            Must be in range (0,1)
        return_boolean : boolean_type
    Returns
    -------
        labels : boolean or binary array_type
            type depends on "return_boolean"
    '''
    if return_boolean:
        return correlations>threshold
    else:
        labels = np.zeros((correlations.shape[0]))
        labels[correlations > threshold] = 1
        return labels


def plot_correlated_wf(original_idx,waveforms,bool_labels,threshold,saveas=None,verbose=True):
    '''Plot '''

    time = np.arange(0,3.5,3.5/waveforms.shape[-1])
    print(f'Number of waveforms above threshold for wf_idx={original_idx} : {sum(bool_labels)}.')
    plt.figure()
    plt.plot(time,waveforms[bool_labels].T,color = (0.6,0.6,0.6),lw=0.5)
    plt.plot(time,np.median(waveforms[bool_labels],axis=0),color = (0.1,0.1,0.1),lw=1, label='Median')
    plt.plot(time,waveforms[original_idx,:],color = (1,0,0),lw=1, label='Original')

    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(\mu V)$')
    plt.title(f'Waveforms such that corr > {threshold} to "Original" ')
    plt.legend()
    if saveas is not None:
        plt.savefig(saveas,dpi=150)
    if verbose:
        plt.show()
    plt.close()

def sim_measure2():
    '''
    ...

    Parameters
    ----------
    
    Returns
    -------
    '''
    pass

if __name__ == "__main__":
    '''
    ######## TESTING: #############
    Shape of waveforms: (136259, 141).
    Shape of timestamps: (136259, 1).
    OBS takes about 6.4 milliseconds to call "get_event_rates()" (mean of 100 runs)
    '''

    import numpy as np
    from scipy.io import loadmat
    from event_rate_first import * 
    import matplotlib.pyplot as plt
    import time
    print()
    print('Loading matlab files...')
    print()
    wf_name = '../matlab_files/gg_waveforms-R10_IL1B_TNF_03.mat'
    ts_name = '../matlab_files/gg_timestamps.mat'

    waveforms = loadmat(wf_name)
    #print(f' keys of matlab file: {waveforms.keys()}')
    waveforms = waveforms['waveforms']
    timestamps = loadmat(ts_name)['gg_timestamps']
    print('MATLAB files loaded succesfully...')
    print()
    print(f'Shape of waveforms: {waveforms.shape}.')
    print()
    print(f'Shape of timestamps: {timestamps.shape}.')
    print()
    assert waveforms.shape[0] == timestamps.shape[0], 'Missmatch of waveforms and timestamps shape.'
    # CREATE LABELS FOR TESTS
    manually_label = False
    if manually_label:
        print('Setting labels manually... ')
        labels = np.zeros((waveforms.shape[0]))
        first_injection_time = 30*60
        second_injection_time = 60*60

        labels[timestamps[:,0] < first_injection_time] = 1
        labels[(first_injection_time < timestamps[:,0]) & (timestamps[:,0] < second_injection_time)] = 2
        labels[timestamps[:,0] > second_injection_time] = 3


    # Create test linear timestamps
    #time_test = np.arange(0,60*60*1.5,60*60*1.5/136259)
    std_waveforms = standardise_wf(waveforms)
    start = time.time()
    # ------------------------------------------------------------------------------------
    # --------------------- TEST FUNCTIONS: ----------------------------
    # ------------------------------------------------------------------------------------
    threshold = 0.7
    runs_for_time = 0
    i = 27
    path_to_fig = None
    path_to_fig_ev = None
    for threshold in [0.7,0.8,0.9]:
        for i in range(10,100,20):
            runs_for_time += 1
            correlations = wf_correlation(i,std_waveforms)
            bool_labels = label_from_corr(correlations,threshold=threshold,return_boolean=True )
            event_rates, real_clusters = get_event_rates(timestamps[:,0],bool_labels,bin_width=1,consider_only=1)
            delta_ev, ev_stats = delta_ev_measure(event_rates)
            #path_to_fig = '../Figs_TESTS/correlation_meassure_relates/WF_thres_'+str(threshold)+'_wf_'+str(i)+'.png'
            plot_correlated_wf(i,std_waveforms,bool_labels,threshold,saveas=path_to_fig,verbose=False )
            
            #path_to_fig_ev = '../Figs_TESTS/correlation_meassure_relates/EV_thres_'+ str(threshold) +'_wf_' + str(i)+'.png'
            plot_event_rates(event_rates,timestamps,noise=0,conv_width=100,saveas=path_to_fig_ev, verbose=False)
            print(f'The change in event rate is: {delta_ev}')
            print()
            print(f'Event_rate stats : {ev_stats}')
    end = time.time()
    print(f' Mean time for calculating labels and event_rate : {(end-start)/runs_for_time * 1000} ms')
    #print(f'event rates shape: {event_rates.shape}')
    #print(f'Real clusters: {real_clusters}')
    bool_labels = label_from_corr(correlations,threshold=threshold, return_boolean=True)
    plot_correlated_wf(i,std_waveforms,bool_labels,threshold,saveas=None)
    plot_event_rates(event_rates,timestamps,noise=0,conv_width=100)
    #plt.show()

