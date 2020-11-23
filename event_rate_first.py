import numpy as np
import matplotlib.pyplot as plt

def get_event_rates(timestamps,labels,bin_width=1):
    '''
    Calculates event rate of labeled waveforms. This by counting the number of occurances in sliding
    one second window of the corresponding timestamps.
    
    Parameters
    ----------
    timestaps : (number_of_waveforms, ) array_like 
            Vector containing timestamp for each waveform in seconds from started recording. 
            
    labels : (number_of_waveforms, ) array_like
            Integer valued vector -- encoding which custer each timestampt waveform belong to,
    
    bin_width : (1,) Integer 
            If 1 then evemt rate is calculated in Hz. (wf/second)
    
    Returns
    -------
    event_rates : (total_time_in_seconds, number_of_clusters) array_like 
            Number of occurances of labeled waveforms in each one second window during time
            of recording. 

    real_clusters : (number_of_real_clusters,) python_list
            Returns the clusters which has a mean event rate larger than a threshold of 0.1
    '''
    assert timestamps.shape[0] == labels.shape[0], f'Missmatch of labels and timestamps shape: ts: {timestamps.shape} & lb: {labels.shape}'
    
    clusters = np.unique(labels) # Get the different clusters for whcich to calculate event rate..
    bin_edges = np.arange(0,timestamps[-1]+1,1) # Must include rightmost edge in "np.histogram"
    event_rate_results = np.empty((bin_edges.shape[0]-1,clusters.shape[0]))

    real_clusters = []
    for cluster_idx,cluster in enumerate(clusters):
        event_count = np.histogram(timestamps[labels==cluster],bin_edges)
        event_rate_results[:,cluster_idx] = event_count[0]
        #cluster_idx += 1
        if np.mean(event_count[0]) > 0.5: # 0.1 in MATLAB
            real_clusters.append(cluster)
        
    return event_rate_results, real_clusters

def delta_ev_measure(event_rates):
        '''
        Calculates measure of how event-rate differs before and after injections. 
        i.e changes as 30min and 60min into recording.

        Could explore many differnt times of measures. especially take variance into account.
        However mu/var for instance could end up with division by zero..
        For now, a simple difference in mean of event-rates during each period will be considered.

        TODO: Could add an input of "real clusters" to assure positive event rate to be able to 
        divide by variance..

        Parameters
        ----------
        event_rates : total_time_in_seconds, number_of_clusters) array_like 
            Number of occurances of labeled waveforms in each one second window during time
            of recording. 
        
        Returns
        -------
        delta_ev : (2, number_of_clusters) array_like
                Changes in mean event rate after the two injections for each cluster.

        ev_stats : (3, number_of_clusters) array_like
                mean and variance of event rates for all three periods for each cluster.
        
        '''
        num_intervals = 3
        num_clusters = event_rates.shape[-1]
        # OBS. recording last longer than 90 minutes. 
        # Could change to end time but then intervals have different lengths.
        injection_times = [0, 60*30, 60*60, 60*90] # injections occur 30 and 60 min into recording (in seconds).
        
        interval_ev_means = np.empty((num_intervals, num_clusters))
        interval_ev_var = np.empty((num_intervals, num_clusters))

        for i in range(num_intervals):
                interval_ev_means[i,:] = np.mean(event_rates[injection_times[i]:injection_times[i+1]],axis=0) 
                interval_ev_var[i,:] = np.var(event_rates[injection_times[i]:injection_times[i+1]],axis=0) 
        
        # Could explore many differnt times of measures...
        # For now, difference in mean event-rate will be considered.
        delta_ev = interval_ev_means[1:,:] - interval_ev_means[:-1,:]

        stats = [interval_ev_means, interval_ev_var]
        return delta_ev, stats
        
def plot_event_rates(event_rates,timestamps, conv_width=100, clusters=None, saveas=None):
    '''
    Plots event rates by smoothing kernel average of width convolution_window.
    convolution done including boundary effects but returns vector of same size.

    Parameters
    ----------
    event_rates: (total_time_in_seconds, number_of_clusters) array_like
            Number of occurances of labeled waveforms in each one second window during time
            of recording. 
    conv_width: Integer_like
            Size of smoothing kernel window for plotting
    clusters : (number_of_clusters, ) array_like
        Contains integers encoding which cluster each event_rate corresponds to.
        If "-1" is in clusters it is interpreted as noise.
        If clusters is None all event_rates is plotted in the same way.. 
    Returns
    -------
    '''
    end_time = timestamps[-1]
    number_of_obs = event_rates[:,0].shape[0]
    #time_of_recording_in_seconds = event_rates[:,0].shape[0]
    time = np.arange(0,end_time,end_time/number_of_obs) / 60 # To minutes
    conv_kernel = np.ones((conv_width))* 1/conv_width
    #colors = ['r','k','g']
    for i,ev in enumerate(event_rates.T):
        smothed_ev = np.convolve(ev,conv_kernel,'same')
        plt.plot(time.T, smothed_ev, linestyle='-',lw=0.5, label=f'CAP cluster {i}') #color=colors[i%3]
    
    plt.xlabel('Time of recording (min)')
    plt.ylabel('Event rate (CAPs/second)') 
    plt.title('Event Rate')
    plt.legend() 

    if saveas is not None:
            plt.savefig(saveas, dpi=150)
    plt.show()


if __name__ == "__main__":
    '''
    ######## TESTING: #############
    Shape of waveforms: (136259, 141).
    Shape of timestamps: (136259, 1).
    OBS takes about 6.4 milliseconds to call "get_event_rates()" (mean of 100 runs)
    '''

    import numpy as np
    from scipy.io import loadmat
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
    labels = np.zeros((waveforms.shape[0]))
    first_injection_time = 30*60
    second_injection_time = 60*60

    labels[timestamps[:,0] < first_injection_time] = 1
    labels[(first_injection_time < timestamps[:,0]) & (timestamps[:,0] < second_injection_time)] = 2
    labels[timestamps[:,0] > second_injection_time] = 3


    # Create test linear timestamps
    time_test = np.arange(0,60*60*1.5,60*60*1.5/136259)

    start = time.time()
    # ------------------------------------------------------------------------------------
    # --------------------- TEST FUNCTIONS: ----------------------------
    # ------------------------------------------------------------------------------------
    runs_for_time = 1
    for i in range(runs_for_time):
        event_rates, real_clusters = get_event_rates(timestamps[:,0],labels,bin_width=1)

    end = time.time()
    print(f' Mean time for calculating event_rate : {(end-start)/runs_for_time * 1000} ms')
    print(f'event rates shape: {event_rates.shape}')
    print(f'Real clusters: {real_clusters}')

    plot_event_rates(event_rates,timestamps,conv_width=100)
    plt.show()

