import numpy as np
import time
import matplotlib.pyplot as plt
from wf_similarity_measures import *
import warnings

def get_event_rates(timestamps,labels,bin_width=1,consider_only=None):
    '''
    Calculates event rate of labeled waveforms. This by counting the number of occurances in a sliding
    one second window of the corresponding timestamps for each label.
    
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
    
    bin_edges = np.arange(0,timestamps[-1]+1,1) # Must include rightmost edge in "np.histogram"
    if consider_only is None:
        clusters = np.unique(labels) # Get the different clusters for whcich to calculate event rate..
        event_rate_results = np.empty((bin_edges.shape[0]-1,clusters.shape[0]))
        real_clusters = []
        for cluster_idx,cluster in enumerate(clusters):
                event_count = np.histogram(timestamps[labels==cluster],bin_edges)
                event_rate_results[:,cluster_idx] = event_count[0]
                #cluster_idx += 1
                if np.mean(event_count[0]) > 0.5: # 0.1 in MATLAB
                        real_clusters.append(cluster)

    else:
        cluster = np.array((consider_only))       
        event_rate_results = np.empty((bin_edges.shape[0]-1,1))
        real_clusters = []
        event_count = np.histogram(timestamps[labels==cluster],bin_edges)
        event_rate_results[:,0] = event_count[0]
        #cluster_idx += 1
        if np.mean(event_count[0]) > 0.5: # 0.1 in MATLAB
                real_clusters.append(cluster)

    return event_rate_results, real_clusters
def get_average_ev(ev_stats):
    """
    Extracts average event_rate and variance for full period using outputs of "delta_ev_measure()"
    OBS: assuming ev_stats for one cluster...
    Parameters
    ----------
    ev_stats :[interval_ev_means, interval_ev_std] python_list
        interval_ev_means: (3, number_of_clusters) array_like
        interval_ev_std : (3, number_of_clusters) array_like
            mean and standard deviation of event rates for all three periods for each cluster.
    Returns
    -------
    tot_mean : float
        mean over complete recording
    tot_std : float
        standard deviation over complete recording

    """
    assert np.isnan(np.sum(ev_stats))==False, 'Nans in "ev_stats"'
    means = ev_stats[0]
    stds = ev_stats[1]

    tot_means = np.mean(means,axis=0)
    tot_std = np.mean(stds,axis=0)    
    assert np.isnan(np.sum(tot_means))==False, 'Nans in "ev_stats"'
    assert np.isnan(np.sum(tot_std))==False, 'Nans in "ev_stats"'
    return tot_means,tot_std


def delta_ev_measure(event_rates,timestamps = None):
        '''
        Calculates measure of how event-rate differs before and after injections. 
        i.e changes at 30min and 60min into recording.

        Could explore many different types of measures. especially take variance into account.
        However mu/var for instance could end up with division by zero..
        For now, a simple difference in mean of event-rates during each period will be considered.

        TODO: Could add an input of "real clusters" to assure positive event rate to be able to 
        divide by variance..

        Parameters
        ----------
        event_rates : (total_time_in_seconds, number_of_clusters) array_like 
                Number of occurances of labeled waveforms in each one second window during time
                of recording. 

        timestamps : ()

        Returns
        -------
        delta_ev : (number_of_injections, number_of_clusters) array_like
                Changes in mean event rate after the two injections for each cluster.

        ev_stats :[interval_ev_means, interval_ev_std] python_list
            interval_ev_means: (3, number_of_clusters) array_like
            interval_ev_std : (3, number_of_clusters) array_like
                mean and standard deviation of event rates for all three periods for each cluster.

        '''
        num_intervals = 3
        num_clusters = event_rates.shape[-1]
        # OBS. recording last longer than 90 minutes. 
        # Could change to end time but then intervals have different lengths.
        if timestamps is not None:
            assert timestamps[0] < 60*30, f'Invalid time range. Start time {timestamps[0]}, need to be before first injection.'
            assert timestamps[-1] > 60*60, f'Invalid time range. End time {timestamps[-1]}, need to be After second injection.'
            injection_times = [np.int(timestamps[0][0]), 60*30, 60*60, np.int(timestamps[-1][0])]
        else:
            warnings.warn('No timestamps given to "delta_ev_measure()". Assumes full time of recording.')
            injection_times = [0, 60*30, 60*60, 60*90] # injections occur 30 and 60 min into recording (in seconds).

        interval_ev_means = np.empty((num_intervals, num_clusters))
        interval_ev_std = np.empty((num_intervals, num_clusters))

        for i in range(num_intervals):
                interval_ev_means[i,:] = np.mean(event_rates[injection_times[i]:injection_times[i+1]],axis=0) 
                interval_ev_std[i,:] = np.std(event_rates[injection_times[i]:injection_times[i+1]],axis=0) 
        assert np.isnan(np.sum(interval_ev_means))==False, 'Nans in "interval_ev_means"'
        assert np.isnan(np.sum(interval_ev_std))==False, 'Nans in "interval_ev_std"'
        # Could explore many differnt times of measures...
        # For now, difference in mean event-rate will be considered.
        delta_ev = interval_ev_means[1:,:] - interval_ev_means[:-1,:]

        stats = [interval_ev_means, interval_ev_std]
        return delta_ev, stats

def ev_label(delta_ev,ev_stats,n_std=1):
    '''
    Give waveform a label encoding how the event rate change at time of injections.
    The label is vector with 3 dimensions. The three values corresponds 
    to "increase after first injection", "increase after second injection", "consant" -- respectively.

    Parameters
    ----------
    delta_ev : (number_of_injections, number_of_clusters) array_like
                Changes in mean event rate after the two injections for each cluster.
                (This function will only need/get "number_of_clusters" = 1.)
    ev_stats :[interval_ev_means, interval_ev_std] python_list
        interval_ev_means: (3, number_of_clusters) array_like
        interval_ev_std : (3, number_of_clusters) array_like
            mean and standard deviation of event rates for all three periods for each cluster.
    n_std : float
        number of standard deviations that mean hase to change after injection for it to be 
        considered as increase/decrease
    Returns
    -------
        label : :(3, number_of_clusters) array_like

    Example
    -------
        label = [0,1,0] corresponds to increase in activity after second injection. 
        label = [1,1,0] corresponds to increase in activity after both injections. 
    '''
    # TODO OBS for now it only accepts one label and one delta_ev..
    interval_ev_std = ev_stats[1][:2] #.reshape((-1,2)) # Get standard deviation for first two periods.

    # Define baseline standard deviation for second injection as mean of first two periods of recording..
    interval_ev_std[-1] = (interval_ev_std[0] + interval_ev_std[1])/2

    ev_label = np.zeros((3,delta_ev.shape[-1]))

    # Find if there is a sufficient increase in event rates after injections: 
    is_increase = delta_ev > (n_std*interval_ev_std)
    if True in is_increase:
        is_increase = np.append(is_increase,np.array((False)).reshape((1,1)),axis=0)
        ev_label[is_increase] = 1
        #print(ev_label)
    else:
        ev_label[-1] = 1
        #print(ev_label)

    return ev_label

def get_ev_labels(wf_std,timestamps,threshold=0.6,saveas=None, similarity_measure='corr',
                    assumed_model_varaince=0.5,n_std_threshold=1):
    '''
    Complete pipeline of labeling standardised waveforms based on change in event rates.
    Steps in process:
        * Use similarity_measure to cluster wavefomes assuming each observation as "candidate-wf"
        * Calculate event-rate from resulting cluster above
        * Calculate the change in event rate at time of injection as well as mean/variance for the three periods 
        * get ev_labels using the threshold: "mean-event rate increasing at least n std_deviations  after injection".
    
    Parameters
    ----------
        wf_std : (number_of_waveforms, dim_of_waveforms) array_like 
            Standardised/Preprocessed waveforms to label with ev_labels.
        timestaps : (number_of_waveforms, ) array_like 
            Vector containing timestamp for each waveform in seconds from started recording.
        threshold : float
            Gives either the minimum correlation using 'corr' or epsilon in gaussain annulus theorem for 'ssq' 
        similarity_measure : 'corr' or 'ssq'
            specifies which similarity measure to use for initial event-rate calculations.
            'corr' : correlation similarity measure
            'ssq' : sum of squares (gaussian annulus theorem) similarity measure
        assumed_model_varaince : float
            The  model variance assumed in ssq-similarity measure. i.e variance in N(x_candidate,sigma^2*I)  
        n_std_threshold : float
            Number of standard deviation which the mean-even-rate need to increase for a candidate-CAP to 
            be labeled as "likely to encode cytokine-info". 
    Returns
    -------
        ev_labels : (n_wf,) array_like

        ev_stats_tot : (2,n_wf)
            (tot_mean, tot_std)
    '''
    print('Initiating event-rate labeling')
    #n_std_threshold = 1

    n_wf = wf_std.shape[0]
    ev_labels = np.zeros((3,n_wf))
    ev_stats_tot = np.zeros((2,n_wf))
    if similarity_measure=='corr':
        print(f'Using Correlation as similarity measure...')
        print(f'threshold : {threshold}')
        print(f'n_std_threshold : {n_std_threshold}')
        print()
        sub_steps = 1000
        ii = 0
        t0 = time.time()
        prev_substep = 0
        #wf_downsampled = wf_std/assumed_model_varaince # Will no longer be normalised--not suitible for corr..??
        for sub_step in np.arange(sub_steps,n_wf,sub_steps):
            #print(sub_step)
            i_range = np.arange(prev_substep,sub_step)
            correlations = wf_correlation(i_range,wf_std)
            for corr_vec in correlations.T:
                bool_labels = label_from_corr(corr_vec,threshold=threshold,return_boolean=True)
                event_rates, real_clusters = get_event_rates(timestamps[:,0],bool_labels,bin_width=1,consider_only=1)
                delta_ev, ev_stats = delta_ev_measure(event_rates,timestamps=timestamps)
                tot_mean,tot_std = get_average_ev(ev_stats)
                ev_stats_tot[:,ii] = np.array((tot_mean,tot_std)).reshape(2,)
                #ev_labels = ev_label(delta_ev,ev_stats,n_std=1)
                ev_labels[:,ii] = ev_label(delta_ev,ev_stats,n_std=n_std_threshold)[:,0]
                ii +=1
            prev_substep = sub_step            
            if ii%(sub_steps*10)==0:
                print(f'On waveform {ii} in event-rate labeling')        
                ti = time.time()
                ETA_t =  n_wf * (ti-t0)/(ii) - (ti-t0) 
                print(f'ETA: {round(ETA_t)} seconds..')
                print()
    if similarity_measure=='ssq':
        print(f'Using Sum of squares (gaussian annulus theorem) as similarity measure. Paramterers:')
        print(f'assumed_model_varaince : {assumed_model_varaince}')
        print(f'n_std_threshold : {n_std_threshold}')
        print(f'threshold : {threshold}')
        print()
        # assumed_model_varaince = 0.5
        ii = 0
        t0 = time.time()
        wf_downsampled = wf_std[:,::2]/assumed_model_varaince # 0.5 in CVAE atm. 
        #n_wf = wf_downsampled.shape[0]
        #ev_labels = np.zeros((3,n_wf))
        #ev_stats_tot = np.zeros((2,n_wf))
        for candidate_idx in range(n_wf):
            bool_labels, _ = similarity_SSQ(candidate_idx, wf_downsampled, epsilon=threshold)
            event_rates, real_clusters = get_event_rates(timestamps[:,0],bool_labels,bin_width=1,consider_only=1)
            delta_ev, ev_stats = delta_ev_measure(event_rates,timestamps=timestamps)
            tot_mean,tot_std = get_average_ev(ev_stats)
            ev_stats_tot[:,ii] = np.array((tot_mean,tot_std)).reshape(2,)
            #ev_labels = ev_label(delta_ev,ev_stats,n_std=1)
            ev_labels[:,ii] = ev_label(delta_ev,ev_stats,n_std=n_std_threshold)[:,0]
            ii +=1
            if ii%1000==0:
                print(f'On waveform {ii} in event-rate labeling')        
                ti = time.time()
                ETA_t =  n_wf * (ti-t0)/(ii) - (ti-t0) 
                print(f'ETA: {round(ETA_t)} seconds..')
                print()
    if saveas is not None:
        np.save(saveas,ev_labels)
        np.save(saveas+'tests_tot',ev_stats_tot)
    return ev_labels, ev_stats_tot



        
def plot_event_rates(event_rates,timestamps, conv_width=100, noise=None, saveas=None,verbose=True):
    '''
    Plots event rates by smoothing kernel average of width "conv_width".
    convolution done including boundary effects but returns vector of same size.

    Parameters
    ----------
    event_rates: (total_time_in_seconds, number_of_clusters) array_like
            Number of occurances of labeled waveforms in each one second window during time
            of recording. 
    conv_width: Integer_like
            Size of smoothing kernel window for plotting
    noise :  integer_like
        Integers encoding which cluster is to be considered as noise.
       ((( qqq: old If "-1" is in clusters it is interpreted as noise. )))
        If noise is None, then all event_rates is plotted in the same way..
    Returns
    -------
    '''
    warnings.warn('Old version of function plot_event_rates() is being used. Use version in plot_functions_wf.py instead.',DeprecationWarning)
    end_time = timestamps[-1]
    number_of_obs = event_rates[:,0].shape[0]
    #time_of_recording_in_seconds = event_rates[:,0].shape[0]
    time = np.arange(0,end_time,end_time/number_of_obs) / 60 # To minutes
    conv_kernel = np.ones((conv_width))* 1/conv_width

    #plt.figure()
    #colors = ['r','k','g']
    if noise is not None:
        print('Noise...')
        for i,ev in enumerate(event_rates.T):
            if i != noise:
                smothed_ev = np.convolve(ev,conv_kernel,'same')
                plt.plot(time.T, smothed_ev, linestyle='-',lw=0.5, label=f'CAP cluster {i}') #color=colors[i%3]

    else:
        print('No given noise..')
        for i,ev in enumerate(event_rates.T):
            smothed_ev = np.convolve(ev,conv_kernel,'same')
            plt.plot(time.T, smothed_ev, linestyle='-',lw=0.5, label=f'CAP cluster {i}') #color=colors[i%3]
    
    plt.xlabel('Time of recording (min)')
    plt.ylabel('Event rate (CAPs/second)') 
    plt.title('Event Rate')
    plt.legend() 

    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose:
        plt.show()
    #plt.close()    



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
    from preprocess_wf import standardise_wf
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
    wf_std = standardise_wf(waveforms) 
    get_ev_labels(wf_std,timestamps,threshold=0.001,saveas=None, similarity_measure='ssq')

    quit()

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
        delta, ev_stats = delta_ev_measure(event_rates)
        mean_,std_ = get_average_ev(ev_stats)
        
    end = time.time()
    print(f' Mean time for calculating event_rate : {(end-start)/runs_for_time * 1000} ms')
    print(f'event rates shape: {event_rates.shape}')
    print(f'Real clusters: {real_clusters}')

    plot_event_rates(event_rates,timestamps,conv_width=100)
    plt.show()

