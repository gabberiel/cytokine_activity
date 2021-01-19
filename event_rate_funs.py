import numpy as np
import time
import matplotlib.pyplot as plt
from wf_similarity_measures import wf_correlation,similarity_SSQ
#from plot_functions_wf import plot_event_rates
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
def __get_average_ev__(ev_stats):
    """
    Extracts average event_rate and variance for full period using outputs of "__delta_ev_measure__()"
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


def __delta_ev_measure__(event_rates,timestamps = None):
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
            warnings.warn('No timestamps given to "__delta_ev_measure__()". Assumes full time of recording.')
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

def __ev_label__(delta_ev,ev_stats,n_std=1, new_variance_periods=True):
    '''
    Give waveform a label encoding how the event rate change at time of injections.
    The label is vector with 3 dimensions. The three values corresponds 
    to "increase after first injection", "increase after second injection", "consant" -- respectively.

    Parameters
    ----------
    delta_ev : (number_of_injections(=2), number_of_clusters) array_like
                Changes in mean event rate after the two injections for each cluster.
                (This function will only need/get "number_of_clusters" = 1. ??)
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
    # Define baseline standard deviation for second injection as mean of first two periods of recording..
    if new_variance_periods:
        # Will probably reduce variance threshold assuming variance is lower during second period
        interval_ev_std = np.empty(ev_stats[1][:2].shape)
        var_thres_first = (ev_stats[1][0] + ev_stats[1][1])/2 *0.5 # Mean variance over firts two periods
        var_thres_second = (ev_stats[1][0] + ev_stats[1][1] + ev_stats[1][2] )/3 # Mean variance over whole recording
        interval_ev_std[0] = var_thres_first
        interval_ev_std[1] = var_thres_second
    else:
        interval_ev_std = ev_stats[1][:2] #.reshape((-1,2)) # Get standard deviation for first two periods.
        interval_ev_std[-1] = (interval_ev_std[0] + interval_ev_std[1])/2

    __ev_label__ = np.zeros((3,delta_ev.shape[-1]))

    # Find if there is a sufficient increase in event rates after injections: 
    is_increase = delta_ev > (n_std*interval_ev_std)
    if True in is_increase:
        largest_increase = np.argmax(delta_ev-(n_std*interval_ev_std))
        __ev_label__[largest_increase] = 1
        #is_increase = np.append(is_increase,np.array((False)).reshape((1,1)),axis=0)
        #__ev_label__[is_increase] = 1
        #print(__ev_label__)
    else:
        __ev_label__[-1] = 1
        #print(__ev_label__)

    return __ev_label__

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
                delta_ev, ev_stats = __delta_ev_measure__(event_rates,timestamps=timestamps)
                tot_mean,tot_std = __get_average_ev__(ev_stats)
                ev_stats_tot[:,ii] = np.array((tot_mean,tot_std)).reshape(2,)
                #ev_labels = __ev_label__(delta_ev,ev_stats,n_std=1)
                ev_labels[:,ii] = __ev_label__(delta_ev,ev_stats,n_std=n_std_threshold)[:,0]
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
        print(f'assumed_model_varaince = {assumed_model_varaince}')
        print(f'n_std_threshold = {n_std_threshold}')
        print(f'Epsilon = {threshold}')
        print()
        # assumed_model_varaince = 0.5
        ii = 0
        t0 = time.time()
        if assumed_model_varaince is not False:
            wf_downsampled = wf_std[:,::2]/assumed_model_varaince # 0.5 in CVAE atm. 
        else:
            wf_downsampled = wf_std[:,::2] 
        #n_wf = wf_downsampled.shape[0]
        #ev_labels = np.zeros((3,n_wf))
        #ev_stats_tot = np.zeros((2,n_wf))
        for candidate_idx in range(n_wf):
            if assumed_model_varaince is not False:
                bool_labels, _ = similarity_SSQ(candidate_idx, wf_downsampled, epsilon=threshold,standardised_input=True)
            else:
                bool_labels, _ = similarity_SSQ(candidate_idx, wf_downsampled, epsilon=threshold,standardised_input=False)

            event_rates, real_clusters = get_event_rates(timestamps[:,0],bool_labels,bin_width=1,consider_only=1)
            delta_ev, ev_stats = __delta_ev_measure__(event_rates,timestamps=timestamps)
            tot_mean,tot_std = __get_average_ev__(ev_stats)
            ev_stats_tot[:,ii] = np.array((tot_mean,tot_std)).reshape(2,)
            #ev_labels = __ev_label__(delta_ev,ev_stats,n_std=1)
            ev_labels[:,ii] = __ev_label__(delta_ev,ev_stats,n_std=n_std_threshold)[:,0]
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
        print(f'EV_labels succesfully saved as : {saveas}')
    return ev_labels, ev_stats_tot
"""
def evaluate_cytokine_candidates(waveforms, timestamps, hpdp, k_labels, injection=1, similarity_measure='ssq', similarity_thresh=0.4, 
                            assumed_model_varaince=0.5, k=1, SD_min=1, saveas='saveas_not_specified', verbose=False):
    '''
    Evaluates the results of clustered hpdp using the median of each hpdp cluster as a "cytokine-candidate". 
    We make use of the specified similarity measure to find the corresponding event-rate of each candidate and evaluates 
    if there is a sufficient increase in the firing rate at time of injection. If so, the mice under consideration is considered
    to be a "responder". of the corresponding cytokine. 
    
    The evaluation of the event-rate increase is defined as follows,
    - Measure standart deviation, SD, of the baseline activity. (10-30 min from initial recording.)
    - Measure mean firing rate, MU, 4 min before the considered injection.
    - Measure past-injection firing rate, EV_past. (10-30 min after injection.)
    - Set threshold to k*max(SD_min,SD) and consider mice as responer if EV_past > k*max(SD_min,SD) for at least 1/3 of the considered 
      past-injection-time-interval. (7 out of 20 min.)
    
    The SD_min paramater is manually set to prevent the method to be sensitive to insignificant changes in event-rate. 
    
    Parameters
    ----------
    waveforms : (n_wf,d_wf), array_like
        The waveforms which are used for similarity measure to evaluate candidates.
    timestamps : (n_wf,) array_like
        Corresponding timestamps
    hpdp : (n_hpdp, d_wf) array_lika
        The high probability datapointes under consideration to find cytokine-candidate.
    k_labels : (n_hpdp,) array_like
        labels for hpdp -- should correpond to the different maximas of conditional pdf.
    
    saveas : 'path/to/save_fig' string_like _or_ None
            If None then the figure is not saved
        verbose : Booleon
            True => plt.show()
    
    Returns
    -------
    if return_candidates is True:
        candidate_wf : (n_clusters, d_wf) array_like
            The median of each hpdp-cluster.
    
    '''

    # Times of interest (in seconds)
    t0_baseline_SD = 10*60 # Initial time for measure baseline SD
    time_baseline_MU = 4*60 # length og time period measureing  baseline MU

    if injection==1:
        t_injection = 30*60 # Time of first injection
    elif injection==2:
        t_injection = 60*60 # Time of second injection
    assert timestamps[0] < 60*30, f'Invalid time range. Start time {timestamps[0]}, need to be before first injection.'
    assert timestamps[-1] > 60*60, f'Invalid time range. End time {timestamps[-1]}, need to be After second injection.'

    k_clusters = np.unique(k_labels)  
    candidate_wf = np.empty((k_clusters.shape[0],waveforms.shape[-1]))
    prel_results = [] # Store result about responders. If no responders, then this remains empty. 
    print(f'Shape of test-dataset (now considers all observations): {waveforms.shape}')
    for cluster in k_clusters:
        hpdp_cluster = hpdp[k_labels==cluster]
        MAIN_CANDIDATE = np.median(hpdp_cluster,axis=0) # Median more robust to outlier. should however not be a problem using DBSCAN..

        added_main_candidate_wf = np.concatenate((MAIN_CANDIDATE.reshape((1,MAIN_CANDIDATE.shape[0])),waveforms),axis=0)
        assert np.sum(MAIN_CANDIDATE) == np.sum(added_main_candidate_wf[0,:]), 'Something wrong in concatenate..'

        if similarity_measure=='corr':
            print('Using "corr" to evaluate final result')
            correlations = wf_correlation(0,added_main_candidate_wf)
            bool_labels = label_from_corr(correlations,threshold=similarity_thresh,return_boolean=True)
        if similarity_measure=='ssq':
            print('Using "ssq" to evaluate final result')
            if assumed_model_varaince is None:
                #added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh,standardised_input=False)
            else:
                added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh)
        event_rate, _ = get_event_rates(timestamps,bool_labels[1:],bin_width=1,consider_only=1)
        
        _, baseline_SD = get_ev_stats(event_rate,start_time=t0_baseline_SD, end_time=30*60)
        baseline_MU,_ =  get_ev_stats(event_rate,start_time=t_injection-time_baseline_MU, end_time=t_injection)
        #baseline_MU_2,_ =  get_ev_stats(event_rate,timestamps,start_time=t_injection_2-time_baseline_MU, end_time=t_injection_2)
        
        SD_thesh = k * np.max((SD_min, baseline_SD))
        cytokine_stats = get_ev_stats(event_rate,start_time=t_injection+10*60, end_time=t_injection+30*60, 
                                            compare_to_theshold=(baseline_MU+SD_thesh), conv_width=5)
        #print(f'Cytokine candidate responder result for injection 1 is : {cytokine_stats[2]}')
        #print(f'Cytokine candidate responder result for injection 2 is : {second_cytokine_stats[2]}')
        if cytokine_stats[2] is True:
            #plt.figure(1)
            #bool_labels[bool_labels==True] = cluster
            #plt.plot(event_rate)
            #plt.title(f'Responder-cluster = {cluster}')
            #plt.show()
            #plot_event_rates(event_rate,timestamps,noise=None,conv_width=100,saveas=saveas+'Main_cand_ev', verbose=True, cluster=cluster)
            prel_results.append(np.array([MAIN_CANDIDATE, cytokine_stats]))
            print(f'CAP nr. {cluster} found to have a sufficient increase in firing rate for injection {injection}.')
    return prel_results
        
def get_ev_stats(event_rate,start_time=10*60, end_time=90*60, compare_to_theshold=None,conv_width=5):
    '''
    Get event-rate mean and standard deviation for a specified time-period.
    If "compare_to_threshold" is not None, then the EV is compared to thresh to see if we have a "responder".
    Called by "evaluate_cytokine_candidates()".

    Returns
    -------
    if "compare_to_threshold" is not None:
        returns: MU, SD, responder, time_above_thresh
    else: 
        returns: MU, SD
    
    MU : float
        Mean event rate of considered period
    SD : float
        Mean Standard deviation of considered period
    responder : booleon
        True is the event-rate of specified period is larger than thesh for 1/3 of the period. 
    time_above_thresh : float
        How much time in seconds that the EV is above thresh.
    '''
    T = end_time-start_time # Length of time interval in seconds

    event_rate_ = event_rate[start_time:end_time] # Event-rate under time-period of interest.
    SD = np.std(event_rate_) # Standart deviation (SD) of period of interest
    MU = np.mean(event_rate_) # Mean (MU) event-rate under period of interest
    if compare_to_theshold is not None:
        responder  = False
        conv_kernel = np.ones((conv_width))* 1/conv_width
        smoothed_EV = np.convolve(np.squeeze(event_rate_),conv_kernel,'same') # Smooth out event_rate
        EV_above_thres = smoothed_EV > compare_to_theshold # Find all Event-rates larger then threshold
        time_above_thresh = np.sum(EV_above_thres) # Total time, in seconds, above specified threshold. 
        if time_above_thresh > T/3: # If responder, The EV has to be higher than thresh for at least 1/3 of the time period.
            responder = True
        return MU, SD, responder, time_above_thresh
    else:
        return MU, SD
"""

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
        delta, ev_stats = __delta_ev_measure__(event_rates)
        mean_,std_ = __get_average_ev__(ev_stats)
        
    end = time.time()
    print(f' Mean time for calculating event_rate : {(end-start)/runs_for_time * 1000} ms')
    print(f'event rates shape: {event_rates.shape}')
    print(f'Real clusters: {real_clusters}')

    plot_event_rates(event_rates,timestamps,conv_width=100)
    plt.show()

