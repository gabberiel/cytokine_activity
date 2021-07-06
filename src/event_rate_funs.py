'''
Contains functions related to calculating event-rates and labeling waveforms.

Functions:
    * get_event_rates()
        (timestamps, labels) --> (event-rate)
    * __get_EV_label()
        (event-rate) --> (label)
    * __get_EV_stats()
        (event-rate, t-period) --> (ev-mean, ev-std)
    * get_ev_labels()
        (waveforms, timestamps) --> (labels)
'''

import time
import numpy as np
import matplotlib.pyplot as plt
from wf_similarity_measures import wf_correlation, similarity_SSQ

def get_event_rates(timestamps, hypes, labels=None, consider_only=None):
    '''
    Calculates event rate of labeled waveforms in Hz or as fraction of total event-rate. 
    This by counting the number of CAP-occurances in a sliding one-second window of the 
    corresponding timestamps for each label. 

    If hypes['labeling']['relative_EV'] is ``True``, then the relative event-rate is calculated.\\
    elif False, The absolute event-rate is calculated in Hz.
    
    Called during labeling of CAPs from "get_ev_labels()".

    Parameters
    ----------
    timestaps : (number_of_waveforms, ) array_like 
            Vector containing timestamp for each waveform in seconds from started recording. 
            
    hypes : .json file
        Containing hyperparameters
            relative_EV : Boolean
                weahter or not to use relative EV. That is CAP_EV / TOT_EV.

    labels : (number_of_waveforms, ) array_like or ``None``
            If ``None``: Total event-rate is returned. \\
            Else: Integer or Boolean valued vector -- encoding which custer each timestampt-waveform belongs to. \\
            If Boolean values, then "consider_only"=1 will show event-rate for all "True-labeled" waveforms. 

    consider_only : ``None`` or integer
        If ``None``: Loops through all different clusters from "labels". \\
        If integer : Only considers the label equal to "consider_only"

    Returns
    -------
    event_rates : (total_time_in_seconds, number_of_clusters) array_like 
            Number of occurances of labeled waveforms in each one-second window during time of recording. \\
            i.e CAPs / second for all times of the input.

    '''
    
    relative_EV = hypes['labeling']['relative_EV']
    
    bin_edges = np.arange(0, timestamps[-1] + 1 ,1) # Must include rightmost edge in "np.histogram"
    if labels is None:
        event_rate_results = np.histogram(timestamps, bin_edges)[0]
        return event_rate_results.reshape(-1,1)

    assert timestamps.shape[0] == labels.shape[0], f'Missmatch of labels and timestamps shape: ts: {timestamps.shape} & lb: {labels.shape}'

    if consider_only is None:
        clusters = np.unique(labels) # Get the different clusters for whcich to calculate event rate..
        event_rate_results = np.empty((bin_edges.shape[0] - 1, clusters.shape[0]))
        for cluster_idx, cluster in enumerate(clusters):
                event_count = np.histogram(timestamps[labels==cluster], bin_edges)
                event_rate_results[:,cluster_idx] = event_count[0]
    else:
        cluster = np.array((consider_only))       
        event_rate_results = np.empty((bin_edges.shape[0] - 1, 1))
        event_count = np.histogram(timestamps[labels==cluster], bin_edges)
        event_rate_results[:,0] = event_count[0]

    if relative_EV:
        tot_ev = np.histogram(timestamps, bin_edges)[0]
        conv_width  = 10
        conv_kernel = np.ones((conv_width))* 1/conv_width
        tot_smoothed_EV = np.convolve(np.squeeze(tot_ev), conv_kernel, 'same')
        non_zero_idx = tot_smoothed_EV > 0
        event_rate_norm = event_rate_results / tot_smoothed_EV.reshape(-1,1)
        event_rate_results[non_zero_idx] = event_rate_norm[non_zero_idx]

    return event_rate_results

def __get_EV_label(event_rate, hypes):
    '''
    Returns a label as a vector with 3 dimensions. The three values corresponds to:
    "increase after first injection", "increase after second injection", "consant" -- respectively.

    For each injection time, as specified in hypes.json, the input is labeled as 
    "increase after injection" if:

        ``post-injection-mean > pre-injection-mean + k * max(SD_min, SD_baseline)``
    for at least 1/3 of the post-injection time period.


    Parameters
    ----------
    event_rates : (total_time_in_seconds, 1) array_like 
        Number of occurances of a CAP-waveform in each one second window during time
        of recording. Or (depending on hypes), the procentage of total event-rate during full recording
    hypes : .json file
        Containing hyperparameters

    Returns
    -------
        label : (3,) array_like

    Example
    -------
        label = [1,0,0] corresponds to increase in activity after first injections. 
        label = [0,1,0] corresponds to increase in activity after second injection. 
    '''
    # All times in hypes should be in minutes..
    t0_baseline_SD =  hypes["labeling"]["t0_baseline_SD"]
    time_baseline_MU =  hypes["labeling"]["time_baseline_MU"]
    k = hypes["labeling"]["k_SD"]
    SD_min = hypes["labeling"]["SD_min"]
    injection_t_period = hypes["experiment_setup"]["injection_t_period"] # 30 for Zanos. 10 for KI.
    t_delay_post_injection = hypes["experiment_setup"]["t_delay_post_injection"] 

    ev_label = np.zeros((3,))
    results = np.zeros((2,))
    increase = False
    # Find if there is a sufficient increase in event rates after each injections: 
    _, baseline_SD = __get_EV_stats(event_rate, start_time=t0_baseline_SD*60, end_time=injection_t_period*60)
    for injection in [1,2]:
        t_injection = injection_t_period * 60 * injection # Time of injection
        baseline_MU, _ =  __get_EV_stats(event_rate, start_time=t_injection-time_baseline_MU*60, end_time=t_injection)
        SD_thesh = k * np.max((SD_min,  baseline_SD))

        start_t_for_stats = t_injection + t_delay_post_injection * 60
        end_t_for_stats = t_injection + injection_t_period * 60 

        cytokine_stats = __get_EV_stats(event_rate, start_time=start_t_for_stats, end_time=end_t_for_stats, 
                                          compare_to_theshold=baseline_MU + SD_thesh, conv_width=10)
        if cytokine_stats[2] is True:
            increase = True
            results[injection-1] = cytokine_stats[3]

    if increase is True:
        # Set label to "increase after injection". (for the largest injection.) 
        largest_increase = np.argmax(results)
        ev_label[largest_increase] = 1

    else:
        # Set label to "No increase after injection"
        ev_label[-1] = 1
    return ev_label

def __get_EV_stats(event_rate, start_time=10*60, 
                   end_time=90*60, 
                   compare_to_theshold=None,
                   conv_width=5):
    '''
    Get event-rate mean and standard deviation for a specified time-period.
    If "compare_to_threshold" is not None, then the EV is compared to thresh to see if we have a "response". \\
    "response" is retured as ``True`` if the EV is larger than the threshold value for at least 1/3 of the 
    specified time-period.


    Called by "__get_EV_label()" and "__evaluate_cytokine_candidates__()".

    Parameters
    ----------
    event_rates : (total_time_in_seconds, 1) array_like 
        Number of occurances of a CAP-waveform in each one second window during time
        of recording. 
    start_time : scalar
        start time in second
    end_time : scalar
        end time in second
    compare_to_theshold : None or scalar.
        None => return only MU_local, SD_local \\
        scalar => The EV is compared to compare_to_theshold-scalar value to see if we have a "response".
    conv_width : Int.
        Only used if "compare_to_theshold" is not None. \\
        The width of smoothing window that is applied to event-rate before comparing it to the 
        "compare_to_theshold"-value

    Returns
    -------
    if "compare_to_threshold" is not None:
        ( MU_local, SD_local, response, time_above_thresh )
    else: 
        ( MU_local, SD_local )
    where: 
    MU_local : float
        Mean event rate of considered period
    SD_local : float
        Mean Standard deviation of considered period
    response : booleon
        True is the event-rate of specified period is larger than thesh for 1/3 of the period. 
    time_above_thresh : float
        How much time in seconds that the EV is above thresh.
    '''

    T = end_time-start_time # Length of time interval in seconds

    EV_local = event_rate[start_time:end_time] # Event-rate under time-period of interest.
    SD_local = np.std(EV_local) # Standart deviation (SD_local) of period of interest
    MU_local = np.mean(EV_local) # Mean (MU_local) event-rate under period of interest

    if compare_to_theshold is not None:
        response  = False 
        conv_kernel = np.ones((conv_width)) * 1 / conv_width
        smoothed_EV = np.convolve(np.squeeze(EV_local), conv_kernel,'same') # Smooth out event-rate
        EV_above_thres = smoothed_EV > compare_to_theshold # Find all Event-rates larger then threshold
        time_above_thresh = np.sum(EV_above_thres) # Total time in seconds above specified threshold (since each element in event-rate is <=> 1 second). 
        
        if time_above_thresh > T/3: # If response, The EV has to be higher than thresh for at least 1/3 of the time period.
            response = True
        return MU_local, SD_local, response, time_above_thresh

    else:
        return MU_local, SD_local


def get_ev_labels(wf_std, timestamps, hypes, saveas=None): 
    '''
    Complete pipeline of labeling standardised waveforms based on change in event rates. \\
    Steps in process:
        * Use similarity_measure to cluster wavefomes assuming each observation as "candidate-wf".
        * Calculate event-rate from resulting "similarity-cluster".
        * Calculate the change in event rate at time of injection as well as mean/variance for the three periods. 
        * Get ev_labels using the threshold: 

                ``post-injection-mean > pre-injection-mean + k * max(SD_min, SD_baseline)``

         for at least 1/3 of the post-injection time period.

    In practise that is, for each CAP:
        * First : ``similarity-boolean-vector <-- wf_correlation(wf_std..)`` / ``similarity_SSQ(wf_std..)``
        * Secnodly :  ``Event-rate <-- get_event_rates(similarity-boolean-vector..)``
        * Finally, ``EV-label <-- __get_EV_label(Event-rate..)``
    
    Parameters
    ----------
        wf_std : (number_of_waveforms, dim_of_waveforms) array_like 
            Standardised/Preprocessed waveforms to label with ev_labels.
        timestamps : (number_of_waveforms, ) array_like 
            Vector containing timestamp for each waveform in seconds from started recording.
        hypes : .json file
            Containing the hyperparameters:
                threshold : float
                    Gives either the minimum correlation using 'corr' or epsilon in gaussain annulus theorem for 'ssq'.
                similarity_measure : 'corr' or 'ssq'
                    specifies which similarity measure to use for initial event-rate calculations. \\
                    'corr' : correlation similarity measure \\
                    'ssq' : sum of squares (gaussian annulus theorem) similarity measure
                assumed_model_varaince : float
                    The  model variance assumed in ssq-similarity measure. i.e variance in N(x_candidate,sigma^2*I)  
        saveas : ``None`` or String.
            If saveas is not ``None``:
                ev_labels : saved as, "saveas" + ".npy" \\
                ev_stats_tot : saved as, "saveas" + "tests_tot.npy"

    Returns
    -------
    (ev_labels, ev_stats_tot)
        ev_labels : (3, n_wf) array_like
            e.g. (0,0,1) for a waveform without increase after injection.
        ev_stats_tot : (2, n_wf)
            (tot_mean, tot_std)

    Saves:
    ------
    If saveas is not ``None``:
        ev_labels : saved as, "saveas" + ".npy" \\
        ev_stats_tot : saved as, "saveas" + "tests_tot.npy"
    '''
    print('Initiating event-rate labeling.. \n')

    # *** Extract hyperparameters from json file: **
    similarity_measure = hypes["labeling"]["similarity_measure"]
    assumed_model_varaince = hypes["labeling"]["assumed_model_varaince"]
    threshold = hypes["labeling"]["similarity_thresh"]
    ssq_downsample = hypes["labeling"]["ssq_downsample"]       # Downsample waveforms to speed up analysis using "ssq"
    start_time = hypes["experiment_setup"]["start_time"] 
    end_time = hypes["experiment_setup"]["end_time"] 
    # ***********************************************
    
    n_wf = wf_std.shape[0]
    ev_labels = np.zeros((3, n_wf))         # Store Labels
    ev_stats_tot = np.zeros((2, n_wf))      # Store Total ev mean and standard deviation

    ii = 0
    t0 = time.time()
    if similarity_measure=='corr':
        print(f'Using Correlation as similarity measure...')
        print(f'threshold : {threshold}')
        sub_steps = 1000
        prev_substep = 0

        for sub_step in np.arange(sub_steps, n_wf, sub_steps):
            # This sub_step approach is used to speed up computations with mat-mult.
            # Can however not use all wf for matmult since this creates memory issues.. (allocates to much memory.)
            i_range = np.arange(prev_substep,sub_step)
            correlations = wf_correlation(i_range, wf_std)
            for corr_vec in correlations.T:
                bool_labels = corr_vec > threshold
                event_rates = get_event_rates(timestamps[:,0], hypes, labels=bool_labels, consider_only=1)
                
                ev_labels[:,ii] = __get_EV_label(event_rates, hypes)
                tot_mean, tot_std = __get_EV_stats(event_rates, start_time=start_time * 60, 
                                                    end_time=end_time * 60)
                ev_stats_tot[:,ii] = np.array((tot_mean, tot_std)).reshape(2,)
                ii +=1

            prev_substep = sub_step            
            if ii % (sub_steps*10) == 0:
                print(f'On waveform {ii} in event-rate labeling')        
                ti = time.time()
                ETA_t =  n_wf * (ti - t0) / (ii) - (ti - t0) 
                print(f'ETA: {round(ETA_t)} seconds.. \n')

    if similarity_measure=='ssq':
        print(f'Using Sum of squares (gaussian annulus theorem) as similarity measure. Paramterers:')
        print(f'assumed_model_varaince = {assumed_model_varaince}')
        print(f'Epsilon = {threshold} \n')

        if assumed_model_varaince is not False:
            # Divide by assumed variance. Allows for more/less similarity within "similarity-clusters"
            wf_downsampled = wf_std[:,::ssq_downsample]/assumed_model_varaince # 0.5 in CVAE atm. 
        else:
            wf_downsampled = wf_std[:,::ssq_downsample] 

        # Loop through and lable all observed CAPs :
        for candidate_idx in range(n_wf):
            if assumed_model_varaince is not False:
                bool_labels, _ = similarity_SSQ(candidate_idx, wf_downsampled, 
                                                epsilon=threshold, standardised_input=True)
            else:
                bool_labels, _ = similarity_SSQ(candidate_idx, wf_downsampled,
                                                epsilon=threshold, standardised_input=False)

            event_rates = get_event_rates(timestamps[:,0], hypes, labels=bool_labels, consider_only=1)
            
            ev_labels[:,ii] = __get_EV_label(event_rates, hypes)

            tot_mean, tot_std = __get_EV_stats(event_rates, 
                                                 start_time=start_time*60, 
                                                 end_time=end_time*60)
            ev_stats_tot[:,ii] = np.array((tot_mean, tot_std)).reshape(2,)
            ii +=1
            if ii%1000==0:
                print(f'On waveform {ii} in event-rate labeling')        
                ti = time.time()
                ETA_t =  n_wf * (ti-t0)/(ii) - (ti-t0) 
                print(f'ETA: {round(ETA_t)} seconds.. \n')
    if saveas is not None:
        np.save(saveas, ev_labels)
        np.save(saveas + 'stats_tot', ev_stats_tot)
        print(f'EV_labels succesfully saved as : {saveas}')

    return ev_labels, ev_stats_tot

