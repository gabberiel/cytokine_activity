'''
Functions for similarity measure between waveforms.

Functions:
    * wf_correlation()
        (candidate_idx, waveforms) --> (correlations)
    * similarity_SSQ()
        (candidate_idx, waveforms) --> (similarity-Boolean)
'''
import numpy as np
import warnings
import matplotlib.pyplot as plt

def wf_correlation(candidate_idx, std_waveforms):
    '''
    Waveform similarity measure based on correlation. 
    Computes the correlation between the different standardised std_waveforms and the candidate_idx-waveform.

    Parameters
    ----------
        candidate_idx : integer_like
            Specifices which waveform to get event rate from. 
        std_waveforms : (number_of_std_waveforms, size_of_waveform) array_like
            standardised waveforms. (mean subtracted and devided by variance)
    
    Returns
    -------
        correlations : (number_of_std_waveforms, 1) array_like
            Corr(wf, std_waveforms) 
    '''
    assert np.isnan(np.sum(std_waveforms))==False, '[wf_correlation()] Nans in "waveforms"'
    
    correlations = np.matmul(std_waveforms , std_waveforms[candidate_idx,:].T) / (std_waveforms.shape[-1] - 1)
    
    assert np.isnan(np.sum(correlations))==False, '[wf_correlation()] Nans in "correlations"'

    return correlations


def similarity_SSQ(candidate_idx, waveforms, epsilon=0.1, standardised_input=True):
    '''
    Similarity measure of waveforms based on Annulus theorem of Gaussians.
    H0: All observations (i.e waveforms) are assumed to be distributed as N(μ_x, σ^2*I),
    where mu_x is the candidate wavefrom specified by candidate_idx. 

    Annulus theorem: \\
    A n-dimensional spherical gaussian has all but at most 3*exp{−cnε^2} 
    of the probability mass in the annulus: n(1−ε) <= ||x|| <= n(1+ε). for some fixed constant c. \\
    The similarity check is however simply: ||x|| <= n(1+ε)

    OBS: extremely time-consuming to do for all waveforms...

    Parameters
    ----------
    candidate_idx : integer
        Specifies the index of the waveform which is assumed to be candidate for mean.
    wavefroms : (num_of_wf, dim_of_waveforms) array_like
        All waveforms used for similarity evaluation.
    epsilon : float
        parameter for test-statistic.. ish
    standardised_input : Boolean
        Whether the input is assumed to be standardised. 
        ``True`` => "ssq" is applied to input as it is. \\
        ``False`` => candidate_standardised / (np.var(candidate_wf)) before
        "ssq" is applied.

    Returns
    -------
    (similarity_evaluation, upper_buond)
        similarity_evaluation : (num_of_wf,) array_like - Boolean
            True for similar waveforms
        upper_buond : scalar
            Theoretical convergence rate of mass towards annulus
    '''
    n = waveforms.shape[1]
    upper_buond = 3*np.exp(-n*(epsilon**2)) # Theoretical convergence rate of mass towards annulus
    candidate_wf = waveforms[candidate_idx,:]
    candidate_standardised = waveforms - candidate_wf # Mean shifted

    if standardised_input is not True:
        # All waveforms are after this assumed to be iid N(0, var*I)
        candidate_standardised = candidate_standardised / (np.var(candidate_wf)) 
    # Sum of Squares:
    ssq = np.sum(np.square(candidate_standardised),axis=1) # Under H0, we should have: (n(1-epsilon) < ssq < n(1+epsilon)), with prob. = 1 as n --> inf.
    similarity_evaluation =  ssq < n*(1+epsilon) # Those waveform s.t. ssq/sqrt(n) is close to 1 is assumed to be in the same cluster..
    
    return similarity_evaluation, upper_buond


if __name__ == "__main__":
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    from load_and_GD_funs import load_mat_file
    from preprocess_wf import standardise_wf
    from event_rate_funs import get_event_rates, __new_ev_labeling__
    from plot_functions_wf import plot_similar_wf, plot_event_rates

    with open('../hypes/test_runs.json', 'r') as f:
        hypes = json.load(f)
    wf_path = '../../matlab_files/wfR10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05'
    ts_path = '../../matlab_files/tsR10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05'

    waveforms =  load_mat_file(wf_path, 'waveforms')
    timestamps = load_mat_file(ts_path, 'timestamps')

    assert waveforms.shape[0] == timestamps.shape[0], 'Missmatch of waveforms and timestamps shape.'
    # CREATE LABELS FOR TESTS
    manually_label = False
    if manually_label:
        print('Setting labels manually... ')
        labels = np.zeros((waveforms.shape[0]))
        first_injection_time = 30*60
        second_injection_time = 60*60

        pre_injection = waveforms[timestamps[:,0] < first_injection_time] 
        past_first = waveforms[(first_injection_time < timestamps[:,0]) & (timestamps[:,0] < second_injection_time)] 
        past_second = waveforms[timestamps[:,0] > second_injection_time] 


    # Create test linear timestamps
    #time_test = np.arange(0,60*60*1.5,60*60*1.5/136259)
    std_waveforms = standardise_wf(waveforms)
    start = time.time()
    # ------------------------------------------------------------------------------------
    # --------------------- TEST FUNCTIONS: ----------------------------
    # ------------------------------------------------------------------------------------
    threshold = 0.7
    runs_for_time = 0
    i_range = np.arange(2000)

    for threshold in [0.7]:
        correlations = wf_correlation(i_range,std_waveforms)
        for corr_vec in correlations.T: #range(20,100,20):
            #print(corr_vec.shape)
            runs_for_time += 1
            bool_labels = corr_vec > threshold
            event_rates = get_event_rates(timestamps[:,0], bool_labels, 
                                          bin_width=1, consider_only=1)
            ev_labels = __new_ev_labeling__(event_rates, hypes)

    end = time.time()
    print(f' Mean time for calculating labels and event_rate : {(end-start)/runs_for_time * 1000} ms')
    print(f'event rates shape: {event_rates.shape}')
    idx_range = [i for i in range(10,1000,100)]
    correlations = wf_correlation(idx_range,std_waveforms)
    for i,corr_vec in enumerate(correlations.T):
        bool_labels = corr_vec > threshold
        plt.figure(1)
        plot_similar_wf(idx_range[i], std_waveforms, bool_labels, threshold)
        plt.show()

