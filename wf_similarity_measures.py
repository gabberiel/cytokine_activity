import numpy as np
import warnings
import matplotlib.pyplot as plt

def wf_correlation(main_idx,std_waveforms):
    '''
    Waveform similarity measure based on correlation. 
    Computes the correlation between the different standardised std_waveforms and the main_idx-waveform.

    Parameters
    ----------
        std_waveforms : (number_of_std_waveforms, size_of_waveform) array_like
            standardised waveforms. (mean subtracted and devided by variance)
        
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


def similarity_SSQ(candidate_idx, waveforms, epsilon=0.1, var=1, standardised_input=True):
    '''
    qqq: TODO, Remove "var" from imputs..
    Similarity measure of waveforms based on Annulus theorem of Gaussians.
    H0: All observations (i.e waveforms) are assumed to be distributed as N(μ_x, σ^2*I),
    where mu_x is the candidate wavefrom specified by candidate_idx. 

    Annulus theorem: A n-dimensional spherical gaussian has all but at most 3*exp{−cnε^2} 
    of the probability mass in the annulus: n(1−ε) <= ||x|| <= n(1+ε). for some fixed constant c.
    
    OBS: extremely time-consuming to do for all waveforms...

    Parameters
    ----------
    candidate_idx : integer
        ...
    wavefroms : (num_of_wf, dim_of_waveforms) array_like
        
    epsilon : float
        parameter for test-statistic.. ish

    Returns
    -------
    similarity_evaluation : (num_of_wf,) array_like - Booleon
        True for similar waveforms
    '''
    n = waveforms.shape[1]
    upper_buond = 3*np.exp(-n*(epsilon**2)) # Theoretical convergence rate of mass towards annulus
    candidate_wf = waveforms[candidate_idx,:]
    candidate_standardised = waveforms - candidate_wf # Mean shifted
    assert np.sum(candidate_standardised[candidate_idx,:]) == 0, 'Mean shift in ssq not correct'
    # TODO: Which type of variance to use..? 
    if standardised_input is not True:
        candidate_standardised = candidate_standardised / (np.var(candidate_wf)*var) # All elements now assumed to be iid N(0,var)
    #print(candidate_standardised.shape)
    # Sum of Squares:
    ssq = np.sum(np.square(candidate_standardised),axis=1) # Under H0, we should have: (n(1-epsilon) < ssq < n(1+epsilon)), with prob. = 1 as n --> inf.
    similarity_evaluation = (n*(1-epsilon) < ssq ) & ( ssq < n*(1+epsilon)) # Those waveform s.t. ssq/sqrt(n) is close to 1 is assumed to be in the same cluster..
    return similarity_evaluation, upper_buond

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



def plot_correlated_wf(candidate_idx,waveforms,bool_labels,threshold,saveas=None,verbose=True):
    '''
    Plots wavefroms specified as True in bool_label. candidate_idx gives index for the Candidate-wavform under consideration.

    Will show plot if verbose is Ture.
    Will save figure if saveas is a valid path.

    Parameters
    ----------
        candidate_idx : integer
            index of main-waveform
        waveforms : (number_of_waveforms, size_of_waveform) array_like
            waveforms -- should be standardised
        bool_labels : (number_of_waveforms,) boolean_type
            labels of True/False as if they belong to same class as the main-waveform (baased on "threshold")
        threshold : float
            Threshold used when labeling waveforms
        saveas : 'path/to/save_fig' string_like _or_ None
            If None then the figure is not saved
    verbose : Booleon
        True => plt.show()
    Returns
    -------
        None
    '''
    warnings.warn('Old version of function plot_correlated_wf() is being used. Use version in plot_functions_wf.py instead.',DeprecationWarning)

    nr_of_wf_in_cluster = np.sum(bool_labels)
    print(f'Number of waveforms above threshold for wf_idx={candidate_idx} : {nr_of_wf_in_cluster}.')
    # If there is more than 1000 wavforms in cluster, then 500 indexes is sampled to speed up plotting.

    if np.sum(bool_labels)>500:
        true_idx = np.where(bool_labels==True)
        idx_sample = np.random.choice(true_idx[0], size=500, replace=False)
        new_bool_labels = np.zeros(bool_labels.shape)
        new_bool_labels[idx_sample] = 1 
        bool_labels = new_bool_labels == 1 # Convert to booleon
        print('Plotting 500...')

    time = np.arange(0,3.5,3.5/waveforms.shape[-1])
    
    #plt.figure()
    plt.plot(time,waveforms[bool_labels].T,color = (0.6,0.6,0.6),lw=0.5)
    plt.plot(time,np.median(waveforms[bool_labels],axis=0),color = (0.1,0.1,0.1),lw=1, label='Median')
    plt.plot(time,waveforms[candidate_idx,:],color = (1,0,0),lw=1, label='Candidate')

    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(\mu V)$')
    plt.title(f'W.F. s.t. corr > {threshold}. candidate wf: {candidate_idx}, N_cluster = {nr_of_wf_in_cluster}')
    plt.legend(loc='upper right')
    if saveas is not None:
        plt.savefig(saveas,dpi=150)
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
    from event_rate_funs import * 
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
    path_to_fig = None
    path_to_fig_ev = None
    label_res = [None,]
    for threshold in [0.7]:
        correlations = wf_correlation(i_range,std_waveforms)
        for corr_vec in correlations.T: #range(20,100,20):
            #print(corr_vec.shape)
            runs_for_time += 1
            bool_labels = label_from_corr(corr_vec,threshold=threshold,return_boolean=True )
            event_rates, real_clusters = get_event_rates(timestamps[:,0],bool_labels,bin_width=1,consider_only=1)
            delta_ev, ev_stats = delta_ev_measure(event_rates)
            #path_to_fig = '../Figs_TESTS/correlation_meassure_relates/WF_thres_'+str(threshold)+'_wf_'+str(i)+'.png'
            ev_labels = ev_label(delta_ev,ev_stats,n_std=1)

            if ev_labels[-1] == 0:
                print('joro')
                label_res.append(ev_label)
            #print(f'The event_rate_label is : {ev_labels}')
            #print()
            #plot_correlated_wf(i,std_waveforms,bool_labels,threshold,saveas=path_to_fig,verbose=True )
            
            #path_to_fig_ev = '../Figs_TESTS/correlation_meassure_relates/EV_thres_'+ str(threshold) +'_wf_' + str(i)+'.png'
            #plot_event_rates(event_rates,timestamps,noise=None,conv_width=100,saveas=path_to_fig_ev, verbose=True)
            
            #print(f'The event_rate_label is : {delta_ev}')
            #print()
            #print(f'Event_rate stats : {ev_stats}')
    end = time.time()
    print(f' Mean time for calculating labels and event_rate : {(end-start)/runs_for_time * 1000} ms')
    print(label_res)
    #print(f'event rates shape: {event_rates.shape}')
    #print(f'Real clusters: {real_clusters}')
    #bool_labels = label_from_corr(correlations,threshold=threshold, return_boolean=True)
    #plot_correlated_wf(i,std_waveforms,bool_labels,threshold,saveas=None)
    #plot_event_rates(event_rates,timestamps,noise=None,conv_width=100)
    #plt.show()

