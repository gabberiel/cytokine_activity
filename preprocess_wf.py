import numpy as np
def standardise_wf(waveforms):
    '''
    Standardise the waveforms by subtracting the mean and devide by the variance of each observation. 

    Parameters
    ----------
        waveforms : (number_of_waveforms, size_of_waveform) array_like
            Original waveforms
    Retruns
    ----------
        std_waveforms : (number_of_waveforms, size_of_waveform) array_like
            Standardised waveforms
    '''
    mean = np.mean(waveforms, axis=-1)
    std  = np.std(waveforms, axis=-1)  
    waveforms = waveforms - mean[:,None]
    std_waveforms = waveforms/std[:,None]
    return std_waveforms



def get_desired_shape(waveforms,timestamps,start_time=15,end_time=90,dim_of_wf=141, desired_num_of_samples=None):
    '''
    Returns waveforms of the dimension specified by "dim_of_waveform" which are observed in the time interval ["start_time","end_time"].
    If desired_num_of_samples is not None, then every k:th observation is used to get as close as possible to the desired sample size.

    Parameters
    ----------
    

    Returns
    -------

    '''

    d0_wf = waveforms.shape[0]
    firts_15_idx = np.where(timestamps<start_time*60)[0] # Find how many datapoints that corresponds to first 15 min of recording:  
    past_90_idx = np.where(timestamps>end_time*60)[0] # Find how many datapoints that corresponds to last 5 min of recording: 

    # Look at how the general event rate is over time and also look at the mean event-rate???

    start = round(firts_15_idx[-1]/1000)*1000 # Get even number of waveforms.. for the moment needed for the corr-labeling to work properly
    top_range = round((past_90_idx[0]-1)/1000)*1000
    if desired_num_of_samples is not None:
        number_of_obs = top_range-start 
        n_down = round(number_of_obs/desired_num_of_samples)
        n_down = np.max([n_down,1]) # Assure it does not become 0..
        use_range = np.arange(start,top_range,n_down) # REDUCE NUMBER OF CAPS UNDER CONSIDEERATION FOR EFFICENCY
    else:
        use_range = np.arange(start,top_range)
    waveforms = waveforms[use_range,:]
    timestamps = timestamps[use_range]
    print(waveforms.shape)
    # ************************************************************
    # ** Enforce all recording to have the same waveform dim. ****
    # ************************************************************
    # Enforce all recording to have the same waveform dim. (141):
    if d0_wf != 141:
        wf = np.zeros((waveforms.shape[0],141))
        if d0_wf >= 141:
            wf[:,:] = waveforms[:,:141] # disregard last dimensions of waveform..
        else:
            wf[:,:d0_wf] = waveforms[:,:] # The last elements remain zero..
        del(waveforms) 
        waveforms = wf # Let waveform point on the waveforms of the standard dimension.


    return waveforms, timestamps


def apply_mean_ev_threshold(waveforms,timestamps,mean_event_rates,ev_threshold=1,ev_labels=None):
    '''
    Returns waveforms with corresponding timestamps which has a mean event rate above ev_threshold.
    TODO : correct output shape of labels???? 
    '''
    idx = np.where(mean_event_rates>ev_threshold)
    if ev_labels is not None:
        high_occurance_ev_labels = ev_labels[:,idx]
        high_occurance_wf = waveforms[idx,:]
        high_occurance_ts = timestamps[idx]
        return high_occurance_wf[0,:,:], high_occurance_ts[:,0], high_occurance_ev_labels[:,0,:].T
    else:
        high_occurance_wf = waveforms[idx,:]
        high_occurance_ts = timestamps[idx]
        return high_occurance_wf[0,:,:],high_occurance_ts[:,0]


def apply_max_amplitude_thresh(waveforms,timestamps,maxamp_threshold=80):
    '''
    Removes CAPs with amplitude above threeshold value..
    '''
    keep_idx = np.all((waveforms < maxamp_threshold) & (waveforms > -maxamp_threshold) , axis=1)
    return waveforms[keep_idx],timestamps[keep_idx]


if __name__ == "__main__":
    pass