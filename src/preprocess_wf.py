import numpy as np

def standardise_wf(waveforms, hypes):
    '''
    Standardise the waveforms by subtracting the mean and devide by the variance or max of each observation. 

    Parameters
    ----------
    waveforms : (number_of_waveforms, dim_of_waveform) array_like
        Original waveforms
    hypes : .json file
            Containing hyperparameters. In this function:

            standardise_type : str. ("max" or "var")
                "max" => standardise on max-values that is allowed for a wf (from amplitude threshold) \\
                "var" => standadise om the variance of each wf.
            maxamp_threshold : scalar. (used if "standardise_type"=="max")
                The amplitude threshold.


    Retruns
    ----------
    std_waveforms : (number_of_waveforms, dim_of_waveform) array_like
        Standardised waveforms

    '''
    standardise_type = hypes["preprocess"]["standardise_type"]
    standardise_type = hypes["preprocess"]["standardise_type"]
    
    mean = np.mean(waveforms, axis=-1)
    waveforms = waveforms - mean[:,None]

    if standardise_type == "var":
        std  = np.std(waveforms, axis=-1)  
        std_waveforms = waveforms/std[:,None]
    elif standardise_type == "max":
        max_val =  hypes["preprocess"]["maxamp_threshold"]
        std_waveforms = waveforms / max_val
    else:
        print('Invalid argument for "standardise_type" in hypes.')
        raise
    return std_waveforms



def get_desired_shape(waveforms,timestamps, hypes, training=True):
    '''
    Uses hyperparams from the "hypes" .json file and returns waveforms of the dimension specified by "dim_of_waveform",
    which are observed in the time interval ["start_time","end_time"].
    
    If desired_num_of_samples is not None, then every k:th observation is used to get as close as possible to the desired sample size.

    To achieve the desired number of dimensions/samples, each CAP is either cut of at specified dim, or extended with zeros for missing dimensions.

    Parameters
    ----------
    waveforms : (number_of_waveforms, dim_of_waveform) array_like
            CAP-waveforms
    timestaps : (number_of_waveforms, ) array_like 
            Vector containing timestamp for each waveform in seconds.
    hypes : .json file
            Containing hyperparameters.

    Returns
    -------
    (waveforms, timestamps)
    waveforms : (new_number_of_waveforms, new_dim_of_waveform) array_like

    timestamps : (new_number_of_waveforms, ) array_like
    '''
    # ** Extract hyperparameters from json file: **
    start_time = hypes["experiment_setup"]["start_time"] # Specifies time to consider given in minutes
    end_time = hypes["experiment_setup"]["end_time"] # Specifies time to consider given in minutes
    prefered_dim_of_wf = hypes["preprocess"]["dim_of_wf"] # Number of dimensions to be used. (number of samples in each CAP)
    if training:
        desired_num_of_samples = hypes["preprocess"]["desired_num_of_samples"]
    else:
        desired_num_of_samples = hypes["preprocess_for_eval"]["desired_num_of_samples"]
    # **********************************************
    d0_wf = waveforms.shape[1]
    # Find waveforms Before recording time of interest.
    first_idx = np.where(timestamps<start_time*60)[0] 
    if list(first_idx):
        # There excisted a timestep prior to start_time
        start = first_idx[-1] 
    else:
        # No timestamp prior to start_time
        start = 0
    # Find waveforms after recording time of interest:
    if timestamps[-1] > end_time*60:
        past_90_idx = np.where(timestamps>end_time*60)[0] 
        top_range = past_90_idx[0]
    else:
        top_range = len(timestamps)

    if desired_num_of_samples is not None:
        number_of_obs = top_range-start 
        n_down = round(number_of_obs/desired_num_of_samples)
        n_down = np.max([n_down,1]) # Assure it does not become 0..
        use_range = np.arange(start,top_range,n_down).astype(int) # REDUCE NUMBER OF CAPS UNDER CONSIDEERATION FOR EFFICENCY
    else:
        use_range = np.arange(start,top_range).astype(int)
    waveforms = waveforms[use_range,:]
    timestamps = timestamps[use_range]
    # ************************************************************
    # ** Enforce all CAPs to have the same waveform dim. (zanos: prefered_dim_of_wf=141) ****
    # ************************************************************
    if d0_wf != prefered_dim_of_wf:
        wf = np.zeros((waveforms.shape[0],prefered_dim_of_wf))
        if d0_wf >= prefered_dim_of_wf:
            wf[:,:] = waveforms[:,:prefered_dim_of_wf] # disregard last dimensions of waveform..
        else:
            wf[:,:d0_wf] = waveforms[:,:] # The last elements remain zero..
        del(waveforms) 
        waveforms = wf # Let waveform point on the waveforms of the standard dimension.

    return waveforms, timestamps


def apply_mean_ev_threshold(waveforms, timestamps, mean_event_rates, hypes, ev_labels=None):
    '''
    Applies threshold to only consider "high occurance"-CAPs as specified by a set threshold.
    If "ev_thresh_procentage" is Fasle: 
        Returns waveforms (with corresponding timestamps) which has a mean event rate above ev_threshold.
    If True:
        Calculated threshold based on total event-rate. ev_threshold is then interpreted as a fraction. 
        e.g. ev_threshold=0.01 => CAP-EV must be at least 1% of total ev.

    Parameters
    ----------
    waveforms : (number_of_waveforms, dim_of_waveform) array_like
            CAP-waveforms 
    timestaps : (number_of_waveforms, ) array_like 
            Vector containing timestamp for each waveform in seconds

    mean_event_rate : (n_wf,) array_like
            Estimated mean event rate during the considered recording time for all CAPs.
    hypes : .json file
            Containing hyperparameters
    ev_labels :  (3, n_wf) array_like

    Returns: 
    --------
    high_occurance_wf : (n_ho_wf, dim_of_waveforms) array_like

    high_occurance_ts : (n_ho_wf,) array_like

    high_occurance_ev_labels: (n_ho_labels, 3) array_like
            OBS: transposed shape compared to input to be more consistent..
    '''
    # *** Extract hyperparameters from json file: **
    ev_thresh_procentage = hypes["preprocess"]["ev_thresh_procentage"]
    ev_threshold = hypes["preprocess"]["ev_thresh_fraction"]
    relative_EV = hypes["labeling"]["relative_EV"]
    # **********************************************

    # Select event-rate threshold unique for each recording.. 
    if ev_thresh_procentage:
        if relative_EV:
            # The calculated event-rate is already in procentage of total..
            pass

        else:
            # Find absolute threshold asspecified  procentage of total.
            T  = timestamps[-1]-timestamps[0] # length of used recording in seconds.
            N = waveforms.shape[0]
            mean_event_rate = N/T
            #ev_thresh_procentage = 0.01 # ie 1%
            ev_threshold = mean_event_rate*ev_threshold
            print(f'Overall mean event rate = {mean_event_rate}, and thus ev_threshold = {ev_threshold}')
    
    idx = np.where(mean_event_rates>ev_threshold)

    if ev_labels is not None:
        high_occurance_ev_labels = ev_labels[:,idx]
        high_occurance_wf = waveforms[idx,:]
        high_occurance_ts = timestamps[idx]
        return np.squeeze(high_occurance_wf), np.squeeze(high_occurance_ts), np.squeeze(high_occurance_ev_labels).T  # [0,:,:], [:,0], [:,0,:].T
    
    else:
        high_occurance_wf = waveforms[idx,:]
        high_occurance_ts = timestamps[idx]
        return np.squeeze(high_occurance_wf), np.squeeze(high_occurance_ts)


def apply_amplitude_thresh(waveforms, timestamps, hypes): 
    '''
    Removes CAPs with amplitude above/below threeshold values in hypes .json file.

    Parameters
    ----------
    waveforms : (number_of_waveforms, dim_of_waveform) array_like

    timestaps : (number_of_waveforms, ) array_like 
            Vector containing timestamp for each waveform in seconds
    hypes : .json file
            Containing hyperparameters

    Returns
    -------
    waveforms_pre_thresh : (n_wf_pre_thresh, dim_of_waveform )
    Remaining waveforms after applied threshold
    '''
    # *** Extract hyperparameters from json file: **
    maxamp_threshold = hypes["preprocess"]["maxamp_threshold"]
    minamp_threshold = hypes["preprocess"]["minamp_threshold"]
    # **********************************************
    max_amps = np.max(waveforms, axis=1)
    min_amps = np.min(waveforms, axis=1)
    keep_idx = (max_amps < maxamp_threshold)&(max_amps > minamp_threshold) & (min_amps > -maxamp_threshold) & (min_amps<-minamp_threshold)
    #keep_idx = (max_amps < maxamp_threshold)&(max_amps > minamp_threshold) & (min_amps > -max_amps)
    return waveforms[keep_idx],timestamps[keep_idx]


if __name__ == "__main__":
    pass