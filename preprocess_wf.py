import numpy as np
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