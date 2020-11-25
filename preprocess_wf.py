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