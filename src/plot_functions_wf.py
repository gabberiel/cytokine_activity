'''
Functions for plotting. 

'''
import numpy as np
import matplotlib.pyplot as plt

# Specify Fontsizes:
plt.rcParams.update({'font.size': 18})
#plt.rc('figure', titlesize=44)
plt.rc('axes', labelsize=16)
plt.rc('axes', titlesize=18)
plt.rc('ytick', labelsize=11) 
plt.rc('xtick', labelsize=11)
plt.rc('legend', fontsize=16)    # legend fontsize


def plot_decoded_latent(decoder, resolution=6, saveas=None, verbose=1, ev_label=None):
    '''
    Takes (resolution x resolution) samples from grid in latent space and plots the decoded x-mean.
    We assume x ~ N(mu_x, c*I). The functions then does as follows:
        * sample z. (takes values in evenly spaced grid.)
        * Use decoder to get mu_x = f(z; theta)
        * Plot mu_x in grid-subplots

    Will show plot if verbose = 1. \\
    Will save figure if saveas is a valid path.

    If ev_label is None then it is assumed that the encoder is part of a VAE
    Otherwise a CVAE.

    Parameters
    ----------
    decoder : keras.Model class_instance
        Decoder part of VAE/CVAE
    
    resolution : integer
        Spacing of latent space grid.
    
    saveas : 'path/to/save_fig' string_like
        if None then the figure is not saved
    
    verbose : integer_like
        verbose>0 => plt.show()
    
    ev_label : (num_of_wf, label_dim) array_like or None
        Determens if a vae or cvae model is used.

    Notes
    -----
    Sample (z1,z2) in grid [-2,2] x [-2,2] . 

    z1,z2 in R^2 --- decoder --> waveform in R^waveform_dim

    '''
    print('\nConstructing plot of decoded latent space...\n')
    fig = plt.figure(constrained_layout=True)
    resolution = 6
    spec = fig.add_gridspec(ncols=resolution+1, nrows=resolution+1)
    for i,z1 in enumerate(np.arange(-2,2 + 4/resolution,4/resolution)):
        for j, z2 in enumerate(np.arange(-2,2 + 4/resolution,4/resolution)):
            ax = fig.add_subplot(spec[j, i])
            latent_z = np.array((z1,-z2))
            #print(latent_z)
            latent_z = np.reshape(latent_z, (1,2))
            if ev_label is not None:
                reconstructed = decoder([latent_z.astype(float), ev_label.astype(float)])
            else:
                reconstructed = decoder(latent_z)
            reconstructed = reconstructed.numpy()
            ax.plot(reconstructed[0,0:-1:5]) # speed up plot a bit by only plotting every fifth datapoint. 
            ax.set(xticks=[], yticks=[])
            if j == resolution:
                plt.xlabel(np.round(z1, decimals=1))
            if i == 0:
                plt.ylabel(np.round(-z2, decimals=1))
    for ax in fig.get_axes():
        ax.spines['top'].set_visible(ax.is_first_row())
        ax.spines['bottom'].set_visible(ax.is_last_row())
        ax.spines['left'].set_visible(ax.is_first_col())
        ax.spines['right'].set_visible(ax.is_last_col())
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    #plt.title('Decoded Latent Space')
    if verbose>0:
        plt.show()


def plot_encoded(encoder, data, labels_to_plot=[0,1], saveas=None, verbose=1, ev_label=None, title=None):
    '''
    Display a 2D plot of the latent space mean. \\
    Will show plot if verbose=1. \\
    Will save figure if saveas is a valid path. \\

    If ev_label is ``None`` then it is assumed that the encoder is part of a VAE
    Otherwise a CVAE.

    Parameters
    ----------
    encoder : keras.Model class_instance
        Encoder part of VAE
    
    data : (num_of_wf, size_of_wf) array_like
        Data to be visualised in laten space.

    labels_to_plot : list.
        label-index to plot. \\
        [0,1] => both injections. \\
        [0] => first injection. etc. \\
    
    saveas : 'path/to/save_fig' string_like
        if ``None`` then the figure is not saved
    
    verbose : integer_like
        verbose=1 => plt.show()
    
    ev_label : (num_of_wf, label_dim) array_like or ``None``
        Determens if a vae or cvae model is used.

    '''
    if ev_label is not None:
        # Assumes CVAE
        z_mean, _, _  = encoder([data, ev_label])
    else:
        # Assumes VAE (old)
        z_mean, _, _  = encoder(data)
    z_mean = z_mean.numpy()
    assert z_mean.shape[-1] == 2, '[plot_encoded] PLOT ONLY POSSIBLE FOR 2D LATENT SPACE'
    labels = ['First','Second']
    for label_on in labels_to_plot:
        z_mean_labeled = z_mean[ev_label[:,label_on]==1]
        plt.scatter(z_mean_labeled[:, 0], z_mean_labeled[:, 1], label=labels[label_on])
    plt.xlabel("$\mu_{1}$")
    plt.ylabel("$\mu_{2}$")
    plt.legend()
    if title is not None:
        plt.title(title)
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose==1:
        plt.show()


def plot_rawrec(rawrec, sample_freq=8000, saveas=None, verbose=False, title=None):
    '''
    Plot of raw recording before MATLAB preprocessing.
    Assumes downsampled (5) raw_rec such that the sample frequency is 8000 Hz.

    Parameters
    ----------
    rawrec : (n_samples, ) array_like
        The raw recoding data
    
    sample_freq : int. 
        Sample Frequency to define time-axis.
    
    saveas : 'path/to/save_fig' string_like
        if ``None`` then the figure is not saved
    
    verbose : integer_like
        verbose=1 => plt.show()
    
    title : string, or ``None``.
        if ``None`` => No title

    '''
    timeline = np.arange(0, rawrec.shape[0]) /sample_freq /60 # Timeline in minutes
    plt.plot(timeline, rawrec)#, c=labels)
    plt.xlabel("Time (min)")
    # plt.ylabel("Voltage ($\mu V$)")
    if title is not None:
        plt.title(title)
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose==True:
        plt.show()

def plot_waveforms_grid(waveforms, N, saveas=None, verbose=False, title=None):
    ''' 
    Plot N x N grid of observed CAP-waveforms as : 
        ``waveforms(i) for i in range(N^2)``.

    args:
    -----
    waveforms : (n_wf, dim_wf)
        Waveforms / CAPs to use for plot. \\
        OBS! Must have n_wf >= N^2
    
    N : int
        Grid size. Plots N^2 waveforms.
    
    saveas : 'path/to/save_fig' string_like
        if ``None`` then the figure is not saved
    
    verbose : integer_like
        verbose=1 => plt.show()
    
    title : string, or ``None``.
        if ``None`` => No title

    '''
    num_wf = waveforms.shape[0]
    wf_dim = waveforms.shape[-1]
    t_axis = np.arange(0, 3.49, 3.5/wf_dim)
    for i in range(N*N):
        plt.subplot(N,N,i+1)
        plt.xticks([]), plt.yticks([])
        plt.plot(t_axis.T,waveforms[i,:])
    #plt.xlabel('Time $(ms)$')
    # plt.ylabel('Voltage $(\mu V)$')
    if title is not None:
        plt.title(title)
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose==True:
        plt.show()
        
def plot_waveforms(waveforms, labels=None, saveas=None, verbose=False, title=None):
    ''' 
    
    If labels are given, then the meadian of each waveform-cluster is ploted. 
    x_axis assumes each waveform is 3.5ms long.

    Parameters:
    ----------
    waveforms : (n_wf, dim_wf)
        Waveforms / CAPs to use for plot. \\
        OBS! Must have n_wf >= N^2
    
    labels : (n_wf, 3)
        Grid size. Plots N^2 waveforms.
    
    saveas : 'path/to/save_fig' string_like
        if ``None`` then the figure is not saved
    
    verbose : integer_like
        verbose=1 => plt.show()
    
    title : string, or ``None``.
        if ``None`` => No title
    '''
    num_wf = waveforms.shape[0]
    wf_dim = waveforms.shape[-1]
    if labels is not None:
        t_axis = np.arange(0,3.49,3.5/wf_dim)

        for cluster in np.unique(labels):
            wf_clust = waveforms[labels==cluster]
            plt.plot(t_axis.T, np.median(wf_clust, axis=0).T)# ,color = (0.7,0.7,0.7),lw=0.2)
    #plt.plot(time,np.median(waveforms[ind,:,0],axis=0),color = (0.2,0.2,0.2),lw=1)
    else: 
        t_axis = np.arange(0, 3.49, 3.5 / wf_dim) * np.ones((num_wf, 1))
        plt.plot(t_axis.T, waveforms.T, lw=2)#, color =  (0.2,0.6,0.6))
    plt.xlabel('Time $(ms)$')
    # plt.ylabel('Voltage $(\mu V)$')
    if title is not None:
        plt.title(title)
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose==True:
        plt.show()

def plot_simulated(cvae,waveform,ev_label=None,n=3,var=0.5, saveas=None, verbose=False,title=None):
    '''
    Plots input waveform together with the corresponding predicted mean using the cvae
    and draws n samples from the corresponding distribution: x_sim ~ N(mean,var*I)

    Parameters
    ----------
    cvae : keras.Model 
        Trained conditional vae-model.

    waveform : (1,dim_of_waveform) array_like
        Waveform to reconstruct using cvae 
    ev_label : (1, label_dim) array_like 
        Input to cvae
    n : integer
        how many draws from dist
    var : assumed variance of distribution.
    saveas : 'path/to/save_fig' string_like
        if None then the figure is not saved
    verbose : Boolean
        verbose=True => plt.show()
    '''
    dim_of_waveform = waveform.shape[1]
    x = waveform
    label = ev_label

    x_rec = cvae([x,label]).numpy()
    time = np.arange(0,3.49,3.5/dim_of_waveform)

    #plt.figure()
    if n==1:
        x_sample = np.random.multivariate_normal(x_rec.reshape((dim_of_waveform,)),np.eye(dim_of_waveform)*var)
        plt.plot(time,x_sample,color =  (0,0,0.7),lw=1, label='$x_{sim}$')
    else:
        for i in range(n):
            x_sample = np.random.multivariate_normal(x_rec.reshape((dim_of_waveform,)),np.eye(dim_of_waveform)*var)
            if i==0: # Allow for one label for all sampled waveforms.
                plt.plot(time,x_sample,color =  (0.6,0.6,0.6),lw=0.5, label='$x_{sim}$')
            else:
                plt.plot(time,x_sample,color =  (0.6,0.6,0.6),lw=0.5) #, label='_nolegend_')
    
    plt.plot(time,x.reshape((dim_of_waveform,)),color = (0,0,0),lw=1,label='$x$')
    plt.plot(time,x_rec.reshape((dim_of_waveform,)),color = (1,0,0),lw=1,label='$\mu_{x}$')
    plt.xlabel('Time $(ms)$')
    # plt.ylabel('Voltage $(\mu V)$')
    if title is None:
        plt.title('Simulating $x \sim \mathcal{N}(\mu_x,0.5 \mathcal{I} )$')
    else:
        plt.title(title)
    plt.legend(loc='upper right')
    if saveas is not None:
        plt.savefig(saveas,dpi=150)
    if verbose is True:
        plt.show()
    #plt.close()

def plot_similar_wf(candidate_idx, waveforms, bool_labels,
                    saveas=None, verbose=True,
                    show_clustered=True, 
                    return_cand=False, title=None):
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
    
    verbose : Boolean
        True => plt.show()
    
    show_clustered : boolean
        True => All similar waveforms are plotted as "gray background".
    
    return_cand : Boolean
        True => mean of candidate waveform is returned.

    Returns
    -------
        None
    '''
    
    nr_of_wf_in_cluster = np.sum(bool_labels)
    print(f'Number of waveforms above threshold for wf_idx={candidate_idx} : {nr_of_wf_in_cluster}.')

    if np.sum(bool_labels)>500:
        # If more than 500 wavforms are given, then 500 indexes are sampled to speed up plotting.

        true_idx = np.where(bool_labels==True)
        idx_sample = np.random.choice(true_idx[0], size=500, replace=False)
        new_bool_labels = np.zeros(bool_labels.shape)
        new_bool_labels[idx_sample] = 1 
        bool_labels = new_bool_labels == 1 # Convert to boolean
        #print('Plotting 500...')

    time = np.arange(0,3.49,3.5/waveforms.shape[-1])
    mean_wf = np.mean(waveforms[bool_labels],axis=0)
    #plt.figure()
    if show_clustered:
        plt.plot(time, waveforms[bool_labels].T, color=(0.6,0.6,0.6), lw=0.5)
        plt.plot(time, mean_wf, color = (1,0,0), lw=1, label='Mean')
        plt.plot(time, waveforms[candidate_idx,:], color = (0.1,0.1,0.1), lw=1, label='Candidate')
    else:    
        plt.plot(time, mean_wf, lw=3) #, label='Cluster '+str(cluster))

    plt.xlabel('Time $(ms)$')
    # plt.ylabel('Voltage $(\mu V)$')
    if title is None:
        plt.title('CAPs similar to "Candidate".')
    else:
        plt.title(title)
    plt.legend(loc='upper right')
    if saveas is not None:
        plt.savefig(saveas,dpi=150)
    if verbose:
        plt.show()
    if return_cand:
        return mean_wf
    

      
def plot_event_rates(event_rates, timestamps, hypes, 
                     conv_width=100, 
                     tot_EV=False,
                     linewidth=2, 
                     saveas=None, verbose=True, 
                     title='Event-Rate'):
    '''
    Plots event rates for each "waveform-cluster".
    A smoothing kernel is applied of width "conv_width".
    The convolution will result in some boundary effects, therefore
    the plot removes "conv_width" from start and end of event-rate plots.

    Parameters
    ----------
    event_rates: (total_time_in_seconds, number_of_clusters) array_like
            Number of occurances of labeled waveforms in each one second window during time
            of recording. 

    timestamps : (n_waceforms_of_interest, ) array_like
        timestamps of the waveforms of interest. 
        Only used to define end time of event-rate plot.

    conv_width: Integer_like
            Size of smoothing kernel window.
    
    tot_EV : Boolean
        True if total event-rate is to be plotted. This input is used to make sure that the y-label
        is correct when a relative event-rate is used in analysis. 
    
    linewidth : Integer
        linewidth in plot.
    
    saveas : 'path/to/save_fig' string_like _or_ None
            If None then the figure is not saved
    
    verbose : Boolean
        True => plt.show()
    
    title : string.
        Title of plot

    Returns
    -------
    None
    '''
    relative_EV = hypes['labeling']['relative_EV']  # Used for y-label
    injection_t_period = hypes['experiment_setup']['injection_t_period']  # Used for y-label

    end_time = timestamps[-1]
    number_of_obs = event_rates[:, 0].shape[0]
    time = np.arange(0,end_time,end_time/number_of_obs) / 60 # To minutes
    conv_kernel = np.ones((conv_width))* 1/conv_width
    max_ev = 0
    for ev in event_rates.T:
        smothed_ev = np.convolve(ev, conv_kernel,'same')
        plt.plot(time[conv_width:-conv_width].T, smothed_ev[conv_width:-conv_width], linestyle='-', lw=linewidth)
        max_ev_prel = np.max(smothed_ev)
        if max_ev < max_ev_prel:
            max_ev = max_ev_prel
            
    # Plot lines at injection times:        
    plt.vlines(injection_t_period, 0, max_ev, linestyles='--', colors='k')
    plt.vlines(injection_t_period*2, 0, max_ev, linestyles='--', colors='k')

    plt.xlabel('Time of Recording (min)')

    if (not relative_EV) or tot_EV:
        plt.ylabel('Event-Rate (CAPs/sec)')
    else:
        plt.ylabel('Fraction of tot EV')

    plt.title(title)

    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose:
        plt.show()


def plot_amplitude_hist(waveforms, saveas=None, verbose=True):
    """
    Plots histogram distribution of min/max - amplitude of 
    CAP-waveforms. 
    Mainly used to determine amplitide thresholds. 

    args:
    -----
    waveforms : (n_wf, dim_wf) array_like
        Waveforms to find distribution of max/min amplitudes.
    
    saveas : 'path/to/save_fig' string_like _or_ None
        If None then the figure is not saved
    
    verbose : Boolean
        True => plt.show()
    """
    max_amps = np.max(waveforms,axis=1)
    min_amps = np.min(waveforms,axis=1)

    print(f'shape of max_amps: {max_amps.shape}')
    plt.hist(max_amps, bins=100, density=False)
    plt.hist(min_amps, bins=100, density=False)

    plt.xlabel('Max/Min Amplitude')
    plt.title('Distribution Max/Min Amplitudes')
    if saveas is not None:
        plt.savefig(saveas+'tot_mean_ev',dpi=150)
    if verbose:
        plt.show()


def plot_event_rate_stats_hist(ev_stats_tot, saveas=None, verbose=False):
    """
    Plots histrogram-distribution of mean event-rate and
    standard deviation obtained during the labeling of CAPs.
    "ev_stats_tot" is saved as .npy file by the function "get_ev_labels()".
    """
    plt.hist(ev_stats_tot[0],density=True,bins=40)
    plt.xlabel('$\mu_{EV}$')
    plt.title('Distribution of Mean Event-Rate.')
    if saveas is not None:
        plt.savefig(saveas+'tot_mean_ev',dpi=150)
    if verbose is True:
        plt.show()
    plt.close()
    plt.hist(ev_stats_tot[1],bins=40,density=True)
    plt.xlabel('$\sigma_{EV}$')
    plt.title('Distribution of Event-Rate Standard Deviations.')
    if saveas is not None:
        plt.savefig(saveas+'tot_std_ev',dpi=150)
    if verbose is True:
        plt.show()
    plt.close()