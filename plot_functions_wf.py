import numpy as np
import matplotlib.pyplot as plt

# VERISON USED FOR ARAMS CODE ON WAVEFORM DATA
def plot_decoded_latent(decoder,resolution=6,saveas=None, verbose=1,ev_label=None):
    '''
    Takes (resolution x resolution) samples from grid in latent space and plots the decoded x-mean.
    We assume x ~ N(mu_x,I). The functions then does as follows:
        * sample z. (takes values in evenly spaced grid.)
        * Use decoder to get mu_x = f(z;theta)
        * Plot mu_x in grid-subplots


    Will show plot if verbose=1.
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

    z1,z2 in R^2 --- decoder --> waveform in R^141

    '''
    print()
    print('Constructing plot of decoded latent space...')
    print()
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
                reconstructed = decoder.predict([latent_z,ev_label])
            else:
                reconstructed = decoder.predict(latent_z)
            ax.plot(reconstructed[0,0:-1:5])
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
    plt.savefig(saveas, dpi=150)
    #plt.title('Decoded Latent Space')
    if verbose==1:
        plt.show()


def plot_encoded(encoder, data, saveas=None,verbose=1,ev_label=None):
    '''
    Display a 2D plot of the latent space mean. 
    Will show plot if verbose=1.
    Will save figure if saveas is a valid path.

    If ev_label is None then it is assumed that the encoder is part of a VAE
    Otherwise a CVAE.

    Parameters
    ----------
    encoder : keras.Model class_instance
        Encoder part of VAE
    data : (num_of_wf, size_of_wf) array_like
        Data to be visualised in laten space.
    saveas : 'path/to/save_fig' string_like
        if None then the figure is not saved
    verbose : integer_like
        verbose>0 => plt.show()
    ev_label : (num_of_wf, label_dim) array_like or None
        Determens if a vae or cvae model is used.

    '''
    if ev_label is not None:
        z_mean, log_var_out, z  = encoder.predict([data,ev_label])
    else:
        z_mean, log_var_out, z  = encoder.predict(data)
    
    assert z_mean.shape[-1] == 2, 'PLOT ONLY WORK FOR 2D LATENT SPACE'
    #plt.figure(figsize=(10, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1])#, c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose==1:
        plt.show()


def plot_rawrec(rawrec, sample_freq=8000, saveas=None,verbose=False):
    '''
    Plot of raw recording before MATLAB preprocessing.
    Assumes downsampled raw_rec such that the sample frequency is 8000 Hz
    Parameters
    ----------

    '''
    timeline = np.arange(0,rawrec.shape[0]) /sample_freq /60 # Timeline in minutes
    #plt.figure(figsize=(10, 10))
    plt.plot(timeline, rawrec)#, c=labels)
    plt.xlabel("Time (min)")
    plt.ylabel("Voltage ($\mu V$)")
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose==True:
        plt.show()


        
def plot_waveforms(waveforms,labels=None,saveas=None,verbose=False):
    ''' 
    If labels are given, then the meadian of each waveform - cluster is ploted. 
    x_axis assumes each waveform is 3.5ms long.
    '''
    num_wf = waveforms.shape[0]
    wf_size = waveforms.shape[-1]
    cols = ['r','b',]
    if labels is not None:
        t_axis = np.arange(0,3.5,3.5/wf_size)

        for cluster in labels:
            wf_clust = waveforms[labels==cluster]
            plt.plot(t_axis.T,np.median(wf_clust,axis=0).T)# ,color = (0.7,0.7,0.7),lw=0.2)
    #plt.plot(time,np.median(waveforms[ind,:,0],axis=0),color = (0.2,0.2,0.2),lw=1)
    else: 
        t_axis = np.arange(0,3.5,3.5/wf_size)*np.ones((num_wf,1))
        plt.plot(t_axis.T,waveforms.T)
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(\mu V)$')
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose==True:
        plt.show()

def plot_simulated(cvae,waveform,ev_label=None,n=3,var=0.5, saveas=None, verbose=False):
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
    verbose : Booleon
        verbose=True => plt.show()
    '''
    dim_of_waveform = waveform.shape[1]
    x = waveform
    label = ev_label

    x_rec = cvae.predict([x,label])
    time = np.arange(0,3.5,3.5/dim_of_waveform)

    #plt.figure()
    plt.plot(time,x.reshape((dim_of_waveform,)),color = (0,0,0),lw=1,label='$x$')
    plt.plot(time,x_rec.reshape((dim_of_waveform,)),color = (1,0,0),lw=1,label='$\mu_x$')
    for i in range(n):
        x_sample = np.random.multivariate_normal(x_rec.reshape((dim_of_waveform,)),np.eye(dim_of_waveform)*var)
        if i==0: # Allow for one label for all sampled waveforms. -- Does not seem to work...
            plt.plot(time,x_sample,color = (0.1,0.1,0.1),lw=0.3, label='$x_{sim}$')
        else:
            plt.plot(time,x_sample,color = (0.1,0.1,0.1),lw=0.3) #, label='_nolegend_')
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(\mu V)$')
    plt.title('Model Assessment: Simulating $x \sim \mathcal{N}(\mu_x,0.5 \mathcal{I} )$')
    plt.legend(loc='upper right')
    if saveas is not None:
        plt.savefig(saveas,dpi=150)
    if verbose is True:
        plt.show()
        #plt.close()

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
    
    nr_of_wf_in_cluster = np.sum(bool_labels)
    print(f'Number of waveforms above threshold for wf_idx={candidate_idx} : {nr_of_wf_in_cluster}.')
    # If there is more than 1000 wavforms in cluster, then 500 indexes is sampled to speed up plotting.

    if np.sum(bool_labels)>500:
        true_idx = np.where(bool_labels==True)
        idx_sample = np.random.choice(true_idx[0], size=500, replace=False)
        new_bool_labels = np.zeros(bool_labels.shape)
        new_bool_labels[idx_sample] = 1 
        bool_labels = new_bool_labels == 1 # Convert to booleon
        #print('Plotting 500...')

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