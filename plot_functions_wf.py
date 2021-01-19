import numpy as np
import matplotlib.pyplot as plt
import preprocess_wf
from wf_similarity_measures import wf_correlation, label_from_corr, similarity_SSQ
from event_rate_funs import get_event_rates
'''
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 28
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("figure", figsize=(10, 8))
'''

plt.rcParams.update({'font.size': 18})
#plt.rc('figure', titlesize=44)
plt.rc('axes', labelsize=16)
plt.rc('axes', titlesize=18)
plt.rc('ytick', labelsize=11) 
plt.rc('xtick', labelsize=11)
plt.rc('legend', fontsize=16)    # legend fontsize
#plt.rc(set_minor_formatter(FormatStrFormatter('% 1.2f'))

#plt.rc("figure", figsize=(10, 8))

def plot_decoded_latent(decoder,resolution=6,saveas=None, verbose=1,ev_label=None):
    '''
    TODO : Fix title/axes..

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
    if verbose==1:
        plt.show()


def plot_encoded(encoder, data, saveas=None,verbose=1,ev_label=None,title=None):
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
    
    assert z_mean.shape[-1] == 2, 'PLOT ONLY POSSIBLE FOR 2D LATENT SPACE'
    #plt.figure(figsize=(10, 10))
    labels = ['First','Second']
    for label_on in [0,1]:
        z_mean_labeled = z_mean[ev_label[:,label_on]==1]
        plt.scatter(z_mean_labeled[:, 0], z_mean_labeled[:, 1], label=labels[label_on])
    #plt.scatter(z_mean[:, 0], z_mean[:, 1])#, c=labels)
    #plt.colorbar()
    plt.xlabel("$\mu_{1}$")
    plt.ylabel("$\mu_{2}$")
    plt.legend()
    if title is not None:
        plt.title(title)
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose==1:
        plt.show()


def plot_rawrec(rawrec, sample_freq=8000, saveas=None,verbose=False,title=None):
    '''
    Plot of raw recording before MATLAB preprocessing.
    Assumes downsampled (5) raw_rec such that the sample frequency is 8000 Hz
    Parameters
    ----------

    '''
    timeline = np.arange(0,rawrec.shape[0]) /sample_freq /60 # Timeline in minutes
    #plt.figure(figsize=(10, 10))
    plt.plot(timeline, rawrec)#, c=labels)
    plt.xlabel("Time (min)")
    # plt.ylabel("Voltage ($\mu V$)")
    if title is not None:
        plt.title(title)
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose==True:
        plt.show()

def plot_waveforms_grid(waveforms,N, saveas=None,verbose=False,title=None):
    ''' 
    Plot N x N observed CAP-waveforms in grid. 
    '''
    num_wf = waveforms.shape[0]
    wf_dim = waveforms.shape[-1]
    t_axis = np.arange(0,3.5,3.5/wf_dim)
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
        
def plot_waveforms(waveforms,labels=None,saveas=None,verbose=False,title=None):
    ''' 
    If labels are given, then the meadian of each waveform-cluster is ploted. 
    x_axis assumes each waveform is 3.5ms long.
    '''
    num_wf = waveforms.shape[0]
    wf_dim = waveforms.shape[-1]
    if labels is not None:
        t_axis = np.arange(0,3.5,3.5/wf_dim)

        for cluster in labels:
            wf_clust = waveforms[labels==cluster]
            plt.plot(t_axis.T,np.median(wf_clust,axis=0).T)# ,color = (0.7,0.7,0.7),lw=0.2)
    #plt.plot(time,np.median(waveforms[ind,:,0],axis=0),color = (0.2,0.2,0.2),lw=1)
    else: 
        t_axis = np.arange(0,3.5,3.5/wf_dim)*np.ones((num_wf,1))
        plt.plot(t_axis.T,waveforms.T,lw=2)#, color =  (0.2,0.6,0.6))
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
    verbose : Booleon
        verbose=True => plt.show()
    '''
    dim_of_waveform = waveform.shape[1]
    x = waveform
    label = ev_label

    x_rec = cvae.predict([x,label])
    time = np.arange(0,3.5,3.5/dim_of_waveform)

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

def plot_similar_wf(candidate_idx,waveforms,bool_labels,threshold,saveas=None,verbose=True,
                        show_clustered=True,cluster=None, return_cand=False, title=None):
    '''
    TODO Remove "Threshold" input and write more in docstring
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
    mean_wf = np.mean(waveforms[bool_labels],axis=0)
    #plt.figure()
    if show_clustered:
        plt.plot(time,waveforms[bool_labels].T,color = (0.6,0.6,0.6),lw=0.5)
        plt.plot(time,mean_wf,color = (1,0,0),lw=1, label='Mean')
        plt.plot(time,waveforms[candidate_idx,:],color = (0.1,0.1,0.1),lw=1, label='Candidate')
    else:    
        plt.plot(time,mean_wf,lw=3) #, label='Cluster '+str(cluster))
        #plt.plot(time,waveforms[candidate_idx,:],color = (1,0,0),lw=1, label='Candidate')

    plt.xlabel('Time $(ms)$')
    # plt.ylabel('Voltage $(\mu V)$')
    if title is None:
        plt.title('CAPs similar to "Candidate".')
    else:
        plt.title(title)
    #plt.title(f'W.F. s.t. corr > {threshold}. candidate wf: {candidate_idx}, N_cluster = {nr_of_wf_in_cluster}')
    plt.legend(loc='upper right')
    if saveas is not None:
        plt.savefig(saveas,dpi=150)
    if verbose:
        plt.show()
    if return_cand:
        return mean_wf
    #plt.close()

      
def plot_event_rates(event_rates,timestamps, conv_width=100, noise=None, saveas=None,verbose=True,cluster=None,title=None):
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

    if noise is not None:
        print('Noise...')
        for i,ev in enumerate(event_rates.T):
            if i != noise:
                smothed_ev = np.convolve(ev,conv_kernel,'same')
                plt.plot(time.T, smothed_ev, linestyle='-',lw=0.5, label=f'CAP cluster {i}') #color=colors[i%3]
                plt.legend() 
    else:
        #print('No given noise..')
        for i,ev in enumerate(event_rates.T):
            smothed_ev = np.convolve(ev,conv_kernel,'same')
            if cluster is not None:
                plt.plot(time[conv_width:-conv_width].T, smothed_ev[conv_width:-conv_width], linestyle='-',lw=1) #, label=f'Cluster {cluster}') #color=colors[i%3]
            else:
                plt.plot(time[conv_width:-conv_width].T, smothed_ev[conv_width:-conv_width], linestyle='-',lw=2) #, label=f'Cluster {i}') #color=colors[i%3]
    
    plt.xlabel('Time of Recording (min)')
    plt.ylabel('Event-Rate (CAPs/sec)')
    if title is None: 
        plt.title('Event-Rate')
    else:
        plt.title(title)
    #if cluster is not None:
    #    plt.legend() 

    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose:
        plt.show()


def plot_amplitude_hist(waveforms, saveas=None,verbose=True):
    
    max_amps = np.max(waveforms,axis=1)
    min_amps = np.min(waveforms,axis=1)

    print(f'shape of max_amps: {max_amps.shape}')
    plt.hist(max_amps,bins=100,density=False)
    plt.hist(min_amps,bins=100,density=False)

    plt.xlabel('Max Amplitude')
    plt.title('Distribution Max amplitudes')
    if saveas is not None:
        plt.savefig(saveas+'tot_mean_ev',dpi=150)
    if verbose:
        plt.show()


def plot_event_rate_stats_hist(ev_stats_tot,saveas=None,verbose=False):
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
"""
def evaluate_hpdp_candidates(wf0,ts0,hpdp,k_labels,similarity_measure='corr', similarity_thresh=0.4, 
                            assumed_model_varaince=0.5,saveas='saveas_not_specified',verbose=False, return_candidates=False):
    '''
    Evaluates the results of clustered hpdp. Uses the specified similarity measure to find the event rate using the median
    of each hpdp cluster as "main candidate". The clusters as specified by k_labels.
    
    Parameters
    ----------
    wf0 : (n_wf,d_wf), array_like
        The waveforms which are used for similarity measure to evaluate candidates.
    ts0 : (n_wf,) array_like
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
    k_clusters = np.unique(k_labels)  
    candidate_wf = np.empty((k_clusters.shape[0],wf0.shape[-1]))
    for cluster in k_clusters:
        hpdp_cluster = hpdp[k_labels==cluster]
        MAIN_CANDIDATE = np.median(hpdp_cluster,axis=0) # Median more robust to outlier..

        added_main_candidate_wf = np.concatenate((MAIN_CANDIDATE.reshape((1,MAIN_CANDIDATE.shape[0])),wf0),axis=0)
        assert np.sum(MAIN_CANDIDATE) == np.sum(added_main_candidate_wf[0,:]), 'Something wrong in concatenate..'
        

        # QUICK FIX FOR WAVEFORMS AMPLITUDE INCREASING AFTER GD-- standardise it.
        # Should not be needed if GD works properly...
        
        #added_main_candidate_wf = preprocess_wf.standardise_wf(added_main_candidate_wf)

        print(f'Shape of test-dataset (now considers all observations): {added_main_candidate_wf.shape}')
        # Get correlation cluster for Delta EV - increased_second hpdp
        
        if similarity_measure=='corr':
            print('Using "corr" to evaluate final result')
            correlations = wf_correlation(0,added_main_candidate_wf)
            bool_labels = label_from_corr(correlations,threshold=similarity_thresh,return_boolean=True)
        if similarity_measure=='ssq':
            print('Using "ssq" to evaluate final result')
            if assumed_model_varaince is False:
                #added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh,standardised_input=False)
            else:
                added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh)
        event_rates, _ = get_event_rates(ts0,bool_labels[1:],bin_width=1,consider_only=1)
        plt.figure(1)
        median_wf = plot_similar_wf(0,added_main_candidate_wf,bool_labels,similarity_thresh,saveas=saveas+'Main_cand_wf',
                            verbose=False, show_clustered=False,cluster=cluster,return_cand=True)
        candidate_wf[cluster,:] = median_wf
        plt.figure(2)
        bool_labels[bool_labels==True] = cluster
        plot_event_rates(event_rates,ts0,noise=None,conv_width=100,saveas=saveas+'Main_cand_ev', verbose=False,cluster=cluster) 
    plt.figure(3)
    #plt.hist(ts0,bins=200)
    event_rates, _ = get_event_rates(ts0,np.ones((ts0.shape[0],)),bin_width=1,consider_only=1)
    plot_event_rates(event_rates,ts0,noise=None,conv_width=100,saveas=saveas+'overall_EV', verbose=False)     
    
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose:
        plt.show()
    if return_candidates:
        return candidate_wf
"""