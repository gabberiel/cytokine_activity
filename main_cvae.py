# ************************************************************
# Main file in cytokine-identification process. 
#
# Master Thesis, KTH , Gabriel Andersson 
# ************************************************************

import numpy as np
import matplotlib.pyplot as plt
from os import path
from plot_functions_wf import *
from main_functions import *
from sklearn.cluster import KMeans, DBSCAN
from event_rate_first import *
import preprocess_wf 
from wf_similarity_measures import *

# OBS assumes existance of the standardise

# ************************************************************
# ******************** Parameters ****************************
# ************************************************************

similarity_measure='corr'

# VAE training params:
continue_train = True
nr_epochs = 100 # if all train data is used -- almost no loss-decrease after 100 batches..
batch_size = 128

view_vae_result = True # True => reqires user to give input if to continue the script to pdf-GD or not.. 
view_GD_result = True # This reqires user to give input if to continue the script to clustering or not.

run_DBscan = False
run_KMeans = False

verbose_main = 1

# pdf GD params: 
# For first attempt of CVAE. m=2000 sufficient for very low variance in hpdp. All converged to same cluster. (looking at latent space..)
# After 3000 iterations, this cluster split in two but the two are basically encoding the same waveform
m=1000 # Number of steps 
gamma=0.01 # learning_rate
eta=0.005 # Noise variable -- adds white noise with variance eta to datapoints during GD.


# DBSCAN params
db_eps = 0.1 # max_distance to be considered as neighbours 
db_min_sample = 35 # Minimum members in neighbourhood to not be regarded as Noise.

# Shape of waveforms: (136259, 141)
#training_idx = np.arange(10000) # initial testing
#training_idx = np.arange(0,131000,10)

# ************************************************************
# ******************** Paths *********************************
# ************************************************************
#recording = 'amp_thresh_R10_Exp2_71516_BALBC_TNF_05ug_IL1B_35ngperkg_10.mat'
# General: 
recording = 'R10_Exp2_71516_BALBC_TNF_05ug_IL1B_35ngperkg_10.mat'
path_to_wf = '../matlab_files/waveforms'+recording+'.mat' 
path_to_ts = '../matlab_files/timestamps'+recording+'.mat'

# FOR TESTING:
recording = 'new_computer_deleteme3'
path_to_wf = '../matlab_files/gg_waveforms-R10_IL1B_TNF_03'+'.mat'
path_to_ts = '../matlab_files/gg_timestamps'+'.mat'

# FOR SAVING FIGURES
unique_string_for_run = 'corr_2_dec'+recording
# tf weight-file:
path_to_weights = 'models/'+unique_string_for_run
# Numpy file:
path_to_hpdp = "../numpy_files/numpy_hpdp/"+unique_string_for_run
path_to_EVlabels = "../numpy_files/EV_labels/"+unique_string_for_run

# ************************************************************
# ******************** Load Files ****************************
# ************************************************************

# TODO Move preprocessing from "load_waveforms function"
load_data = True
if load_data:
    waveforms, mean, std = load_waveforms(path_to_wf,'waveforms',standardize=False, verbose=1)
    timestamps = load_timestamps(path_to_ts,'gg_timestamps',verbose=1)
    n0_wf = waveforms.shape[0]
    d0_wf = waveforms.shape[1] 

    print(f'Shape of training data: {waveforms.shape}')

# ************************************************************
# ****** Cut first and last part of recording to ensure stable
#        sleep state during recording ************************
# ************************************************************
remove = 5000 
top_range = round((n0_wf-remove)/1000)*1000
use_range = np.arange(remove,top_range)
waveforms = waveforms[use_range,:]
timestamps = timestamps[use_range]
print(waveforms.shape)
# ************************************************************
# ** Enforce all recording to have the same waveform dim. ****
# ************************************************************

# ************************************************************
# ******************** Preprocess ****************************
# ************************************************************
# Enforce all recording to have the same waveform dim. (141):
wf = np.zeros((waveforms.shape[0],141))
if d0_wf >= 141:
    wf[:,:] = waveforms[:,:141] # disregard last dimensions of waveform..
else:
    wf[:,:d0_wf] = waveforms[:,:] # The last elements remain zero..
del(waveforms) 
waveforms = wf # Let waveform point on the waveforms of the standard dimension.

# Standardise waveforms
waveforms = preprocess_wf.standardise_wf(waveforms)

# ************************************************************
# ******************** Event-rate Labeling *******************
# ************************************************************

if path.isfile(path_to_EVlabels+'.npy'):
    print()
    print(f'Loading saved EV-labels from {path_to_EVlabels}')
    print()
    ev_labels = np.load(path_to_EVlabels+'.npy')
    ev_stats_tot = np.load(path_to_EVlabels+'tests_tot.npy') 
else:
    ev_labels, ev_stats_tot = get_ev_labels(waveforms,timestamps,threshold=0.5,saveas=path_to_EVlabels,similarity_measure=similarity_measure)

print()
print(f'Number of wf which ("icreased after first","increased after second", "constant") = {np.sum(ev_labels,axis=1)} ')

# ho := High Occurance, ts := timestamps
wf_ho, ts_ho, ev_label_ho = preprocess_wf.apply_mean_ev_threshold(waveforms,timestamps,ev_stats_tot[0],ev_threshold=1,ev_labels=ev_labels)

print(f'After EV threshold: ("icreased after first","increased after second", "constant") = {np.sum(ev_label_ho,axis=0)} ')

number_of_occurances= np.sum(ev_label_ho,axis=0)
# ************************************************************
# ******************** Train/Load model **********************
# ************************************************************
print()
print('*********************** Tensorflow Blaj *************************************')
print()

encoder,decoder,cvae = train_model(wf_ho, nr_epochs=nr_epochs, batch_size=batch_size, path_to_weights=path_to_weights, 
                                        continue_train=continue_train, verbose=1, ev_label=ev_label_ho)
print()
print('******************************************************************************')
print()

#view_vae_result = False # This reqires user to give input if to continue the script to GD or not.
if view_vae_result:
    save_figure = 'figures/encoded_decoded/' + unique_string_for_run
    
    plot_decoded_latent(decoder,saveas=save_figure+'_decoded_constant',verbose=verbose_main,ev_label=np.array((0,0,1)).reshape((1,3)))
    plot_decoded_latent(decoder,saveas=save_figure+'_decoded_increase_second',verbose=verbose_main,ev_label=np.array((0,1,0)).reshape((1,3)))
    plot_decoded_latent(decoder,saveas=save_figure+'_decoded_increase_first',verbose=verbose_main,ev_label=np.array((1,0,0)).reshape((1,3)))

    continue_to_run_GD = input('Continue to gradient decent of pdf? (yes/no) :')

    all_fine = False
    while all_fine==False:
        if continue_to_run_GD=='no':
            exit()
        elif continue_to_run_GD=='yes':
            print('Continues to "run_GD"')
            all_fine = True
        else:
            continue_to_run_GD = input('Invalid input, continue to gradient decent of pdf? (yes/no) :')


# ************************************************************
# ** Perform GD on pdf to find high prob. data-points (hpdp) *
# ************************************************************  


run_GD = True
#view_GD_result = True # This reqires user to give input if to continue the script to clustering or not.
label_on = 1
waveforms_increase_second = wf_ho[ev_label_ho[:,label_on]==1]
ev_label_corr_shape = np.zeros((waveforms_increase_second.shape[0],3))
ev_label_corr_shape[:,label_on] = 1
print(f'waveforms_increase_second : {waveforms_increase_second.shape}')
if run_GD:
    print()
    print('Running pdf_GD to get hpdp...')
    print()
    # To easy computational load -- only every 20th data-point is used..
    hpdp = pdf_GD(cvae, waveforms_increase_second,ev_label=ev_label_corr_shape, m=m, gamma=gamma, eta=eta, path_to_hpdp=path_to_hpdp,verbose=verbose)
    #hpdp = pdf_GD(vae, wf_train, m=m, gamma=gamma, eta=eta, path_to_hpdp=path_to_hpdp,verbose=verbose)

    if view_GD_result:
        save_figure = 'figures/hpdp/' + unique_string_for_run
        print(f'Visualising decoded latent space of hpdp...')
        print()
        plot_encoded(encoder, hpdp, saveas=save_figure+'_encoded_hpdp', verbose=1,ev_label=ev_label_corr_shape)        
        continue_to_Clustering = input('Continue to Clustering? (yes/no) :')

        all_fine = False
        while all_fine==False:
            if continue_to_Clustering=='no':
                exit()
            elif continue_to_Clustering=='yes':
                print('Continues to "run_GD"')
                all_fine = True
            else:
                continue_to_Clustering = input('Invalid input, continue to Clustering? (yes/no) :')

else:
    print()
    print('Skipps over pdf_GD...')
    print()


# ************************************************************
# *********** Inference from increased EV hpdp ***************
# ************************************************************
gd_runs = '_3000'
plot_hpdp_assesments = False
if plot_hpdp_assesments:
    bool_labels = np.ones((hpdp.shape[0])) == 1 # Label all as True (same cluster) to plot the average form of increased EV-hpdp
    saveas = 'figures/hpdp/'+unique_string_for_run+gd_runs
    plot_correlated_wf(0,hpdp,bool_labels,None,saveas=saveas+'_wf',verbose=True)

    #PLOT ENCODED wf_increase_second... :
    save_figure = 'figures/encoded_decoded/'+unique_string_for_run + '_ho_second'
    plot_encoded(encoder, waveforms_increase_second, ev_label =ev_label_corr_shape, saveas=save_figure+'_encoded', verbose=True)
    plot_encoded(encoder, hpdp, saveas=save_figure+'_encoded_hpdp', verbose=1,ev_label=ev_label_corr_shape) 

#MAIN_CANDIDATE = np.median(hpdp,axis=0)
MAIN_CANDIDATE = np.mean(hpdp,axis=0)

added_main_candidate_wf = np.concatenate((MAIN_CANDIDATE.reshape((1,MAIN_CANDIDATE.shape[0])),waveforms),axis=0)
assert np.sum(MAIN_CANDIDATE) == np.sum(added_main_candidate_wf[0,:]), 'Something wrong in concatenate..'

# Get correlation cluster for Delta EV - increased_second hpdp
MAIN_THRES = 0.6
saveas = 'figures/event_rate_labels/'+unique_string_for_run+gd_runs
if similarity_measure=='corr':
    correlations = wf_correlation(0,added_main_candidate_wf)
    bool_labels = label_from_corr(correlations,threshold=MAIN_THRES,return_boolean=True)
if similarity_measure=='ssq':
    added_main_candidate_wf = added_main_candidate_wf/0.6 # Assumed var in ssq
    bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=0.2)
print(f'Bool labels = {bool_labels[:10]}...')
event_rates, real_clusters = get_event_rates(timestamps,bool_labels[1:],bin_width=1,consider_only=1)
plot_correlated_wf(0,added_main_candidate_wf,bool_labels,MAIN_THRES,saveas=saveas+'Main_cand'+'_wf',verbose=True )

plot_event_rates(event_rates,timestamps,noise=None,conv_width=20,saveas=saveas+'Main_cand'+'_ev', verbose=True) 


exit()
# ************************************************************
# ******************** OLD STUFF FROM HERE ON ****************
# ************************************************************


# ************************************************************
# ******************** Cluster wf using hpdp *****************
# ******************** This to access labels *****************
# ************************************************************


if run_KMeans:
    print()
    print('Running KMeans on hpdp...')
    print()
    kmeans = KMeans(n_clusters=10, random_state=0).fit(hpdp)
    labels = kmeans.labels_
else:
    print()
    print('Skipps over KMeans...')
    print()

if run_DBscan:
    print()
    print('Running DBSCAN on hpdp...')
    print()
    dbscan = DBSCAN(eps=db_eps, min_samples=db_min_sample, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
    hpdp_latent_mean,_,_ = encoder.predict(hpdp)
    dbscan.fit(hpdp_latent_mean)
    labels = dbscan.labels_ #  Noisy samples are given the label -1
else:
    print()
    print('Skipps over DBSCAN...')
    print()


# ************************************************************
# ******************** Event Rate *****************
# ************************************************************
if 'labels' in locals():
    run_event_rate = True
else:
    run_event_rate = False

if run_event_rate:
    print('Calculating Event rates...')
    event_rates, real_clusters = get_event_rates(ts_train,labels,bin_width=1)
    delta_ev, ev_stats = delta_ev_measure(event_rates)
    print(f'Delta_ev_measure : {delta_ev}')

    #event_rates, real_clusters = get_event_rates(ts_train,labels,bin_width=1)
    print(f'Real cluster (with mean event_rate over 0.5 is CAPs {real_clusters})')
    plot_event_rates(event_rates,ts_train,saveas=save_figure+'_event_rate', conv_width=30)
    #plot_event_rates(event_rates,ts_train,saveas=save_figure+'_event_rate', conv_width=30)

# ************************************************************
# ******************** General PLOTTING ******************************
# ************************************************************

X = waveforms[0:10,:]
X_rec = vae.predict(X)
plot_waveforms(X,labels=None)
plt.show()
plot_waveforms(X_rec,labels=None)
plt.show()


if verbose>1:
    print()
    print(f'Plotting waveforms of each cluster if labels are specified...')
    print()
    plot_waveforms(hpdp[0:10,:],labels=None)
    print()
    print(f'Visualising decoded latent space of hpdp...')
    print()
    plot_encoded(encoder, hpdp[0:1000,:], saveas=save_figure+'_hpdp_encoded', verbose=verbose)
    print()
    print(f'Visualising decoded latent space...')
    print()
    #plot_decoded_latent(decoder,saveas=save_figure+'_decoded',verbose=1)
    plot_encoded(encoder, wf_train, saveas=save_figure+'_encoded', verbose=1)






