# ************************************************************
# Main file in cytokine-identification process. 
#
# Master Thesis, KTH , Gabriel Andersson 
# ************************************************************

import numpy as np
import matplotlib.pyplot as plt
from plot_functions_wf import *
from main_functions import *
from sklearn.cluster import KMeans, DBSCAN
from event_rate_first import *

# ************************************************************
# ******************** Parameters ****************************
# ************************************************************

# VAE training params:
continue_train = False
nr_epochs = 100 # if all train data is used -- almost no loss-decrease after 100 batches..
batch_size = 128

view_vae_result = True # True => reqires user to give input if to continue the script to pdf-GD or not.. 
view_GD_result = True # This reqires user to give input if to continue the script to clustering or not.

run_DBscan = False
run_KMeans = True


verbose = 1

# pdf GD params: 
m=0 # Number of steps 
gamma=0.01 # learning_rate
eta=0.01 # Noise variable -- adds white noise with variance eta to datapoints during GD.


# DBSCAN params
db_eps = 0.2 # max_distance to be considered as neighbours 
db_min_sample = 200 # Minimum members in neighbourhood to not be regarded as Noise.

# Shape of waveforms: (136259, 141)
#training_idx = np.arange(10000) # initial testing
training_idx = np.arange(0,131000,10)

# ************************************************************
# ******************** Paths ****************************
# ************************************************************

path_to_wf = '../matlab_files/gg_waveforms-R10_IL1B_TNF_03.mat' 
path_to_ts = '../matlab_files/gg_timestamps.mat'

save_figure = 'figures/18nov_train_from_all_data_203'
# tf weight-file:
path_to_weights = 'models/18_nov_train_from_all_data_203'
# Numpy file:
path_to_hpdp = "../numpy_files/numpy_hpdp/18_nov_train_from_all_data_204"

# ************************************************************
# ******************** Load Files ****************************
# ************************************************************
load_data = True
if load_data:
    waveforms, mean, std = load_waveforms(path_to_wf,'waveforms',standardize=True, verbose=1)
    timestamps = load_timestamps(path_to_ts,'gg_timestamps',verbose=1)
    
    # Extract Training data:
    #wf_train = waveforms[training_idx]
    #ts_train = timestamps[training_idx]
    wf_train = waveforms
    ts_train = timestamps
    print(f'Shape of training data: {wf_train.shape}')

# ************************************************************
# ******************** Train/Load model **********************
# ************************************************************
print()
print('*********************** Tensorflow Blaj *************************************')
print()

encoder,decoder,vae = train_model(wf_train, nr_epochs=nr_epochs, batch_size=batch_size, path_to_weights=path_to_weights, 
                                        continue_train=continue_train, verbose=1)
print()
print('******************************************************************************')
print()

#view_vae_result = False # This reqires user to give input if to continue the script to GD or not.
if view_vae_result:
    plot_decoded_latent(decoder,saveas=save_figure+'_decoded',verbose=1)
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
if run_GD:
    print()
    print('Running pdf_GD to get hpdp...')
    print()
    # To easy computational load -- only every 20th data-point is used..
    hpdp = pdf_GD(vae, wf_train[0::20], m=m, gamma=gamma, eta=eta, path_to_hpdp=path_to_hpdp,verbose=verbose)
    #hpdp = pdf_GD(vae, wf_train, m=m, gamma=gamma, eta=eta, path_to_hpdp=path_to_hpdp,verbose=verbose)

    if view_GD_result:
        print(f'Visualising decoded latent space of hpdp...')
        print()
        plot_encoded(encoder, hpdp, saveas=save_figure+'_hpdp_encoded', verbose=1)        
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
run_event_rate = True
if run_event_rate:
    print('Calculating Event rates...')
    event_rates, real_clusters = get_event_rates(ts_train[0::20],labels,bin_width=1)
    delta_ev, ev_stats = delta_ev_measure(event_rates,timestamps,labels)

    #event_rates, real_clusters = get_event_rates(ts_train,labels,bin_width=1)
    print(f'Real cluster (with mean event_rate over 0.5 is CAPs {real_clusters})')
    plot_event_rates(event_rates,ts_train[0::20],saveas=save_figure+'_event_rate', conv_width=30)
    #plot_event_rates(event_rates,ts_train,saveas=save_figure+'_event_rate', conv_width=30)

# ************************************************************
# ******************** General PLOTTING ******************************
# ************************************************************
# Plot hpdp:
if verbose>1:
    print()
    print(f'Plotting waveforms of each cluster if labels are specified...')
    print()
    plot_waveforms(hpdp,labels=None)
    print()
    print(f'Visualising decoded latent space of hpdp...')
    print()
    plot_encoded(encoder, hpdp, saveas=save_figure+'_hpdp_encoded', verbose=verbose)
    print()
    print(f'Visualising decoded latent space...')
    print()
    plot_decoded_latent(decoder,saveas=save_figure+'_decoded',verbose=1)
    plot_encoded(encoder, wf_train, saveas=save_figure+'_encoded', verbose=1)






