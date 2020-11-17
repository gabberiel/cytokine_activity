import numpy as np
import matplotlib.pyplot as plt
from plot_functions_wf import *
from main_functions import *
from sklearn.cluster import KMeans
from event_rate_first import *

# ************************************************************
# ******************** Parameters ****************************
# ************************************************************
continue_train = False
nr_epochs = 50
verbose = 1

m=0
gamma=0.01
eta=0.01

training_idx = np.arange(10000)

# ************************************************************
# ******************** Paths ****************************
# ************************************************************

path_to_wf = 'matlab_files/gg_waveforms-R10_IL1B_TNF_03.mat'
path_to_ts = 'matlab_files/gg_timestamps.mat'

saved_weights = 'models/first_test_101'
save_figure = 'figures/wf_latent_101'
path_to_weights = 'models/main_funs'

path_to_hpdp = "numpy_hpdp/second_run"
# ************************************************************
# ******************** Load Files ****************************
# ************************************************************

waveforms, mean, std = load_waveforms(path_to_wf,'waveforms',standardize=True, verbose=verbose)
timestamps = load_timestamps(path_to_ts,'gg_timestamps',verbose=verbose)

# Extract Training data:
wf_train = waveforms[training_idx]
# ************************************************************
# ******************** Train/Load model ***************************
# ************************************************************

encoder,decoder,vae = train_model(wf_train, nr_epochs=5, batch_size=128, path_to_weights=path_to_weights, 
                                        continue_train=continue_train, verbose=verbose)

# ************************************************************
# ** Perform GD on pdf to find high prob. data-points (hpdp) *
# ************************************************************  


hpdp = pdf_GD(vae, wf_train[:1000], m=m, gamma=gamma, eta=eta, path_to_hpdp=path_to_hpdp,verbose=verbose)


# ************************************************************
# ******************** Cluster wf using hpdp *****************
# ************************************************************


kmeans = KMeans(n_clusters=5, random_state=0).fit(hpdp)
labels = kmeans.labels_

# ************************************************************
# ******************** Event Rate *****************
# ************************************************************
print('Calculating Event rates...')
event_rates, real_clusters = get_event_rates(timestamps[:1000],labels,bin_width=1)
print(f'Real cluster (with mean event_rate over 0.1 is CAPs {real_clusters})')
plot_event_rates(event_rates, conv_width=20)
plt.show()

# ************************************************************
# ******************** PLOTTING ******************************
# ************************************************************
# Plot hpdp:
if verbose>1:
    print()
    print(f'Plotting waveforms of each cluster if labels are specified...')
    print()
    plot_waveforms(hpdp,labels=None)
if verbose>1:
    print()
    print(f'Visualising decoded latent space of hpdp...')
    print()
    plot_encoded(encoder, hpdp, saveas=save_figure+'hpdp_encoded', verbose=verbose)
if verbose>1:
    print()
    print(f'Visualising decoded latent space...')
    print()
    plot_decoded_latent(decoder,saveas=save_figure+'_decoded',verbose=1)
    plot_encoded(encoder, wf_train, saveas=save_figure+'_encoded', verbose=1)






