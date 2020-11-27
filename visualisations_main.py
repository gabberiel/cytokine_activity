# ************************************************************
# Running visualisations of workflow. 
# Results -- tuning hyper params etc.
# ************************************************************

import numpy as np
import matplotlib.pyplot as plt
from plot_functions_wf import *
from main_functions import *
from sklearn.cluster import KMeans, DBSCAN
from event_rate_first import *
import preprocess_wf 
from wf_similarity_measures import *

from scipy import stats

from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# ************************************************************
# ******************** Parameters ****************************
# ************************************************************

view_vae_result = False # True => reqires user to give input if to continue the script to pdf-GD or not.. 
view_GD_result = False # This reqires user to give input if to continue the script to clustering or not.

# ************************************************************
# ******************** Paths ****************************
# ************************************************************


# MATLAB FILES:
path_to_matlab_wf = '../matlab_files/gg_waveforms-R10_IL1B_TNF_03.mat' 
path_to_matlab_ts = '../matlab_files/gg_timestamps.mat'

# tf weight-file:
path_to_model_weights = 'models_tests/dense_vae_27nov'


# ********* Figures saveas: ************
save_figure = 'figures/27_nov'

# ******* Numpy files ***********
path_to_hpdp = "../numpy_files/numpy_hpdp/21nov_first_full_training"
path_to_EVlabels = "../numpy_files/EV_labels/27nov_130000"

# ************************************************************
# ******************** Load Files ****************************
# ************************************************************
load_data = True
if load_data:
    waveforms, mean, std = load_waveforms(path_to_matlab_wf,'waveforms',standardize=True, verbose=1)
    timestamps = load_timestamps(path_to_matlab_ts,'gg_timestamps',verbose=1)
    
# ************************************************************
# ******************** Preprocess ****************************
# ************************************************************
# Standardise wavefroms
waveforms = preprocess_wf.standardise_wf(waveforms)

# ************************************************************
# ******************** Event-rate Labeling *******************
# ************************************************************

if path.isfile(path_to_EVlabels+'.npy'):
    print()
    print(f'Loading saved EV-labels from {path_to_EVlabels}')
    print()
    ev_labels_wf = np.load(path_to_EVlabels+'.npy')
    ev_stats_tot = np.load(path_to_EVlabels+'tests_tot.npy') 

else:
    print(f'OBS: did not find {path_to_EVlabels}...')
    #ev_labels = get_ev_labels(waveforms,timestamps,threshold=0.6,saveas=path_to_EVlabels)
# ************************************************************
# ******************** Event-rate-stats **********************
# ************************************************************
plot_ev_stats = True
if plot_ev_stats:
    saveas = 'figures_tests/event_rate_stats/27nov_130000'
    plt.hist(ev_stats_tot[0],bins=100,density=True)
    plt.xlabel('Mean Event Rate')
    plt.title('Distribution of mean event rate.')
    plt.savefig(saveas+'tot_mean_ev',dpi=150)
    #plt.show()
    plt.close()
    plt.hist(ev_stats_tot[1],bins=100,density=True)
    plt.xlabel('Event Rate standard deviation')
    plt.title('Distribution of event rate standard deviations.')
    plt.savefig(saveas+'tot_std_ev',dpi=150)
    #plt.show()

# ************************************************************
# ************* PLOT High occurance WF: **********************
# ************************************************************

#high_occurance_wf, high_occurance_ts = apply_mean_ev_threshold(waveforms,timestamps,mean_event_rates,ev_threshold=1)

# TODO : Include to sort out the high occurance event rates.
waveforms, timestamps = preprocess_wf.apply_mean_ev_threshold(waveforms,timestamps,ev_stats_tot[0],ev_threshold=1)

# TODO : Also sort so plotting is only done for increased EV-label... :

threshold = 0.6
saveas = 'figures_tests/event_rate_labels/aa_nov27_high_occurance_n_std_1_'
for i in np.arange(0,waveforms.shape[0],1000): #range(20,100,20):
    correlations = wf_correlation(i,waveforms)
    bool_labels = label_from_corr(correlations,threshold=threshold,return_boolean=True )
    event_rates, real_clusters = get_event_rates(timestamps,bool_labels,bin_width=1,consider_only=1)
    delta_ev, ev_stats = delta_ev_measure(event_rates)#,timestamps=timestamps)
    ev_labels = ev_label(delta_ev,ev_stats,n_std=1)
    plot_correlated_wf(i,waveforms,bool_labels,threshold,saveas=saveas+'_wf'+str(i),verbose=False)
    plot_event_rates(event_rates,timestamps,noise=None,conv_width=20,saveas=saveas+'_ev'+str(i), verbose=False) 

exit()
# ************************************************************
# ******************** Load model ****************************
# ************************************************************
print()
print('*********************** Tensorflow Blaj *************************************')
print()

encoder,decoder,vae = train_model(waveforms, path_to_weights=path_to_model_weights,continue_train=False)
#encoder,decoder,cvae = train_model(waveforms, path_to_weights=path_to_model_weights,continue_train=False,ev_label=ev_labels_wf)
print()
print('******************************************************************************')
print()


# ************************************************************
# **** PLOT Simulated wf. from N(mu_x,I) *********************
# ************************************************************
plot_simulatated_path_from_model = False
if plot_simulatated_path_from_model:
    for jj in [10,212,3120,10000]:
        saveas = 'figures_tests/model_assessment/cvae_wf_'+str(jj)
        x = waveforms[jj,:]
        print(jj)
        x_rec = vae.predict(x.reshape((1,141))).reshape((141,))
        time = np.arange(0,3.5,3.5/x.shape[0])

        plt.figure()
        plt.plot(time,x,color = (0,0,0),lw=1,label='$x$')
        plt.plot(time,x_rec,color = (1,0,0),lw=1,label='$\mu_x$')

        for i in range(4):
            x_sample = np.random.multivariate_normal(x_rec,np.eye(x.shape[0])*0.5)
            if i==0:
                plt.plot(time,x_sample,color = (0.1,0.1,0.1),lw=0.3, label='$x_{sim}$')
            else:
                plt.plot(time,x_sample,color = (0.1,0.1,0.1),lw=0.3)

            #plt.plot(time,waveforms[original_idx,:],color = (1,0,0),lw=1, label='Original')
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Voltage $(\mu V)$')
        plt.title('Model Assessment: Simulating $x \sim \mathcal{N}(\mu_x,0.5 \mathcal{I} )$')
        plt.legend(loc='upper right')
        if saveas is not None:
            plt.savefig(saveas,dpi=150)
        plt.close()
        #plt.show()
    print('Done.')

# ************************************************************
# ******** Plot examples of event-rates from EV_labeles ******
# ************************************************************
plot_wf_and_ev_for_the_different_ev_labels = False
if plot_wf_and_ev_for_the_different_ev_labels:
    threshold = 0.6
    idx_increase_after_first= np.where(ev_labels_wf[0,:]==1)
    idx_increase_after_second = np.where(ev_labels_wf[1,:]==1)
    idx_constant_throughout = np.where(ev_labels_wf[2,:]==1)
    print(idx_increase_after_first[0][100:300:100])
    saveas = 'figures_tests/event_rate_labels/nov27_ev_incr_first_n_std_1'
    for i in idx_increase_after_first[0][100:300:100]: #range(20,100,20):
        correlations = wf_correlation(i,waveforms)
        bool_labels = label_from_corr(correlations,threshold=threshold,return_boolean=True )
        event_rates, real_clusters = get_event_rates(timestamps[:,0],bool_labels,bin_width=1,consider_only=1)
        delta_ev, ev_stats = delta_ev_measure(event_rates)
        ev_labels = ev_label(delta_ev,ev_stats,n_std=1)
        plot_correlated_wf(i,waveforms,bool_labels,threshold,saveas=saveas+str(i)+'_wf',verbose=False )
        plot_event_rates(event_rates,timestamps,noise=None,conv_width=20,saveas=saveas+str(i)+'_ev', verbose=False) 

    # ********* Increase Afrer Second Injection
    saveas = 'figures_tests/event_rate_labels/nov27_ev_incr_second_n_std_1'
    for i in idx_increase_after_second[0][100:300:100]: #range(20,100,20):
        correlations = wf_correlation(i,waveforms)
        bool_labels = label_from_corr(correlations,threshold=threshold,return_boolean=True )
        event_rates, real_clusters = get_event_rates(timestamps[:,0],bool_labels,bin_width=1,consider_only=1)
        delta_ev, ev_stats = delta_ev_measure(event_rates)
        ev_labels = ev_label(delta_ev,ev_stats,n_std=1)
        plot_correlated_wf(i,waveforms,bool_labels,threshold,saveas=saveas+str(i)+'_wf',verbose=False)
        plot_event_rates(event_rates,timestamps,noise=None,conv_width=20,saveas=saveas+str(i)+'_ev', verbose=False) 



# Assming we have loaded the standardised waveforms : std_waveforms

#import matplotlib.gridspec as gridspec
# ************************************************************
# ******** Quick look at ACF/PACF  ***************************
# ************************************************************
plot_acf_pacf = False
if plot_acf_pacf:
    i=0
    saveas = 'figures_tests/acf_pacf/26_nov'
    for j in range(1000,2000,300):
        fig, (ax0, ax1, ax3) = plt.subplots(ncols=3, constrained_layout=True,figsize=(12,3))
        print(j)
        plot_acf(waveforms[j,:],ax=ax0)
        plot_pacf(waveforms[j,:],ax=ax1)
        ax3.plot(waveforms[j,:])
        i+=1
        plt.savefig(saveas+'_wf_'+str(j)+'.png',dpi=150)
        plt.close()
        #plt.show()



# ************************************************************
# ******** Cluster-Results using "Test-statistic"  ***********
# ************************************************************
plot_test_of_test_statistic = True
if plot_test_of_test_statistic:
    for c in [1021,2312,99210]:
        saveas = 'figures_tests/test_statistic/26_nov_'
        test_stat = waveforms - waveforms[c,:]
        print(test_stat.shape)
        mean = np.zeros((test_stat.shape[-1]))
        var = np.eye(test_stat.shape[-1])
        probs = stats.multivariate_normal.pdf(test_stat,mean,var)*1e57
        threshold = 1e-25
        bool_labels = probs>threshold
        #sum(bool_labels)
        plot_correlated_wf(c,waveforms,bool_labels,threshold,saveas=saveas+'thres_'+str(threshold)+'_wf_'+str(c),verbose=False )