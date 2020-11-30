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
path_to_model_weights = 'models/cvae_27nov_deleteme'

# **************** WHAT TO PLOT: ************************
verbose_main=True

plot_ev_stats = False
saveas_ev_stats = 'figures_tests/event_rate_stats/27nov_130000'

plot_simulatated_path_from_model = False
saveas_simulatated_path_from_model = 'figures_tests/model_assessment/cvae_27nov_deleteme'

plot_wf_and_ev_for_the_different_ev_labels = False
saveas_wf_and_ev_for_the_different_ev_labels = 'figures_tests/event_rate_labels/nov27_ev_incr'

plot_acf_pacf = False
savefig_acf_pacf = 'figures_tests/acf_pacf/26_nov'

plot_test_of_test_statistic = False
savefig_test_of_test_statistic = 'figures_tests/test_statistic/26_nov_'




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
#plot_ev_stats = True
#saveas_ev_stats = 'figures_tests/event_rate_stats/27nov_130000'
if plot_ev_stats:
    saveas = saveas_ev_stats
    plt.hist(ev_stats_tot[0],bins=100,density=True)
    plt.xlabel('Mean Event Rate')
    plt.title('Distribution of mean event rate.')
    plt.savefig(saveas+'tot_mean_ev',dpi=150)
    if verbose_main is True:
        plt.show()
    plt.close()
    plt.hist(ev_stats_tot[1],bins=100,density=True)
    plt.xlabel('Event Rate standard deviation')
    plt.title('Distribution of event rate standard deviations.')
    plt.savefig(saveas+'tot_std_ev',dpi=150)
    if verbose_main is True:
        plt.show()

# ************************************************************
# ************* PLOT High occurance WF: **********************
# ************************************************************

#high_occurance_wf, high_occurance_ts = apply_mean_ev_threshold(waveforms,timestamps,mean_event_rates,ev_threshold=1)

# Consider only waveforms with high occurance event rates.
print(f'Number of wf which ("icreased after first","increased after second", "constant") = {np.sum(ev_labels_wf,axis=1)} ')

# ho := High Occurance, ts := timestamps
wf_ho, ts_ho, ev_label_ho = preprocess_wf.apply_mean_ev_threshold(waveforms,timestamps,ev_stats_tot[0],ev_threshold=1,ev_labels=ev_labels_wf)

print(f'After EV threshold: ("icreased after first","increased after second", "constant") = {np.sum(ev_label_ho,axis=0)} ')
#waveforms, timestamps = preprocess_wf.apply_mean_ev_threshold(waveforms,timestamps,ev_stats_tot[0],ev_threshold=1)

# Plotting EV etc. done for High occurance CAPs... :
plot_ho_EVs = False
if plot_ho_EVs:
    threshold = 0.6
    saveas = 'figures_tests/event_rate_labels/aa_nov27_high_occurance_n_std_1_'
    for i in np.arange(0,wf_ho.shape[0],10000): #range(20,100,20):
        correlations = wf_correlation(i,wf_ho)
        bool_labels = label_from_corr(correlations,threshold=threshold,return_boolean=True )
        event_rates, real_clusters = get_event_rates(ts_ho,bool_labels,bin_width=1,consider_only=1)
        delta_ev, ev_stats = delta_ev_measure(event_rates)#,timestamps=ts_ho)
        ev_labels = ev_label(delta_ev,ev_stats,n_std=1)
        plot_correlated_wf(i,wf_ho,bool_labels,threshold,saveas=saveas+'_wf'+str(i),verbose=verbose_main)
        plot_event_rates(event_rates,ts_ho,noise=None,conv_width=20,saveas=saveas+'_ev'+str(i), verbose=verbose_main) 


# ************************************************************
# ******************** Train/Load model **********************
# ************************************************************
print()
print('*********************** Tensorflow Blaj *************************************')
print()

encoder,decoder,cvae = train_model(wf_ho, nr_epochs=0, batch_size=128, path_to_weights=path_to_model_weights, 
                                        continue_train=False, verbose=1, ev_label=ev_label_ho)
print()
print('******************************************************************************')
print()


# ************************************************************
# **** PLOT Simulated wf. from N(mu_x,I) (CVAE) *********************
# ************************************************************
#plot_simulatated_path_from_model = True
#saveas_simulatated_path_from_model = 'figures_tests/model_assessment/cvae_wf_'
if plot_simulatated_path_from_model:
    for jj in [10,212,3120,10000]:
        saveas = saveas_simulatated_path_from_model+str(jj)
        x = wf_ho[jj,:].reshape((1,141))
        label = ev_label_ho[jj,:].reshape((1,3))

        plot_simulated(cvae,x,ev_label=label,n=3,var=0.5, saveas=saveas, verbose=verbose_main)

    print('Done.')


# ************************************************************
# ******** Plot examples of event-rates from EV_labeles ******
# ************************************************************
#plot_wf_and_ev_for_the_different_ev_labels = False
#saveas_wf_and_ev_for_the_different_ev_labels = 'figures_tests/event_rate_labels/nov27_ev_incr'
if plot_wf_and_ev_for_the_different_ev_labels:
    threshold = 0.6
    idx_increase_after_first= np.where(ev_labels_wf[0,:]==1)
    idx_increase_after_second = np.where(ev_labels_wf[1,:]==1)
    idx_constant_throughout = np.where(ev_labels_wf[2,:]==1)
    
    print(idx_increase_after_first[0][100:300:100])
    saveas = saveas_wf_and_ev_for_the_different_ev_labels + '_first_n_std_1'
    for i in idx_increase_after_first[0][100:300:100]: #range(20,100,20):
        correlations = wf_correlation(i,waveforms)
        bool_labels = label_from_corr(correlations,threshold=threshold,return_boolean=True )
        event_rates, real_clusters = get_event_rates(timestamps[:,0],bool_labels,bin_width=1,consider_only=1)
        delta_ev, ev_stats = delta_ev_measure(event_rates)
        ev_labels = ev_label(delta_ev,ev_stats,n_std=1)
        plot_correlated_wf(i,waveforms,bool_labels,threshold,saveas=saveas+str(i)+'_wf',verbose=verbose_main )
        plot_event_rates(event_rates,timestamps,noise=None,conv_width=20,saveas=saveas+str(i)+'_ev', verbose=verbose_main) 

    # ********* Increase Afrer Second Injection
    saveas = saveas_wf_and_ev_for_the_different_ev_labels+'_second_n_std_1'
    for i in idx_increase_after_second[0][100:300:100]: #range(20,100,20):
        correlations = wf_correlation(i,waveforms)
        bool_labels = label_from_corr(correlations,threshold=threshold,return_boolean=True )
        event_rates, real_clusters = get_event_rates(timestamps[:,0],bool_labels,bin_width=1,consider_only=1)
        delta_ev, ev_stats = delta_ev_measure(event_rates)
        ev_labels = ev_label(delta_ev,ev_stats,n_std=1)
        plot_correlated_wf(i,waveforms,bool_labels,threshold,saveas=saveas+str(i)+'_wf',verbose=verbose_main)
        plot_event_rates(event_rates,timestamps,noise=None,conv_width=20,saveas=saveas+str(i)+'_ev', verbose=verbose_main) 



# Assming we have loaded the standardised waveforms : std_waveforms

#import matplotlib.gridspec as gridspec
# ************************************************************
# ******** Quick look at ACF/PACF  ***************************
# ************************************************************
#plot_acf_pacf = False
#savefig_acf_pacf = 'figures_tests/acf_pacf/26_nov'
if plot_acf_pacf:
    i=0
    saveas = savefig_acf_pacf
    for j in range(1000,2000,300):
        fig, (ax0, ax1, ax3) = plt.subplots(ncols=3, constrained_layout=True,figsize=(12,3))
        print(j)
        plot_acf(waveforms[j,:],ax=ax0)
        plot_pacf(waveforms[j,:],ax=ax1)
        ax3.plot(waveforms[j,:])
        i+=1
        plt.savefig(saveas+'_wf_'+str(j)+'.png',dpi=150)
        plt.close()
        if verbose_main is True:
            plt.show()



# ************************************************************
# ******** Cluster-Results using "Test-statistic"  ***********
# ************************************************************
#plot_test_of_test_statistic = False
#savefig_test_of_test_statistic = 'figures_tests/test_statistic/26_nov_'
if plot_test_of_test_statistic:
    for c in [1021,2312,99210]:
        saveas = savefig_test_of_test_statistic
        test_stat = waveforms - waveforms[c,:]
        print(test_stat.shape)
        mean = np.zeros((test_stat.shape[-1]))
        var = np.eye(test_stat.shape[-1])
        probs = stats.multivariate_normal.pdf(test_stat,mean,var)*1e57
        threshold = 1e-25
        bool_labels = probs>threshold
        #sum(bool_labels)
        plot_correlated_wf(c,waveforms,bool_labels,threshold,saveas=saveas+'thres_'+str(threshold)+'_wf_'+str(c),verbose=verbose_main )