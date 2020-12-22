# ************************************************************
# Running visualisations of workflow. 
# Results -- tuning hyper params etc.
# ************************************************************

import numpy as np
import matplotlib.pyplot as plt
from os import path, scandir
import preprocess_wf 
from main_functions import load_waveforms, load_timestamps, train_model, pdf_GD 
from wf_similarity_measures import wf_correlation, similarity_SSQ, label_from_corr
from event_rate_first import get_ev_labels, get_event_rates,delta_ev_measure
from plot_functions_wf import *
from evaluation import run_DBSCAN_evaluation, run_evaluation, run_visual_evaluation
from scipy.spatial.distance import cdist

from scipy import stats

from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# ************************************************************
# ******************** Parameters ****************************
# ************************************************************
start_time = 15; end_time = 90
desired_num_of_samples = 20000 # Subsample using 
max_amplitude = 500 # Remove CAPs with max amplitude higher than the specified value. (Micro Volts)
min_amplitude = 2 # Remove CAPs with max amplitude lower than the specified value. (Micro Volts)
ev_thresh_procentage = 0.005 #  *100 => %


view_vae_result = False # True => reqires user to give input if to continue the script to pdf-GD or not.. 
view_GD_result = False # This reqires user to give input if to continue the script to clustering or not.

# ************************************************************
# ******************** Paths ****************************
# ************************************************************

matlab_directory = '../matlab_files'

start_string = "21_dec_20k_ampthresh2"

#Interesting recordings: 
matlab_file = 'R10_Exp2_7.13.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_08' # 'R10'  / 'R12' for all case/control
#matlab_file = 'R10_6.30.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_05' # 'R10'  / 'R12' for all case/control


# MATLAB FILES:
path_to_matlab_wf = matlab_directory + '/wf' + matlab_file
path_to_matlab_ts = matlab_directory + '/ts' + matlab_file

# tf weight-file:
path_to_model_weights = 'models/cvae_27nov_deleteme'

unique_string_for_run = start_string + matlab_file
unique_for_figs = start_string + matlab_file.replace('.','_')
# **************** WHAT TO PLOT: ************************
verbose_main=True

plot_ev_stats = False
saveas_ev_stats = 'figures_tests/event_rate_stats/' + unique_for_figs

plot_simulatated_path_from_model = False
saveas_simulatated_path_from_model = 'figures_tests/model_assessment/' + unique_for_figs

view_vae_result = False
saveas_vae_result = 'figures_tests/encoded_decoded/' + unique_for_figs

plot_wf_and_ev_for_the_different_ev_labels = True
saveas_wf_and_ev_for_the_different_ev_labels = 'figures_tests/event_rate_labels/' + unique_for_figs

plot_acf_pacf = True
savefig_acf_pacf = 'figures_tests/acf_pacf/' + unique_for_figs

plot_test_of_test_statistic = True
savefig_test_of_test_statistic = 'figures_tests/test_statistic/' +  unique_for_figs




# ******* Numpy files ***********
path_to_hpdp = "../numpy_files/numpy_hpdp/" + unique_string_for_run
path_to_EVlabels = "../numpy_files/EV_labels/" + unique_string_for_run

# ************************************************************
# ******************** Load Files ****************************
# ************************************************************
load_data = True
if load_data:
    waveforms = load_waveforms(path_to_matlab_wf,'waveforms', verbose=1)
    timestamps = load_timestamps(path_to_matlab_ts,'timestamps',verbose=1)
    
# ************************************************************
# ******************** Preprocess ****************************
# ************************************************************
# Standardise wavefroms
waveforms,timestamps = preprocess_wf.apply_amplitude_thresh(waveforms,timestamps,maxamp_threshold=max_amplitude, minamp_threshold=min_amplitude) # Remove "extreme-amplitude" CAPs-- otherwise risk that pdf-GD diverges..
waveforms,timestamps = preprocess_wf.get_desired_shape(waveforms,timestamps,start_time=start_time,end_time=end_time,dim_of_wf=141,desired_num_of_samples=None)
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
print(f'Number of wf which ("icreased after first","increased after second", "constant") = {np.sum(ev_labels,axis=1)} ')

# ho := High Occurance, ts := timestamps
wf_ho, ts_ho, ev_label_ho = preprocess_wf.apply_mean_ev_threshold(waveforms,timestamps,ev_stats_tot[0],ev_threshold=ev_thresh_procentage,ev_labels=ev_labels,ev_thresh_procentage=True)

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
    for jj in [10,112,220]:
        saveas = saveas_simulatated_path_from_model+str(jj)
        x = wf_ho[jj,:].reshape((1,141))
        label = ev_label_ho[jj,:].reshape((1,3))

        plot_simulated(cvae,x,ev_label=label,n=3,var=0.5, saveas=saveas, verbose=verbose_main)

    print('Done.')


# ************************************************************
# ******** Plot examples of event-rates from EV_labeles ******
# ************************************************************
if view_vae_result:
            #save_figure = 'figures/encoded_decoded/' + unique_for_figs
            plot_decoded_latent(decoder,saveas=saveas_vae_result+'_decoded_constant',verbose=verbose_main,ev_label=np.array((0,0,1)).reshape((1,3)))
            plot_decoded_latent(decoder,saveas=saveas_vae_result+'_decoded_increase_first',verbose=verbose_main,ev_label=np.array((1,0,0)).reshape((1,3)))
            plot_decoded_latent(decoder,saveas=saveas_vae_result+'_decoded_increase_second',verbose=verbose_main,ev_label=np.array((0,1,0)).reshape((1,3)))


# ************************************************************
# ******** Plot examples of event-rates from EV_labeles ******
# ************************************************************
#plot_wf_and_ev_for_the_different_ev_labels = False
#saveas_wf_and_ev_for_the_different_ev_labels = 'figures_tests/event_rate_labels/nov27_ev_incr'
if plot_wf_and_ev_for_the_different_ev_labels:
    threshold = 0.6
    idx_increase_after_first= np.where(ev_labels[0,:]==1)
    idx_increase_after_second = np.where(ev_labels[1,:]==1)
    idx_constant_throughout = np.where(ev_labels[2,:]==1)
    
    print(idx_increase_after_first[0][100:300:100])
    saveas = saveas_wf_and_ev_for_the_different_ev_labels + '_first_n_std_1'
    for i in idx_increase_after_first[0][100:300:100]: #range(20,100,20):
        correlations = wf_correlation(i,waveforms)
        bool_labels = label_from_corr(correlations,threshold=threshold,return_boolean=True )
        event_rates, real_clusters = get_event_rates(timestamps[:,0],bool_labels,bin_width=1,consider_only=1)
        delta_ev, ev_stats = delta_ev_measure(event_rates)
        #ev_labels,_ = get_ev_labels(waveforms,timestamps,threshold=similarity_thresh,saveas=path_to_EVlabels,similarity_measure=similarity_measure, assumed_model_varaince=assumed_model_varaince,
        #                                            n_std_threshold=n_std_threshold)
        plot_correlated_wf(i,waveforms,bool_labels,threshold,saveas=saveas+str(i)+'_wf',verbose=verbose_main )
        plot_event_rates(event_rates,timestamps,noise=None,conv_width=20,saveas=saveas+str(i)+'_ev', verbose=verbose_main) 

    # ********* Increase Afrer Second Injection
    saveas = saveas_wf_and_ev_for_the_different_ev_labels+'_second_n_std_1'
    for i in idx_increase_after_second[0][100:300:100]: #range(20,100,20):
        correlations = wf_correlation(i,waveforms)
        bool_labels = label_from_corr(correlations,threshold=threshold,return_boolean=True )
        event_rates, real_clusters = get_event_rates(timestamps[:,0],bool_labels,bin_width=1,consider_only=1)
        delta_ev, ev_stats = delta_ev_measure(event_rates)
        #ev_labels,_ = get_ev_labels(waveforms,timestamps,threshold=similarity_thresh,saveas=path_to_EVlabels,similarity_measure=similarity_measure, assumed_model_varaince=assumed_model_varaince,
        #                                            n_std_threshold=n_std_threshold)
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

# TODO : -Fix plots of encoded/decoded Latent space.
#        -Event-rate plots? ---Njaa, finns i main..
# ************************************************************
# ******** Cluster-Results using "Test-statistic"  ***********
# ************************************************************
#plot_test_of_test_statistic = False
#savefig_test_of_test_statistic = 'figures_tests/test_statistic/26_nov_'
if plot_test_of_test_statistic:
    for c in [1021,2312]:
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