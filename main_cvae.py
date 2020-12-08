# ************************************************************
# Main file in cytokine-identification process. 
#
# Master Thesis, KTH , Gabriel Andersson 
# ************************************************************

import numpy as np
import matplotlib.pyplot as plt
from os import path
import preprocess_wf 
from main_functions import load_waveforms, load_timestamps, train_model, pdf_GD
from wf_similarity_measures import wf_correlation,similarity_SSQ
from event_rate_first import get_ev_labels, get_event_rates, similarity_SSQ, label_from_corr
from plot_functions_wf import *

from sklearn.cluster import KMeans

matlab_files = ['R10_6.27.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_01', 'R10_6.27.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_01', 
            'R10_6.28.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_03', 'R10_6.28.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_02',
            'R10_6.28.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_02', 'R10_6.28.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_03',
            'R10_6.29.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_04', 'R10_6.29.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_04',
            'R10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05']

figure_strings = ['R10_6_27_16_BALBC_IL1B_35ngperkg_TNF_05ug_01','R10_6_27_16_BALBC_TNF_0_5ug_IL1B_35ngperkg_01',
                'R10_6_28_16_BALBC_IL1B_35ngperkg_TNF_05ug_03', 'R10_6_28_16_BALBC_TNF_05ug_IL1B_35ngperkg_02', 
                'R10_6_28_16_BALBC_IL1B_35ngperkg_TNF_05ug_02', 'R10_6_28_16_BALBC_TNF_05ug_IL1B_35ngperkg_03',
                'R10_6_29_16_BALBC_IL1B_35ngperkg_TNF_05ug_04', 'R10_6_29_16_BALBC_TNF_05ug_IL1B_35ngperkg_04', 
                'R10_6_30_16_BALBC_IL1B_35ngperkg_TNF_05ug_05']

file_names = {'matlab_files':matlab_files, 'figure_strings':figure_strings}



# ************************************************************
# ***************** HYPERPARAMERS ****************************
# ************************************************************

similarity_measure='ssq'

similarity_thresh = 0.1 # Gives either the minimum correlation using 'corr' or epsilon in gaussain annulus theorem for 'ssq'
assumed_model_varaince = 0.7 # The  model variance assumed in ssq-similarity measure. i.e variance in N(x_candidate,sigma^2*I)   

# Should maby fix such that threshold is TNF/IL-1beta specific since there is substantially more findings for TNF..
# Otherwise maby 0.2..?
n_std_threshold = 0.2 #(0.5)  # Number of standard deviation which the mean-even-rate need to increase for a candidate-CAP to be labeled as "likely to encode cytokine-info".
ev_threshold = 0.02 # The minimum mean event rate for each observed CAP for it to be considered in the analysis. (Otherwise regarded as noise..)
#ev_threshold = 0.005 # Downsample=4
verbose_main = 1

downsample = 1 

# pdf-GD params: 
m=500 # Number of steps in pdf-gradient decent
gamma=0.02 # learning_rate in GD.
eta=0.005 # Noise variable -- adds white noise with variance eta to datapoints during GD.
# VAE training params:
continue_train = False
nr_epochs = 50 # if all train data is used -- almost no loss-decrease after 100 batches..
batch_size = 128

view_vae_result = False # True => reqires user to give input if to continue the script to pdf-GD or not.. 
view_GD_result = False # This reqires user to give input if to continue the script to clustering or not.
plot_hpdp_assesments = True


# ************************************************************
# ******************** Paths *********************************
# ************************************************************
# General: 

unique_start_string = '7_dec_nstd_02and01_no_ds'
run_i = 4
# LOOP Through to run analysis of all recodrings over night.. : 
for matlab_file in file_names['matlab_files'][run_i:]:
    path_to_wf = '../matlab_files/wf'+matlab_file+'.mat' 
    path_to_ts = '../matlab_files/ts'+matlab_file+'.mat'
    unique_string_for_run = unique_start_string+matlab_file
    unique_string_for_figs = unique_start_string + file_names['figure_strings'][run_i]
    print(path_to_wf)
    path_to_weights = 'models/'+unique_string_for_run
    # Numpy file:
    path_to_hpdp = "../numpy_files/numpy_hpdp/"+unique_string_for_run #'deleteme2' #saved version for middle case
    path_to_EVlabels = "../numpy_files/EV_labels/"+unique_string_for_run
    path_to_cytokine_candidate = '../numpy_files/cytokine_candidates/'+unique_string_for_run
    
    
    run_i+=1
    # FOR TESTING:
    #recording = 'amp_thresh_R10_Exp2_71516_BALBC_TNF_05ug_IL1B_35ngperkg_10.mat'
    #recording = 'new_computer_deleteme3'
    #path_to_wf = '../matlab_files/gg_waveforms-R10_IL1B_TNF_03'+'.mat'
    #path_to_ts = '../matlab_files/gg_timestamps'+'.mat'


    # General: 
    #recording = 'R10_6.27.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_01'
    #recording = 'R10_6.27.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_01'
    #recording = 'R10_6.28.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_03'
    #recording = 'R10_6.28.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_02' # Strange behaviour of EV during last period..

    #recording = 'R10_6.28.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_02'
    #recording = 'R10_6.28.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_03'
    #recording = 'R10_6.29.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_04'
    #recording = 'R10_6.29.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_04'
    #recording = 'R10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05'



    #path_to_wf = '../matlab_files/wf'+recording+'.mat' 
    #path_to_ts = '../matlab_files/ts'+recording+'.mat'

    #unique_start_string = '7_dec_nstd_02and01_no_ds'
    #unique_string_for_run = 'aa_using_new_var_period_3_test_4_dec'+recording
    #unique_string_for_run = unique_start_string+recording


    # FOR SAVING FIGURES ****Not allowed to contain . or () signs****
    #unique_string_for_figs = unique_start_string +'R10_6_27_16_BALBC_IL1B_35ngperkg_TNF_05ug_01' 
    #unique_string_for_figs = unique_start_string +'R10_6_27_16_BALBC_TNF_0_5ug_IL1B_35ngperkg_01'
    #unique_string_for_figs = unique_start_string +'R10_6_28_16_BALBC_IL1B_35ngperkg_TNF_05ug_03'
    #unique_string_for_figs = unique_start_string +'R10_6_28_16_BALBC_TNF_05ug_IL1B_35ngperkg_02'

    #unique_string_for_figs = unique_start_string + 'R10_6_28_16_BALBC_IL1B_35ngperkg_TNF_05ug_02'
    #unique_string_for_figs = unique_start_string +  'R10_6_28_16_BALBC_TNF_05ug_IL1B_35ngperkg_03'
    #unique_string_for_figs = unique_start_string +  'R10_6_29_16_BALBC_IL1B_35ngperkg_TNF_05ug_04'
    #unique_string_for_figs = unique_start_string +  'R10_6_29_16_BALBC_TNF_05ug_IL1B_35ngperkg_04'
    #unique_string_for_figs = unique_start_string +  'R10_6_30_16_BALBC_IL1B_35ngperkg_TNF_05ug_05'



    # tf weight-file:
    #path_to_weights = 'models/'+unique_string_for_run
    # Numpy file:
    #path_to_hpdp = "../numpy_files/numpy_hpdp/"+unique_string_for_run #'deleteme2' #saved version for middle case
    #path_to_EVlabels = "../numpy_files/EV_labels/"+unique_string_for_run
    #path_to_cytokine_candidate = '../numpy_files/cytokine_candidates/'+unique_string_for_run

    # ************************************************************
    # ******************** Load Files ****************************
    # ************************************************************

    # TODO Move preprocessing from "load_waveforms function"
    load_data = True
    if load_data:
        waveforms = load_waveforms(path_to_wf,'waveforms', verbose=1)
        timestamps = load_timestamps(path_to_ts,'timestamps',verbose=1)
        n0_wf = waveforms.shape[0]
        d0_wf = waveforms.shape[1] 



    wf0,ts0 = preprocess_wf.get_desired_shape(waveforms,timestamps,start_time=10,end_time=90,dim_of_wf=141) # No downsampling. Used for evaluation 

    # ************************************************************
    # ******************** Preprocess ****************************
    # ************************************************************


    # Cut first and last part of recording to ensure stable sleep state during recording etc.:
    waveforms,timestamps = preprocess_wf.get_desired_shape(waveforms,timestamps,start_time=15,end_time=90,dim_of_wf=141,downsample=downsample)
    # Standardise waveforms
    waveforms = preprocess_wf.standardise_wf(waveforms)
    #wf0 = preprocess_wf.standardise_wf(wf0)



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
        ev_labels, ev_stats_tot = get_ev_labels(waveforms,timestamps,threshold=similarity_thresh,saveas=path_to_EVlabels,similarity_measure=similarity_measure, assumed_model_varaince=assumed_model_varaince,
        n_std_threshold=n_std_threshold)

    print()
    print(f'Number of wf which ("icreased after first","increased after second", "constant") = {np.sum(ev_labels,axis=1)} ')

    # ho := High Occurance, ts := timestamps
    wf_ho, ts_ho, ev_label_ho = preprocess_wf.apply_mean_ev_threshold(waveforms,timestamps,ev_stats_tot[0],ev_threshold=ev_threshold,ev_labels=ev_labels)

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
        save_figure = 'figures/encoded_decoded/' + unique_string_for_figs
        
        plot_decoded_latent(decoder,saveas=save_figure+'_decoded_constant',verbose=verbose_main,ev_label=np.array((0,0,1)).reshape((1,3)))
        plot_decoded_latent(decoder,saveas=save_figure+'_decoded_increase_first',verbose=verbose_main,ev_label=np.array((1,0,0)).reshape((1,3)))
        plot_decoded_latent(decoder,saveas=save_figure+'_decoded_increase_second',verbose=verbose_main,ev_label=np.array((0,1,0)).reshape((1,3)))

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
        label_on = 1
        hpdp_list = []
        found_ho_wf = number_of_occurances[:-1]>0
        for label_on in [0,1]: # Either 0 or 1 
            waveforms_increase = wf_ho[ev_label_ho[:,label_on]==1]
            print(f'waveforms_increase injection {label_on+1} : {waveforms_increase.shape}')
            if waveforms_increase.shape[0] > 3000: # Speed up process during param search..
                waveforms_increase = waveforms_increase[::4,:]
                ev_label_corr_shape = np.zeros((waveforms_increase.shape[0],3))
                ev_label_corr_shape[:,label_on] = 1
            else:
                ev_label_corr_shape = np.zeros((waveforms_increase.shape[0],3))
                ev_label_corr_shape[:,label_on] = 1

            hpdp = pdf_GD(cvae, waveforms_increase,ev_label=ev_label_corr_shape, m=m, gamma=gamma, eta=eta, path_to_hpdp=path_to_hpdp+str(label_on),verbose=verbose_main)
            hpdp_list.append(hpdp)
            if view_GD_result:
                save_figure = 'figures/hpdp/' + unique_string_for_figs
                print(f'Visualising decoded latent space of hpdp...')
                print()
                plot_encoded(encoder, hpdp, saveas=save_figure+'_encoded_hpdp'+str(label_on), verbose=1,ev_label=ev_label_corr_shape) 
        if view_GD_result:       
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
#gd_runs = '_3000'
#plot_hpdp_assesments = False
if plot_hpdp_assesments:
    cytokine_candidates = np.empty((2,waveforms.shape[-1])) # To save the main candidates
    for label_on in [0,1]:
        hpdp = hpdp_list[label_on]
        bool_labels = np.ones((hpdp.shape[0])) == 1 # Label all as True (same cluster) to plot the average form of increased EV-hpdp
        saveas = 'figures/hpdp/'+unique_string_for_figs
        plot_correlated_wf(0,hpdp,bool_labels,None,saveas=saveas+'_wf'+str(label_on),verbose=True)
        waveforms_increase = wf_ho[ev_label_ho[:,label_on]==1]
        #print(f'waveforms_increase injection {label_on+1} : {waveforms_increase.shape}')
        if waveforms_increase.shape[0] > 3000: # Speed up process during param search..
            waveforms_increase = waveforms_increase[::4,:]
            ev_label_corr_shape = np.zeros((waveforms_increase.shape[0],3))
            ev_label_corr_shape[:,label_on] = 1
        else:
            ev_label_corr_shape = np.zeros((waveforms_increase.shape[0],3))
            ev_label_corr_shape[:,label_on] = 1   
        #PLOT ENCODED wf_increase... :
        #ev_label_corr_shape = np.zeros((hpdp.shape[0],3))
        #ev_label_corr_shape[:,label_on] = 1
        save_figure = 'figures/encoded_decoded/'+unique_string_for_figs + '_ho_second'
        plot_encoded(encoder,  waveforms_increase, ev_label =ev_label_corr_shape, saveas=save_figure+'_encoded'+str(label_on), verbose=True)
        plot_encoded(encoder, hpdp, saveas=save_figure+'_encoded_hpdp'+str(label_on), verbose=1,ev_label=ev_label_corr_shape,title='Encoded hpdp') 

        K_string  = input('Number of clusters? (integer) :')
        kmeans = KMeans(n_clusters=int(K_string), random_state=0).fit(hpdp)
        k_labels = kmeans.labels_
        saveas = 'figures/event_rate_labels/'+unique_string_for_figs + str(label_on)
        possible_wf_candidates = evaluate_hpdp_candidates(wf0,ts0,hpdp,k_labels,saveas=saveas,similarity_measure='ssq',
                                similarity_thresh=0.4, assumed_model_varaince=0.5,verbose=True, return_candidates=True)
        
        k_candidate  = input('which CAP-cluster seems most likely to encode the cytokine? (integer or None) :')
        if k_candidate != 'None':
            cytokine_candidates[label_on,:] = possible_wf_candidates[int(k_candidate),:]
    if k_candidate is not None:
        np.save(path_to_cytokine_candidate, cytokine_candidates)

exit()            

#MAIN_CANDIDATE = np.median(hpdp,axis=0)
for label_on in [0,1]:
    hpdp = hpdp_list[label_on]
    MAIN_CANDIDATE = np.median(hpdp,axis=0)
    similarity_thresh = 0.5 #using var=1 -- Bad alternative, wfs not similar enough...

    added_main_candidate_wf = np.concatenate((MAIN_CANDIDATE.reshape((1,MAIN_CANDIDATE.shape[0])),wf0),axis=0)
    assert np.sum(MAIN_CANDIDATE) == np.sum(added_main_candidate_wf[0,:]), 'Something wrong in concatenate..'
    
    # QUICK FIX FOR WAVEFORMS AMPLITUDE INCREASING AFTER GD-- standardise it.
    # Should not be needed if GD works properly...
    added_main_candidate_wf = preprocess_wf.standardise_wf(added_main_candidate_wf)
    print(f'number of waveforms considered during tests: {added_main_candidate_wf.shape}')
    # Get correlation cluster for Delta EV - increased_second hpdp
    # MAIN_THRES = 0.6
    saveas = 'figures/event_rate_labels/'+unique_string_for_figs
    if similarity_measure=='corr':
        print('Using "corr" to evaluate final result')
        correlations = wf_correlation(0,added_main_candidate_wf)
        bool_labels = label_from_corr(correlations,threshold=similarity_thresh,return_boolean=True)
    if similarity_measure=='ssq':
        print('Using "ssq" to evaluate final result')
        added_main_candidate_wf = added_main_candidate_wf/0.5  # (0.7) Assumed var in ssq
        bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh)
    
    event_rates, real_clusters = get_event_rates(ts0,bool_labels[1:],bin_width=1,consider_only=1)
    plot_correlated_wf(0,added_main_candidate_wf,bool_labels,similarity_thresh,saveas=saveas+'Main_cand'+'_wf'+str(label_on),verbose=True )
    plot_event_rates(event_rates,ts0,noise=None,conv_width=20,saveas=saveas+'Main_cand'+'_ev'+str(label_on), verbose=True) 


exit()



# ************************************************************
# ************************************************************
# ************************************************************
# ******************** OLD STUFF FROM HERE ON ****************
# ************************************************************
# ************************************************************
# ************************************************************


# ************************************************************
# ******************** Cluster wf using hpdp *****************
# ******************** This to access labels *****************
# ************************************************************

'''
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



'''
