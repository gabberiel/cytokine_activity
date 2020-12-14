
import numpy as np
import matplotlib.pyplot as plt
from os import path, scandir
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
similarity_thresh = 0.3 # Gives either the minimum correlation using 'corr' or epsilon in gaussain annulus theorem for 'ssq'
assumed_model_varaince = 0.5 # The  model variance assumed in ssq-similarity measure. i.e variance in N(x_candidate,sigma^2*I)   



def number_of_reponders(directory, start_string='', end_string=''):
    responders = []
    for entry in scandir(directory):
        if entry.path.startswith(directory+start_string) & entry.path.endswith(end_string):# and entry.is_file():
            print(entry.path)
            result = np.load(entry.path, allow_pickle=True)
            responder_bool = False
            for injection_res in result:
                if len(injection_res)==0:
                    pass
                else:
                    responder_bool=True
                    
                    if injection_res.shape==(2,):
                        plt.plot(injection_res[0])
                        plt.title(entry.path[25:])
                        plt.show()
                    else:
                        times_above_thresh = np.zeros((len(injection_res,)))
                        for ii,responder in enumerate(injection_res):
                            stats = responder[1]
                            times_above_thresh[ii] = stats[3] # time above threshold
                            #waveform = responder[0]
                        main_candidate = np.argmax(times_above_thresh)
                        plt.plot(injection_res[main_candidate][0])
                        plt.title(entry.path[25:])
                        plt.show()
                    
            if responder_bool:
                responders.append(1)
            else: 
                responders.append(0)
    print(f'Number of responders: {np.sum(responders)} out of {len(responders)}')
    return responders

recording_candidate = 'R10_Exp2_7.15.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_10'
unique_start_string = '10_dec_nstd_02and01_ds1_ampthresh' # on second to last file in this run..
# ************************************************************
# ******************** Start Evaluation ****************************
# ************************************************************
run_i = 0
directory = '../numpy_files/cytokine_candidates'
start_string = "\\14_dec_unique_threshs_saline"
#start_string = "\\13_dec_unique_threshs_ampthresh2"
end_string = 'new_test.npy'
number_of_reponders(directory, start_string=start_string, end_string=end_string)

'''
responders = []
for entry in scandir(directory):
    if entry.path.startswith(directory+"\\13_dec_unique_threshs_ampthresh2") & entry.path.endswith('new_test.npy'):# and entry.is_file():
        print(entry.path)
        result = np.load(entry.path, allow_pickle=True)
        responder_bool = False
        for injection_res in result:
            if len(injection_res)==0:
                pass
            else:
                responder_bool=True
                if injection_res.shape==(2,):
                    plt.plot(injection_res[0])
                    plt.title(entry.path[25:])
                    plt.show()
                else:
                    times_above_thresh = np.zeros((len(injection_res,)))
                    for ii,responder in enumerate(injection_res):
                        stats = responder[1]
                        times_above_thresh[ii] = stats[3] # time above threshold
                        #waveform = responder[0]
                    main_candidate = np.argmax(times_above_thresh)
                    plt.plot(injection_res[main_candidate][0])
                    plt.title(entry.path[25:])
                    plt.show()
        if responder_bool:
            responders.append(1)
        else: 
            responders.append(0)
print(f'Number of responders: {np.sum(responders)} out of {len(responders)}')
'''
# LOOP Through to run analysis of all recodrings over night.. : 
exit()
for matlab_file in file_names['matlab_files']:#[:4]:
    path_to_wf = '../matlab_files/wf'+matlab_file+'.mat' 
    path_to_ts = '../matlab_files/ts'+matlab_file+'.mat'
    unique_string_for_run = unique_start_string+matlab_file
    unique_string_for_figs = unique_start_string + file_names['figure_strings'][run_i]
    path_to_cytokine_candidate = '../numpy_files/cytokine_candidates/'+unique_start_string + recording_candidate
    path_to_weights = 'models/'+unique_string_for_run
    # Numpy file:
    path_to_hpdp = "../numpy_files/numpy_hpdp/"+unique_string_for_run #'deleteme2' #saved version for middle case
    path_to_EVlabels = "../numpy_files/EV_labels/"+unique_string_for_run

    print()
    print(f'DATA FILE : {matlab_file}')
    load_data = True
    if load_data:
        wf0 = load_waveforms(path_to_wf,'waveforms', verbose=0)
        ts0 = load_timestamps(path_to_ts,'timestamps',verbose=0)
        cytokine_candidates = np.load(path_to_cytokine_candidate+'.npy')
        n0_wf = wf0.shape[0]
        d0_wf = wf0.shape[1] 
    label_on=0
    cluster='main'
    wf0,ts0 = preprocess_wf.get_desired_shape(wf0,ts0,start_time=1,end_time=90,dim_of_wf=141,downsample=None)
    #wf0 = preprocess_wf.standardise_wf(wf0)
    for i in range(2):
        plt.plot(np.arange(0,3.5,3.5/141),cytokine_candidates[i,:])
    plt.show()

    for MAIN_CANDIDATE in cytokine_candidates:
        label_on +=1

        added_main_candidate_wf = np.concatenate((MAIN_CANDIDATE.reshape((1,MAIN_CANDIDATE.shape[0])),wf0),axis=0)
        assert np.sum(MAIN_CANDIDATE) == np.sum(added_main_candidate_wf[0,:]), 'Something wrong in concatenate..'
        
        # QUICK FIX FOR WAVEFORMS AMPLITUDE INCREASING AFTER GD-- standardise it.
        # Should not be needed if GD works properly...
        added_main_candidate_wf = preprocess_wf.standardise_wf(added_main_candidate_wf)
        print(f'Shape of test-dataset (now considers all observations): {added_main_candidate_wf.shape}')
        # Get correlation cluster for Delta EV - increased_second hpdp
        # MAIN_THRES = 0.6
        saveas = 'figures/event_rate_labels/'+unique_string_for_figs
        if similarity_measure=='corr':
            print('Using "corr" to evaluate final result')
            correlations = wf_correlation(0,added_main_candidate_wf)
            bool_labels = label_from_corr(correlations,threshold=similarity_thresh,return_boolean=True)
        if similarity_measure=='ssq':
            print('Using "ssq" to evaluate final result')
            added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
            bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh)
        event_rates, _ = get_event_rates(ts0,bool_labels[1:],bin_width=1,consider_only=1)
        plt.figure(1)
        plot_correlated_wf(0,added_main_candidate_wf,bool_labels,similarity_thresh,saveas=saveas+'Main_cand'+'_wf'+str(label_on),
                            verbose=False, show_clustered=False,cluster=cluster)
        plt.figure(2)
        #bool_labels[bool_labels==True] = cluster
        plot_event_rates(event_rates,ts0,noise=None,conv_width=100,saveas=saveas+'Main_cand'+'_ev'+str(label_on), verbose=False,cluster=cluster) 
        plt.figure(3)
        #plt.hist(ts0,bins=200)
        event_rates, _ = get_event_rates(ts0,np.ones((ts0.shape[0],)),bin_width=1,consider_only=1)
        plot_event_rates(event_rates,ts0,noise=None,conv_width=100,saveas=saveas+'overall_EV', verbose=False) 
        plt.show()