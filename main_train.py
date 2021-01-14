# ************************************************************
# Main file in cytokine-identification process. 
#
# Master Thesis, KTH , Gabriel Andersson 
# ************************************************************
import numpy as np
import matplotlib.pyplot as plt
from os import path, scandir
import preprocess_wf 
from load_and_GD_funs import load_waveforms, load_timestamps, get_pdf_model,run_pdf_GD  
from wf_similarity_measures import wf_correlation, similarity_SSQ, label_from_corr
from event_rate_funs import get_ev_labels, get_event_rates
from plot_functions_wf import plot_decoded_latent
from evaluation import run_DBSCAN_evaluation, run_evaluation, run_visual_evaluation
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans, DBSCAN

'''
directory = '../matlab_files2'
for entry in scandir(directory):
    if entry.path.startswith(directory+"\\tsR10"):# and entry.is_file():
        matlab_file = entry.path[19:]
        print(entry.path[19:-4])        
'''
# ************************************************************
# ***************** HYPERPARAMERS ****************************
# ************************************************************

similarity_measure='ssq' # From testing, correlation give worse results..
#similarity_thresh = 0.7 # For corrrelation
similarity_thresh = 0.1 # Gives either the minimum correlation using 'corr' or "epsilon" in gaussain annulus theorem using 'ssq' (sum of squares)
assumed_model_varaince = 0.7 # The  model variance assumed in ssq-similarity measure. i.e variance in N(x_candidate,sigma^2*I) 
# Setting the assumed model variance to 0.5 as in CVAE yeild un unsufficient number of similar CAPs.. Thus set a bit higher for labeling..

# Should maby fix such that threshold is TNF/IL-1beta specific since there is substantially more CAPs labeles as increase after TNF injection.. 
# This. however, differs quite a lot from one recording to another..
# Otherwise maby 0.2..?

n_std_threshold = 0.2 #(0.5)  # Number of standard deviation which the mean-even-rate need to increase after injection for a candidate-CAP to be labeled as "likely to encode cytokine-info".

#ev_threshold = 0.02 # The minimum mean event rate for each observed CAP for it to be considered in the analysis. (Otherwise regarded as noise..)
#ev_threshold = 0.005 # Downsample=4 # Mabe good with 0.005 for aprox 37000 observations..

# downsample = 2 # Only uses every #th observation during the analysis for efficiency. 
desired_num_of_samples = None #40000 # Subsample using 
max_amplitude = 500 # Remove CAPs with max amplitude higher than the specified value. (Micro-Volts)
min_amplitude = 2 # Remove CAPs with max amplitude lower than the specified value. (Micro-Volts)
ev_thresh_fraction = 0.005 # Fraction of total event-rate used for thresholding. -- i.e 0.5%

# Time interval of recording used for training:
start_time = 15; end_time = 90

# pdf-GD params: 
run_GD = True
m=2000 # Number of steps in pdf-gradient decent
gamma=0.02 # learning_rate in GD.
eta=0.005 # Noise variable -- adds white noise with variance eta to datapoints during GD.

# VAE training params:
continue_train = False
nr_epochs = 120 # if all train data is used -- almost no loss-decrease after 100 batches..
batch_size = 128

# ****** If using 2D-latent space dimension: *********
view_cvae_result = False # True => reqires user to give input if to continue the script to pdf-GD or not.. 
view_GD_result = False # This reqires user to give input if to continue the script to clustering or not.
plot_hpdp_assesments = False # Cluster and evaluate hpdp to find cytokine-candidate CAP manually inspecting plots.
# ***********************

run_automised_assesment = True # Cluster and evaluate hpdp by defined quantitative measure.

# Evaluation Parameters using k*max(SD_min,SD) as threshold for "significant increase in ev." 
#SD_min_eval = 0.2 # Min value of SD s.t. mice is not classified as responder for insignificant increase in EV.
SD_min_eval = 0.3 #0.3 Min value of SD s.t. mice is not classified as responder for insignificant increase in EV.
k_SD_eval = 2.5 #2.5 # k-param in k*max(SD_min,SD) 


# Use DBSCAN on labeled data independent of everything after labeling to see if we obtain similar results.
run_DBscan = False
# DBSCAN params
#db_eps = 16 # max_distance to be considered as neighbours 
#db_min_sample = 4 # Minimum members in neighbourhood to not be regarded as Noise.
db_eps = 0.2 # max_distance to be considered as neighbours 
db_min_sample = 4 # Minimum members in neighbourhood to not be regarded as Noise.


standardise_waveforms = True
verbose_main = 1

# ************************************************************
# ******************** Paths *********************************
# ************************************************************

# OBS: The string for saving tensorflow weights are not allowed to be too long.  
# raises utf-8 encoding errors.. max ~250 characters..

# **********************************************************************************
# *********** Specify unique sting for saving files for a run: *********************

#unique_start_string = '15_dec_30000_max200__ampthresh5' # on second to last file in this run..
#unique_start_string = '17_dec_30000_max200_ampthresh5_saline'
#unique_start_string = '14_dec_unique_threshs_saline' # on second to last file in this run..
#unique_start_string = '15_dec_30000_max200__ampthresh5' # on second to last file in this run..
#unique_start_string = '15_dec_30000_max200_ampthresh5_new'
#unique_start_string = '17_dec_30000_max500_clean'

#unique_start_string = '22_dec_30k_ampthresh2' # Fine for results? 
#unique_start_string = '21_dec_30k_paramsearch' # Fine for results? 

#unique_start_string = '7_jan_40k_100epochs'

unique_start_string = 'finalrun_first'

# ***** Specify path to directory of recordings ******* 
directory = '../matlab_files'
# *****************************************************

# ***** Specify the starting scaracters in filename of recordings to analyse *****
# if "ts" is not specified, then all files will be run twise since we have one file for timestamps and one for CAP-waveform with identical names, exept the starting ts/wf.
start_string = '\\tsR10_Exp' #.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05' # Since each recording has two files in directory (waveforms and timestamps)-- this is solution to only get each recording once.
# *****************************************************

number_of_skipped_files = 0
for entry in scandir(directory):
    if entry.path.startswith(directory+start_string): # Find unique recording string. tsR10 for all cytokine injections, tsR12 for saline. 
        #matlab_file = entry.path[19:-4] # Find unique recording string'
        matlab_file = entry.path[len(directory+'\\ts'):-len('.mat')] # extract only the matlab_file name from string.
        print()
        print('*******************************************************************************')
        print(f'Starting analysis on recording : {matlab_file}')
        print()
        path_to_wf = directory + '/wf'+matlab_file +'.mat' # This is how the waveforms are saved. path + wf+MATLAB_FILE_NAME.mat
        path_to_ts = directory + '/ts'+matlab_file +'.mat' # This is how the timestamps are saved. path + ts+MATLAB_FILE_NAME.mat

        # ***** Define paths and name to save files/figures: ********
        unique_string_for_run = unique_start_string+matlab_file
        unique_string_for_figs = unique_start_string + matlab_file.replace('.','_') # '.' not allowed in plt.savefig.. 
        path_to_weights = 'models/'+unique_string_for_run
        # Numpy file paths:
        path_to_hpdp = "../numpy_files/numpy_hpdp/"+unique_string_for_run 
        path_to_EVlabels = "../numpy_files/EV_labels/"+unique_string_for_run
        path_to_cytokine_candidate = '../numpy_files/cytokine_candidates/'+unique_string_for_run
        
        
        # ************************************************************
        # ******************** Load Files ****************************
        # ************************************************************
        waveforms = load_waveforms(path_to_wf,'waveforms', verbose=1)
        timestamps = load_timestamps(path_to_ts,'timestamps',verbose=1)

        # ************************************************************
        # ******************** Preprocess ****************************
        # ************************************************************
        # Cut first and last part of recording to ensure stable "sleep-state" during recording.
        # Furthermore a downsampling is applied to speed up training. 

        wf0,ts0 = preprocess_wf.get_desired_shape(waveforms,timestamps, start_time=10,end_time=90, 
                                                    dim_of_wf=141,desired_num_of_samples=None) # No downsampling. Used for evaluation 
        
        print(f'Shape before amplitude threshold : {waveforms.shape}')
        waveforms,timestamps = preprocess_wf.apply_amplitude_thresh(waveforms,timestamps,maxamp_threshold=max_amplitude, minamp_threshold=min_amplitude) # Remove "extreme-amplitude" CAPs-- otherwise risk that pdf-GD diverges..
        print()
        print(f'Shape after amplitude threshold : {waveforms.shape}')
        print()
        waveforms,timestamps = preprocess_wf.get_desired_shape(waveforms,timestamps,start_time=start_time,end_time=end_time,dim_of_wf=141,desired_num_of_samples=desired_num_of_samples)
        print(f'Shape after shape-preprocessing : {waveforms.shape}')
        # ****** Standardise waveforms for more stable training. ********
        if standardise_waveforms:
            print()
            print(f'Standardises waveforms...')
            print()
            waveforms = preprocess_wf.standardise_wf(waveforms)
            wf0 = preprocess_wf.standardise_wf(wf0)

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

        # ************************************************************
        # ******************* Event-rate Threshold *******************
        # ************************************************************
        print()
        print(f'Number of wf which ("icreased after first","increased after second", "constant") = {np.sum(ev_labels,axis=1)} ')

        # ho := High Occurance, ts := timestamps
        wf_ho, ts_ho, ev_label_ho = preprocess_wf.apply_mean_ev_threshold(waveforms,timestamps,ev_stats_tot[0],ev_threshold=ev_thresh_fraction,ev_labels=ev_labels,ev_thresh_procentage=True)

        print(f'After EV threshold: ("icreased after first","increased after second", "constant") = {np.sum(ev_label_ho,axis=0)} ')
        #if np.sum(np.sum(ev_label_ho,axis=0)[:1]) < 500:
        #    print('An inssuficient number of CAPs where labeled as "likely to encode cytokine". Will go to next file..')
        #    break
        # number_of_occurances= np.sum(ev_label_ho,axis=0)
        
        # ************************************************************
        # ******************** Train/Load model **********************
        # ************************************************************
        if run_DBscan is False:
            print()
            print('*********************** Tensorflow Blaj *************************************')
            print()
            encoder,decoder,cvae = get_pdf_model(wf_ho, nr_epochs=nr_epochs, batch_size=batch_size, path_to_weights=path_to_weights, 
                                                    continue_train=continue_train, verbose=1, ev_label=ev_label_ho)
            print()
            print('******************************************************************************')
            print()

        #view_cvae_result = False # This reqires user to give input if to continue the script to GD or not.
        if view_cvae_result:
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
        # *  Perform GD on pdf to find high prob. data-points (hpdp) *
        # ************************************************************  
        if run_GD:
            print()
            print('Running pdf_GD to get hpdp...')
            print()
            hpdp_list = run_pdf_GD(wf_ho,cvae,ev_label_ho, labels_to_evaluate=[0,1], m=m, gamma=gamma, eta=eta,matlab_file=matlab_file,
                unique_string_for_figs=unique_string_for_figs, path_to_hpdp=path_to_hpdp,verbose=False,view_GD_result=view_GD_result,encoder=encoder)
            '''
            label_on = 1
            hpdp_list = []
            found_ho_wf = number_of_occurances[:-1]>0 # TODO: Not used, delete? 
            for label_on in [0,1]: # Either 0 or 1 
                waveforms_increase = wf_ho[ev_label_ho[:,label_on]==1]

                print(f'waveforms_increase injection {label_on+1} : {waveforms_increase.shape}')
                if waveforms_increase.shape[0] == 0:
                    waveforms_increase = np.append(np.zeros((1,141)),waveforms_increase).reshape((1,141))
                    ev_label_corr_shape = np.zeros((waveforms_increase.shape[0],3))
                    ev_label_corr_shape[:,label_on] = 1
                    print('*************** OBS ***************')
                    print(f'No waveforms with increased event rate at injection {label_on+1} was found.')
                    print(f'This considering the recording {matlab_file}')
                elif waveforms_increase.shape[0] > 3000: # Speed up process during param search..
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
            '''
        else:
            print()
            print('Skipps over pdf_GD...')
            print()
        
        # ************************************************************
        # *********** Inference from increased EV hpdp ***************
        # ************************************************************
        if plot_hpdp_assesments:
            #cytokine_candidates = np.empty((2,waveforms.shape[-1])) # To save the main candidates
            #print('Remove noise by reconstructing means..')
            #wf0_means = cvae.predict([waveforms,ev_labels.T])
            #print('Done..')
            run_visual_evaluation(wf0,ts0,hpdp_list,encoder,labels_to_evaluate=[0,1],clustering_method='k-means', db_eps=0.15, db_min_sample=5,
                     similarity_measure='ssq', similarity_thresh=0.4, assumed_model_varaince=0.5, unique_string_for_figs=unique_string_for_figs,
                     path_to_cytokine_candidate=path_to_cytokine_candidate)

        # ************************************************************
        # ******** Look for responders using hpdp-cluster ************
        # ************************************************************
        if run_automised_assesment:
            saveas = path_to_cytokine_candidate+'auto_assesment'
            run_evaluation(wf0,ts0,hpdp_list,encoder,k_SD_eval=k_SD_eval,SD_min_eval=SD_min_eval,labels_to_evaluate=[0,1], k_clusters=8, 
                            similarity_measure='ssq', similarity_thresh=0.4, assumed_model_varaince=0.5, 
                            db_eps=db_eps, db_min_sample=db_min_sample,saveas=saveas)
            
        # Run DBSCAN on labeled data to see if the obtained results are similar. 
        if run_DBscan:
            saveas = 'figures/dbscan/'+unique_string_for_figs + 'DBSCAN'
            np_saveas = path_to_cytokine_candidate + 'DBSCAN'
            run_DBSCAN_evaluation(wf_ho,wf0,ts0,ev_label_ho,labels_to_evaluate=[0,1], saveas=saveas, np_saveas=np_saveas, 
                                    db_eps=4, db_min_sample=3,matlab_file=matlab_file,similarity_measure='ssq',
                                    similarity_thresh=similarity_thresh, assumed_model_varaince=assumed_model_varaince)


print('Finished successfully')

'''
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
'''