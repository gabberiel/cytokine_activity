# ************************************************************
# Main file in cytokine-identification process. 
#
# Master Thesis, KTH , Gabriel Andersson
# ************************************************************
import numpy as np
import matplotlib.pyplot as plt
import preprocess_wf
import json
from os import path, scandir
from load_and_GD_funs import load_waveforms, load_timestamps, get_pdf_model, run_pdf_GD  
from wf_similarity_measures import wf_correlation, similarity_SSQ, label_from_corr
from event_rate_funs import get_ev_labels, get_event_rates
from plot_functions_wf import plot_decoded_latent, plot_amplitude_hist
from evaluation import run_DBSCAN_evaluation, run_evaluation, run_visual_evaluation
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans, DBSCAN


continue_train = False    # Tensorflow CVAE-model.
run_GD = True
# ****** If using 2D-latent space dimension: *********
view_cvae_result = False    # True => reqires user to give input if to continue-
                           # the script to pdf-GD or not.
view_GD_result = False    # This reqires user to give input if to continue the-
                          # script to clustering or not.
plot_hpdp_assesments = False    # Cluster and evaluate hpdp to find -
                                # cytokine-candidate CAP manually inspecting plots.
# ***********************

run_automised_assesment = True    # Cluster and evaluate hpdp by defined quantitative measure.
run_DBscan = False

verbose_main = 1

# *****************************************************************************
# *********** Specify unique sting for saving files for a run: ****************

# training_start_title = '22_dec_30k_ampthresh2' # Fine for results? 
# training_start_title = '21_dec_30k_paramsearch' # Fine for results? 
# training_start_title = '7_jan_40k_100epochs'
# training_start_title = 'jan15_20k_amp_1_1000'

training_start_title = 'test_runs'
# ************************************************************
# ************ LOAD HYPERPARAMERS ****************************
# ************************************************************

with open('hypes/'+ training_start_title+'.json', 'r') as f:
    hypes = json.load(f)

# ***** Specify path to directory of recordings ******* 
directory = '../matlab_files'
# *****************************************************

# ***** Specify the starting scaracters in filename of recordings to analyse *****
# if "ts" is not specified, then all files will be run twise since we have one file for timestamps and one for CAP-waveform with identical names, exept the starting ts/wf.
rec_start_string = '\\tsR10' #.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05' # Since each recording has two files in directory (waveforms and timestamps)-- this is solution to only get each recording once.
# rec_start_string = '\\tsR10_6.27.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_01'
# *****************************************************

for entry in scandir(directory):
    if entry.path.startswith(directory+rec_start_string):    # Find unique recording string. tsR10 for all cytokine injections, tsR12 for saline. 
        #matlab_file = entry.path[19:-4] # Find unique recording string'
        matlab_file = entry.path[len(directory+'\\ts'):-len('.mat')]    # extract only the matlab_file name from string.
        print()
        print('*******************************************************************************')
        print(f'Starting analysis on recording : {matlab_file}')
        print()
        path_to_wf = directory + '/wf'+matlab_file + '.mat'    # This is how the waveforms are saved. path + wf+MATLAB_FILE_NAME.mat
        path_to_ts = directory + '/ts'+matlab_file + '.mat'    # This is how the timestamps are saved. path + ts+MATLAB_FILE_NAME.mat

        # ***** Define paths and name to save files/figures: ********
        unique_string_for_run = training_start_title+matlab_file
        unique_string_for_figs = training_start_title + matlab_file.replace('.', '_')    # '.' not allowed in plt.savefig
        path_to_weights = 'models/'+unique_string_for_run
        # Numpy file paths:
        path_to_hpdp = "../numpy_files/numpy_hpdp/" + unique_string_for_run 
        path_to_EVlabels = "../numpy_files/EV_labels/" + unique_string_for_run
        path_to_cytokine_candidate = '../numpy_files/cytokine_candidates/' + unique_string_for_run
        
        
        # ************************************************************
        # ******************** Load Files ****************************
        # ************************************************************
        waveforms = load_waveforms(path_to_wf, 'waveforms')
        timestamps = load_timestamps(path_to_ts, 'timestamps')

        # ************************************************************
        # ******************** Preprocess ****************************
        # ************************************************************
        # Cut first and last part of recording to ensure stable "sleep-state" during recording.
        # Furthermore a downsampling is applied to speed up training. 

        wf0, ts0 = preprocess_wf.get_desired_shape(waveforms, timestamps, hypes)    # No downsampling. Used for evaluation 
        #plot_amplitude_hist(waveforms)

        print(f'Shape before amplitude threshold : {waveforms.shape} \n')
        # Remove "extreme-amplitude" CAPs -- otherwise risk that pdf-GD diverges:
        waveforms, timestamps = preprocess_wf.apply_amplitude_thresh(waveforms, timestamps, hypes) 
        print(f'Shape after amplitude threshold : {waveforms.shape} \n')

        waveforms, timestamps = preprocess_wf.get_desired_shape(waveforms, timestamps, hypes)
        print(f'Shape after shape-preprocessing : {waveforms.shape}')
        # ****** Standardise waveforms for more stable training. ********
        standardise_waveforms = hypes["preprocess"]["standardise_waveforms"]
        if standardise_waveforms:
            print(f' \n Standardises waveforms... \n')
            waveforms = preprocess_wf.standardise_wf(waveforms)
            wf0 = preprocess_wf.standardise_wf(wf0)

        # ************************************************************
        # ******************** Event-rate Labeling *******************
        # ************************************************************
        if path.isfile(path_to_EVlabels+'.npy'):
            print(f' \n Loading saved EV-labels from {path_to_EVlabels} \n')
            ev_labels = np.load(path_to_EVlabels + '.npy')
            ev_stats_tot = np.load(path_to_EVlabels + 'tests_tot.npy') 
        else:
            ev_labels, ev_stats_tot = get_ev_labels(waveforms, timestamps, 
                                                    hypes, saveas=path_to_EVlabels)
        # ************************************************************
        # ******************* Event-rate Threshold *******************
        # ************************************************************
        print(f'\n Number of wf which : \
                ("icreased after first","increased after second", "constant") = {np.sum(ev_labels,axis=1)} ')

        # ho := High Occurance, ts := timestamps
        wf_ho, ts_ho, ev_label_ho = preprocess_wf.apply_mean_ev_threshold(waveforms, timestamps, 
                                                                          ev_stats_tot[0], hypes, 
                                                                          ev_labels=ev_labels)
        print(f'After EV threshold : \
                ("icreased after first","increased after second", "constant") = {np.sum(ev_label_ho,axis=0)} ')

        # ************************************************************
        # ******************** Train/Load model **********************
        # ************************************************************
        if run_DBscan is False:
            print('\n *********************** Tensorflow Blaj ************************************* \n')
            encoder, decoder, cvae = get_pdf_model(wf_ho, hypes, 
                                                   path_to_weights=path_to_weights, 
                                                   continue_train=continue_train, 
                                                   verbose=1, ev_label=ev_label_ho)
            print('\n ****************************************************************************** \n')

        if view_cvae_result:
            save_figure = 'figures/encoded_decoded/' + unique_string_for_figs
            plot_decoded_latent(decoder, saveas=save_figure+'_decoded_constant',
                                verbose=verbose_main, ev_label=np.array((0,0,1)).reshape((1,3)))
            plot_decoded_latent(decoder, saveas=save_figure+'_decoded_increase_first', 
                                verbose=verbose_main, ev_label=np.array((1,0,0)).reshape((1,3)))
            plot_decoded_latent(decoder, saveas=save_figure+'_decoded_increase_second', 
                                verbose=verbose_main, ev_label=np.array((0,1,0)).reshape((1,3)))

            continue_to_run_GD = input('Continue to gradient decent of pdf? (yes/no) :')

            all_fine = False
            while all_fine==False:
                if continue_to_run_GD == 'no':
                    exit()
                elif continue_to_run_GD == 'yes':
                    print('Continues to "run_GD"')
                    all_fine = True
                else:
                    continue_to_run_GD = input('Invalid input, continue to gradient decent of pdf? (yes/no) :')


        # ************************************************************
        # *  Perform GD on pdf to find high prob. data-points (hpdp) *
        # ************************************************************  
        if run_GD:
            print('\n Running pdf_GD to get hpdp... \n')
            hpdp_list = run_pdf_GD(wf_ho, cvae, ev_label_ho, hypes, matlab_file=matlab_file,
                                   unique_string_for_figs=unique_string_for_figs,
                                   path_to_hpdp=path_to_hpdp,
                                   verbose=False,
                                   view_GD_result=view_GD_result,
                                   encoder=encoder)
        else:
            print('\n Skipps over pdf_GD... \n')
        # ************************************************************
        # *********** Inference from increased EV hpdp ***************
        # ************************************************************
        if plot_hpdp_assesments:
            print('\n Running visual evaluation of the resulting hpdp using gradient descent... \n')
            run_visual_evaluation(wf0, ts0, hpdp_list,
                                  encoder, hypes, 
                                  unique_string_for_figs=unique_string_for_figs,
                                  path_to_cytokine_candidate=path_to_cytokine_candidate)

        # ************************************************************
        # ******** Look for responders using hpdp-cluster ************
        # ************************************************************
        if run_automised_assesment:
            saveas = path_to_cytokine_candidate+'auto_assesment'
            run_evaluation(wf0, ts0, hpdp_list, 
                        encoder, hypes, saveas=saveas)

        # Run DBSCAN on labeled data to see if the obtained results are similar. 
        if run_DBscan:
            saveas = 'figures/dbscan/'+unique_string_for_figs + 'DBSCAN'
            np_saveas = path_to_cytokine_candidate + 'DBSCAN'
            run_DBSCAN_evaluation(wf_ho, wf0, ts0, 
                                  ev_label_ho, hypes, 
                                  saveas=saveas, 
                                  np_saveas=np_saveas, 
                                  matlab_file=matlab_file)


print('Finished successfully')
