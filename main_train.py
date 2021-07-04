'''
 ************************************************************
 Main file for identifying cytokine-encoding CAPs. 
 i.e, the first script to run after the MATLAB preprocessing of the raw recordings.

 Loads preprocessed "waveforms" (wf)- and "timestamps" (ts)- data. (.mat files)
 These are the resulting CAPs from the MATLAB preprocessing of the raw VN-recordings.
 
 The "output" of the analysis in this files are a number of saved files, which is 
 loaded in the script for the second and last step of the analysis; "main_evaluation.py".

 The following data is saved:
    * event-rate labels : "path_to_EVlabels" + ".npy"
    * event-rate stats : "path_to_EVlabels" + "tests_tot.npy"
        (EV-Mean, EV-SD) for full time of recording for each CAP.
    * CVAE-network weights : 
    * hpdp : "path_to_hpdp" + "0" or "1" + ".npy"
        High probability data points. Result from CVAE-gradient descent for each injection-label.
    * cytokine_candidates : "path_to_cytokine_candidate" + "auto_assesment" + .npy

 --------------------------------------
 Master Thesis, KTH , Gabriel Andersson
 ************************************************************
 Notation used throughout:
 wf := waveforms, CAP := compound action potential (wf/waveform/CAPa are used interchangeably).
 ho := High Occurance, ts := timestamps
 EV := event-rate [CAPs/sec]
 hpdp := high probability data points.
 GD := Gradient Descent
 pdf := probability density function
'''
import numpy as np
import matplotlib.pyplot as plt
import json
from os import path, scandir, sys
from sklearn.cluster import DBSCAN

sys.path.insert(1,'src/')   # Add directory to path for function-imports

import preprocess_wf
from load_and_GD_funs import load_mat_file, get_pdf_model, run_pdf_GD  
from event_rate_funs import get_ev_labels 
from plot_functions_wf import plot_decoded_latent, plot_amplitude_hist 
from evaluation import run_DBSCAN_evaluation, run_evaluation, run_visual_evaluation


continue_train = False    # Tensorflow CVAE-model.
run_GD = True
zero_GD_iterations = True # True if you only want to e.g. only rerun assessment 
# ****** If using 2D-latent space dimension: *********
view_cvae_result = False    # True => reqires user to give input if to continue-
                            # -the script to pdf-GD or not.
view_GD_result = False    # This reqires user to give input if to continue the-
                          # -script to clustering or not.
plot_hpdp_assesments = False    # Cluster and evaluate hpdp to find -
                               # -cytokine-candidate CAP manually inspecting plots.
# ***********************

run_automised_assesment = True    # Cluster and evaluate hpdp by defined quantitative measure.
run_DBscan = False

verbose_main = 1

# *****************************************************************************
# Specify unique title for the run.
training_start_title =  'zanos_0702' # 
training_start_title =  'chan_pre_proced_KI' # 
# *****************************************************************************

# ************************************************************
# ************ LOAD HYPERPARAMERS ****************************
# ************************************************************

with open('hypes/'+ training_start_title+'.json', 'r') as f:
    hypes = json.load(f)

if zero_GD_iterations:
    # This assumes that the full training has been complete, and the run is only for assessment..
    hypes['pdf_GD']['m'] = 0


# ***** Specify path to directory of recordings ******* 
directory = '../matlab_files'
directory = 'MATLAB/preprocessed2'
# *****************************************************

# ***** Specify the starting scaracters in filename of recordings to analyse *****
# If "ts" is not specified, then all files will be run twise, since we have 
# one file for timestamps and one for CAP-waveform with identical names, 
# exept the starting ts/wf.
rec_start_string = 'R10' #.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05' # Since each recording has two files in directory (waveforms and timestamps)-- this is solution to only get each recording once.
# rec_start_string = '\\tsR10_Exp2_7.20'   # .16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_15'

# rec_start_string = 'R10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05'
rec_start_string = '_final' # '_10min_KCl_10min_210617_122538A-000'
# *****************************************************
labeling_results = {}
# Loop through all saved recordings .mat files in specified directory.
for entry in scandir(directory):

    # Find unique recording string. tsR10 for all cytokine injections, tsR12 for saline. 
    if entry.name.startswith('ts' + rec_start_string): 
        # Extract only the matlab_file name from full file name.
        matlab_file = entry.path[len(directory+'/ts'):-len('.mat')] 
        print('\n' + '*'*80 + '\n')
        print(f'Starting analysis on recording : {matlab_file}')

        # This is titles of the waveforms and timestamps files saved from matlab.
        path_to_wf = directory + '/wf'+matlab_file + '.mat'    # path + wf+MATLAB_FILE_NAME.mat
        path_to_ts = directory + '/ts'+matlab_file + '.mat'    # path + ts+MATLAB_FILE_NAME.mat

        # Define paths and titles to save files/figures.
        unique_string_for_run = training_start_title+matlab_file

        # '.' not allowed in plt.savefig
        unique_string_for_figs = training_start_title + matlab_file.replace('.', '_') 
        path_to_weights = 'models/'+unique_string_for_run

        # Numpy file paths:
        path_to_hpdp = "../numpy_files/numpy_hpdp/" + unique_string_for_run 
        path_to_EVlabels = "../numpy_files/EV_labels/" + unique_string_for_run
        path_to_cytokine_candidate = '../numpy_files/cytokine_candidates/' + unique_string_for_run
        
        # ************************************************************
        # ******************** Load Files ****************************
        # ************************************************************
        waveforms = load_mat_file(path_to_wf, 'waveforms')
        timestamps = load_mat_file(path_to_ts, 'timestamps')
        

        # ************************************************************
        # ******************** Preprocess ****************************
        # ************************************************************
        # Cut first and last part of recording to ensure stable "sleep-state" during recording.
        # Furthermore a downsampling is applied to speed up training. 

        # First, saves copy without any downsampling. Used for evaluation:
        wf0, ts0 = preprocess_wf.get_desired_shape(waveforms, timestamps, hypes, training=False )
        #plot_amplitude_hist(waveforms)

        print(f'Shape before amplitude threshold : {waveforms.shape} \n')
        # Remove "extreme-amplitude" CAPs -- otherwise risk that pdf-GD diverges:
        waveforms, timestamps = preprocess_wf.apply_amplitude_thresh(waveforms, timestamps, 
                                                                     hypes) 
        print(f'Shape after amplitude threshold : {waveforms.shape} \n')

        waveforms, timestamps = preprocess_wf.get_desired_shape(waveforms, timestamps, 
                                                                hypes, training=True )
        print(f'Shape after shape-preprocessing : {waveforms.shape}')

        standardise_waveforms = hypes["preprocess"]["standardise_waveforms"]
        if standardise_waveforms:
            # Standardise waveforms for more stable training.
            print(f' \n Standardises waveforms... \n')
            waveforms = preprocess_wf.standardise_wf(waveforms, hypes)
            wf0 = preprocess_wf.standardise_wf(wf0, hypes)

        # ************************************************************
        # ******************** Event-rate Labeling *******************
        # ************************************************************

        # Look if labels has been saved in earlier run. If so, the .npy files are loaded.
        if path.isfile(path_to_EVlabels+'.npy'):
            print(f' \n Loading saved EV-labels from {path_to_EVlabels} \n')
            ev_labels = np.load(path_to_EVlabels + '.npy')
            ev_stats_tot = np.load(path_to_EVlabels + 'tests_tot.npy')
        else:
            # Else, run labeling method. 
            ev_labels, ev_stats_tot = get_ev_labels(waveforms, timestamps, 
                                                    hypes, saveas=path_to_EVlabels)
        
        # ************************************************************
        # ******************* Event-rate Threshold *******************
        # Remove CAPs with a mean estimated EV below set threshold.
        print(f'\n Number of wf which : \
                ("icreased after first","increased after second", "constant") = \
                    {np.sum(ev_labels,axis=1)} ')
        wf_ho, ts_ho, ev_label_ho = preprocess_wf.apply_mean_ev_threshold(waveforms, timestamps, 
                                                                          ev_stats_tot[0], hypes, 
                                                                          ev_labels=ev_labels)
        print(f'After EV threshold : \
                ("icreased after first","increased after second", "constant") = \
                {np.sum(ev_label_ho,axis=0)} ')
        labeling_results.update({matlab_file : [np.sum(ev_labels,axis=1), np.sum(ev_label_ho,axis=0)]})
        # ************************************************************
        # ******************** Train/Load model **********************
        # ************************************************************
        # Loads trained CVAE/VAE probability model if it exsists in specified path.
        # Otherwise initiates training of model.
        if run_DBscan is False:
            print('\n *********************** Tensorflow Blaj ************************************* \n')
            encoder, decoder, cvae = get_pdf_model(wf_ho, hypes, 
                                                   path_to_weights=path_to_weights, 
                                                   continue_train=continue_train, 
                                                   verbose=1, ev_label=ev_label_ho)
            print('\n ****************************************************************************** \n')
        # qqq: TODO: REMOVE THIS ??:
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
        # *************** pdf-Gradient Descent ***********************
        # ************************************************************        
        # Perform GD on pdf to find high prob. data-points (hpdp)   
        if run_GD:
            print('\n Running pdf_GD to get hpdp... \n')
            hpdp_list = run_pdf_GD(wf_ho, cvae, ev_label_ho, hypes, 
                                   matlab_file=matlab_file,
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
        # Run visual evaluation where user specify the number of clusters etc.
        if plot_hpdp_assesments:
            print('\n Running visual evaluation of the resulting hpdp using gradient descent... \n')
            run_visual_evaluation(wf0, ts0, hpdp_list,
                                  encoder, hypes, 
                                  unique_title_for_figs=unique_string_for_figs,
                                  path_to_save_candidate=path_to_cytokine_candidate)
        # Evaluation with prespecified number of clusters etc. No user input required. 
        if run_automised_assesment:
            saveas = path_to_cytokine_candidate + 'auto_assesment'
            run_evaluation(wf0, ts0, hpdp_list, 
                           encoder, hypes, saveas=saveas)

        # ************************************************************
        # ************************ DBSCAN ****************************
        # ************************************************************
        # Run DBSCAN on labeled data to see if the obtained results are similar. 
        if run_DBscan:
            saveas = 'figures/dbscan/' + unique_string_for_figs + 'DBSCAN'
            np_saveas = path_to_cytokine_candidate + 'DBSCAN'
            run_DBSCAN_evaluation(wf_ho, wf0, ts0, 
                                  ev_label_ho, hypes, 
                                  saveas=saveas, 
                                  np_saveas=np_saveas, 
                                  matlab_file=matlab_file)

# json_label_res = json.dumps(labeling_results)
# with open('label_dictionaries/'+ training_start_title+'.json', 'w') as f:
#     f.write(json_label_res)
#     f.close()
for key in labeling_results.keys():
    print(f'{key}: \n {labeling_results[key]}\n')
print('Finished successfully!')
