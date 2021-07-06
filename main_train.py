'''
 ************************************************************
 Main file for identifying cytokine-encoding CAPs. 
 i.e, the first script to run after the MATLAB preprocessing of the raw recordings.

 Loads preprocessed "waveforms" (wf)- and "timestamps" (ts)- data. (.mat files)
 These are the resulting CAPs from the MATLAB preprocessing of the raw VN-recordings.

 OBS! The script assumes that the stored waveforms and timestamp - .mat files are saved as
 "wf" + "..." and "ts" + "..." . I.e. that the filenames starts with "wf" and "ts" respectively.
 
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

The start of the script defines boolean variables that defines what is to be run, verbosity etc.

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
import textwrap
from os import path, scandir, sys

sys.path.insert(1,'src/')   # Add directory to path for function-imports

import preprocess_wf
from load_and_GD_funs import load_mat_file, get_pdf_model, run_pdf_GD  
from event_rate_funs import get_ev_labels, get_event_rates 
from plot_functions_wf import plot_decoded_latent, plot_amplitude_hist, plot_event_rates 
from evaluation import run_DBSCAN_evaluation, run_evaluation, run_visual_evaluation

# NEW TESTS:
from new_clustering_approach import runTSNE, runDBSCAN

continue_train = False    # Tensorflow CVAE-model.
run_GD = True             # CVAE/GD

zero_GD_iterations = True # True if you only want to e.g. only rerun assessment

# ****** If using 2D-latent space dimension: *********
view_cvae_result = False       # True => reqires user to give input if to continue-
                               # -the script to pdf-GD or not.
view_GD_result = False         # This reqires user to give input if to continue the-
                               # -script to clustering or not.
plot_hpdp_assesments = False   # Cluster and evaluate hpdp to find -
                               # -cytokine-candidate CAP manually inspecting plots.
# ***********************

run_automised_assesment = True    # Cluster and evaluate hpdp by defined quantitative measure.
run_DBscan = False

verbose_main = 1

# *****************************************************************************
# Specify unique title for the run.
training_start_title =  'zanos_0702' # 
training_start_title =  'all_channels_2_KI' # 
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
"""
If empty string '', is specified then all files in directory will be run.
For Zanos: 'R10'-files will run analysis on cytokine-injection-files. 
'R12'-files => Saline.
Only run Analysis on one experiment by specifying e.g. 'R10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05'
"""

# rec_start_string = 'R10' 
# rec_start_string = 'R10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05'
rec_start_string = '_final2'    #Baseline_10min_Saline' # '_10min_KCl_10min_210617_122538A-000'
# *****************************************************
labeling_results = {}       # Us
for entry in scandir(directory):
    # Loop through all saved recordings .mat files in specified directory.
    if not entry.name.endswith('.mat'):
        # Skip files that is not on .mat format -- go to next file in directory
        continue
    if entry.name.startswith('ts' + rec_start_string):
        # Find unique recording string. tsR10 for all cytokine injections, tsR12 for saline. 

        # Extract only the matlab_file name from full file name:
        matlab_file = entry.path[len(directory+'/ts'):-len('.mat')] 
        print('\n' + '*'*80 + '\n')
        print(f'Starting analysis on recording : {matlab_file}')

        # This is the titles of the waveforms and timestamps files saved from matlab.
        path_to_wf = directory + '/wf' + matlab_file + '.mat'    # path + wf+MATLAB_FILE_NAME.mat
        path_to_ts = directory + '/ts' + matlab_file + '.mat'    # path + ts+MATLAB_FILE_NAME.mat

        # Define paths and titles to save files/figures.
        unique_string_for_run = training_start_title + matlab_file      # the unique part for this run. 

        # '.' not allowed in plt.savefig.. replace with '_' :
        unique_string_for_figs = training_start_title + matlab_file.replace('.', '_')

        # Path to save/Load Tensorflow weights. 
        path_to_weights = 'models/' + unique_string_for_run

        # Numpy file paths. This is where the actual results from the run is saved. (together with tf-weights..)
        path_to_hpdp = "../numpy_files/numpy_hpdp/" + unique_string_for_run 
        path_to_EVlabels = "../numpy_files/EV_labels/" + unique_string_for_run
        path_to_cytokine_candidate = '../numpy_files/cytokine_candidates/' + unique_string_for_run
        
        # ************************************************************
        # ******************** Load Files ****************************
        # ************************************************************
        '''
        Load the MATLAB preprocessed files waveforms and timestamps. 
        These are saved with the keys "waveforms" and "timestamps", respectively.
        '''
        waveforms = load_mat_file(path_to_wf, 'waveforms')  
        timestamps = load_mat_file(path_to_ts, 'timestamps')
        

        # ************************************************************
        # ******************** Preprocess ****************************
        # ************************************************************
        '''
        Cut first and last part of recording to ensure consistensy and stable "sleep-state" during recording.
        These "cut-times" are specified in hypes.
        Furthermore a downsampling can be applied to speed up training. Using e.g. every second CAP.
        '''
        # First, saves copy without any downsampling. Used for evaluation:
        wf0, ts0 = preprocess_wf.get_desired_shape(waveforms, timestamps, hypes, training=False)
        
        ''' Uncomment below to see how the amplitudes are distributed as a amplitude-threshold-assessment: '''
        # plot_amplitude_hist(waveforms)

        '''
        Remove "extreme-amplitude" CAPs. That is CAPs where :
        max-amp > max_amp_thresh.
        max-amp < min_amp_thresh.
        min-amp > - max_amp_thresh.
        min-amp > - min_amp_thresh.
        Without max-amp thresh, the pdf-GD often diverges.
        The min-amp thresh is thought to remove CAPs with a "baseline" too far from zero.
        '''

        print(f'Shape before amplitude threshold : {waveforms.shape} \n')
        
        waveforms, timestamps = \
            preprocess_wf.apply_amplitude_thresh(waveforms, timestamps, hypes) 

        print(f'Shape after amplitude threshold : {waveforms.shape} \n')

        waveforms, timestamps =  \
            preprocess_wf.get_desired_shape(waveforms, timestamps, hypes, training=True)
        print(f'Shape after shape-preprocessing : {waveforms.shape}')
        '''
        Standardising the waveforms is nessessary for the proposed similarity measures to work properly.
        It does remove some information regarding amplitues, but hopefully enough information is kept in the 
        remaining shape to find CAPs correlated with injection-event. 
        Furthermore, not being reliant on the amplitudes could make the analysis less dependent on the sensitivity of the
        electrode..
        '''
        standardise_waveforms = hypes["preprocess"]["standardise_waveforms"]
        if standardise_waveforms:
            print(f'Standardises waveforms... \n')
            waveforms = preprocess_wf.standardise_wf(waveforms, hypes)
            wf0 = preprocess_wf.standardise_wf(wf0, hypes)

        # ************************************************************
        # ******************** Event-rate Labeling *******************
        # ************************************************************
        '''
        Go through the preprocessed waveforms to estimate the event-rate of each.
        The CAPs are labeled based on change of event-rate at injection event.
        '''
        if path.isfile(path_to_EVlabels+'.npy'):
            # Look if labels has been saved in previous run. If so, the .npy files are loaded.
            print(f' \n Loading saved EV-labels from {path_to_EVlabels} \n')
            ev_labels = np.load(path_to_EVlabels + '.npy')
            ev_stats_tot = np.load(path_to_EVlabels + 'tests_tot.npy') # "stats_tot.npy" for new runs, "tests_tot.npy" for old..

        else:
            # Else, run labeling method from scratch. 
            ev_labels, ev_stats_tot = get_ev_labels(waveforms, timestamps, 
                                                    hypes, saveas=path_to_EVlabels)
        
        # ************************************************************
        # ******************* Event-rate Threshold *******************
        # ************************************************************
        # Remove CAPs with a mean estimated EV below set threshold.
        
        print(f'''\n Number of wf which :
              ("icreased after first","increased after second", "constant") :
              {np.sum(ev_labels, axis=1)} ''')

        wf_ho, ts_ho, ev_label_ho = preprocess_wf.apply_mean_ev_threshold(waveforms, 
                                                                          timestamps, 
                                                                          ev_stats_tot[0], 
                                                                          hypes, 
                                                                          ev_labels=ev_labels)
        print(f'''After EV threshold :
              ("icreased after first","increased after second", "constant") :
              {np.sum(ev_label_ho, axis=0)} ''')

        # For some verbosity regarding labeling. Printed in the end of run.
        labeling_results.update({matlab_file : [np.sum(ev_labels,axis=1), np.sum(ev_label_ho,axis=0)]})
        
        # ************************************************************
        # ******************** Train/Load model **********************
        # ************************************************************
        '''
        Loads trained CVAE/VAE probability model if it exists at specified path.
        Otherwise initiates training of model.

        '''
        if run_DBscan is False:
            print('\n *********************** Tensorflow VERBOSITY ************************************* \n')
            encoder, decoder, cvae = get_pdf_model(wf_ho, hypes, 
                                                   path_to_weights=path_to_weights, 
                                                   continue_train=continue_train, 
                                                   verbose=1, ev_label=ev_label_ho)
            print('\n ****************************************************************************** \n')

        if view_cvae_result:
            # Possible verbosity. Show plots of decoded latent-space. [-2,2]x[-2,2] grid.

            save_figure = 'figures/encoded_decoded/' + unique_string_for_figs
            plot_decoded_latent(decoder, saveas=save_figure+'_decoded_constant',
                                verbose=verbose_main, ev_label=np.array((0,0,1)).reshape((1,3)))
            plot_decoded_latent(decoder, saveas=save_figure+'_decoded_increase_first', 
                                verbose=verbose_main, ev_label=np.array((1,0,0)).reshape((1,3)))
            plot_decoded_latent(decoder, saveas=save_figure+'_decoded_increase_second', 
                                verbose=verbose_main, ev_label=np.array((0,1,0)).reshape((1,3)))

        # ************************************************************
        # *************** pdf-Gradient Descent ***********************
        # ************************************************************        
        # Perform CVAE-GD on pdf to find high prob. data-points (hpdp)   
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
        
        if plot_hpdp_assesments:
            # Run visual evaluation where user specifies the number of clusters etc. based on
            # Figures of Latent space after GD.
            print('\n Running visual evaluation of the resulting hpdp using gradient descent... \n')
            run_visual_evaluation(wf0, ts0, hpdp_list,
                                  encoder, hypes, 
                                  unique_title_for_figs=unique_string_for_figs,
                                  path_to_save_candidate=path_to_cytokine_candidate)

        if run_automised_assesment:
            # Evaluation with prespecified number of clusters etc. No user input required. 

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
            '''
            # #####################%%%%%%%%%%%%¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤###################
            # TODO: TESTS OF T-SNE + DBSCAN Approach TODO TODO TODO TODO TODO TODO
            # Runs t-SNE dimensionallity reduction on waveforms and clusters the 
            # 2-dim manyfold using DBSCAN. 

            tsne_wf = runTSNE(wf0[0:55000, :], reduced_dim=2, verbose=True)
            # tsne_wf = np.load('aa.npy')
            persistent_hom = []
            eps_range =  np.arange(1.5, 10, 0.2)
            for epsilon in eps_range:
                labels = runDBSCAN(tsne_wf,  db_eps=epsilon, db_min_sample=100)
                n_clusters = np.sum(np.unique(labels)) - 1
                persistent_hom.append( n_clusters )
            plt.plot(eps_range, persistent_hom)
            plt.show()
            event_rate_results = get_event_rates(timestamps[0:55000], hypes, labels=labels, consider_only=None)
            plot_event_rates(event_rate_results, timestamps[0:55000], conv_width=10)    

            # #####################%%%%%%%%%%%%¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤###################
            '''

for key in labeling_results.keys():
    # Print the number of waveforms with each label for all recordings.
    print(f'{key}: \n {labeling_results[key]}\n')

print('Finished successfully!')
