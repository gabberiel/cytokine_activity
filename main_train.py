'''
 ************************************************************
 Main file for identifying CAPs "correlated" with injection event. 
 i.e, the first script to run after the MATLAB preprocessing of the raw recordings.

 Loads preprocessed "waveforms" (wf)- and "timestamps" (ts)- data. (.mat files)
 These are the resulting CAPs from the MATLAB preprocessing of the raw VN-recordings.

 OBS! The script assumes that the stored waveforms and timestamp - .mat files are saved as
 "wf" + "shared_recoding_name" and "ts" + "shared_recoding_name" . I.e. that the filenames 
 starts with "wf" and "ts" for each recording.

 The hyperparameters of the analysis is given by a .json file, denoted "hypes" and given as 
 input to most function as a dictionary.
 
 The "output" of the analysis in this script is a number of saved files, which are loaded if 
 you want to continue training at some stage, or by plotting some assessments in "main_visualisations.py".
 The final results "cytokine_candidates" are loaded in "main_evaluation.py" as a last step in 
 the analysis. The best results are then plotted and the best CAPs out of the candidates can 
 be saved.

 The following data is saved:
    * event-rate labels : "path_to_EVlabels" + ".npy"
        3-dim one-hot encoded vector for each CAP-waveform (after python-preprocessing)
    * event-rate stats : "path_to_EVlabels" +  "stats_tot.npy" (alt. "tests_tot.npy" for old runs.)
        (EV-Mean, EV-SD) for full time of recording for each CAP.
    * CVAE-network weights : "path_to_weights" 
        Tensorflow network weights. Observe that too long file-names can cause utf-8 encoding error.
        (If the string to save weights exceeds approx 255 characters..)
    * hpdp : "path_to_hpdp" + ("0" _or_ "1") + ".npy"
        High probability data points. Result from CVAE-gradient descent for each injection-label.
    * cytokine_candidates : "path_to_cytokine_candidate" + "auto_assesment" + .npy
        A Candidate-CAP for each observed cluster in the hpdp-data.

The start of the script defines boolean variables that defines what is to be run, verbosity etc.

 --------------------------------------
 Gabriel Andersson 2021-08
 ************************************************************

 Notation used throughout:
 wf := waveforms, CAP := compound action potential (wf/waveform/CAPs are used interchangeably).
 ts := timestamps, ho := High Occurance, 
 EV := event-rate [CAPs/sec] or [% of total EV].
 hpdp := high probability data points.
 GD := Gradient Descent
 pdf := probability density function

'''
import json
import numpy as np
from os import path, scandir, sys

sys.path.insert(1,'src/')   # Add src-directory to path for function-imports

# import matplotlib.pyplot as plt
import preprocess_wf
from load_and_GD_funs import load_mat_file, get_pdf_model, run_pdf_GD  
from event_rate_funs import get_ev_labels 
from plot_functions_wf import plot_decoded_latent, plot_amplitude_hist 
from evaluation import run_DBSCAN_evaluation, run_evaluation, run_visual_evaluation


continue_train = False    # Tensorflow CVAE-model.
run_GD = True             # CVAE-GD
zero_GD_iterations = True # True if you only want to run assessment. (i.e load hpdp but not continue GD.)

manual_assesments = True      # Cluster and evaluate hpdp to find -
                               # -cytokine-candidate CAP manually inspecting plots.
run_automised_assesment = True    # Cluster and evaluate hpdp by defined quantitative measure.

run_DBscan = False

verbose_main = 1

# ****** If using 2D-latent space dimension: *********
view_cvae_result = False       # True => plots grid of decoded latent variables.

view_GD_result = True         # Plots encoded latent-variable before/after GD.
# ***********************

# *****************************************************************************
# Specify unique title for the training-run. (hype-file title)
hypes_file_name =  'zanos_0702' # 
hypes_file_name =  'all_channels_2_KI' # 
hypes_file_name =  'one_chan_KI_30min' # 
# hypes_file_name =  'all_chan_KI_10min2' # 

# *****************************************************************************

# ************************************************************
# ************ LOAD HYPERPARAMERS ****************************
# ************************************************************

with open('hypes/'+ hypes_file_name+'.json', 'r') as f:
    hypes = json.load(f)

if zero_GD_iterations:
    # This assumes that the full training has been complete, and the run is only for assessment..
    hypes['pdf_GD']['m'] = 0


# ***** Specify path to directory of recordings ******* 
# directory = '../matlab_files'   # Zanos recordings
directory = 'MATLAB/preprocessed2'  # KI recordings

# *****************************************************
# ***** Specify the starting characters in filename of recordings to analyse *****
"""
If you would like run the analysis on one recording that is saved as 
    "ts2020New_Saline_recordings_wooho" & "wf2020New_Saline_recordings_wooho",
then you should set: 
    rec_start_string= "2020New_Saline_recordings_wooho".

If "rec_start_string" is an empty string:'', then the analysis will be run on all recordings 
in the specified directory.

For Example, "R10" => the Zanos-cytokine-injections, and "R12" => the Zanos saline injection. (Assuming correct directory..)

Check file names in the "directory" folder to specify "rec_start_string".
OBS! To long file name will create "utf-8" error when saving tensorflow weights..
"""

# rec_start_string = 'R10' 
# rec_start_string = '210715_30min_baseline_30min'
rec_start_string = '_final3_210715_30min'
rec_start_string = '210715_30min_baseline_30min_ZYA_injection_30min_KCl_A-000_second'
# rec_start_string = '_final2Baseline' # Baseline_10min'    #Baseline_10min_Saline' # '_10min_KCl_10min_210617_122538A-000'

# *****************************************************
labeling_results = {}       # Used for final prints about number of CAPs with each label for the different recordings.
for entry in scandir(directory):
    # Loop through all saved recordings .mat files in specified directory.

    if not entry.name.endswith('.mat'):
        # Skip files that is not on .mat format -- go to next file in directory
        continue

    if entry.name.startswith('ts' + rec_start_string):
        # A recording of interest is found in directory. The analysis will be now be applied on it. 

        # Extract only the matlab_file name from full file-name:
        matlab_file = entry.path[len(directory+'/ts'):-len('.mat')] 
        print('\n' + '*'*80 + '\n')
        print(f'Starting analysis on recording : {matlab_file}')

        # ************************************************************
        # ********** Define paths to save files/figures **************
        # ************************************************************

        # Define the unique part string to save files/figures for this run.
        unique_string_for_run = hypes_file_name + matlab_file   # the unique part of saves for this run. 

        # '.' not allowed in plt.savefig.. replace with '_' :
        unique_string_for_figs = hypes_file_name + matlab_file.replace('.', '_')

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
        # The following is the titles of the waveforms and timestamps files saved from MATLAB preprocessing.
        path_to_wf = directory + '/wf' + matlab_file + '.mat'    # path + wf + recording_name.mat
        path_to_ts = directory + '/ts' + matlab_file + '.mat'    # path + ts + recording_name.mat

        waveforms = load_mat_file(path_to_wf, 'waveforms')  
        timestamps = load_mat_file(path_to_ts, 'timestamps')
        
        # ************************************************************
        # ******************** Preprocess ****************************
        # ************************************************************
        '''
        Cut first and last part of recording to ensure consistensy and stable "sleep-state" during recording.
        These "cut-times" are specified in "hypes['experiment_setup']['start/end_time']"
        Furthermore, a downsampling can be applied to speed up training. Using e.g. every second CAP.
        The downsampling is useful as a first step to see if the parameters are suitible for the EV-labeling.
        (since this can be a bit time-consuming.)
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
        It possibly removes some information regarding amplitues, but hopefully enough information is kept in the 
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
        Go through the preprocessed waveforms to make a first estimation of the event-rate for each.
        The CAPs are labeled based on the change of this event-rate at injection event.
        The hyperparameters for the type of similarity measure, thresholds etc. are 
        specified in "hypes['labeling'][...]"
        '''

        if path.isfile(path_to_EVlabels+'.npy'):
            # Look if labels has been saved in previous run. If so, the .npy files are loaded.
            # If you wish to make a new run using the same hypes-file, simply delete the saved EV-label file.
            print(f' \n Loading saved EV-labels from {path_to_EVlabels} \n')
            ev_labels = np.load(path_to_EVlabels + '.npy')
            ev_stats_tot = np.load(path_to_EVlabels + 'stats_tot.npy') # "stats_tot.npy" for new runs, "tests_tot.npy" for old..

        else:
            # Else, run labeling method from scratch. 
            ev_labels, ev_stats_tot = get_ev_labels(waveforms, timestamps, 
                                                    hypes, saveas=path_to_EVlabels)
        
        # ************************************************************
        # ******************* Event-rate Threshold *******************
        # ************************************************************
        ''' Remove CAPs with an estimated mean EV below set threshold. '''

        print(f'''\n Number of CAPs that :
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

        # For some verbosity regarding labeling. Printed in the end of run:
        labeling_results.update({matlab_file : [np.sum(ev_labels,axis=1), np.sum(ev_label_ho,axis=0)]})
        
        # ************************************************************
        # ******************** Train/Load model **********************
        # ************************************************************
        '''
        Loads trained CVAE/VAE probability model if it exists at specified path.
        Otherwise, initiates training of model from scratch.
        If model exists, the boolean param "continue_train" determines if more training is to be run 
        or not. (Specified in the begining of script.)
        '''

        if run_DBscan is False:
            print('\n *********************** TENSORFLOW VERBOSITY ************************************* \n')
            encoder, decoder, cvae = get_pdf_model(wf_ho, hypes, 
                                                   path_to_weights=path_to_weights, 
                                                   continue_train=continue_train, 
                                                   verbose=1, ev_label=ev_label_ho)
            print('\n ' + '*'*80 + ' \n')

        if view_cvae_result:
            ''' Possible verbosity. Show plots of decoded latent-space. [-2,2]x[-2,2] grid.
            This is a bit time consuming. '''

            save_figure = 'figures/encoded_decoded/' + unique_string_for_figs
            plot_decoded_latent(decoder, saveas=save_figure+'_decoded_increase_first', 
                                verbose=verbose_main, ev_label=np.array((1,0,0)).reshape((1,3)))
            plot_decoded_latent(decoder, saveas=save_figure+'_decoded_increase_second', 
                                verbose=verbose_main, ev_label=np.array((0,1,0)).reshape((1,3)))
            plot_decoded_latent(decoder, saveas=save_figure+'_decoded_constant',
                                verbose=verbose_main, ev_label=np.array((0,0,1)).reshape((1,3)))

        # ************************************************************
        # *************** pdf-Gradient Descent ***********************
        # ************************************************************        
        ''' 
        Perform the CVAE-gradient decent on pdf to find high probability data-points (hpdp). 
        "run_pdf_GD()" checks if saved hpdp (.npy files) exists or if the GD should be run from scratch.
        If you wish to load saved hpdp and not continue the GD, specify "zero_GD_iterations"=True in the 
        beginning of script. (files are loaded but 0 iterations is to be run..)

        The hyperparameters for the pdf-GD are specified in "hypes['pdf_GD'][...]"
        '''

        if run_GD:
            print(f'\n Running pdf_GD to get hpdp for {hypes["pdf_GD"]["m"]} iterations... \n')
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
        
        if manual_assesments:
            # Run visual evaluation where user specifies the number of clusters etc. based on
            # plot of Latent space after GD.
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
        # ************************ DBSCAN (Not really used..) ********
        # ************************************************************
        # Run DBSCAN on labeled data (high-occurance waveforms) to see if the obtained results are similar..
        # However not really feasible due to the high dimensionallity of waveforms. e.g. 141 for Zanos.

        if run_DBscan:
            saveas = 'figures/dbscan/' + unique_string_for_figs + 'DBSCAN'
            np_saveas = path_to_cytokine_candidate + 'DBSCAN'
            run_DBSCAN_evaluation(wf_ho, wf0, ts0, 
                                  ev_label_ho, hypes, 
                                  saveas=saveas, 
                                  np_saveas=np_saveas, 
                                  matlab_file=matlab_file)


for key in labeling_results.keys():
    # Print the number of waveforms with each label for all recordings.
    print(f'{key}: \n {labeling_results[key]}\n')

print('Finished successfully!')
