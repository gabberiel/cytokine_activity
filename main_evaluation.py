'''
Main script for visual evaluation of the cytokine "responders".

File should be run after "main_train.py".
The evaluation assumes that the the resulting .npy files in "cytokine_candidates" from main_train.py exists.


'''
import numpy as np
import matplotlib.pyplot as plt
import json
from os import sys
sys.path.insert(1,'src/')

from evaluation import find_reponders, eval_candidateCAP_on_multiple_recordings

responder_directory = '../numpy_files/cytokine_candidates/' # path to saved numpy result files.

save_main_candidates_CAPs = True 

eval_on_multiple_recordings = False
load_saved_candidate = False # Set To False unless you wish to look for similarity of saved Candidate-CAPs over multiple recordings.

verbose = False
# hypes_file_name = 'one_chan_KI_10min' # 'HereWeGoAgain'   # Specify title of the "run" to evaluate.

# hypes_file_name = 'all_channels_2_KI' # 'HereWeGoAgain'   # Specify title of the "run" to evaluate.
# hypes_file_name = 'one_chan_KI_30min' # 'HereWeGoAgain'   # Specify title of the "run" to evaluate.
hypes_file_name = 'all_chan_KI_10min' # 'HereWeGoAgain'   # Specify title of the "run" to evaluate.

# end_string = 'auto_assesment.npy'   # This is how the files were saved. 

# ************* Interesting recordings: *********************

'''
If you would like to consider a recording that is saved as 
    "ts2020New_Saline_recordings_wooho" & "wf2020New_Saline_recordings_wooho"
then you should set: 
    rec_start_string= "2020New_Saline_recordings_wooho"

If "rec_start_string" is an empty string:'', then all recordings used for the specifed run (hypes_file_name), 
will be taken into consideration. 

For Example, "R10" => the Zanos-cytokine-injections, and "R12" => the Zanos saline injection.

OBS! You need to set the directory in the hype-file at: hypes["dir_and_files"]["matlab_dir"]
'''

# rec_start_string = 'R10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05'
rec_start_string = '_final2Baseline' #_10min_KCl_10min_210617_122538A-015'
# rec_start_string = '210715_30min_baseline_30min_ZYA_injection_30min_KCl_A-000_second' #_10min_KCl_10min_210617_122538A-015'

with open('hypes/' + hypes_file_name + '.json', 'r') as f:
    hypes = json.load(f)

# ****************************************************************************
# ** Find the responders from the files saved by "run_evaluation()" in main **
# ****************************************************************************

if not load_saved_candidate:
    '''
    Runs evaluation. Otherwise, assumes the intension is to only load stored CAP-candodate and 
    Evaluate their event-rate in multiple recordings. (Later in script.)
    '''
    saveas = 'figures/Responders/' + hypes_file_name
    responders, main_candidates = find_reponders(responder_directory, hypes, 
                                                start_string=hypes_file_name, 
                                                end_string='auto_assesment.npy',
                                                specify_recordings=rec_start_string, 
                                                saveas=saveas, 
                                                verbose=verbose, 
                                                return_main_candidates=True)
    if save_main_candidates_CAPs:
        saveas_responder_caps = '../numpy_files/responder_CAPs/' + hypes_file_name
        np.save(saveas_responder_caps + '.npy', main_candidates)

# ****************************************************************************
# ***** Use CAP-candidate to get EV in multiple different recordings  ********
# ****************************************************************************
'''
Get the event-rate of a CAP-candidate in multiple different recordings.
This to see if there is any similarities of CAPs in between different mice.
Can either load saved CAP-candidates in specified path or use the found "main_candidates" from above.

Loads Candidate-CAPs that has been saved as "responders" specified by "hypes_file_name".
Calculates event-rate for candidate-CAPs in all recordings compatible with "rec_start_string".
i.e. All recordings that are saved as "ts"+ "rec_start_string" + ...
'''
hypes_file_name = "all_channels_2_KI"
consider_modified_candidate = True

if eval_on_multiple_recordings:
    if load_saved_candidate:
        # Do not use candidates from above, but a previously saved result. 

        path_to_cytokine_candidate = '../numpy_files/responder_CAPs/' + hypes_file_name
        main_candidates = np.load(path_to_cytokine_candidate + '.npy', allow_pickle=True)

    

    if consider_modified_candidate:
        '''
        After loading the main-candidates, they could be combined / manipulated here to look for 
        CAP-shapes with consistent event-rates over multiple recordings..
        
        i.e. use only singel Candidate as defined below for all specified recordings:
        '''

        candidate_CAP = ( main_candidates[2,:] + main_candidates[0,:] ) / 2
        
        saveas2 = 'figures/cap_eval/modified_' + str(hypes['evaluation']['similarity_threshold']).replace('.','_')
        print(f'\n{"*"*40} \nStarting with a new Candidate-CAP!\n{"*"*40}\n')
        eval_candidateCAP_on_multiple_recordings(candidate_CAP, hypes,
                                            file_name=rec_start_string,
                                            saveas=saveas2, 
                                            verbose=verbose)
    else:
        candidate_count = 0
        for candidate_CAP in main_candidates:
            # Loop through all loaded candidate-CAPs (responders) for all specified recordings.
            candidate_count += 1
            saveas2 = 'figures/cap_eval/' + str(candidate_count) + '_'
            print(f'\n{"*"*40} \nStarting with a new Candidate-CAP!\n{"*"*40}\n')
            eval_candidateCAP_on_multiple_recordings(candidate_CAP, hypes,
                                                file_name=rec_start_string,
                                                saveas=saveas2, 
                                                verbose=verbose)

