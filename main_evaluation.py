'''
Main script for Evaluation of cytokine candidates to find "responders".

File should be run after "main_train.py".
The evaluation assumes that the the resulting .npy files from main_train.py exists.

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
load_saved_candidate = False

verbose = True
training_start_title = 'zanos_0702' # 'HereWeGoAgain'   # Specify title of the "run" to evaluate.
training_start_title = 'all_channels_2_KI' # 'HereWeGoAgain'   # Specify title of the "run" to evaluate.

end_string = 'auto_assesment.npy'   # This is how the files were saved. 

# ************* Interesting recordings: *********************

'''
If "specify_recording" is an empty string:'', then all recordings used for the specifed run (training_start_title), 
will be taken into consideration. 
If, 'R10'-- the cytokine injections, and 'R12' -- the saline injection.
If a specific recording is of interest then this file name is given. 
E.g: 'R10_Exp2_7.13.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_08'
'''

# specify_recording = 'R10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05'
specify_recording = '' #_10min_KCl_10min_210617_122538A-015'

with open('hypes/' + training_start_title + '.json', 'r') as f:
    hypes = json.load(f)

# ****************************************************************************
# ** Find the responders from the files saved by "run_evaluation()" in main **
# ****************************************************************************
if not load_saved_candidate:
    # Runs evaluation. Otherwise, assumes the intension is to only load stored CAP-candodate and 
    # Evaluate over multiple recordings. (Later in script.)
    saveas = 'figures/Responders/' + training_start_title
    responders, main_candidates = find_reponders(responder_directory, hypes, 
                                                start_string=training_start_title, 
                                                end_string=end_string,
                                                specify_recordings=specify_recording, 
                                                saveas=saveas, 
                                                verbose=verbose, 
                                                return_main_candidates=True)
    if save_main_candidates_CAPs:
        saveas_responder_caps = '../numpy_files/responder_CAPs/' + training_start_title
        np.save(saveas_responder_caps + '.npy', main_candidates)

# ****************************************************************************
# ***** Use CAP as candidate over multiple different recordings  *************
# ****************************************************************************
'''
See how the event-rate is of a found CAP-candidate over multiple different recordings.
This to see if there is any similarities of CAPs in between different mice.
Can either load saved CAP-candidates in specified path or use the found "main_candidates" from above.
'''
if eval_on_multiple_recordings:
    if load_saved_candidate:
        # Do not use candidates from above, but a previously saved result. 
        recording_candidate = '_finalBaseline_10min_LPS_10min_KCl_10min_210617_103421'
        unique_start_string = training_start_title

        path_to_cytokine_candidate = '../numpy_files/responder_CAPs/'+unique_start_string
        main_candidates = np.load(path_to_cytokine_candidate + '.npy', allow_pickle=True)

    for candidate_CAP in main_candidates:
        saveas2 = 'figures/cap_eval/aaa' + recording_candidate
        eval_candidateCAP_on_multiple_recordings(candidate_CAP, hypes,
                                            file_name=specify_recording,
                                            saveas=saveas2, 
                                            verbose=verbose)

