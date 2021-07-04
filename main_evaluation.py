'''
Evaluation of cytokine candidates to find "responders".
File is run after training in main. 

'''
import numpy as np
import matplotlib.pyplot as plt
import json
from os import sys
sys.path.insert(1,'src/')

from evaluation import find_reponders, eval_candidateCAP_on_multiple_recordings

responder_directory = '../numpy_files/cytokine_candidates/' # path to saved numpy result files.

save_main_candidates_CAPs = True
load_saved_candidate = True

verbose = True
training_start_title = 'zanos_0702' # 'HereWeGoAgain'   # Specify title of the "run" to evaluate.
training_start_title = 'chan_pre_proced_KI' # 'HereWeGoAgain'   # Specify title of the "run" to evaluate.

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
quit()
# ****************************************************************************
# ***** Use CAP as candidate over multiple different recordings  *************
# ****************************************************************************
# Can either load a saved CAP-candidates in specified path or use the found "main_candidates" from above.
if load_saved_candidate:
    recording_candidate = '_finalBaseline_10min_LPS_10min_KCl_10min_210617_103421'
    unique_start_string = training_start_title

    path_to_cytokine_candidate = '../numpy_files/responder_CAPs/'+unique_start_string
    main_candidates = np.load(path_to_cytokine_candidate + '.npy', allow_pickle=True)
else:
    candidate_CAP = main_candidates[0]   # Select one of the found CAP-candidates. 
    plt.plot(candidate_CAP)
    plt.show()

for candidate_CAP in main_candidates:
    saveas2 = 'figures/cap_eval/aaa' + recording_candidate
    eval_candidateCAP_on_multiple_recordings(candidate_CAP, hypes,
                                         file_name=specify_recording,
                                         saveas=saveas2, 
                                         verbose=verbose)

