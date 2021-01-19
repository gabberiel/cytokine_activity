# Evaluation of cytokine candidates to find "responders".
# File is run after te training in main. 
# In main, the  

import numpy as np
from evaluation import find_reponders, eval_candidateCAP_on_multiple_recordings
import matplotlib.pyplot as plt

directory = '../numpy_files/cytokine_candidates' # path to saved numpy result files.

#matlab_directory = '../matlab_saline'
matlab_directory = '../matlab_files'

load_saved_candidate = False

start_string = "\\17_dec_30000_max200_ampthresh5_saline" # Saline injection recordings used as control
start_string = "\\21_dec_20k_ampthresh2"
start_string = "\\22_dec_30k_ampthresh2"
start_string = "\\7_jan_40k_100epochs"
#start_string = "\\12_jan_40k_100epochs"

start_string = '\\finalrun_first'
end_string = 'auto_assesment.npy' 
# ************* Interesting recordings: *********************
# An empty string:'', will take all recordings used for the specifed run (start_string), into consideration. 'R10' the cytokine injections and 'R12' the saline injection.
# If a specific recording is of interest then this file name is given. E.g: 'R10_Exp2_7.13.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_08'

#specify_recording = 'R10_Exp2_7.13.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_08' 
specify_recording = 'R10_6.30.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_05'
specify_recording = 'R10_Exp2_7.13.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_08'
#specify_recording = 'R10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05'
specify_recording = 'R12'
# ****************************************************************************
# ** Find the responders from the files saved by "run_evaluation()" in main **
# ****************************************************************************
saveas = 'figures/Responders/'+start_string
responders, main_candidates = find_reponders(directory, start_string=start_string, end_string=end_string,
                            specify_recordings=specify_recording, saveas=saveas, verbose=True, 
                            matlab_directory=matlab_directory, return_main_candidates=True)
exit()
# ****************************************************************************
# ***** Use CAP as candidate over multiple different recordings  *************
# ****************************************************************************
# Can either load a saved CAP-candidates in specified path or use the "main_candidates" above.
if load_saved_candidate:
    recording_candidate = 'R10_Exp2_7.15.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_10'
    unique_start_string = '10_dec_nstd_02and01_ds1_ampthresh' 
    path_to_cytokine_candidate = '../numpy_files/cytokine_candidates/'+unique_start_string + recording_candidate
    main_candidates = np.load(path_to_cytokine_candidate+'.npy')
else:
    candidate_CAP = main_candidates[0]
    plt.plot(candidate_CAP)
    plt.show()

#for candidate_CAP in main_candidates:
saveas2 = 'figures/cap_eval/'+'TNF_cand_second_18_jan'
eval_candidateCAP_on_multiple_recordings(candidate_CAP,matlab_directory,file_name='',similarity_measure='ssq',
                                        similarity_thresh=0.3,assumed_model_varaince=0.5,saveas=saveas2,verbose=False)

