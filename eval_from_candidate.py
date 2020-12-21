
import numpy as np
from evaluation import find_reponders, eval_candidateCAP_on_multiple_recordings
import matplotlib.pyplot as plt



directory = '../numpy_files/cytokine_candidates'
start_string = "\\17_dec_30000_max200_ampthresh5_saline" # Saline injection recordings used as control
#end_string = 'new_test.npy'

end_string = 'auto_assesment.npy'
saveas = 'figures/Responders/'+start_string
#matlab_directory = '../matlab_saline'
matlab_directory = '../matlab_files'
saveas2 = 'figures/cap_eval/'+'AAA_TEST_cand2'

load_saved_candidate = False
start_string = "\\21_dec_20k_ampthresh2"

#Interesting recordings: 
#specify_recording = 'R10_Exp2_7.13.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_08' # 'R10'  / 'R12' for all case/control
specify_recording = 'R10_6.30.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_05' # 'R10'  / 'R12' for all case/control

# ****************************************************************************
# ** Find the responders from the files saved by "run_evaluation()" in main **
# ****************************************************************************
responders, main_candidates = find_reponders(directory, start_string=start_string, end_string=end_string,
                            specify_recordings=specify_recording, saveas=saveas, verbose=True, 
                            matlab_directory=matlab_directory, return_main_candidates=True)

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
eval_candidateCAP_on_multiple_recordings(candidate_CAP,matlab_directory,similarity_measure='ssq',
                                        similarity_thresh=0.4,assumed_model_varaince=0.5,saveas=saveas2)

