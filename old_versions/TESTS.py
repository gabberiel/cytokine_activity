# TESTS

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
from plot_functions_wf import plot_decoded_latent,plot_amplitude_hist
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

desired_num_of_samples = 20000 #40000 # Subsample using 
max_amplitude = 500 #500 # Remove CAPs with max amplitude higher than the specified value. (Micro-Volts)
min_amplitude = 2 #2 # Remove CAPs with max amplitude lower than the specified value. (Micro-Volts)
ev_thresh_fraction = 0.005 # Fraction of total event-rate used for thresholding. -- i.e 0.5%

# Time interval of recording used for training:
start_time = 15; end_time = 90

# pdf-GD params: 
run_GD = True
m= 0 #2000 # Number of steps in pdf-gradient decent
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

run_automised_assesment = False # Cluster and evaluate hpdp by defined quantitative measure.

# Evaluation Parameters using k*max(SD_min,SD) as threshold for "significant increase in ev." 
SD_min_eval = 0.3 #0.3 Min value of SD s.t. mice is not classified as responder for insignificant increase in EV.
k_SD_eval = 2.5 #2.5 # k-param in k*max(SD_min,SD) 


# Use DBSCAN on labeled data independent of everything after labeling to see if we obtain similar results.
run_DBscan = False
# DBSCAN params
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

#unique_start_string = '22_dec_30k_ampthresh2' # Fine for results? 
#unique_start_string = '21_dec_30k_paramsearch' # Fine for results? 

#unique_start_string = '7_jan_40k_100epochs'

unique_start_string = 'finalrun_first'
unique_start_string = 'jan15_20k_amp_1_1000'

# ***** Specify path to directory of recordings ******* 
directory = '../matlab_files'
# *****************************************************

# ***** Specify the starting scaracters in filename of recordings to analyse *****
# if "ts" is not specified, then all files will be run twise since we have one file for timestamps and one for CAP-waveform with identical names, exept the starting ts/wf.
start_string = '\\tsR10_Exp3' #.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05' # Since each recording has two files in directory (waveforms and timestamps)-- this is solution to only get each recording once.
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
        plot_amplitude_hist(waveforms)


        print(f'Shape before amplitude threshold : {waveforms.shape}')
        waveforms,timestamps = preprocess_wf.apply_amplitude_thresh(waveforms,timestamps,maxamp_threshold=max_amplitude, minamp_threshold=min_amplitude) # Remove "extreme-amplitude" CAPs-- otherwise risk that pdf-GD diverges..
        print()
        print(f'Shape after amplitude threshold : {waveforms.shape}')
        print()
        waveforms,timestamps = preprocess_wf.get_desired_shape(waveforms,timestamps,start_time=start_time,end_time=end_time,dim_of_wf=141,desired_num_of_samples=desired_num_of_samples)
        print(f'Shape after shape-preprocessing : {waveforms.shape}')