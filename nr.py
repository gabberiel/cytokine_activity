# CLASS TO EVALUATE RESULTS FROM NEURAL RECORDINGS
import numpy as np
import matplotlib.pyplot as plt
from os import path, scandir
import preprocess_wf 
from main_functions import load_waveforms, load_timestamps, train_model, pdf_GD, run_evaluation, run_DBSCAN_evaluation
from wf_similarity_measures import wf_correlation, similarity_SSQ, label_from_corr
from event_rate_first import get_ev_labels, get_event_rates, evaluate_cytokine_candidates
from plot_functions_wf import *
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans, DBSCAN

# ************************************************************
# ***************** HYPERPARAMERS ****************************
# ************************************************************

similarity_measure='ssq' # From testing, correlation give worse results..
#similarity_thresh = 0.7 # For corrrelation
similarity_thresh = 0.1 # Gives either the minimum correlation using 'corr' or epsilon in gaussain annulus theorem for 'ssq'
assumed_model_varaince = 0.7 # The  model variance assumed in ssq-similarity measure. i.e variance in N(x_candidate,sigma^2*I)   

# Should maby fix such that threshold is TNF/IL-1beta specific since there is substantially more findings for TNF..
# Otherwise maby 0.2..?
n_std_threshold = 0.2 #(0.5)  # Number of standard deviation which the mean-even-rate need to increase for a candidate-CAP to be labeled as "likely to encode cytokine-info".
#ev_threshold = 0.02 # The minimum mean event rate for each observed CAP for it to be considered in the analysis. (Otherwise regarded as noise..)
#ev_threshold = 0.005 # Downsample=4 # Mabe good with 0.005 for aprox 37000 observations..

# downsample = 2 # Only uses every #th observation during the analysis for efficiency. 
desired_num_of_samples = 30000
max_amplitude = 200
ev_thresh_procentage = 0.005 # ie 1%

# Time interval of recording used for training:
start_time = 15; end_time = 90

# pdf-GD params: 
run_GD = False
m=0 # Number of steps in pdf-gradient decent
gamma=0.02 # learning_rate in GD.
eta=0.005 # Noise variable -- adds white noise with variance eta to datapoints during GD.

# VAE training params:
continue_train = False
nr_epochs = 40 # if all train data is used -- almost no loss-decrease after 100 batches..
batch_size = 128

view_vae_result = False # True => reqires user to give input if to continue the script to pdf-GD or not.. 
view_GD_result = False # This reqires user to give input if to continue the script to clustering or not.
plot_hpdp_assesments = False # Cluster and evaluate hpdp to find cytokine-candidate CAP
run_automised_assesment = False

# Evaluation Parameters using k*max(SD_min,SD) as threshold for "significant increase in ev." 
SD_min_eval = 0.15 # Min value of SD s.t. mice is not classified as responder for insignificant increase in EV.
k_SD_eval = 2 # k-param in k*max(SD_min,SD) 


# Use DBSCAN on labeled data independent of everything after labeling to see if we obtain similar results.
run_DBscan = True
# DBSCAN params
#db_eps = 6 # max_distance to be considered as neighbours 
#db_min_sample = 4 # Minimum members in neighbourhood to not be regarded as Noise.
db_eps = 0.2 # max_distance to be considered as neighbours 
db_min_sample = 4 # Minimum members in neighbourhood to not be regarded as Noise.


standardise_waveforms = True
verbose_main = 1
# ************************************************************
# ******************** Paths *********************************
# ************************************************************

# General: 

#unique_start_string = '14_dec_unique_threshs_saline' # on second to last file in this run..
unique_start_string = '15_dec_30000_max200__ampthresh5' # on second to last file in this run..


# LOOP Through all recordings in specified directory: 
directory = '../matlab_saline'
directory = '../matlab_files2'

class nr():
    '''
    Evaluation class 
    '''

    def __init__(self,waveforms,timestamps,):
        self.wf0 = waveforms
        self.ts0 = timestamps
    

    def set():
        pass