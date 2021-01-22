## Hyperparameters explained:

similarity_measure='ssq' # From testing, correlation give worse results..
similarity_thresh = 0.1 # Gives either the minimum correlation using 'corr' or "epsilon" in gaussain annulus theorem using 'ssq' (sum of squares)
assumed_model_varaince = 0.7 # The  model variance assumed in ssq-similarity measure. i.e variance in N(x_candidate,sigma^2*I) 
# Setting the assumed model variance to 0.5 as in CVAE yeild un unsufficient number of similar CAPs.. Thus set a bit higher for labeling..

# Should maby fix such that threshold is TNF/IL-1beta specific since there is substantially more CAPs labeles as increase after TNF injection.. 
# This. however, differs quite a lot from one recording to another..
# Otherwise maby 0.2..?

n_std_threshold = 0.2 # (0.5)  # Number of standard deviation which the mean-even-rate need to increase after injection for a candidate-CAP to be labeled as "likely to encode cytokine-info".

max_amplitude = 500 # Remove CAPs with max amplitude higher than the specified value. (Micro-Volts)
min_amplitude = 2 # Remove CAPs with max amplitude lower than the specified value. (Micro-Volts)
ev_thresh_fraction = 0.005 # Fraction of total event-rate used for thresholding. -- i.e 0.5%

desired_num_of_samples = None # 40000 # Subsample using every nth datapoint to get as close to given number as possible. 
# Time interval of recording used for training:
start_time = 15; end_time = 90

# pdf-GD params: 
run_GD = True
m = 0 #2000 # Number of steps in pdf-gradient decent
gamma = 0.02 # learning_rate in GD.
eta = 0.005 # Noise variable -- adds white noise with variance eta to datapoints during GD.

# CVAE training params:
continue_train = False
nr_epochs = 120 # if all train data is used -- almost no loss-decrease after 100 batches..
batch_size = 128

# ****** If using 2D-latent space dimension: *********
view_cvae_result = False # True => reqires user to give input if to continue the script to pdf-GD or not.. 
view_GD_result = False # This reqires user to give input if to continue the script to clustering or not.
plot_hpdp_assesments = False # Cluster and evaluate hpdp to find cytokine-candidate CAP manually inspecting plots.
# ***********************

run_automised_assesment = True # Cluster and evaluate hpdp by defined quantitative measure.

# Evaluation Parameters using k*max(SD_min,SD) as threshold for "significant increase in ev." 
SD_min_eval = 0.3 #0.3 Min value of SD s.t. mice is not classified as responder for insignificant increase in EV.
k_SD_eval = 2.5 #2.5 # k-param in k*max(SD_min,SD) 


# Use DBSCAN on labeled data independent of everything after labeling to see if we obtain similar results.
run_DBscan = False
# DBSCAN params
db_eps = 0.2 # max_distance to be considered as neighbours 
db_min_sample = 4 # Minimum members in neighbourhood to not be regarded as Noise.


---
## General Notes:
* The amplitude threshold decrease the number of waveforms from around 100000 to 30000. -- Will not be used. 
* Looking at a sample of the availible raw recordings, it seems reasonable to disregard the first 15 minutes of the recording since there is a lot more activity compared to the 15 minutes prior to the first injection.
* Also cut the CAPs that occur after 90 minutes of the recording.

## MATLAB Preprocessing steps:
* Apply Gain
* Butterworth high-pass filter
* Downsample with Nd=5
* Adaptive threshold
* Amplidute threshold. -- Assume that we need a sufficiently strong signal for it to rise above the noise level. Also resonalbe considering that we then standardise the waveforms, causing more similarity between low and high amplitude waveforms. 

 
## Notes regarding parameters etc for different recordings

**First final run**
desired_num_of_samples = None # Subsample using 
max_amplitude = 500 # Remove CAPs with max amplitude higher than the specified value. (Micro-Volts)
min_amplitude = 2 # Remove CAPs with max amplitude lower than the specified value. (Micro-Volts)
ev_thresh_fraction = 0.005 # Fraction of total event-rate used for thresholding. -- i.e 0.5%

# Time interval of recording used for training:
start_time = 15; end_time = 90

# pdf-GD params: 
run_GD = True
m= 2000 # Number of steps in pdf-gradient decent
gamma=0.02 # learning_rate in GD.
eta=0.005 # Noise variable -- adds white noise with variance eta to datapoints during GD.

# VAE training params:
continue_train = False
nr_epochs = 120 # if all train data is used -- almost no loss-decrease after 100 batches..
batch_size = 128


**First Testing example**

**R10_6.27.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_01**
Showed promising results with the parametre settings:
* similarity_measure='ssq'
* similarity_thresh = 0.5 
* assumed_model_varaince = 0.7    
* n_std_threshold = 0.5  
* ev_threshold = 0.01 

pdf-GD params: 
* m=500 # Number of steps in pdf-gradient decent
* gamma=0.02 # learning_rate in GD.
* eta=0.005 # Noise variable -- adds white noise with variance eta to datapoints during GD.

After run with downsample=2. The run is saved as test_4_dec...


**R10_6.27.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_01**
Using the same setting as above-- reasonalble number of waveforms in each "ev-cluster". Maby a bit too many in "increase after second".. The results using tha labeled waveforms without pdf_GD not really good.. After GD,  



**R10_6.30.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_05.mat**
use_range = np.arange(5000,130000)


**R10_Exp2_71516_BALBC_TNF_05ug_IL1B_35ngperkg_10.mat** :
use_range = np.arange(5000,100000)



# General: 
#recording = 'R10_6.27.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_01'
#recording = 'R10_6.27.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_01'
recording = 'R10_6.28.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_03'
#recording = 'R10_6.28.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_02' # Strange behaviour of EV during last period..

#recording = 'R10_6.28.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_02'
#recording = 'R10_6.28.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_03'
#recording = 'R10_6.29.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_04'
#recording = 'R10_6.29.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_04'
#recording = 'R10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05'



path_to_wf = '../matlab_files/wf'+recording+'.mat' 
path_to_ts = '../matlab_files/ts'+recording+'.mat'

unique_start_string = '7_dec_nstd_02and01_no_ds'
#unique_string_for_run = 'aa_using_new_var_period_3_test_4_dec'+recording
unique_string_for_run = unique_start_string+recording


# FOR SAVING FIGURES ****Not allowed to contain . or () signs****
#unique_string_for_figs = unique_start_string +'R10_6_27_16_BALBC_IL1B_35ngperkg_TNF_05ug_01' 
#unique_string_for_figs = unique_start_string +'R10_6_27_16_BALBC_TNF_0_5ug_IL1B_35ngperkg_01'
unique_string_for_figs = unique_start_string +'R10_6_28_16_BALBC_IL1B_35ngperkg_TNF_05ug_03'
#unique_string_for_figs = unique_start_string +'R10_6_28_16_BALBC_TNF_05ug_IL1B_35ngperkg_02'

#unique_string_for_figs = unique_start_string + 'R10_6_28_16_BALBC_IL1B_35ngperkg_TNF_05ug_02'
#unique_string_for_figs = unique_start_string +  'R10_6_28_16_BALBC_TNF_05ug_IL1B_35ngperkg_03'
#unique_string_for_figs = unique_start_string +  'R10_6_29_16_BALBC_IL1B_35ngperkg_TNF_05ug_04'
#unique_string_for_figs = unique_start_string +  'R10_6_29_16_BALBC_TNF_05ug_IL1B_35ngperkg_04'
#unique_string_for_figs = unique_start_string +  'R10_6_30_16_BALBC_IL1B_35ngperkg_TNF_05ug_05'


matlab_files = ['R10_6.27.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_01', 'R10_6.27.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_01', 
            'R10_6.28.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_03', 'R10_6.28.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_02',
            'R10_6.28.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_02', 'R10_6.28.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_03',
            'R10_6.29.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_04', 'R10_6.29.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_04',
            'R10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05']
matlab_files_amp_thres = ['R10_Exp2_7.13.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_08','R10_Exp2_7.13.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_09',
            'R10_Exp2_7.13.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_08', 'R10_Exp2_7.15.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_10']

figure_strings = ['R10_6_27_16_BALBC_IL1B_35ngperkg_TNF_05ug_01','R10_6_27_16_BALBC_TNF_0_5ug_IL1B_35ngperkg_01',
                'R10_6_28_16_BALBC_IL1B_35ngperkg_TNF_05ug_03', 'R10_6_28_16_BALBC_TNF_05ug_IL1B_35ngperkg_02', 
                'R10_6_28_16_BALBC_IL1B_35ngperkg_TNF_05ug_02', 'R10_6_28_16_BALBC_TNF_05ug_IL1B_35ngperkg_03',
                'R10_6_29_16_BALBC_IL1B_35ngperkg_TNF_05ug_04', 'R10_6_29_16_BALBC_TNF_05ug_IL1B_35ngperkg_04', 
                'R10_6_30_16_BALBC_IL1B_35ngperkg_TNF_05ug_05']
figure_strings_amp_thres = ['R10_Exp2_7_13_16_BALBC_IL1B_35ngperkg_TNF_0_5ug_08','R10_Exp2_7_13_16_BALBC_IL1B_35ngperkg_TNF_0_5ug_09',
            'R10_Exp2_7_13_16_BALBC_TNF_0_5ug_IL1B_35ngperkg_08', 'R10_Exp2_7_15_16_BALBC_IL1B_35ngperkg_TNF_0_5ug_10']

#file_names = {'matlab_files':matlab_files, 'figure_strings':figure_strings}
file_names = {'matlab_files':matlab_files_amp_thres, 'figure_strings':figure_strings_amp_thres}
