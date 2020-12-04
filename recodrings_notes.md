

## General Notes:
* The amplitude threshold decrease the number of waveforms from around 100000 to 30000. -- Will not be used. 
* Looking at a sample of the availible raw recordings, it seems reasonable to disregard the first 15 minutes of the recording since there is a lot more activity compared to the 15 minutes prior to the first injection.
* Also cut the CAPs that occur after 90 minutes of the recording.

## MATLAB Preprocessing steps:
* Apply Gain
* Butterworth high-pass filter
* Downsample with Nd=5
* Adaptive threshold

 
## Notes regarding parameters etc for different recordings
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