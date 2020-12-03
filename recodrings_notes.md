

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


**R10_6.30.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_05.mat**
use_range = np.arange(5000,130000)


**R10_Exp2_71516_BALBC_TNF_05ug_IL1B_35ngperkg_10.mat** :
use_range = np.arange(5000,100000)