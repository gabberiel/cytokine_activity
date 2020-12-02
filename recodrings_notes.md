
## Notes regarding parameters etc for different recordings

#### General:
* The amplitude threshold decrease the number of waveforms from around 100000 to 30000. -- Will not be used..? 


## MATLAB Preprocessing steps:
* Apply Gain
* Butterworth high-pass filter
* Downsample with Nd=5
* Adaptive threshold


**R10_6.30.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_05.mat**
use_range = np.arange(5000,130000)


**R10_Exp2_71516_BALBC_TNF_05ug_IL1B_35ngperkg_10.mat** :
use_range = np.arange(5000,100000)