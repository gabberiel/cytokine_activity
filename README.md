

# Clustering Neural Signals Related to Cytokine Activity

Neural recordings data as well as raw MATLAB code availible at: 
<public.feinsteininstitute.org/cbem/PNAS%20Submission>.

Find Main Article at <https://www.researchgate.net/publication/325035277_Identification_of_cytokine-specific_sensory_neural_signals_by_decoding_murine_vagus_nerve_activity>. 

## Steps in workflow:

Raw input-file.plx --[Matlab-Preprocess (1)]--> waveforms & timestamps .mat files --[Preprocess and Label (2)]-->  waveforms & timestamps .npy files --[CVAE-training (3)]--> keras.Model 
--[ pdf GD (4)]--> high prob. data-points --[Candidate evaluation looking at Event-rates (5)]--> Result that can be infered from. 


=======
* (1) : **MATLAB preprocessing of raw-recording.** Includes "adaptive threshold" and romoval of "bad"-datapoints influenced by cardiac events etc. In the MATLAB code we apply the steps:
    * Apply Gain
    * Butterworth high-pass filter
    * Downsample with Nd=5
    * Adaptive threshold
**OUT** : waveforms.mat, timestamps.mat

* (2) : **Python preprocessing and label of waveforms.** 
    * Remove observations occuring before 15min and after 90 minutes of recording. (From visual inspection of raw-file.) 
    * Standardisation of waveforms (favourable for Neural Network input.)
    * Event-rate calculation based on similarity measure.
    * Remove data-point which has a mean-event-rate less then specified threshold. (consider these as noise.)
    * Label waveform based on how the event rate changes at the injection-times, representing if they are likely or not to encode cytokine-information.
**OUT** : waveforms.npy, timestamps.npy, ev_labels.npy, (numpy_arrays)

* (3) : **Build and train CVAE model** to achieve approximate probability model -- variational inference approach. Latent representation of data etc. \
**OUT** : Conditional Variational autoencoder + weights (keras.Model)

* (4) **Preform pdf-gradient decent** on I(x)=-log(p|label="increased event-rate"), to find high probability data-points (hpdp) in the probability space . \
**OUT** : hpdp<-->"increase after first injection", hpdp-<-->"increase after second injection" (numpy arrays)

* (5) : **Candidate evaluation looking at Event-rates.** Consider the hpdp given that they are likely to encode cytokine information and consider one such CAP as the main-candidate. Look at Event-rate for the main-canditate and study the event-rate to see if there is a significant increase at time of injection for this candidate CAP. \ 


## Code structure

MAIN files as of 30 november. (Which are to be called directly.)
**main_cvae.py** : Can runs all parts of pipeline depending on settings. -- Mainly to be able to train model over night using 'caffinate'
**visualisations_main.py** : Assumes all files from training exists and load these to visualise results.
**CVAE_pipeline_main.ipnb** : Combination of the above to run one "step of workflow" at a time and visualise results in between.

Source Tree is as follows: 


OBS. main reads matlab files currently assumed to be in a folder "matlab_files" one step back in pwd.
