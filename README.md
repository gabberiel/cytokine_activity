

# Decoding Neural Signals Related to Cytokine Activity
### Author: *Gabriel Andersson*

This reposetory provides the python and matlab code for the master-thesis:



Work is inspired by: <https://www.researchgate.net/publication/325035277_Identification_of_cytokine-specific_sensory_neural_signals_by_decoding_murine_vagus_nerve_activity>. 
They also provided the MATLAB code wich only has been slighlty altered for this work.
Neural recordings data as well as raw MATLAB code availible at: 
<https://public.feinsteininstitute.org/cbem/PNAS%20Submission>.

Find Main Article at <https://www.researchgate.net/publication/325035277_Identification_of_cytokine-specific_sensory_neural_signals_by_decoding_murine_vagus_nerve_activity>. 


## Dependencies & Instructions
1. Install required packages
```
pip install -r requirements.txt
```
2. Download neural recordings (.plx files) at: 
<https://public.feinsteininstitute.org/cbem/PNAS%20Submission>.


## Steps in workflow:
### 1. Conversion of .plx files to .mat 
Convert .plx files to .mat files, using the script ``` convert_plx.mat ```. 

### 2. MATLAB preprocessing
The MATLAB preprocessing can be done for specific file using qqq: .
Alternatively, convert all file in specified  directory using qqq: . 

This saves the CAP-waveforms data into matlab (N x d) matrix together with timestamps (N x 1).
### 3. Main analysis
The rest of the preprocessing and training is run by the script ```main_train.py``` .
Including,
* Similarity Measure and Data ,
* Build and Train Probabilistiv Model (Conditional VAE),
* Gradient Decent of Conditional Distribution,
* Clustering. 

## 4. Evaluation of Resultig CAP-Candidates.
    ```main_evaluation.py```

## 5. Visualisations and Model Assessments.
    ```main_visualisations.py```

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

* (4) **Preform pdf-gradient decent** on I(x) = -log p(x|label="increased event-rate"), to find high probability data-points (hpdp) in the probability space . \
**OUT** : hpdp<-->"increase after first injection", hpdp-<-->"increase after second injection" (numpy arrays)

* (5) : **Candidate evaluation looking at Event-rates.** Consider the hpdp and cluster these using k-means. The median of each cluster is then considered as main-candidate CAPs for encoding cytokine. Looking at the Event-rate for each of the main-canditate to see if there is a significant increase at time of injection or not. \ 


## Code structure

MAIN files as of 30 november. (Which are to be called directly.)
**main_cvae.py** : Can runs all parts of pipeline depending on settings. -- Mainly to be able to train model over night using 'caffinate'
**visualisations_main.py** : Assumes all files from training exists and load these to visualise results.
**CVAE_pipeline_main.ipnb** : Combination of the above to run one "step of workflow" at a time and visualise results in between.

Source Tree is as follows: 


OBS. main reads matlab files currently assumed to be in a folder "matlab_files" one step back in pwd.
