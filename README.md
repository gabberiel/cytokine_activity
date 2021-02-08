

# Decoding Neural Signals Related to Cytokine Activity
### Author: *Gabriel Andersson*

This reposetory provides the python and matlab code for the master-thesis:



Work is inspired by: <https://www.researchgate.net/publication/325035277_Identification_of_cytokine-specific_sensory_neural_signals_by_decoding_murine_vagus_nerve_activity>. 

Which also provided the MATLAB code wich only has been slighlty altered for this work.

Neural recordings data as well as raw MATLAB code availible at: 
<https://public.feinsteininstitute.org/cbem/PNAS%20Submission>.


## Dependencies & Instructions
1. Install required packages
```
pip install -r requirements.txt
```
2. Download neural recordings (.plx files) at: 
<https://public.feinsteininstitute.org/cbem/PNAS%20Submission>.


## Steps in workflow:
### 1. Conversion of .plx files to .mat 
Convert .plx files to .mat files. See "README" in the MATLAB folder. 

### 2. MATLAB preprocessing
For MATLAB-preprocessing steps, see README file in the MATLAB-folder. 

The final results are CAP-waveforms saved as a (N x d) matrix, together with correspondong timestamps (N x 1).
### 3. Main analysis
The rest of the preprocessing and training is run by the script ```main_train.py``` .
Including: \
**Preprocessing and label of waveforms.** 
    * Remove observations occuring before 10- and after 90 minutes of recording. (From visual inspection of raw-file.) 
    * Standardisation of waveforms (favourable for Neural Network input.)
    * Event-rate calculation based on similarity measure.
    * Remove data-point which has a mean event-rate less then specified threshold. (consider these as noise.)
    * Label waveform based on how the event rate changes at the injection-times, representing if they are likely or not to encode cytokine-information. \
    **Returns** : waveforms.npy, timestamps.npy, ev_labels.npy, (numpy_arrays) 

**Build and train CVAE model** to achieve approximate probability model using variational inference. \
**Returns** : Conditional Variational autoencoder + weights (keras.Model)

**Preform pdf-gradient decent** on I(x) = -log p(x|label="increased event-rate"), to find high probability data-points (hpdp) in the probability space . \
**Returns** : hpdp<-->"increase after first injection", hpdp-<-->"increase after second injection" (numpy arrays)


## 4. Evaluation of Resultig CAP-Candidates.
Evaluation is run in ```main_evaluation.py```. \
Considers the hpdp and cluster these using k-means. The mean of each cluster is then considered as main-candidate CAPs for encoding cytokine. The event-rate for each of the main-canditate is considered to see if there is a significant increase after injection or not.
## 5. Visualisations and Model Assessments.
Run ```main_visualisations.py``` with parameters of your choosing.


