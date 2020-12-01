

# Clustering Neural Signals Related to Cytokine Activity

Neural recordings data as well as raw MATLAB code availible at: 
<public.feinsteininstitute.org/cbem/PNAS%20Submission>.

Find Main Article at <https://www.researchgate.net/publication/325035277_Identification_of_cytokine-specific_sensory_neural_signals_by_decoding_murine_vagus_nerve_activity>. 

## Steps in workflow:

Raw input-file.plx --[Preprocess (1)]--> waveforms & timestamps .mat files --[Preprocess (2)]-->  waveforms & timestamps .npy files --[VAE-training (3)]--> keras.Model 
--[ pdf GD (4)]--> high prob. data-points --[Clustering (5)]--> labeled data --[Event-rate (6)]--> Result that can be infered. 


* (1) : MATLAB preprocessing of raw recording. Includes "adaptive threshold" and romoval of "bad"-datapoints based on cardiac events etc.  See steps in MAIN_preprocess.mat

* (2) : Python preprocessing to have optimal form of input to VAE. qqq: Standardisation/frequency-domain ...?

* (3) : Build VAE model to achieve approximate probability model -- variational inference approach. Latent representa
tion of data etc. 

* (4) : Preform gradient decent on $$ \pi $$
* (5) : 

* (6) :

$$ \pi $$
=======
* (1) : **MATLAB preprocessing** of raw recording. Includes "adaptive threshold" and romoval of "bad"-datapoints based on cardiac events etc. \
**OUT** : waveforms.mat, timestamps.mat

* (2) : **Python preprocessing** to have optimal form of input to VAE. qqq: Standardisation/frequency-domain ...? \
**OUT** : waveforms_input.npy (numpy_array)

* (3) : **Build and train VAE model** to achieve approximate probability model -- variational inference approach. Latent representation of data etc. \
**OUT** : Variational autoencoder + weights (keras.Model)

* (4) : Preform gradient decent on $$ I(x)=-log(p) $$ to find high probability data-points (hpdp). \
**OUT** : 

* (5) : qqq: **Clustering** procedure on either hpdp in ots original dimention or latent space using either k-means/ DBSCAN etc. Very important to sort out noise here since the achieved labels together with the "timestamps" determines the event-rate deterministically. \
**OUT** :

* (6) : **Event Rate** calculation -- the occurence rate in CAPs/sec during time of recording. \
**OUT** : 

## Code structure

MAIN files as of 30 november. (Which are to be called directly.)
**main_cvae.py** : Can runs all parts of pipeline depending on settings. -- Mainly to be able to train model over night using 'caffinate'
**visualisations_main.py** : Assumes all files from training exists and load these to visualise results.
**CVAE_pipeline_main.ipnb** : Combination of the above to run one "step of workflow" at a time and visualise results in between.

Source Tree is as follows: 


OBS. main reads matlab files currently assumed to be in a folder "matlab_files" one step back in pwd.
