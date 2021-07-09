# Comments from original zanos-paper: 
Download the following dependencies and add to the matlab path
<https://www.mathworks.com/matlabcentral/fileexchange/52905-dbscan-clustering-algorithm>

<https://lvdmaaten.github.io/drtoolbox/>

<https://plexon.com/wp-content/uploads/2017/08/OmniPlex-and-MAP-Offline-SDK-Bundle_0.zip>

Run nrdemo.m one cell at a time.

There may be parameters you need to set in each cell

# The steps performed in the Raw file preprocessing:
Main script: ```MAIN_preprocess.m```.
1. a) Zanos recordings: .plx-file coversion to .mat files is done by one 
      of the firts cells in "MAIN_preprocess.m". Note that file-name inputs
      must be specified by the characters "", not ''.
    b) KI-recordings: .rhs-file conversionto .mat is done using 
       "convert_rhs_to_mat.m" as in first cell in "MAIN_preprocess.m"

2. Gain of 20 is applied to achieve the units of micro-Volt for Zanos recordings, not KI.

3. High-pass filter is applied (parameter set to 10). Removing small frequencies, assumed unrelated to neural-events. 

4. Downsampling, ( Zanos recodings where preprocessed with Nd=5.)

5. Adaptive threshold. Uses wavelet transform to enhance neural-events (1ms scale) and cardiac-events (5ms scale) seperately. 
A local in time noise-level is estimated using a sliding window (so-cfar filter) and individual events are extracted from a 
threshold based on the local noise-level. Neural-events that co-occure with cardiac-events are disregarded. 
This results in an array of waveforms with corresponding timestamps.

6. Removeal of non-unique timestamps. 

7. Save the found waveforms and corresponding timestamps. 


## More Info
The script ``main_visualisations.m``, is containing cells for plotting and evaluating the different parts of the preprocessing. High-pass filter, cardiac/neurally-enhanced signal, thresholds etc.

The ``readme_visualisations.pdf`` file is a short powerpoint presenting visualisations of the different steps in the preprocessing together with some results from the python code. 

Main steps in **Adaptive threshold:**
  
  * Define width of wavelets to emphasize cardiac / neural events.
  * Define "half-time" or "time-of-peak" of the assumed duration of individual CAP. ( 3.5/2 ms )
  * Perform cwt on raw signal y ; for both neural and cardiac scale.
  * Perform so-cfar filter on neurally enhanced signal (``sig`` in code) and cardiac enhanced signal (``cardiac`` in code), this finds all indicies for both negative and possitive threshold-crossings. (These "crossings" are then combined.)
  * Remove indicies from the neural-threshold-crossing which overlap with the cardiac-events.
  **This has now given us a list of all indicies, each corresponding to a neural event.**
  * Loop through the list of neral-event-indicies and get a local-waveform of the assumed duration that is "max-peak-centered" for each.

    * All this is done in ``NeuralRecording.adaptive_threshold()`` which returns ``"waveforms"`` from the neural-cwt-transformed signal. 
    * However, the timeseries-attribute that is stored in ``NeuralRecording.waveforms`` is containing waveforms from the raw-signal!! This is done a few lines after the call to ``adaptive_threshold()``.

