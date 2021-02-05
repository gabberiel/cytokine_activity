#### From original zanos-paper: ###
Download the following dependencies and add to the matlab path
https://www.mathworks.com/matlabcentral/fileexchange/52905-dbscan-clustering-algorithm
https://lvdmaaten.github.io/drtoolbox/
https://plexon.com/wp-content/uploads/2017/08/OmniPlex-and-MAP-Offline-SDK-Bundle_0.zip

nrdemo.m one cell at a time
There may be parameters you need to set in each cell


###### Gabriel Comments: #########
## The steps performed for thesis:
1. .plx-file coversion to .mat files as done by firts cell in "MAIN_preprocess.m". 
Note that file-name inputs must be specified by the characters "", not ''. 

2. Gain of 20 is applied to achieve the units of micro-Volt.

3. High-pass filter is applied (parameter set to 10). Removing small frequencies, assumed unrelated to neural-events. 

4. Downsampling, with Nd=5.

5. Adaptive threshold. Uses wavelet transform to enhance neural-events and cardiac-events seperately. 
A local in time noise-level is estimated using a sliding window and individual events are extracted from a 
threshold based on the local noise-level. Neural-events that co-occure with cardiac-events are disregarded. 
This results in an array of waveforms with corresponding timestamps.

6. Removeal of non-unique timestamps. 

7. Save the found waveforms and corresponding timestamps. 
