# General comment and observations regarding the Python-code/ analysis
---
## Code Structure

### Recordings --- File Structure
After preprocessing the continious data-files in MATLAB, the observed CAP-waveform instances (wf) and corresponding timestamps (ts) are saved as "wf+recording_title+.mat" and "ts+recording_title+.mat" respectively. These titles are specifided in MATLABs "Preprocessing_MAIN"-script.

### Hyperparameters
All parameters for each different run is specified in .json file. This is loaded in  main-files and passed as 'hype'-input to most function. The params are then accessable as:
hypes["key1"]["key2"].


---
## Preprocessing

---
## Tensorflow
* The string for saving tensorflow weights are not allowed to be too long. Raises utf-8 encoding errors for ~250 characters..


---
## Gradient Decent of Probability Distribution




