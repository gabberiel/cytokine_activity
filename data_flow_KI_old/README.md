## Steps in Workflow of loading .rhs file into MATLAB

### **MAIN STEPS**
  1. Convert .rhs files into .kwd format using the code in ``intan2kwik`` repo.

        - See example files in ``intan2kwik`` directory.

  2. Read the .kwd files in MATLAB using ``h5read(filename, path_to_data_within_.kwd_file)``.

        - 

