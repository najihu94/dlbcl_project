# Whole_SlideIPP #

Pre-processing of ndpi images using IBM article "Whole-slide image preprocessing in Python". <br/>
IPP stands for Image Pre Processing

Tiling done using deephistopath library.

Merging is home made.

## Create conda env associated with the project

Before using the code, you need to create a conda environment and to activate it when running the code. **To know if the instructions are updated**, check the status after it.

**To create the environment:**
``` bash
conda deactivate
conda create -n Whole-SlideIPP python=3.8
conda activate Whole-SlideIPP
conda install h5py
```
Status: Updated :white_check_mark:

**To activate the environment before running the code:**
```bash
conda activate Whole-SlideIPP
```

_In case of issue when creating or using the env_ 

Download the Whole-SlideIPP.yml file and open a terminal in the folder where the file is, and type:
```bash
conda env create -f Whole-SlideIPP.yml
```
Status: Updated :white_check_mark:

