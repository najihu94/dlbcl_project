
import numpy as np
import h5py
import scipy.io

# import potrace
# import os

path2files_previous = '/home/lsancere/These/CMMC/Cheops1_Mount_Projects/Data_General/Predictions/IHC_HE_ScieboData1/HE/Inputs_sample_008_overlappingHor_tiles/HoVerNet_Outputs/'
path2files = '/home/lsancere/These/CMMC/Cheops1_Mount_Projects/Data_General/Predictions/IHC_HE_ScieboData1/HE/Inputs_sample_008_overlappingHor_tiles/HoVerNet_Outputs/_proc/'
filename = 'Staining_CD_sample_008-tile-r4-c25bis-x24576-y3072-w1024-h1024.mat'


# f = h5py.File(path2files + filename, 'r')
# data = f.keys() # to outputs the keys of the file
# data = f.get('data/variable1')
# data = np.array(data) # For converting to a NumPy array
# print(data)


mat = scipy.io.loadmat(path2files_previous + filename)
print(mat)

