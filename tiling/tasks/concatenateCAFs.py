
import os
import numpy as np
import itertools as IT
from PIL import Image
from PIL.Image import Image

"""
Concatenate prediction tiles together to recreate the whole original wsi or part of the wsi.

The corresponding location of the concatenated tiles (with number of row and column as indexes) should be written in the name of the files
The user can choose the width and height (in number of tiles) of the recreated area thanks to the parameters:   
    - rownumbermin, rownumbermax colnumbermin, colnumbermax 
    
This code is to concatenate not overlapping tiles only. To merge concatenated tiles, use mergeCAFs_onlyHor.py
"""


# Parameters

path = '/home/lsancere/These/CMMC/Cheops1_Mount_Projects/Hover_Net_Complete/tensorflow-inference/data_hovernet_inference/tile_mode/Outputs_sample_008_all_tiles'# without last /
nameoftiles = 'sample_008-tile'
rownumbermin, rownumbermax = 5, 94   #can create a code to find it -> not really useful!
colnumbermin, colnumbermax = 14, 95
tileHeight, tileWidth = 1024, 1024 #as to be written
nrows = int((rownumbermax - rownumbermin)+1)
ncols = int((colnumbermax - colnumbermin)+1)



# Functions

def concatenateTiles(nameoftiles, path, nrows, ncols, rownumbermin, colnumbermin, tileHeight, tileWidth):
    display = np.empty((tileHeight*nrows, tileWidth*ncols, 3), dtype=np.uint8)
    numbertiles = int(nrows * ncols)
    progress = 0
    for i, j in IT.product(range(nrows), range(ncols)):
        rownumber = i + rownumbermin  #because the first raw number is not necessarily 0 (part of the wsi)
        colnumber = j + colnumbermin  #because the first column number is not necessarily 0 (part of the wsi)
        progress += 1
        print('Concatenation progress : %d / %d' % (progress, numbertiles))
        path_to_tile = path + '/' + nameoftiles + '-r' + str(rownumber) + '-c' + str(colnumber) + '.png'
        if os.path.exists(path_to_tile):
            image = Image.open(path_to_tile)
            x, y = i * tileHeight, j * tileWidth
            display[x:x + tileHeight, y:y + tileWidth, :] = image
        else:
            blackimage = np.full((tileHeight, tileWidth, 3), 0, dtype=np.uint8)        # If there is no tile in this location, create a full black tile
            x, y = i * tileHeight, j * tileWidth
            display[x:x + tileHeight, y:y + tileWidth, :] = blackimage

    return display



# Use

finalresult = concatenateTiles(nameoftiles, path, nrows, ncols, rownumbermin, colnumbermin, tileHeight, tileWidth)
image = Image.fromarray(finalresult)

Output_path = path + '/ConcResult'
if not os.path.exists(Output_path):
    os.mkdir(Output_path)

image.save(Output_path + '/' + 'Concatenation_' + nameoftiles + '_row_' + str(rownumbermin) + 'to' + str(rownumbermax) + '_col_' + str(colnumbermin) + 'to' + str(colnumbermax) + '.png')
print('Concatenated Image Saved')