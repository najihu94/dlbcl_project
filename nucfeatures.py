"""
This script requires the segmented nuclei contours of the n best image tiles 
(according to the attention score of CLAM). There are two different folders:
one for class 0 (non-recurred patients) and class 1 (recurred patients). The 
nuclei were segmented using HoVer-Net (https://github.com/vqdang/hover_net) 
and the resulting nuclei contours were stored as json files. Using these
contours, this script calculates morphological features of each nucleus.
"""

# import libraries
import numpy as np
from tqdm import tqdm
import os
from os.path import isfile, join, listdir
import cv2
import pickle as pkl
import json
import copy
from skimage import measure

# two iterations to go through the folders of class 0 and class 1
for c in range(0,2):
    json_path = "/path/to/contours/folder/" + str(c) + "/"
    json_files = sorted([f for f in listdir(json_path) if isfile(join(json_path, f))])   
    # initialize an empty dictionary to store data
    data = {
        "id": [],
        "centroid": [],
        "contour": [],
        "type": [],
        "instmap": [],
    }   
    # set image tile size
    instmap_height = 256  # image height
    instmap_width  = 256  # image width    
    savejson = "/path/in/which/to/store/new/files/" + str(c) + "/"
    # cutoff nuclei at the margin of the images have to be removed for feature
    # calculation. This is done in the following loop.
    for file_path in tqdm(json_files):
        file = json_path + file_path
        # load current file
        with open(file) as f:
            data = json.load(f)   
        temp = copy.deepcopy(data)  
        # iterate through all nuclei within a file
        for nuc in tqdm(temp["nuc"].keys()):
            # exclude background and cutoff nuclei at the borders for feature
            # calculation
            if any(0 in sublist for sublist in temp["nuc"][nuc]["contour"]) or any(255 in sublist for sublist in temp['nuc'][nuc]['contour']):
                del data["nuc"][nuc]
        # save new json file with removed cutoff nuclei
        with open(savejson + file_path, 'w') as f:
            json.dump(data, f)
    
    # Now the new files are loaded. The previous saving and this loading step 
    # are not mandatory, it was implemented to verify the correctness of
    # results.
    json_path = "/path/in/which/to/store/new/files/" + str(c) + "/"
    json_files = sorted([f for f in listdir(json_path) if isfile(join(json_path, f))])   
    # iterate through all files         
    for file_path in tqdm(json_files):
        file = json_path + file_path
        # load current file
        with open(file) as f:
            data = json.load(f)   
        # each nucleus should be assigned a unique grayscale value, so that 
        # each nucleus counts as a separate instance    
        grayscale_value = 1  
        # initialize instance map with array of zeros (feature calculation 
        # requires an instancemap)
        instmap = np.zeros((instmap_height, instmap_width, 3), dtype=np.uint8)
        # save the properties of all nuclei of a file in one column of the data
        # dictionary
        for nucleus_id, nucleus_prop in data["nuc"].items():
            data["id"].append(nucleus_id)
            centroid = nucleus_prop["centroid"]
            data["centroid"].append(centroid)
            contour = np.array(nucleus_prop["contour"], dtype=np.int32).reshape((-1, 1, 2))
            data["contour"].append(contour)
            nucleus_type = nucleus_prop["type"]
            data["type"].append(nucleus_type)            
            # draw contours on the instance maps with the current grayscale
            # value for each instance using the contour coordinates
            cv2.drawContours(instmap, [contour], -1, grayscale_value, -1)
            # increment the grayscale value for the next nucleus
            grayscale_value += 1 
        # save all instance maps in one dictionary
        data["instmap"].append(instmap[:,:,0])
    
    # initialize an array containing the features you want to calculate
    properties = [
                    "area",
                    "major_axis_length",
                    "minor_axis_length",
                    "eccentricity",
                    "extent",
                    "orientation",
                    "perimeter",
                    "solidity",
                    ]   
    # append the features as new columns to the existing data dictionary
    for prop in properties:
        data[prop] = []
    # iterate through all instance maps   
    for ins in tqdm(data["instmap"]):    
        # calculate features for the instance map
        features = measure.regionprops_table(ins, properties=properties)
        for prop in properties:
            data[prop].extend(features[prop])
    # initialize list of contours to calculate the feature "roughness" (which is
    # not included in the "measure.regionprops_table" function)
    contours_list = data["contour"]
    rough = []
    # iterate through all contours
    for contour in contours_list:
        # calculate perimeter of nuclei
        perimeter = cv2.arcLength(contour, True)
        # approximate a circle around the contour
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        # calculate the perimeter of the approximated circle
        perimeter_approx = cv2.arcLength(approx, True)
        # roughness can be calculated as the absolute difference between the 
        # perimeter of the approximated circle around the contour and the
        # perimeter of the contour
        rough.append(abs(perimeter_approx - perimeter))
    data['roughness'] = rough
      
    # save the resulting dictionary with all calculated features (the 
    # dictionaries for class 0 and 1 are saved in the same folder but with a
    # filename extention indicating the class)
    with open(os.path.join("/path/to/save/dict/", "class" + str(c) + "_features.pkl"), 'wb') as fh:
            pkl.dump(data, fh) 
