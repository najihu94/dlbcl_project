
from tasks import slideCAFs
from tasks import filterCAFs
from tasks import tilesCAFs
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np

"""
TODO
Explain code and add comment_

The files in the folder should be numbered in ascending order without missing integers 

"""


AllImages_ToProcess = False

FilterAll_TileOnlyOne = True #If AllImages_ToProcess = False, this boolean is not used

# no of wsi
Image_to_process = 3 #If AllImages_ToProcess == True, the image to process will be the tiled image.
# #If AllImages_ToProcess == False, the image to process will be the filtered and tiled image.

if not os.path.exists(slideCAFs.FILTER_DIR):
    os.makedirs(slideCAFs.FILTER_DIR)
if not os.path.exists(slideCAFs.TILE_SUMMARY_DIR):
    os.makedirs(slideCAFs.TILE_SUMMARY_DIR)


if AllImages_ToProcess:
    slideCAFs.multiprocess_training_slides_to_images()
    filterCAFs.CombineFilt_allimages_singleprocess(save=False, display=False, image_num_list=None)  #change name of function if release / final edition
    if FilterAll_TileOnlyOne:
            tilesCAFs.multiprocess_CombineFilt_images_to_tiles(display=True, save_summary=False, save_data=False,
                                                            save_top_tiles=True,
                                                            image_num_list=[Image_to_process])
    else:
            tilesCAFs.multiprocess_CombineFilt_images_to_tiles(display=True, save_summary=False, save_data=False,
                                                            save_top_tiles=True,
                                                            image_num_list=None)
else:
    slideCAFs.training_slide_to_image(Image_to_process)
    filterCAFs.CombineFilt1_singleimage(Image_to_process, DisplaySteps=True, DisplayResults=True, PrintResultPath=True)
    # multiprocess doesn't work on windows
    # tilesCAFs.multiprocess_ComineFilt_images_to_tiles(display=False, save_summary=False, save_data=False,
    #                                                     save_top_tiles=True,
    #                                                     image_num_list=[Image_to_process])
    tilesCAFs.singleprocess_filtered_images_to_tiles(display=True, save_summary=False, save_data=False, save_top_tiles=True,
                                           html=False, image_num_list=[Image_to_process], infos=False)

#%%    
# remove part of string 
str_path = "C:/Users/Hussi/Desktop/tiles_png/001_ver/"  
str_files = [f for f in listdir(str_path) if isfile(join(str_path, f))]
hor = False
ver = True

if hor or ver:
    none = False
else:
    none=True
    
for a in tqdm(range(0,len(str_files))):
    if hor:
        string = "-x"
        # replace "-x" with "bis-x"
        new_string = str_files[a].replace(string, "bis-x")
        os.rename(str_path + str_files[a], str_path + new_string)
    elif ver:
        string = "-c"
        # replace "-c" with "bis-c"
        new_string = str_files[a].replace(string, "bis-c")  
        os.rename(str_path + str_files[a], str_path + new_string) 
    if not none:   
        # remove everything after row and column characters, except ".png"
        final_string, sep, tail = new_string.partition("-x")
        os.rename(str_path + new_string, str_path + final_string + ".png")
    else:
        final_string, sep, tail = str_files[a].partition("-x")
        os.rename(str_path + str_files[a], str_path + final_string + ".png")

#%%
idx_path = "C:/Users/Hussi/Desktop/tiles_png/002/"  
idx_files = [f for f in listdir(idx_path) if isfile(join(idx_path, f))]
idx_r = []
idx_c = []

for b in tqdm(range(0,len(idx_files))):
    head, sep, tail = idx_files[b].partition("-c")
    head2, sep2, tail2 = tail.partition(".png")
    head2 = head2.replace("bis", "")
    idx_c.append(int(head2))
min(idx_c)    
max(idx_c)
#%%

# import cv2
# from tqdm import tqdm

# pat = "C:/Users/Hussi/Desktop/tiles_png/001/"
# files = [f for f in listdir(pat) if isfile(join(pat, f))]
# imli = []
# for i in tqdm(range(0,len(files))):    
#     ima = cv2.imread(pat + files[i])
#     imli.append(ima)    

# ima = cv2.imread(pat + '/wsi_001-tile-r100-c101bis.png')    
