##############################################################################
'''
This script creates json annotation files in the Detectron2 data format that is
required to run the Mask R-CNN from simple instance masks (unique id per object).
To better understand the conversion to the
Detectron2 format, which you can find here:
https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
'''
##############################################################################

############################## IMPORT LIBRARIES ##############################
import numpy as np
from tifffile import imread
from glob import glob
import os
from tqdm import tqdm
import json
from skimage.measure import find_contours, regionprops


def create_annotations_from_masks(mask_files, convert_fct=None):
    '''
    From a list of tif mask files, create a dictionary of annotations for each object present in the mask

    :args mask_files: list of path to mask files, they should be instance masks (i.e. the shapes are filled with unique ids).
                      the mask should be images, openable with tifffile.imread
    :args indices_files: indices of files to consider, corresponding the the list of files `mask_files`
    :args convert_fct: function that takes at least the name from a mask file and transforms it to the name of
                       the corresponding input image file

    :return: dict_annot, for each file, file path, and a list of contour coordinates and bounding boxes coordinates are
             saved in a dictionary
    '''
    # initialize dict
    dict_annot = {'file': [],  # image ID
                  'coordinates': [],  # annotation contour coordinates
                  'class': [],  # class of the annotation
                  'bbox': [],  # bounding box of the annotation
                  }

    for tile_ind in tqdm(range(len(mask_files)), desc=f'Reading the annotation files'.ljust(65)):
        mask_filename = mask_files[tile_ind]
        mask = imread(mask_filename)
        id_objects = np.unique(mask)[1:] # remove background == 0
        # loop to iterate through all objects of the current mask file
        mask_objects_props = regionprops(mask)
        dict_annot['file'].append(convert_fct(mask_files[tile_ind]))
        dict_annot['coordinates'].append([])
        dict_annot['bbox'].append([])
        for i_id_object, id_object in enumerate(id_objects):
            contour = find_contours(mask == id_object)
            # Need contours to be changed from ([x0, y0], ..., [xn, yn]) format to ([x0, y0, ..., xn, yn]).

            # index 0 because if several contours then the other ones are the smaller less relevant holes inside
            # subsampling_factor_contour because too many points in the contour when computed this way
            contour = contour[0]
            # cast the coordinates to int
            contour = contour.astype('int')
            # because of the int casting, it might contain doublons. However, we want to keep the order of the contour,
            # hence I used the indices, that I reorder with np.sort
            _, idx = np.unique(contour, axis=0, return_index=True)
            # ::-1 is to switch between x and y to get the order right for detection (not the same convention as skimage)
            # flatten is to get a ([x0, y0, ..., xn, yn]) vector as stated above
            contour = contour[np.sort(idx), ::-1].flatten()
            # cast in int because before np.int and json don't like numpy types
            dict_annot['coordinates'][-1].append([int(c) for c in contour])

            # The bbox format depends on what setting you use for Detectron2
            # (XYXY or XYWH). I use (XYXY) which means (top left x coordinate, top
            # left y coordinate, top right x coordinate, bottom left y coordinate)
            ymin, xmin, ymax, xmax = mask_objects_props[i_id_object].bbox
            dict_annot['bbox'][-1].append([xmin, ymin, xmax, ymax])

    return dict_annot


def temporary_dict_to_detectron(dict_annot, output_file_path, basedir_im_path='', height=None, width=None,
                                threshold_len_contours=8):
    '''
    Use the pre-defined dict dict_annot to generate the final Detectron2 format and save it to json format

    :args dict_annot: anotation dictionary as output by the function `create_annotations_from_masks`
    :args output_file_path: full path to save the output in a json file
    :args basedir_im_path: optional argument, path that will be prefix the image file names to get the full path
                           (so the detectron network can find them)
    :args height: height of images, optional. If not provided then the image would be opened to get the shape of it
    :args width: width of images, optional. If not provided then the image would be opened to get the shape of it
    :args threshold_len_contours: if the contour is smaller than this threshold, do not add to annotations

    :return: None
    '''
    # initialize target variable which will be in the final Detectron2 format
    target_dicts = []
    for i in range(len(dict_annot['file'])):
        # threshold to remove contours that are too small
        if len(dict_annot['coordinates'][i]) >= threshold_len_contours:
            temp_dict = {}
            filename = os.path.join(basedir_im_path, dict_annot['file'][i])
            temp_dict['file_name'] =  filename # IMPORTANT: needs to be full path!!!
            # iterate the variable for the image ID everytime it changes
            temp_dict['image_id'] = i

            # height and width of image
            if height is None or width is None:
                temp_dict['height'], temp_dict['width'] = imread(filename).shape[:2]
            else:
                temp_dict['height'], temp_dict['width'] = height, width
            # The 'annotation' entry must be a list of dicts. Regarding the bbox
            # mode: you need to match the function BoxMode from detectron2.structure.
            temp_dict['annotations'] = [{
                'bbox': dict_annot['bbox'][i][j],
                'bbox_mode': 0,# aka BoxMode.XYXY_ABS,
                'segmentation': [dict_annot['coordinates'][i][j]],
                'category_id': 0,
            } for j in range(len(dict_annot['bbox'][i]))]

            target_dicts.append(temp_dict)
        '''
        IMPORTANT: This dict contains all annotations of your data set in the correct format.
        Before you train the model you need to split this dict into train and test set.
        For more information, see my script of the Mask R-CNN.
        '''
        ##############################################################################

    print(f'[INFO] length of saved dict (= nb files): {len(target_dicts)}, '
          f'Inside the last file, there are {len(dict_annot["coordinates"])} annotations')
    print(f'[INFO] saving json file...')

    with open(output_file_path, 'w') as json_out:
        json.dump(target_dicts, json_out)

    print(f'[INFO] json file {output_file_path} saved')


if __name__ == '__main__':
    '''
    This script is working when given 2 lists of instance mask files, one for train, the other one for test.
    Train and test annotations will be saved separately in 2 different json files.
    Importantly, in the json files the full path to the corresponding input images should be noted, 
    for the detectron model to be able to find them.
    
    After creating these files,
    do not hesitate to test them with the following detectron functions (cf mask_rcnn script for more details):
    
    visualizer = Visualizer(im[:, :, ::-1], metadata=metadata_, scale=5)
    vis = visualizer.draw_dataset_dict(d)
    plt.imsave(os.path.join(save_path, vis.get_image()[:, :, ::-1])
    
    '''

    DEBUG = False # used to debug the script, only does the first 10 files
    if DEBUG:
        suffix = '_debug'
    else:
        suffix = ''
    ############################# ARGUMENTS ######################################
    basedir = '/home/france/Mounted_dir/'
    if not os.path.exists(basedir):
        raise ValueError(f'basedir {basedir} not found')

    # set the path where your Qupath annotations are stored
    mask_files = [np.sort(glob(os.path.join(basedir, "tissuenet_1.0/masks_train/train_mask_nucleus*.tif"))),
                  np.sort(glob(os.path.join(basedir, "tissuenet_1.0/masks_test/test_mask_nucleus*.tif")))]
    if DEBUG:
        mask_files = [m[:10] for m in mask_files]

    ind_tile_train = len(mask_files[0])
    ind_tile_test = len(mask_files[1])

    # you can give the height and width of your images if it's fixed,
    # if not given, then each file will be open to know its width and height
    # first index in the list is for train, the second one is for test
    height = width = [512, 256]

    output_file_path = os.path.join(basedir, "tissuenet_1.0/")
    # IMPORTANT: because we need the full image path in the json files:
    basedir_im_path = [os.path.join(basedir, "tissuenet_1.0/images_train/"),
                       os.path.join(basedir, "tissuenet_1.0/images_test/")]

    def convert_mask_path_to_image_path(path, *args, old_string='mask_nucleus', new_string='image', **kwargs):
        return os.path.basename(path).replace(old_string, new_string)

    ##############################################################################


    for i, name in enumerate(['train', 'test']):
        dict_annot = create_annotations_from_masks(mask_files[i],
                                                   convert_fct=convert_mask_path_to_image_path)

        temporary_dict_to_detectron(dict_annot,
                                    os.path.join(output_file_path, f'target_dicts_{name}{suffix}.json'),
                                    basedir_im_path=basedir_im_path[i],
                                    height=height[i], width=width[i])
    print('end of for loop')


