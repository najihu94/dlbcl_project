"""
This script converts json annotation files from QuPath to the Detectron2 data
format that is required to run the Mask R-CNN. QuPath is an open source 
software for bioimage analysis. It is especially useful for digital pathology 
applications because it offers a powerful set of tools for working with whole
slide images and tissue micro arrays. One of the main features include powerful
and easy to use annotation tools. My data set contains HE and IHC images and I
use QuPath to annotate cell nuclei. After being done with an annotation task 
you can export and save the annotations as json files by writing a Groovy
script. Groovy is a Java based programming language that is also included in
QuPath. That is, you can write scripts for doing all sorts of things within the
QuPath environment. You can find two ready-to-use groovy scripts in our Github:
one for exporting annotation files and one for exporting binary masks of the
annotations. I also attached one sample QuPath annotation file, so you can see 
what the QuPath format looks like and better understand the conversion to the
Detectron2 format, which you can find here:

https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
"""

import os.path
from pathlib import Path
from typing import Dict, Any, List

import argparse
import cv2
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import json
from detectron2.structures import BoxMode


def create_dictionary(annotations_dir: Path) -> Dict[str, Any]:
    """
    @param annotations_dir directory where QuPath annotations are stored

    I convert the data in two steps. In the first step, I extract the relevant
    information from all QuPath files and store them in one dictionary. The QuPath
    files are also dicts that have many convoluted entries that are irrelevant for
    Detectron2. So this first step can be interpreted as a basic cleaning routine
    to simplify and filter the dicts.
    """

    dict_annot = {'file': [],  # image ID
                  'coordinates': [],  # annotation contour coordinates
                  'class': [],  # class of the annotation
                  'bbox': [],  # bounding box of the annotation
                  }

    # store filenames of all files that are in that path in a list
    annot_files = [f for f in listdir(annotations_dir) if isfile(join(annotations_dir, f))]

    # loop to extract the relevant information from the QuPath files and store them in 'annotations_dir'
    for tile_ind in tqdm(range(0, len(annot_files))):

        with open(os.path.join(annotations_dir, annot_files[tile_ind])) as qupath_file:
            annotations = json.load(qupath_file)
        for annot_ind in range(0, len(annotations)):
            # the coordinates must be polygons, otherwise the format is incorrect
            if annotations[annot_ind]['geometry']['type'] != 'Polygon':
                continue

            dict_annot['file'].append(annot_files[tile_ind])

            dict_annot['coordinates'].append(annotations[annot_ind]['geometry']['coordinates'][0])

            dict_annot['class'].append(annotations[annot_ind]['properties']['classification']['name'])

            xy = annotations[annot_ind]['geometry']['coordinates'][0]
            x, y = [], []
            for cont_ind in range(0, len(xy)):
                x.append(xy[cont_ind][0])
                y.append(xy[cont_ind][1])
            """
            The bbox format depends on what setting you use for Detectron2 (XYXY or XYWH). 
            I use (XYXY) which means (top left x coordinate, top left y coordinate, top right x coordinate, 
            bottom left y coordinate)
            """
            dict_annot['bbox'].append([min(x), min(y), max(x), max(y)])

        """
        Remove 'json' ending from file strings to match input name with target names.
        This is a specific issue that occurred to me, so this might not be necessary for you.
        """
        for file_name in range(0, len(dict_annot['file'])):
            dict_annot['file'][file_name] = dict_annot['file'][file_name].replace('.json', '')

    return dict_annot


def flatten_coordinates(annotations_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    In the QuPath file, the contour coordinates are stored in ([x0, y0], ..., [xn, yn]) format.
    For Detectron2, we need to flatten the structure to ([x0, y0, ..., xn, yn]).
    """

    cont = []
    for cords in range(0, len(annotations_dict['coordinates'])):
        cont.append([item for sublist in annotations_dict['coordinates'][cords] for item in sublist])
    annotations_dict['coordinates'] = cont
    return annotations_dict


def generate_detectron2_dict(annotations_dict: Dict[str, Any],
                             annotations_dir: Path,
                             images_dir: Path) -> List[Dict[str, Any]]:
    """
    We use the pre-defined `annotations_dict` to generate the final Detectron2 format.

    This dict contains all annotations of your data set in the correct format.
    Before you train the model you need to split this dict into train and test set.
    For more information, see my script of the Mask R-CNN.
    """

    # initialize target variable which will be in the final Detectron2 format
    target_dicts_list = []
    # counter variable to set the image ID
    image_id = 0
    for i in tqdm(range(0, len(annotations_dict['file']))):

        # threshold to remove contours that are too small
        if len(annotations_dict['coordinates'][i]) < 8:
            continue

        detectron2_dict = {'file_name': str(os.path.join(annotations_dir, annotations_dict['file'][i]))}
        # iterate the variable for the image ID everytime it changes
        if not (i == 0 or annotations_dict['file'][i] == annotations_dict['file'][i - 1]):
            image_id += 1
        detectron2_dict['image_id'] = image_id
        # height and width of image
        detectron2_dict['height'], detectron2_dict['width'] = \
            cv2.imread(str(os.path.join(images_dir, annotations_dict['file'][i]))).shape[:2]
        """
        The `annotation` entry must be a list of dicts. Regarding the bbox mode: you need to import the function BoxMode 
        from Detectron2.structure which means that Detectron2 must already be installed. Installation instructions can 
        be found in my Mask R-CNN script.
        """
        detectron2_dict['annotations'] = [{
            'bbox': annotations_dict['bbox'][i],
            'bbox_mode': BoxMode.XYXY_ABS,
            'segmentation': [annotations_dict['coordinates'][i]],
            'category_id': 0,   # preset the class label to default value 0
        }]
        # In my case, I have two classes (negative and positive). So in case, the class is 'positive',
        # I change the label to 1
        if annotations_dict['class'][i] == 'Positive':
            detectron2_dict['annotations'][0]['category_id'] = 1

        target_dicts_list.append(detectron2_dict)

    return target_dicts_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=Path, help='Directory of the QuPath annotations folder', required=True)
    parser.add_argument('--images', type=Path, help='Directory of the images folder', required=True)
    parser.add_argument('--output_file', type=Path, help='Path to store aggregated Detectron2 json', required=True)

    args = parser.parse_args()

    d = flatten_coordinates(create_dictionary(args.annotations))

    with open(args.output_file, 'w') as detectron_json:
        json.dump(generate_detectron2_dict(d, args.annotations,  args.images), detectron_json)


