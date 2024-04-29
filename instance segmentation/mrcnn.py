################################ PRESETTINGS #################################
'''
This is a ready-to-use script for using Mask R-CNN on your images. It is based
on Facebook's implementation called detetctron2. First, you need to install it
according to: 
https://detectron2.readthedocs.io/en/latest/tutorials/install.html
Make sure that you have compatible versions of pytorch and CUDA.
I use pytorch 1.9.1 and CUDA 11.1. Detectron2 has special requirements for 
the format of your data set which you can find here:
https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html

In case you use Qupath for annotation and need to convert the Qupath format
to Detectron2 format you can see my script 'qupath2detectron.py'.
Another script can convert instance masks saved as images to compatible 
annotations format, see 'mask2detectron.py'
As written, for basic instance segmentation, you don't need the entries that
come after 'category_id'.

You also need to download some baseline models and weights to set up the model.
You can find them here:
https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md   
I use: R_101_FPN_3x, the corresponding files are: 
mask_rcnn_R_101_FPN_3x.yaml and R-101.pkl

Following basemodels files might also be needed: Base-RCNN-C4.yaml, 
Base-RCNN-DilatedC5.yaml, Base-RCNN-FPN.yaml, Base-RetinaNet.yaml

Some places are commented "TO CHANGE", to help the user spot necessary 
modifications
'''
##############################################################################

############################## IMPORT LIBRARIES ##############################
import numpy as np
import os, json, cv2, random
import pickle as pkl 
from os import listdir, path
from os.path import isfile, join
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
from matplotlib import pyplot as plt

setup_logger()
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
##############################################################################

################################# LOAD DATA ##################################
# set image and  annotation file paths - TO CHANGE
mask_path_train = "/projects/ag-bozek/pwojcik/dlbcl/data/masks/train/"
mask_path_test = "/projects/ag-bozek/pwojcik/dlbcl/data/masks/test/"

# define lists for images and annotation files
mask_files_train = [f for f in listdir(mask_path_train) if isfile(join(mask_path_train, f))]
mask_files_test = [f for f in listdir(mask_path_test) if isfile(join(mask_path_test, f))]

##############################################################################

############################### REGISTER DATA ################################
'''
When using a custom data set, you have to register your data first. For more
information see this link:
https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
'''

# this is a dummy function that is needed for the 'Dataset.register' function
def load_json(mask_path, mask_file):
    with open(os.path.join(mask_path, mask_file),'rb') as fh:
        tar = json.load(fh)
    return tar

# this loop goes through the annotation files of the train and test set,
# registers it, and sets the classes
# TO CHANGE: PATHS AND he_ NAMES, and class names negative/positive
for d in ["train", "test"]:#, "val"]:
    DatasetCatalog.register("he_" + d, lambda d=d: load_json(
                            "/projects/ag-bozek/pwojcik/dlbcl/data/"
                            + d, "target_dicts_" + d + ".json"))
    # In my case, I have 2 classes: 0 (negative) and 1 (positive). So for
    # every integer in your class column of the annotation file you need to 
    # assign a string name for this class
    MetadataCatalog.get("he_" + d).set(thing_classes=["negative", "positive"])
##############################################################################

################################ SET UP MODEL ################################
'''
The easiest way to get started is to load a copy of the default configuration.
This is done using the function get_cfg(). You can then change any parameter
you want. A list of the paramaters can be found here:
https://detectron2.readthedocs.io/en/latest/modules/config.html
'''   

cfg = get_cfg()
cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN" 
# path to the baseline model - TO CHANGE: PATH
model_path = "/projects/ag-bozek/pwojcik/dlbcl/mask_rcnn_R_101_FPN_3x.yaml"
cfg.merge_from_file(model_path)
# path to the model's weights - TO CHANGE: PATH
cfg.MODEL.WEIGHTS = "/projects/ag-bozek/pwojcik/dlbcl/R-101.pkl"
# specify the train data as you defined it in the registration process
cfg.DATASETS.TRAIN = ("he_train",) # TO CHANGE: NAME OF DATASET, cf above
cfg.DATALOADER.NUM_WORKERS = 2
# number of output channels for the feature pyramid network (FPN)
cfg.MODEL.FPN.OUT_CHANNELS = 512
# threshold for non maximum suppression (NMS) or region proposal network (RPN)
cfg.MODEL.RPN.NMS_THRESH = 0.9
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 512
cfg.SOLVER.WEIGHT_DECAY = 0.000001
# batch size
cfg.SOLVER.IMS_PER_BATCH = 2
# learning rate
cfg.SOLVER.BASE_LR = 0.0001  
cfg.SOLVER.MAX_ITER = 10000  
#cfg.SOLVER.GAMMA = 0.2
# The iteration number to decrease learning rate by GAMMA.
#cfg.SOLVER.STEPS = (1000,)

# Maximum number of detections to return per image during inference (100 is
# based on the limit established for the COCO dataset).
cfg.TEST.DETECTIONS_PER_IMAGE = 1000

# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
cfg.SEED = 42

# number of convolutions in the roi mask head
cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = 3  
cfg.MODEL.ROI_MASK_HEAD.CONV_DIM = 512

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
cfg.MODEL.RETINANET.NUM_CONVS = 4

# anchor size: I deal with small cells, so I need to use small anchors. By 
# default Detectron2 is set up to detect bigger objects (cars, persons etc.)
anchor_size = [4, 8, 16, 32, 64]
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [anchor_size] 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# number of classes according to your annotation files - IMPORTANT TO CHANGE
# is not the correct number of classes, throws weird errors
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
# create model with all the set parameters
model = build_model(cfg)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
##############################################################################

################################ TRAIN MODEL #################################
'''
I use the default trainer routine provided by Detectron2. However, you can
also write your own custom training loop. For more information, see:
https://detectron2.readthedocs.io/en/latest/tutorials/training.html
'''
#trainer = DefaultTrainer(cfg)
# if you want to continue from a pretrained model and not start from scratch
# set this parameter to TRUE
#trainer.resume_or_load(resume=False)
# start training
#trainer.train()
##############################################################################
 
################################# TEST MODEL #################################
# load weights from trained model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# set threshold for the ROIs for the testing. If you're having trouble 
# detecting any instances set a very low threshold (I had to use 0.001)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
predictor = DefaultPredictor(cfg)     
figures = []
# specify the name that you gave your test set in the registration process
he_metadata = MetadataCatalog.get("he_test") # TO CHANGE: NAME DATASET
'''
This loop is used to get the prediction results for your test set. The
variable output contains the predicted bounding boxes, classes, masks, and 
scores for all instances. The specific output format can be found here:
https://detectron2.readthedocs.io/tutorials/models.html#model-output-format    
'''
# load annotation files (I stored them as JSON files)

with open(os.path.join(mask_path_test, 'target_dicts_test.json'),'rb') as fh:
    tar_test = json.load(fh)
  
# sort lists to make sure that there's no mix up in the loading process
tar_test = sorted(tar_test, key=lambda d: d['file_name']) 

for d in random.sample(tar_test, 10):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=he_metadata, 
                   scale=1) 
    # the image can be scaled by `scale` when displayed, can be useful when objects 
    # are small and dense
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    '''
    you can also use: 
    vis = visualizer.draw_dataset_dict(d)
    and save the resulting image to display the ground truth annotations, to:
    1. make sure the annotations are correct
    2. compare the annotations and the predictions
    '''
    figures.append(out.get_image()) # POTENTIALLY SAVE IMAGES

    plt.imshow(out.get_image())
    plt.show()
    #with open('/projects/ag-bozek/pwojcik/dlbcl/test_output.json', 'wb') as outfile:
    #    json.dump(figures, outfile)
##############################################################################



    
