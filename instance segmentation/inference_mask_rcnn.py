############################## IMPORT LIBRARIES ##############################
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
from skimage.transform import resize
from glob import glob

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model


##############################################################################
def rescale_im(im, outputtype='uint16', percentiles=(0.5, 99.5)):
    im_min, im_max = np.percentile(im, percentiles)
    im = (im.astype('float') - im_min) / (im_max - im_min)
    im = np.clip(im, a_min=0, a_max=1)
    if outputtype == 'uint16' or outputtype == 'int16' or outputtype == np.uint16 or outputtype == np.int16:
        return (im * (2 ** 16 - 1)).astype(outputtype)
    elif outputtype == 'uint8' or outputtype == 'int8' or outputtype == np.uint8 or outputtype == np.int8:
        return (im * (2 ** 8 - 1)).astype(outputtype)
    else:
        print('[WARNING] output image in float')
    return im


if __name__ == '__main__':
    ############################# ARGUMENTS ######################################

    # get images
    basedir = "/projects/ag-bozek/france/hyperion_data/CLL/2022-03-11/Exported_OME-TIFF_32-bit/2022_03_11_C21-44687_Panel_Titration_I"
    list_files = []
    for roi_dir in os.listdir(basedir):
        list_files.extend(glob(os.path.join(basedir, roi_dir, '*_rescaled_crop.tif')))
    print(len(list_files))

    output_im_path = "/projects/ag-bozek/france/output_mask_rcnn/CLL"
    if not os.path.exists(output_im_path):
        os.mkdir(output_im_path)

    # the 2 following are only for visualization (by using the file names):
    json_path_train = f"/projects/ag-bozek/france/tissuenet_1.0/target_dicts_train.json"
    json_path_test = f"/projects/ag-bozek/france/tissuenet_1.0/target_dicts_test.json"

    model_path = "/home/frose1/mask_rcnn_nets/mask_rcnn_R_101_FPN_3x.yaml"
    model_weights_path = "/home/frose1/mask_rcnn_nets/R-101.pkl"
    output_model_path = "/home/frose1/mask_rcnn_nets/tissuenet"

    # path to json files used for training:
    target_dicts_path = "/projects/ag-bozek/france/tissuenet_1.0/"

    dataset_name_prefix = 'tissuenet_'
    ##############################################################################


    ################################ SET UP MODEL ################################
    '''
    Use the same config as your trained model
    '''

    cfg = get_cfg()
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

    # path to the baseline model
    cfg.merge_from_file(model_path)
    # path to the model's weights
    cfg.MODEL.WEIGHTS = model_weights_path
    # IMPORTANT: you don't need to register any dataset
    cfg.INPUT.FORMAT = "BGR"
    cfg.MODEL.PIXEL_MEAN = [0., 56.6298319, 47.62962295] # from train dataset FR, in order B, G, R

    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT.MIN_SIZE_TEST = 1024 # changed for my images FR
    # Maximum size of the side of the image during testing
    cfg.INPUT.MAX_SIZE_TEST = 1024# changed for my images FR

    # threshold for non maximum suppression (NMS) or region proposal network (RPN)
    cfg.MODEL.RPN.NMS_THRESH = 0.95

    # Maximum number of detections to return per image during inference (100 is
    # based on the limit established for the COCO dataset).
    cfg.TEST.DETECTIONS_PER_IMAGE = 10000

    # Set seed to negative to fully randomize everything.
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    # Disabling all parallelism further increases reproducibility.
    cfg.SEED = 42

    # anchor size: I deal with small cells, so I need to use small anchors. By
    # default Detectron2 is set up to detect bigger objects (cars, persons etc.)
    anchor_size = [2, 4, 8, 16]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [anchor_size]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
    # number of classes according to your annotation files
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # create model with all the set parameters
    cfg.OUTPUT_DIR = output_model_path
    model = build_model(cfg)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    ##############################################################################

    ################################# TEST MODEL #################################
    # load weights from trained model
    cfg.MODEL.WEIGHTS = os.path.join(output_model_path, f"model_final.pth")
    # set threshold for the ROIs for the testing. If you're having trouble
    # detecting any instances set a very low threshold (I had to use 0.001)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01 # original 0.1 FR
    predictor = DefaultPredictor(cfg)

    for im_file in list_files:
        im = cv2.imread(im_file)
        im2 = rescale_im(resize(im, (im.shape[0]*4, im.shape[1]*4, im.shape[2])),
                         outputtype='uint8', percentiles=(0, 100))
        ## Prediction
        outputs = predictor(im2)
        v = Visualizer(im2[:, :, ::-1], scale=5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imsave(os.path.join(output_im_path, f"train_prediction_{os.path.basename(im_file).rstrip('.tif')}.svg"),
                   out.get_image()[:, :, ::-1])
    ##############################################################################



