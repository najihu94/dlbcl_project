# DLBCL Project

## Instance segmentation

 In this folder, there are scripts:
 - [to train a Mask RCNN based on the facebook detectron2](instance%20segmentation/mrcnn.py),
 - to apply a trained version of a mask RCNN to any given image [(inference mode)](instance%20segmentation/inference_mask_rcnn.py),  
 - to convert annotaions (from [QuPath](instance%20segmentation/qpath2detectron.py) or [instance masks](instance%20segmentation/mask2detectron.py)) to the correct detectron format.
