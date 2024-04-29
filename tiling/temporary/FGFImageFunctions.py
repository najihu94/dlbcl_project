import deephistopath.wsi.slideCAFs as slideCAFs
import deephistopath.wsi.util as util
import deephistopath.wsi.filter as filter
from deephistopath.wsi import tiles
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import multiprocessing
from deephistopath.wsi.util import Time

PIL.Image.MAX_IMAGE_PIXELS = 773505280


def CombineFilt1_singleimage(im_num, DisplaySteps=True, DisplayResults=True, PrintResultPath=True):
    """
    Explain
     """
    slideCAFs.training_slide_to_image(im_num)
    img_path = slideCAFs.get_training_image_path(im_num)
    img = slideCAFs.open_image(img_path)
    width, height = img.size
    print(width, height)
    rgb = util.pil_to_np_rgb(img)

    if DisplayResults:
        util.display_img(rgb)

    grayscale = filter.filter_rgb_to_grayscale(rgb)
    if DisplaySteps:
        util.display_img(grayscale)

    # equalize
    normalization = filter.filter_contrast_stretch(grayscale, low=50, high=200)
    if DisplaySteps:
        util.display_img(normalization)

    # threshold
    threshold = filter.filter_threshold(normalization,
                                        threshold=50)  # Could be Otsu Threshold, hysteresis ,and classical threshold)
    hysteresis = filter.filter_hysteresis_threshold(
        normalization)  # Could be Otsu Threshold, hysteresis ,and classical threshold)
    otsu = filter.filter_otsu_threshold(normalization)  # Could be Otsu Threshold, hysteresis ,and classical threshold)
    if DisplaySteps:
        util.display_img(otsu, "otsu")
    # util.display_img(threshold, "threshold")
    # util.display_img(hysteresis, "hysteresis")

    otsu_inv = filter.filter_complement(otsu)
    if DisplaySteps:
        util.display_img(otsu, "otsuinverse")

    # Dilatation
    bin_dilation = filter.filter_binary_dilation(otsu_inv, disk_size=2)
    if DisplaySteps:
        util.display_img(bin_dilation, "bin_dilation")

    # Fill Holes
    fill_holes = filter.filter_binary_fill_holes(bin_dilation)
    # util.display_img(fill_holes, "Fill Holes", bg=True)
    if DisplaySteps:
        util.display_img(fill_holes, "fill_holes")

    # remove small objects
    remove_small_100 = filter.filter_remove_small_objects(fill_holes, min_size=((width * height) / 4))
    if DisplayResults:
        util.display_img(remove_small_100, "fill_holes")

    result_path = slideCAFs.get_filter_image_result(1)
    result = util.pil_to_np_rgb(remove_small_100)

    result = PIL.Image.fromarray(result)
    result.save(result_path)

    img = slideCAFs.open_image(result_path)
    rgbimg = PIL.Image.new("RGB", img.size)
    rgbimg.paste(img)
    rgbimg.save(result_path)

    if PrintResultPath:
        print("Result path %s" % result_path)

    return




# multiprocess_apply_filters_to_images()
