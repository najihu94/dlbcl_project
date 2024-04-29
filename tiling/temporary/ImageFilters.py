import deephistopath.wsi.slideCAFs as slideCAFs
import deephistopath.wsi.util as util
import deephistopath.wsi.filter as filter
from deephistopath.wsi import tiles
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('ktAgg')
import numpy as np

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 773505280



# FILTER

slideCAFs.training_slide_to_image(1)
img_path = slideCAFs.get_training_image_path(1)
img = slideCAFs.open_image(img_path)
rgb = util.pil_to_np_rgb(img)
util.display_img(rgb)

# H&E channels extracted
# hed = filter.filter_rgb_to_hed(rgb)
# hema = filter.filter_hed_to_hematoxylin(hed)
# norm_hema = filter.filter_histogram_equalization(hema)
# util.display_img(norm_hema)
# eosin = filter.filter_hed_to_eosin(hed)
# norm_eosin = filter.filter_histogram_equalization(eosin)
# util.display_img(norm_eosin)


# GRAYSCALE
grayscale = filter.filter_rgb_to_grayscale(rgb)
# util.display_img(grayscale)


# Color filters
# not_green = filter.filter_green_channel(rgb)
# util.display_img(not_green, "Green Channel Filter")
# not_grays = filter.filter_grays(rgb)
# util.display_img(not_grays, "Grey Filter")



# Normalizations
#inverse = filter.filter_complement(grayscale)  # INVERSE
normalization1 = filter.filter_contrast_stretch(rgb)
normalization2 = filter.filter_histogram_equalization(rgb)
normalization3 = filter.filter_adaptive_equalization(rgb)
#util.display_img(normalization)


# Segmentations
#Kmeans Alone
# kmeans_seg = filter.filter_kmeans_segmentation(rgb, n_segments=3000)
# util.display_img(kmeans_seg, "K-Means Segmentation", bg=True)
# otsu_mask = util.mask_rgb(rgb, filter.filter_otsu_threshold(filter.filter_complement(filter.filter_rgb_to_grayscale(rgb)), output_type="bool"))
# util.display_img(otsu_mask, "Image after Otsu Mask", bg=True)
# kmeans_seg_otsu = filter.filter_kmeans_segmentation(otsu_mask, n_segments=3000)
# util.display_img(kmeans_seg_otsu, "K-Means Segmentation after Otsu Mask", bg=True)
#RAG + Kmeans
# rag_thresh = filter.filter_rag_threshold(rgb)
# util.display_img(rag_thresh, "RAG Threshold (9)", bg=True)
# rag_thresh = filter.filter_rag_threshold(rgb, threshold=1)
# util.display_img(rag_thresh, "RAG Threshold (1)", bg=True)
# rag_thresh = filter.filter_rag_threshold(rgb, threshold=20)
# util.display_img(rag_thresh, "RAG Threshold (20)", bg=True)

# Inversion
inverse = filter.filter_complement(grayscale)  # INVERSE


# Thresholding
# threshold = filter.filter_threshold(inverse, threshold=50) #Could be Otsu Threshold, hysteresis ,and classical threshold)
# hysteresis = filter.filter_hysteresis_threshold(complement) #Could be Otsu Threshold, hysteresis ,and classical threshold)
# otsu = filter.filter_otsu_threshold(complement) #Could be Otsu Threshold, hysteresis ,and classical threshold)


# Histogram grayscale
# histogram, bin_edges = np.histogram(normalization1, bins=256, range=(0,255))
# plt.plot(bin_edges[0:-1], histogram)  # <- or here
# plt.show()


# RGB to HSV + infos
tiles.display_image_with_rgb_and_hsv_histograms(rgb)
tiles.display_image_with_rgb_and_hsv_histograms(normalization1)
tiles.display_image_with_rgb_and_hsv_histograms(normalization2)
tiles.display_image_with_rgb_and_hsv_histograms(normalization3)


