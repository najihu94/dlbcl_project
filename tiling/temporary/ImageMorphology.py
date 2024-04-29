import deephistopath.wsi.slideCAFs as slideCAFs
import deephistopath.wsi.util as util
import deephistopath.wsi.filter as filter



slideCAFs.training_slide_to_image(1)
img_path = slideCAFs.get_training_image_path(1)
img = slideCAFs.open_image(img_path)
rgb = util.pil_to_np_rgb(img)
util.display_img(rgb)


# Erosion

# no_grays = filter.filter_grays(rgb, output_type="bool")
# util.display_img(no_grays, "No Grays", bg=True)
# bin_erosion_5 = filter.filter_binary_erosion(no_grays, disk_size=5)
# util.display_img(bin_erosion_5, "Binary Erosion (5)", bg=True)
# bin_erosion_20 = filter.filter_binary_erosion(no_grays, disk_size=20)
# util.display_img(bin_erosion_20, "Binary Erosion (20)", bg=True)

# Dilatation

# no_grays = filter.filter_grays(rgb, output_type="bool")
# util.display_img(no_grays, "No Grays", bg=True)
# bin_dilation_5 = filter.filter_binary_dilation(no_grays, disk_size=5)
# util.display_img(bin_dilation_5, "Binary Dilation (5)", bg=True)
# bin_dilation_20 = filter.filter_binary_dilation(no_grays, disk_size=20)
# util.display_img(bin_dilation_20, "Binary Dilation (20)", bg=True)

# Closing (skip opening)

# no_grays = filter.filter_grays(rgb, output_type="bool")
# util.display_img(no_grays, "No Grays", bg=True)
# bin_closing_5 = filter.filter_binary_closing(no_grays, disk_size=5)
# util.display_img(bin_closing_5, "Binary Closing (5)", bg=True)
# bin_closing_20 = filter.filter_binary_closing(no_grays, disk_size=20)
# util.display_img(bin_closing_20, "Binary Closing (20)", bg=True)

# Remove small objects

# no_grays = filter.filter_grays(rgb, output_type="bool")
# util.display_img(no_grays, "No Grays", bg=True)
# remove_small_100 = filter.filter_remove_small_objects(no_grays, min_size=100)
# util.display_img(remove_small_100, "Remove Small Objects (100)", bg=True)
# remove_small_10000 = filter.filter_remove_small_objects(no_grays, min_size=10000)
# util.display_img(remove_small_10000, "Remove Small Objects (10000)", bg=True)

# Fill holes and remove holes

# no_grays = filter.filter_grays(rgb, output_type="bool")
# fill_holes = filter.filter_binary_fill_holes(no_grays)
# util.display_img(fill_holes, "Fill Holes", bg=True)
# remove_holes_100 = filter.filter_remove_small_holes(no_grays, area_threshold=100, output_type="bool")
# util.display_img(fill_holes ^ remove_holes_100, "Differences between Fill Holes and Remove Small Holes (100)", bg=True)
# remove_holes_10000 = filter.filter_remove_small_holes(no_grays, area_threshold=10000, output_type="bool")
# util.display_img(fill_holes ^ remove_holes_10000, "Differences between Fill Holes and Remove Small Holes (10000)", bg=True)

# Entropy

gray = filter.filter_rgb_to_grayscale(rgb)
util.display_img(gray, "Grayscale")
entropy = filter.filter_entropy(gray, output_type="bool")
util.display_img(entropy, "Entropy")
util.display_img(util.mask_rgb(rgb, entropy), "Original with Entropy Mask")
util.display_img(util.mask_rgb(rgb, ~entropy), "Original with Inverse of Entropy Mask")

# Canny edge detection
gray = filter.filter_rgb_to_grayscale(rgb)
canny = filter.filter_canny(gray, output_type="bool")
util.display_img(canny, "Canny", bg=True)
rgb_crop = rgb[300:900, 300:900]
canny_crop = canny[300:900, 300:900]
util.display_img(rgb_crop, "Original", size=24, bg=True)
util.display_img(util.mask_rgb(rgb_crop, ~canny_crop), "Original with ~Canny Mask", size=24, bg=True)

