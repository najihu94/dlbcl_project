
import deephistopath.wsi.slideCAFs as slideCAFs
import deephistopath.wsi.util as util


### STEP 1 Whole-slide imaging background

## ------ Scale down images
#
# imname = 'Staining_CD_sample_001'
# slideCAFs.open_slide(imname)
# slideCAFs.show_slide(1)
# slideCAFs.slide_info(1)
# slideCAFs.slide_stats()
# slideCAFs.training_slide_to_image(1)


## ------ Image saving, displaying, and conversions

slideCAFs.training_slide_to_image(1)
img_path = slideCAFs.get_training_image_path(1)
img = slideCAFs.open_image(img_path)
rgb = util.pil_to_np_rgb(img)
img.show()
img.save(img_path) # Will overwrite the image because it is the same path
#util.display_img(rgb, "RGB")

