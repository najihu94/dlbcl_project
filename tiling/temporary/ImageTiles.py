import deephistopath.wsi.slideCAFs as slideCAFs
import deephistopath.wsi.util as util
import deephistopath.wsi.filter as filter
import deephistopath.wsi.tilesCAFs as tilesCAFs


tilesCAFs.summary_and_tiles(1, display=True, save_summary=False, save_data=False, save_top_tiles=True)
tilesCAFs.singleprocess_filtered_images_to_tiles(display=False, save_summary=True, save_data=False, save_top_tiles=True,
                                           html=False, image_num_list=None)

