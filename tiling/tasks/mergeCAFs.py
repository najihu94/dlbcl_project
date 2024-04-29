import os
import numpy as np
from PIL.Image import Image
from numpy import asarray
import itertools as IT
from PIL import Image

# from tiler import Merger, Tiler

"""
Code description to add

If more than one unique function, explain each function after definition
"""

# Parameters

path = 'C:/Users/Hussi/Desktop/tiles_png/002'  # without last /
nameoftiles = 'wsi_002-tile'
rownumbermin, rownumbermax = 15, 139  # can create a code to find it -> not really useful!
colnumbermin, colnumbermax = 11, 144
tileHeight, tileWidth = 256, 256  # as to be written for the following (can be deduced by a code also)
# TO ADD: check if these numbers ARE EVEN
nrows = int((rownumbermax - rownumbermin) + 1)
ncols = int((colnumbermax - colnumbermin) + 1)
cropgap = 100  # number of pixel deleted to the end and beginning of a tile to avoid border effects.


# It is replaced by the pixels from the overlapping tiles.


# Dumb Strategy
# Add the check of parity of tile height and width


def mergeTiles(nameoftiles, path, nrows, ncols, rownumbermin, colnumbermin, tileHeight, tileWidth, cropgap):
    display = np.empty((tileHeight * nrows, tileWidth * ncols, 3), dtype=np.uint8)
    numbertiles = int(nrows * ncols)  # 2 times because have to count the overlapping bis tiles
    progress = 0
    for i, j in IT.product(range(nrows), range(ncols)):
        rownumber = i + rownumbermin  # because the first raw number is not necessarily 0 (part of the wsi)
        colnumber = j + colnumbermin  # because the first column number is not necessarily 0 (part of the wsi)
        progress += 1
        path_to_tile = path + '/' + nameoftiles + '-r' + str(rownumber) + '-c' + str(colnumber) + '.png'
        path_to_tilebisHor = path + '/' + nameoftiles + '-r' + str(rownumber) + '-c' + str(colnumber) + 'bis' + '.png'
        path_to_tilebisVer = path + '/' + nameoftiles + '-r' + str(rownumber) + 'bis' + '-c' + str(colnumber) + '.png'
        path_to_previoustileHor = path + '/' + nameoftiles + '-r' + str(rownumber) + '-c' + str(colnumber - 1) + '.png'
        path_to_previoustileVer = path + '/' + nameoftiles + '-r' + str(rownumber - 1) + '-c' + str(colnumber) + '.png'
        # The path_to_previoustile allow identifying if the tile location is in the beginning of a row or a column. If so the behavior
        # is not exactly the same.
        path_to_followingtileHor = path + '/' + nameoftiles + '-r' + str(rownumber) + '-c' + str(colnumber + 1) + '.png'
        path_to_followingtileVer = path + '/' + nameoftiles + '-r' + str(rownumber + 1) + '-c' + str(colnumber) + '.png'
        # The path_to_followingtile allow identifying if the tile location is in the end of a row or a column. If so the behavior
        # is not exactly the same.

        # Create a list of the results of os.path.exists for all path defined above.
        path_check = [os.path.exists(path_to_tile)] + [os.path.exists(path_to_tilebisHor)] + \
                     [os.path.exists(path_to_tilebisVer)] + [os.path.exists(path_to_previoustileHor)] + \
                     [os.path.exists(path_to_previoustileVer)] + [os.path.exists(path_to_followingtileHor)] + \
                     [os.path.exists(path_to_followingtileVer)]

        print(path_check)


        print('Merging progress : %d / %d' % (
        progress - 1, numbertiles))  # Don't put this print at the end of the if statement because of continue
        # statements inbetween


        # Start by considering the 4 corner tiles types (remember that the foreground tiles are not necessarily forming a square!)

        if path_check == [1, 1, 1, 0, 0, 1, 1]:

            # Here we are in the first tile of a row and first tile of a column

            print('path_to_tile row Beginning and column Beginning = ', path_to_tile)
            tile = Image.open(path_to_tile)
            tile = np.array(tile)
            tilecrop = tile[0:tileHeight - cropgap , 0:tileWidth - cropgap]
            tilebisHor = Image.open(path_to_tilebisHor)
            tilebisHor = asarray(tilebisHor)
            tilebisHorcrop = tilebisHor[:, int(tileWidth / 2) - cropgap: int(tileWidth / 2) + cropgap]
            tilebisVer = Image.open(path_to_tilebisVer)
            tilebisVer = asarray(tilebisVer)
            tilebisVercrop = tilebisVer[int(tileHeight / 2) - cropgap: int(tileHeight / 2) + cropgap, :]
            x, y = i * tileHeight, j * tileWidth
            display[x:x + (tileHeight - cropgap), y: y + (tileWidth - cropgap), :] = tilecrop
            display[x:x + tileHeight, y + (tileWidth - cropgap):y + (tileWidth + cropgap), :] = tilebisHorcrop
            display[x + (tileHeight - cropgap): x + (tileHeight + cropgap), y: y + tileWidth, :] = tilebisVercrop
            continue

        elif path_check == [1, 1, 1, 0, 1, 1, 0]:

            # Here we are in the first tile of a row and last tile of a column

            print('path_to_tile row Beginning and column Last = ', path_to_tile)
            tile = Image.open(path_to_tile)
            tile = np.array(tile)
            tilecrop = tile[cropgap: tileHeight - cropgap , 0:tileWidth - cropgap]
            tilebisHor = Image.open(path_to_tilebisHor)
            tilebisHor = asarray(tilebisHor)
            tilebisHorcrop = tilebisHor[:, int(tileWidth / 2) - cropgap: int(tileWidth / 2) + cropgap]
            tilebisVer = Image.open(path_to_tilebisVer)
            tilebisVer = asarray(tilebisVer)
            tilebisVercrop = tilebisVer[int(tileHeight / 2) - cropgap: int(tileHeight / 2), :]
            x, y = i * tileHeight, j * tileWidth
            display[x + cropgap: x + (tileHeight - cropgap), y: y + (tileWidth - cropgap), :] = tilecrop
            display[x:x + tileHeight, y + (tileWidth - cropgap):y + (tileWidth + cropgap), :] = tilebisHorcrop
            display[x + (tileHeight - cropgap): x + tileHeight, y: y + tileWidth, :] = tilebisVercrop
            continue

        elif path_check == [1, 1, 1, 1, 0, 0, 1]:

            # Here we are in the last tile of a row and first tile of a column

            print('path_to_tile row Last and column Beginning = ', path_to_tile)
            tile = Image.open(path_to_tile)
            tile = np.array(tile)
            tilecrop = tile[0:tileWidth - cropgap, cropgap: tileWidth - cropgap]
            tilebisHor = Image.open(path_to_tilebisHor)
            tilebisHor = asarray(tilebisHor)
            tilebisHorcrop = tilebisHor[:, int(tileWidth / 2) - cropgap: int(tileWidth / 2)]
            tilebisVer = Image.open(path_to_tilebisVer)
            tilebisVer = asarray(tilebisVer)
            tilebisVercrop = tilebisVer[int(tileHeight / 2) - cropgap: int(tileHeight / 2) + cropgap, :]
            x, y = i * tileHeight, j * tileWidth
            display[x: x + (tileHeight - cropgap), y + cropgap: y + (tileWidth - cropgap), :] = tilecrop
            display[x:x + tileHeight, y + (tileWidth - cropgap):y + tileWidth, :] = tilebisHorcrop
            display[x + (tileHeight - cropgap): x + (tileHeight + cropgap), y: y + tileWidth, :] = tilebisVercrop
            continue

        elif path_check == [1, 1, 1, 1, 1, 0, 0]:

            # Here we are in the last tile of a row and last tile of a column

            print('path_to_tile row Last and column Last = ', path_to_tile)
            tile = Image.open(path_to_tile)
            tile = np.array(tile)
            tilecrop = tile[cropgap: tileHeight - cropgap , cropgap: tileWidth - cropgap]
            tilebisHor = Image.open(path_to_tilebisHor)
            tilebisHor = asarray(tilebisHor)
            tilebisHorcrop = tilebisHor[:, int(tileWidth / 2) - cropgap: int(tileWidth / 2)]
            tilebisVer = Image.open(path_to_tilebisVer)
            tilebisVer = asarray(tilebisVer)
            tilebisVercrop = tilebisVer[int(tileHeight / 2) - cropgap: int(tileHeight / 2), :]
            x, y = i * tileHeight, j * tileWidth
            display[x + cropgap: x + (tileHeight - cropgap), y + cropgap: y + (tileWidth - cropgap), :] = tilecrop
            display[x:x + tileHeight, y + (tileWidth - cropgap):y + tileWidth, :] = tilebisHorcrop
            display[x + (tileHeight - cropgap): x + tileHeight, y: y + tileWidth, :] = tilebisVercrop
            continue


        # Then we consider the 4 border not corner tiles types (remember that the foreground tiles are not necessarily forming a square!)


        elif path_check == [1, 1, 0 or 1, 0, 1, 1, 1]:

            # Here we are in the first tile of a row and middle of a column

            print('path_to_tile row Beginning and column Middle = ', path_to_tile)
            tile = Image.open(path_to_tile)
            tile = np.array(tile)
            tilecrop = tile[cropgap: tileHeight - cropgap , 0:tileWidth - cropgap]
            tilebisHor = Image.open(path_to_tilebisHor)
            tilebisHor = asarray(tilebisHor)
            tilebisHorcrop = tilebisHor[:, int(tileWidth / 2) - cropgap: int(tileWidth / 2) + cropgap]
            tilebisVer = Image.open(path_to_tilebisVer)
            tilebisVer = asarray(tilebisVer)
            tilebisVercrop = tilebisVer[int(tileHeight / 2) - cropgap: int(tileHeight / 2) + cropgap, :]
            x, y = i * tileHeight, j * tileWidth
            display[x + cropgap: x + (tileHeight - cropgap), y: y + (tileWidth - cropgap), :] = tilecrop
            display[x: x + tileHeight, y + (tileWidth - cropgap):y + (tileWidth + cropgap), :] = tilebisHorcrop
            display[x + (tileHeight - cropgap): x + (tileHeight + cropgap), y: y + tileWidth, :] = tilebisVercrop
            continue

        elif path_check == [1, 1, 0 or 1, 1, 1, 0, 1]:

            # Here we are in the last tile of a row and middle of a column

            print('path_to_tile row Last and column Middle  = ', path_to_tile)
            tile = Image.open(path_to_tile)
            tile = np.array(tile)
            tilecrop = tile[cropgap: tileHeight - cropgap , cropgap: tileWidth - cropgap]
            tilebisHor = Image.open(path_to_tilebisHor)
            tilebisHor = asarray(tilebisHor)
            tilebisHorcrop = tilebisHor[:, int(tileWidth / 2) - cropgap: int(tileWidth / 2)]
            tilebisVer = Image.open(path_to_tilebisVer)
            tilebisVer = asarray(tilebisVer)
            tilebisVercrop = tilebisVer[int(tileHeight / 2) - cropgap: int(tileHeight / 2) + cropgap, :]
            x, y = i * tileHeight, j * tileWidth
            display[x + cropgap: x + (tileHeight - cropgap), y + cropgap: y + (tileWidth - cropgap), :] = tilecrop
            display[x:x + tileHeight, y + (tileWidth - cropgap):y + tileWidth, :] = tilebisHorcrop
            display[x + (tileHeight - cropgap): x + (tileHeight + cropgap), y: y + tileWidth, :] = tilebisVercrop
            continue

        elif path_check == [1, 0 or 1, 1, 1, 0, 1, 1]:

            # Here we are in the middle of a row and first tile of a column

            print('path_to_tile row Middle and column Beginning = ', path_to_tile)
            tile = Image.open(path_to_tile)
            tile = np.array(tile)
            tilecrop = tile[0: tileHeight - cropgap, cropgap: tileWidth - cropgap]
            tilebisHor = Image.open(path_to_tilebisHor)
            tilebisHor = asarray(tilebisHor)
            tilebisHorcrop = tilebisHor[:, int(tileWidth / 2) - cropgap: int(tileWidth / 2) + cropgap]
            tilebisVer = Image.open(path_to_tilebisVer)
            tilebisVer = asarray(tilebisVer)
            tilebisVercrop = tilebisVer[int(tileHeight / 2) - cropgap: int(tileHeight / 2) + cropgap, :]
            x, y = i * tileHeight, j * tileWidth
            display[x: x + (tileHeight - cropgap), y + cropgap: y + (tileWidth - cropgap), :] = tilecrop
            display[x:x + tileHeight, y + (tileWidth - cropgap): y + (tileWidth + cropgap), :] = tilebisHorcrop
            display[x + (tileHeight - cropgap): x + (tileHeight + cropgap), y: y + tileWidth, :] = tilebisVercrop
            continue

        elif path_check == [1, 0 or 1, 1, 1, 1, 1, 0]:

            # Here we are in the middle of a row and last tile of a column

            print('path_to_tile row Middle and column Last = ', path_to_tile)
            tile = Image.open(path_to_tile)
            tile = np.array(tile)
            tilecrop = tile[cropgap: tileHeight - cropgap, cropgap: tileWidth - cropgap]
            tilebisHor = Image.open(path_to_tilebisHor)
            tilebisHor = asarray(tilebisHor)
            tilebisHorcrop = tilebisHor[:, int(tileWidth / 2) - cropgap: int(tileWidth / 2) + cropgap]
            tilebisVer = Image.open(path_to_tilebisVer)
            tilebisVer = asarray(tilebisVer)
            tilebisVercrop = tilebisVer[int(tileHeight / 2) - cropgap: int(tileHeight / 2), :]
            x, y = i * tileHeight, j * tileWidth
            display[x + cropgap: x + (tileHeight - cropgap), y + cropgap: y + (tileWidth - cropgap), :] = tilecrop
            display[x:x + tileHeight, y + (tileWidth - cropgap): y + (tileWidth + cropgap), :] = tilebisHorcrop
            display[x + (tileHeight - cropgap): x + tileHeight, y: y + tileWidth, :] = tilebisVercrop
            continue


        # Then we consider the core tiles

        elif os.path.exists(path_to_tile) and os.path.exists(path_to_tilebisHor):

            # Here we are in the middle of a row and of a column
            # Inside this elif, the other ones were not fulfilled because of continue statement

            print('path_to_tile row Middle and column Middle = ', path_to_tile)
            tile = Image.open(path_to_tile)
            tile = np.array(tile)
            tilecrop = tile[cropgap: tileHeight - cropgap , cropgap: tileWidth - cropgap]
            tilebisHor = Image.open(path_to_tilebisHor)
            tilebisHor = asarray(tilebisHor)
            tilebisHorcrop = tilebisHor[:, int(tileWidth / 2) - cropgap: int(tileWidth / 2) + cropgap]
            tilebisVer = Image.open(path_to_tilebisVer)
            tilebisVer = asarray(tilebisVer)
            tilebisVercrop = tilebisVer[int(tileHeight / 2) - cropgap: int(tileHeight / 2) + cropgap, :]
            x, y = i * tileHeight, j * tileWidth
            display[x + cropgap: x + (tileHeight - cropgap), y + cropgap: y + (tileWidth - cropgap), :] = tilecrop
            display[x: x + tileHeight, y + (tileWidth - cropgap): y + (tileWidth + cropgap), :] = tilebisHorcrop
            display[x + (tileHeight - cropgap): x + (tileHeight + cropgap), y: y + tileWidth, :] = tilebisVercrop


        # Then we consider exceptions and issues


        elif os.path.exists(path_to_tile) and not os.path.exists(path_to_tilebisHor) and not os.path.exists(path_to_tilebisVer):

            # This shouldn't happen. We can imagine that the tile is so closed to the border of the image
            # That an overlapping tile cannot be generated


            print('Needs overlapping tiles! If you don\'t have use concatenateCAFs.py')
        # Finally, we fill with background tiles to have a square image as a result


        else:
            blackimage = np.full((tileHeight, tileWidth, 3), 0,
                                 dtype=np.uint8)  # If there is no tile in this location, create a full black tile
            x, y = i * tileHeight, j * tileWidth
            display[x:x + tileHeight, y:y + tileWidth, :] = blackimage

    return display


# Use

finalresult = mergeTiles(nameoftiles, path, nrows, ncols, rownumbermin, colnumbermin, tileHeight, tileWidth, cropgap)
image = Image.fromarray(finalresult)

Output_path = path + '/MergeResult'
if not os.path.exists(Output_path):
    os.mkdir(Output_path)

image.save(Output_path + '/' + 'Merged_' + nameoftiles + '_row_' + str(rownumbermin) + 'to' + str(
    rownumbermax) + '_col_' + str(colnumbermin) + 'to' + str(colnumbermax) + '.png')
# import cv2
# cv2.imwrite(Output_path + '/' + 'Merged_' + nameoftiles + '_row_' + str(rownumbermin) + 'to' + str(
#     rownumbermax) + '_col_' + str(colnumbermin) + 'to' + str(colnumbermax) + '.ome.tif', cv2.cvtColor(np.uint8(finalresult), cv2.COLOR_RGB2BGR))
print('Merged Image Saved')

# Strategy from https://github.com/the-lay/tiler

# def mergeTiles2(nameoftiles, path, nrows, ncols, rownumbermin, colnumbermin, tileHeight, tileWidth):
#     display = np.empty((tileHeight * nrows, tileWidth * ncols, 3), dtype=np.uint8)
#     numbertiles = (int(nrows * ncols)) * 2  # 2 times because have to count the overlapping bis tiles
#     listrowtiles = []  # final list of all rows with merged tiles to rebuild original image
#     for i in range(nrows):
#         # For each row, will merge overlapping tiles from this row
#         rownumber = i + rownumbermin  # because the first raw number is not necessarily 0 (part of the wsi)
#         listtiles = []  # list of all tiles in the current row towards merging all tiles of this row
#         nbrtiles = 0  # number of tiles in the current row
#         for j in range(ncols):
#             colnumber = j + colnumbermin
#             path_to_tile = path + '/' + nameoftiles + '-r' + str(rownumber) + '-c' + str(colnumber) + '.png'
#             path_to_tilebis = path + '/' + nameoftiles + '-r' + str(rownumber) + '-c' + str(colnumber) + 'bis' + '.png'
#             if os.path.exists(path_to_tile):
#                 image = Image.open(path_to_tile)
#                 tilearray = np.array(image)  # add non-overlapping tiles to the list of tiles
#                 listtiles.append(tilearray)
#                 nbrtiles += 1
#             if os.path.exists(path_to_tilebis):
#                 image = Image.open(path_to_tilebis)
#                 tilearray = np.array(image)
#                 listtiles.append(
#                     tilearray)  # add corresponding bis overlapping tiles to the list of tiles towards merging
#                 nbrtiles += 1
#
#         tiler = Tiler(
#             data_shape=(tileHeight, tileWidth * nbrtiles),
#             tile_shape=(tileHeight, tileWidth, 3),
#             overlap=(round(tileHeight / 2), round(tileWidth / 2), 0),
#             channel_dimension=2,
#         )
#         merger = Merger(tiler=tiler, window="overlap-tile")
#
#         for k in range(nbrtiles):
#             merger.add(k + 1, listtiles[k])
#
#         final_row = merger.merge()  # Results of merging overlapping tiles in this row
#         x, y = i * tileHeight, nbrtiles * tileWidth
#         display[x:x + tileHeight, y:y + tileWidth, :] = image
#
#         # listrowtiles.append(final_row)
