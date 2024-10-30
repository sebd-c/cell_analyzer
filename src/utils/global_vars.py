# global variables module
# Code destined to storing global
# variables used in multiple scripts.

######################################################################
# defining global variables

IMG_WIDTH = 1608
IMG_HEIGHT = 1608
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH)
FIELDS_NUM = 38
SLICES_NUM = 151
CHANNELS_NUM = 3
PIXEL_COUNT = IMG_WIDTH * IMG_HEIGHT
FIG_SIZE = (20, 12)
UPDATE_TIME = 0.1
MEMORY_LIMIT = 90
COLOR_DICT = {'red': 'rgb(255,0,0)',
              'green': 'rgb(0,255,0)',
              'yellow': 'rgb(255,255,0)',
              'blue': 'rgb(0,0,255)'}

# scale: 0.11um/px
DISTANCES_DICT = {'embryo': 362.5,  # 39.8um
                  'nucleus': 47.8,  # 5.3um
                  'territory': 10.6,  # 1.2um
                  'overlap': 5.45}  # 0.6um

# end of current module
