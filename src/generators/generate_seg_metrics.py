# generate segmentation dfs module
import numpy as np
from math import ceil
print('initializing...')  # noqa

# Code destined to generating
# segmentation mask based df.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from argparse import ArgumentParser
from cv2 import imread
from cv2 import boundingRect
from cv2 import drawContours
from cv2 import FILLED
from cv2 import split
from cv2 import imwrite
from cv2 import rotate
from cv2 import ROTATE_90_CLOCKWISE
from cv2 import ROTATE_90_COUNTERCLOCKWISE
from cv2 import fillPoly
from copy import copy
from numpy import uint8 as np_uint8
from numpy import ndarray
from numpy import pad
from pandas import concat, read_pickle
from pandas import DataFrame
from os.path import join
#from numpy import max
# from numpy import min
from numpy import mean
from numpy import median
from numpy import sum
from numpy import unique
from numpy import isin
import tifffile
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_contour_centroid
from src.utils.aux_funcs import get_area_box
from src.utils.aux_funcs import get_contour_rratio
from src.utils.aux_funcs import get_contour_ellipse_feats
from src.utils.aux_funcs import get_contour_roundness
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import get_files_in_folder
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import make_contour_label
from src.utils.aux_funcs import get_unique_ids
from src.utils.aux_funcs import get_coords
from src.utils.aux_funcs import get_boundaries_mask
from src.utils.aux_funcs import apply_mask
from src.utils.aux_funcs import make_crop
from src.utils.aux_funcs import make_crop_rotate
from src.utils.aux_funcs import save_img


print('all required libraries successfully imported.')  # noqa


#####################################################################

# module specific aux functions
def make_single_crop(image: ndarray,
                     contour:ndarray,
                     cx: int or float,
                     cy: int or float,
                     max_width: int or float,
                     max_height: int or float,
                     output_folder: str,
                     crop_name: str
                     ) -> None:
    """
    Given an image, a contour, the contour's centroid (x, y),
    and the supposed size of the crop,
    returns a crop
    """
    # define a pixel intensity
    pixel_intensity = 255

    # getting img shape
    shape = image.shape

    # for img border setting
    img_h, img_w = image.shape

    # creates mask to hold cytoplasm contour
    # for noise cleaning process outside object
    mask = np.zeros(shape, dtype=np.uint8)

    # filling mask image
    drawContours(mask, [contour], -1, pixel_intensity, FILLED)

    # apply mask to make a clean image
    clean_image = apply_mask(image=image,
                             mask=mask,
                             pixel_intensity=pixel_intensity)

    # get object metrics for flipping conditional
    _, _, w, h = boundingRect(contour)

    # if the object is in another rotation, flip it
    if w > h:

        # vertex settings per object
        # with safe border limitations
        x1 = cx - max_height / 2
        x1 = max(0, int(x1))

        x2 = cx + max_height / 2
        x2 = min(img_w, int(x2))

        y1 = cy - max_width / 2
        y1 = max(0, int(y1))

        y2 = cy + max_width / 2
        y2 = min(img_h, int(y2))

        # image cropping
        crop = make_crop_rotate(image=clean_image,
                                x1=x1,
                                x2=x2,
                                y1=y1,
                                y2=y2,
                                max_width=max_width,
                                max_height=max_height)

    else:

        # vertex settings per object
        x1 = cx - max_width / 2
        x1 = max(0, int(x1))

        x2 = cx + max_width / 2
        x2 = min(img_w, int(x2))

        y1 = cy - max_height / 2
        y1 = max(0, int(y1))

        y2 = cy + max_height / 2
        y2 = min(img_h, int(y2))

        # image cropping
        crop = make_crop(image=clean_image,
                         x1=x1,
                         x2=x2,
                         y1=y1,
                         y2=y2,
                         max_width=max_width,
                         max_height=max_height
                         )

    # saving crops
    save_img(output_folder=output_folder,
             file_name=crop_name,
             img_to_save=crop)

    return

def get_image_iou(ground_truth_img: str,
                  predicted_img: str,
                  ) -> None:
    """
    Given a path to two grayscale images,
    one being a ground truth mask, the other a
    predicted mask, returns the number of
    TP, TN, FP and FN.
    """

    # reading ground truth grayscale image
    gt_image = imread(ground_truth_img,
                      -1)

    # reading grayscale image
    p_image = imread(predicted_img,
                     -1)


    return


def get_folder_iou(gt_input_folder: str,
                   predicted_input_folder: str,
                   image_extension: str,
                   output_folder: str
                   ) -> None:
    """
    Given a path to a folder containing
    grayscale images of gt images,
    a path to a folder containing
    grayscale images of predicted images
    and an output folder to iou results and plots,
    saves analysis in said folder
    """
    # getting image files in respective input folder
    gt_files = get_files_in_folder(path_to_folder=gt_input_folder,
                                   extension=image_extension)

    # getting image files in respective input folder
    predicted_files = get_files_in_folder(path_to_folder=predicted_input_folder,
                                          extension=image_extension)

    image_files_num = len(gt_files)

    # iterating over image files
    for file_index, gt_file in enumerate(gt_files, 1):
        # printing progress message
        base_string = 'generating crop image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=image_files_num)

        # getting respective predictive file
        predicted_file = predicted_files[file_index - 1]

        # making current cytoplasm img input path
        gt_input_path = join(gt_input_folder,
                             gt_file)

        # analogous getting current og image input/output paths
        pred_input_path = join(predicted_input_folder,
                              predicted_file)

        # make and save image crops
        make_image_crops(cyto_path=cyto_input_path,
                         nuc_path=nuc_input_path,
                         phase_path=phase_input_path,
                         image_name=cyto_file,
                         df=df,
                         max_width=max_width,
                         max_height=max_height,
                         cyto_output_folder=cyto_output_folder,
                         nuc_output_folder=nuc_output_folder,
                         phase_output_folder=phase_output_folder)

    # printing execution message
    print(f'output saved to {cyto_output_folder}')
    print('analysis complete!')

    return


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'generate crops module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # grayscale input folder param
    parser.add_argument('-gt', '--gt-folder',
                        dest='gt_folder',
                        required=True,
                        help='defines path to folder containing gt grayscale images')

    # grayscale input folder param
    parser.add_argument('-p', '--pred-folder',
                        dest='pred_folder',
                        required=True,
                        help='defines path to folder containing gt grayscale images')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output analysis')

        # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict


######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting cytoplasm input folder
    cyto_folder = args_dict['cyto_folder']

    # getting nuclei input folder
    nuclei_folder = args_dict['nuclei_folder']

    # getting phase input folder
    phase_folder = args_dict['phase_folder']

    # path to segmentation dataframe
    df_path = args_dict['df_path']

    # getting images extension
    images_extension = args_dict['images_extension']

    # getting cytoplasm output folder
    cyto_output_folder = args_dict['cyto_output_folder']

    # getting nuclei output folder
    nuclei_output_folder = args_dict['nuclei_output_folder']

    # getting phase output folder
    phase_output_folder = args_dict['phase_output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input

    # running function to get the segmentation df for a folder
    make_folder_crops(cyto_input_folder=cyto_folder,
                      nuc_input_folder=nuclei_folder,
                      phase_input_folder=phase_folder,
                      image_extension=images_extension,
                      df_path=df_path,
                      cyto_output_folder=cyto_output_folder,
                      nuc_output_folder=nuclei_output_folder,
                      phase_output_folder=phase_output_folder,
                      )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
