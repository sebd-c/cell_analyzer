# generate segmentation dfs module
print('initializing...')  # noqa

# Code destined to generating
# segmentation mask based df.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
import cv2 as cv
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from src._execution_formatting import enter_to_continue
from src._execution_formatting import print_progress_message
from src._execution_formatting import get_files_in_folder
from src._execution_formatting import print_execution_parameters
from src.imaging._img_subprocess import apply_mask
from src.imaging._img_subprocess import make_crop
from src.imaging._img_subprocess import make_crop_rotate
from src._io import save_img


print('all required libraries successfully imported.')  # noqa


#####################################################################

# module specific aux functions
def make_single_crop(image: np.ndarray,
                     contour: np.ndarray,
                     cx: int | float,
                     cy: int | float,
                     max_width: int | float,
                     max_height: int | float,
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
    cv.drawContours(mask, [contour], -1, pixel_intensity, cv.FILLED)

    # apply mask to make a clean image
    clean_image = apply_mask(image=image,
                             mask=mask,
                             pixel_intensity=pixel_intensity)

    # get object metrics for flipping conditional
    _, _, w, h = cv.boundingRect(contour)

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
                                max_width=int(max_width),
                                max_height=int(max_height)
                                )

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
                         max_width=int(max_width),
                         max_height=int(max_height)
                         )

    # saving crops
    save_img(output_folder=output_folder,
             file_name=crop_name,
             img_to_save=crop)

    return

def make_image_crops(cyto_path: str,
                     nuc_path: str,
                     phase_path: str,
                     image_name: str,
                     df: pd.DataFrame,
                     max_width: int | float,
                     max_height: int | float,
                     cyto_output_folder: str,
                     nuc_output_folder: str,
                     phase_output_folder: str,
                     unique_df_output_folder: str
                     ) -> None:
    """
    Given a path to a grayscale image,
    a dataframe containing each object's centroid,
    id, origin image and contour,
    generates crops of all of that image's objects,
    and saves the results in the output folder.
    """

    # reading grayscale image
    cyto_image = cv.imread(cyto_path,
                           -1)

    # reading grayscale image
    nuc_image = cv.imread(nuc_path,
                       -1)

    # reading grayscale image
    phase_image = cv.imread(phase_path,
                            -1)

    # takeout channel from img name
    image_name = image_name.replace('green', '')
    image_name = image_name.replace('red', '')
    image_name = image_name.replace('phase', '')

    # segment the df to loop only on this image's objects
    image_df = df[df['cyto_image_name'] == image_name]

    # since there are repeated cytoplasms, delete duplicates to make a unique df
    unique_df = image_df.drop_duplicates(subset=['image_name', 'cyto_id'], keep='first')

    crop_name_list = []

    # loop not to join different contours
    for index, row in unique_df.iterrows():

        # fixing name just for order
        split_img_name = image_name.split('.')

        # making crop name for saving
        crop_name = 'img_' + split_img_name[0] + '_crop_' + str(row['cyto_id']) + '.' + split_img_name[1]

        # getting df values
        contour = row['cyto_contour']
        cx = row['cyto_cx']
        cy = row['cyto_cy']

        # append new crop name to list
        crop_name_list.append(crop_name)

        # make crops
        make_single_crop(image=cyto_image,
                         contour=contour,
                         cx=cx,
                         cy=cy,
                         max_width=max_width,
                         max_height=max_height,
                         output_folder=cyto_output_folder,
                         crop_name=crop_name
                         )

        make_single_crop(image=nuc_image,
                         contour=contour,
                         cx=cx,
                         cy=cy,
                         max_width=max_width,
                         max_height=max_height,
                         output_folder=nuc_output_folder,
                         crop_name=crop_name
                         )

        make_single_crop(image=phase_image,
                         contour=contour,
                         cx=cx,
                         cy=cy,
                         max_width=max_width,
                         max_height=max_height,
                         output_folder=phase_output_folder,
                         crop_name=crop_name
                         )

    # make new column in dataframe to maintain relation directly to crops
    unique_df['crop_name'] = crop_name_list

    # create the path to save the output path
    output_path = join(unique_df_output_folder,
                       'uniq_obj_df.pickle')

    # saving df
    unique_df.to_pickle(output_path)

    return


def make_folder_crops(cyto_input_folder: str,
                      nuc_input_folder: str,
                      phase_input_folder: str,
                      image_extension: str,
                      df_path: str,
                      cyto_output_folder: str,
                      nuc_output_folder: str,
                      phase_output_folder: str,
                      unique_df_output_folder: str
                      ) -> None:
    """
    Given a path to a folder containing
    grayscale images of an object and
    a dataframe containing each object's centroid,
    id, origin image and contour,
    generates crops of all objects,
    and saves the results in the output folder.
    """

    # read csv
    df = pd.read_pickle(df_path)

    # variables to save width/height values
    max_height = float('-inf')
    max_width = float('-inf')

    # defining crop shape
    for index, row in df.iterrows():

        # use open cv bb to get min rect shape
        x, y, w, h = cv.boundingRect(row['cyto_contour'])

        # as a big object in height or width are the same
        # set one variable to always be the bigger one (some objects will be "flipped" in the end)
        width = w if w < h else h
        height = w if w > h else h

        # compare with maxes saved
        max_width = max_width if max_width > width else width
        max_height = max_height if max_height > height else height

    # getting image files in respective input folder
    cyto_files = get_files_in_folder(path_to_folder=cyto_input_folder,
                                      extension=image_extension)

    # getting image files in respective input folder
    nuc_files = get_files_in_folder(path_to_folder=nuc_input_folder,
                                     extension=image_extension)

    # getting image files in respective input folder
    phase_files = get_files_in_folder(path_to_folder=phase_input_folder,
                                    extension=image_extension)

    image_files_num = len(cyto_files)

    # iterating over image files
    for file_index, cyto_file in enumerate(cyto_files, 1):
        # printing progress message
        base_string = 'generating crop image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=image_files_num)

        # getting respective nuc file
        nuc_file = nuc_files[file_index - 1]

        # getting respective phase file
        phase_file = phase_files[file_index - 1]

        # making current cytoplasm img input path
        cyto_input_path = join(cyto_input_folder,
                               cyto_file)

        # analogous getting current og image input/output paths
        nuc_input_path = join(nuc_input_folder,
                              nuc_file)

        # analogous getting current og image input/output paths
        phase_input_path = join(phase_input_folder,
                                phase_file)

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
                         phase_output_folder=phase_output_folder,
                         unique_df_output_folder=unique_df_output_folder)

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
    parser.add_argument('-c', '--cyto-folder',
                        dest='cyto_folder',
                        required=True,
                        help='defines path to folder containing original grayscale cytoplasm images')

    # grayscale input folder param
    parser.add_argument('-n', '--nuclei-folder',
                        dest='nuclei_folder',
                        required=True,
                        help='defines path to folder containing original nuclei grayscale images')

    # grayscale input folder param
    parser.add_argument('-p', '--phase-folder',
                        dest='phase_folder',
                        required=True,
                        help='defines path to folder containing phase grayscale-ed images')

    # segmentation dataframe input folder param
    parser.add_argument('-d', '--df-path',
                        dest='df_path',
                        required=True,
                        help='defines path to folder containing full df')

    # segmentation dataframe input folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder to save df wo obj repetitions')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # crops output folder param
    parser.add_argument('-co', '--cyto-output-folder',
                        dest='cyto_output_folder',
                        required=True,
                        help='defines path to output folder (cyto crops)')

    # crops output folder param
    parser.add_argument('-no', '--nuclei-output-folder',
                        dest='nuclei_output_folder',
                        required=True,
                        help='defines path to output folder (nuc crops)')

    # crops output folder param
    parser.add_argument('-po', '--phase-output-folder',
                        dest='phase_output_folder',
                        required=True,
                        help='defines path to output folder (phase crops)')

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

    # path to output folder
    output_folder = args_dict['output_folder']

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
    enter_to_continue()

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
                      unique_df_output_folder=output_folder)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
