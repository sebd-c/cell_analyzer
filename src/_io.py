import cv2 as cv
import numpy as np
from os.path import join
from argparse import ArgumentParser
from src._execution_formatting import enter_to_continue
from src._execution_formatting import print_progress_message
from src._execution_formatting import print_execution_parameters
from src._execution_formatting import get_files_in_folder

##############################################################################
# image merging generation functions
def merge_single_image(nuc_path: str,
                       cyto_path: str,
                       output_path: str
                       ) -> None:
    """
    Given a path to nuclei/cyto images,
    merges them into single array,
    saving it to given output path.
    """

    # reading red/green images
    nucleus = cv.imread(nuc_path, -1)
    # make img rgb for colored outlines

    cytoplasm = cv.imread(cyto_path, -1)
    # make img rgb for colored outlines

    # add(or blend) the images
    merged_img = cv.addWeighted(nucleus, 0.5, cytoplasm, 0.5, 0)

    # saving merged image
    cv.imwrite(output_path,
               merged_img)

    # # getting image halves (important to join them later and max still be 255)
    # red_half = red / 2
    # green_half = green / 2
    #
    # # merging images
    # merged = np_add(red_half, green_half)
    #
    # # converting int type
    # merged = merged.astype(np_uint8)

    return


def merge_multiple_images(nuclei_folder: str,
                          cyto_folder: str,
                          output_folder: str,
                          img_extension: str
                          ) -> None:
    """
    Given a path to folders containing
    red/green images, merges images into
    single array, saving output image in
    given output folder.
    """
    # getting images in input folder
    print('getting images in input folder...')
    nuclei_images = get_files_in_folder(path_to_folder=nuclei_folder,
                                        extension=img_extension)

    # getting images total
    images_num = len(nuclei_images)

    # starting counter for current image index
    current_image_index = 1

    # iterating over images
    for nuclei_image_name in nuclei_images:
        # printing execution message
        progress_base_string = 'merging image #INDEX# of #TOTAL#'
        print_progress_message(base_string=progress_base_string,
                               index=current_image_index,
                               total=images_num)

        # getting current nuclei image input paths
        nucleus_path = join(nuclei_folder,
                            nuclei_image_name)

        # make conversion for cytoplasm path
        cyt_image_name = nuclei_image_name.replace('red', 'green')

        # getting current cytoplasm image input paths
        cyto_path = join(cyto_folder,
                         cyt_image_name)

        # make conversion for joined channels path
        joined_image_name = nuclei_image_name.replace('red', 'joined')

        # getting current image save path
        save_path = join(output_folder,
                         joined_image_name)

        # running merge_single_image function
        merge_single_image(nuc_path=nucleus_path,
                           cyto_path=cyto_path,
                           output_path=save_path)

        # updating current image index
        current_image_index += 1

    # printing execution message
    f_string = f'all {images_num} images merged!'
    print(f_string)

    return

#########################################################################

# file writing
def make_contour_label(contour_index: int,
                       centroid_x: int | float,
                       centroid_y: int | float,
                       color: int | tuple,
                       thickness: int,
                       img_to_label: np.ndarray,
                       contour: np.ndarray,
                       ):
    """
    Writes a single contour label in an image
    """

    # # make img rgb for colored outlines
    # colored_img = cv.cvtColor(img_to_label, cv.COLOR_BGR2RGB)

    # making prep to put outlines and labels
    # fontScale
    font_scale = 1

    # fontStyle
    font_style = cv.LINE_8

    # set font style
    font = cv.FONT_HERSHEY_COMPLEX

    # the following is not related to the dict
    # but using the loop in the contours list
    cv.putText(img_to_label,
               str(contour_index),
               (int(centroid_x), int(centroid_y)),
               font,
               font_scale,
               color,
               thickness,
               font_style)

    # drawing contours in img
    cv.drawContours(img_to_label,
                    [contour],
                    -1,
                    color,
                    thickness)

def save_img(output_folder: str,
             file_name: str,
             img_to_save: np.ndarray
             ) -> None:
    """
    saves image in designated directory
    """
    # make output path
    output_path = join(output_folder, file_name)

    # save image
    cv.imwrite(output_path, img_to_save)

    return