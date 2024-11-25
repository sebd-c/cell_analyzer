######################################################################################
# auxiliary functions module
# generate binary masks module

print('initializing...')  # noqa

# Code destined to generating
# binary masks based on a pixel
# intensity threshold.

######################################################################
# imports
import numpy as np
from cv2 import morphologyEx
from cv2 import GaussianBlur
from cv2 import threshold
from cv2 import dilate
from cv2 import subtract
from cv2 import watershed
from cv2 import connectedComponents
from cv2 import distanceTransform
from cv2 import THRESH_BINARY
from cv2 import DIST_L2
from cv2 import THRESH_OTSU
from cv2 import MORPH_OPEN
from matplotlib import pyplot as plt

print('importing required libraries...')  # noqa
from os.path import join
from numpy import ndarray
from argparse import ArgumentParser
from src.utils.aux_funcs_a import save_stack
from src.utils.aux_funcs_a import load_stack
from src.utils.global_vars import SLICES_NUM
from src.utils.aux_funcs_a import convert_to_8bit
from src.utils.aux_funcs_a import get_dimensions_num
from src.classes.ProgressTracker import ProgressTracker
from src.utils.aux_funcs_a import get_specific_files_in_folder

print('all required libraries successfully imported.')  # noqa


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "generate binary masks based on fluorescence images"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help='defines path to folder containing fluorescence images (3d images OR 2d projections')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folder')

    # kernel size param
    parser.add_argument('-b', '--block_size',
                        dest='block_size',
                        required=False,
                        default=7,
                        help='defines kernel size for GaussianBlur (recommended - embryos: 15 | foci: 5)')

    # lower threshold param
    parser.add_argument('-c', '--subtracted_const',
                        dest='subtracted_const',
                        required=False,
                        default=2,
                        help='defines minimum pixel intensity threshold to be used in mask generation')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines output folder (folder that will contain binary masks)')

    # skip enter param
    parser.add_argument('-s', '--skip-enter',
                        dest='skip_enter',
                        action='store_true',
                        required=False,
                        help='defines whether to suppress "Enter to continue" input before execution')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict


#####################################################################
# progress tracking related functions


class ModuleProgressTracker(ProgressTracker):
    """
    Defines ModuleProgressTracker class.
    """

    # defining ProgressTracker init
    def __init__(self) -> None:
        """
        Initializes a ModuleProgressTracker instance
        and defines class attributes.
        """
        # inheriting attributes and methods from ProgressTracker
        super().__init__()

        # defining current module specific attributes

        # image
        self.images_num = 0
        self.current_image = 0

        # slice
        self.slices_num = 0
        self.current_slice = 0

        # dimension
        self.dimensions_num = 0

    # overwriting class methods (using current module specific attributes)

    def get_progress_string(self) -> str:
        """
        Returns a formated progress
        string, based on current progress
        attributes.
        """
        # assembling current progress string
        progress_string = f''

        # checking if iterations total has already been obtained
        if not self.totals_updated.is_set():

            # updating progress string based on attributes
            progress_string += f'calculating totals...'
            progress_string += f' {self.wheel_symbol}'
            progress_string += f' | images: {self.images_num}'
            progress_string += f' | slices: {self.slices_num}'
            progress_string += f' | iterations: {self.iterations_num}'
            progress_string += f' | elapsed time: {self.elapsed_time_str}'

        # if total iterations already obtained
        else:

            # updating progress string based on attributes
            progress_string += f'generating masks...'
            progress_string += f' {self.wheel_symbol}'
            progress_string += f' | image: {self.current_image}/{self.images_num}'
            progress_string += f' | slice: {self.current_slice}/{self.slices_num}'
            progress_string += f' | progress: {self.progress_percentage_str}'
            progress_string += f' | elapsed time: {self.elapsed_time_str}'
            progress_string += f' | ETC: {self.etc_str}'

        # returning progress string
        return progress_string

    def update_totals(self,
                      args_dict: dict
                      ) -> None:
        """
        Implements module specific method
        to update total iterations num.
        """
        # getting input folder
        input_folder = args_dict['input_folder']

        # getting image extension
        images_extension = args_dict['images_extension']

        # getting images
        images = get_specific_files_in_folder(path_to_folder=input_folder,
                                              extension=images_extension)

        # iterating over files
        for image_name in images:

            # updating progress tracker attributes
            self.images_num += 1

            # getting current image input path
            input_path = join(input_folder,
                              image_name)

            # checking if dimensions num has been set (avoids having to load all images for total calculation)
            if self.dimensions_num == 0:
                # loading current image
                image = load_stack(image_path=input_path)

                # getting dimensions num
                dimensions_num = get_dimensions_num(image=image)

                # updating progress tracker attributes
                self.dimensions_num = dimensions_num

            # starting different conditional block since it has to be executed even when dimensions_num == 0

            #  if image is 2d
            if self.dimensions_num == 2:

                # updating progress tracker attributes
                self.slices_num += 1
                self.iterations_num += 1

            #  if image is 3d
            elif self.dimensions_num == 3:

                # updating progress tracker attributes
                self.slices_num += SLICES_NUM
                self.iterations_num += SLICES_NUM

            # invalid format
            else:

                # defining error message
                e_string = f'Invalid image format.\n'
                e_string += f'Found {self.dimensions_num} dimensions in file.\n'
                e_string += f'Images should be 2d or 3d only.\n'
                e_string += f'Please, check and try again.'

                # quitting
                self.exit(message=e_string)

        # updating totals string
        totals_string = f'totals...'
        totals_string += f' | images: {self.images_num}'
        totals_string += f' | slices: {self.slices_num}'
        totals_string += f' | iterations: {self.iterations_num}'
        self.totals_string = totals_string

        # signaling totals updated
        self.signal_totals_updated()


######################################################################
# defining auxiliary functions


def apply_watershed(image: ndarray,
                    block_size: int,
                    subtracted_const: int,
                    ) -> ndarray:
    """
    Given a 2d array, returns masked
    image, based on given parameters.
    """

    # blurring img
    blur = GaussianBlur(image, (5, 5), 0)

    # adaptative threshold
    # adap_img = adaptiveThreshold(blur,
    #                              255,
    #                              ADAPTIVE_THRESH_GAUSSIAN_C,
    #                              THRESH_BINARY,
    #                              blockSize=block_size,
    #                              C=subtracted_const
    #                              )

    # apply Otsu's binarization
    ret, th_img = threshold(blur, 0, 255, THRESH_BINARY + THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = morphologyEx(th_img, MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = distanceTransform(opening, DIST_L2, 5)
    ret, sure_fg = threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = watershed(image.astype(np.uint8), markers)
    # image[markers == -1] = [255, 0, 0]

    print("threshold was:")
    print(ret)

    # converting image to 8bit
    # image = convert_to_8bit(image=adap_img)
    image = convert_to_8bit(image=markers)

    return image


def generate_binary_mask(input_path: str,
                         block_size: int,
                         subtracted_const: int,
                         output_path: str,
                         progress_tracker: ModuleProgressTracker
                         ) -> None:
    """
    Given a path to an image, a pixel intensity
    threshold, reads image as grayscale and
    binarizes image, applying GaussianBlur
    with given kernel size, saving binary
    mask to given output path.
    """
    # reading current image
    image = load_stack(input_path)

    # getting image mask
    image = apply_watershed(image=image,
                            block_size=block_size,
                            subtracted_const=subtracted_const
                            )

    # converting image to 8bit
    image_mask = convert_to_8bit(image=image)

    # saving image
    save_stack(save_path=output_path,
               image_stack=image_mask)


def generate_binary_masks(input_folder: str,
                          images_extension: str,
                          block_size: int,
                          subtracted_const: int,
                          output_folder: str,
                          progress_tracker: ModuleProgressTracker
                          ) -> None:
    """
    Given a path to a folder containing embryos
    fluorescence images, generates binary
    masks, based on given thresholds.
    """
    # getting images in input folder
    images = get_specific_files_in_folder(path_to_folder=input_folder,
                                          extension=images_extension)

    # iterating over images in input folder
    for image_name in images:
        # updating progress tracker attributes
        progress_tracker.current_image += 1

        # getting current image input/output paths
        current_input_path = join(input_folder,
                                  image_name)
        current_output_path = join(output_folder,
                                   image_name)

        # generating binary mask for current image
        generate_binary_mask(input_path=current_input_path,
                             block_size=block_size,
                             subtracted_const=subtracted_const,
                             output_path=current_output_path,
                             progress_tracker=progress_tracker)


def parse_and_run(args_dict: dict,
                  progress_tracker: ModuleProgressTracker
                  ) -> None:
    """
    Extracts args from args_dict
    and runs module function.
    """
    # getting input folder
    input_folder = args_dict['input_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting block size
    block_size = args_dict['block_size']
    block_size = int(block_size)

    # getting subtraction constant
    subtracted_const = args_dict['subtracted_const']
    subtracted_const = int(subtracted_const)

    # getting output folder
    output_folder = args_dict['output_folder']

    # running generate_binary_masks function
    generate_binary_masks(input_folder=input_folder,
                          images_extension=images_extension,
                          block_size=block_size,
                          subtracted_const=subtracted_const,
                          output_folder=output_folder,
                          progress_tracker=progress_tracker)


######################################################################
# defining main function


def main():
    """Runs main code."""
    # initializing current module progress tracker instance
    progress_tracker = ModuleProgressTracker()

    # running code in separate thread
    progress_tracker.run(function=parse_and_run,
                         args_parser=get_args_dict)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
