import numpy as np

print('initializing...')  # noqa

######################################################################
# imports
print('importing required libraries...')  # noqa
from argparse import ArgumentParser
from os.path import join
from os import makedirs
from shutil import copy2
from random import shuffle
from src.utils.aux_funcs import get_files_in_folder
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

######################################################################
# module specific aux functions

def stratified_split(class_folders: list,
                     input_folder: str,
                     output_folder: str,
                     extension: str,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.2,
                     test_ratio: float = 0.1) -> None:
    """
    Given folders of images (each folder = class),
    split into train/val/test preserving class distribution.
    """

    for class_name in class_folders:
        # get list of files for class
        class_path = join(input_folder, class_name)
        image_files = get_files_in_folder(path_to_folder=class_path,
                                          extension=extension)

        # shuffle for randomness
        shuffle(image_files)

        # split sizes
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        n_test = n_total - n_train - n_val

        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train+n_val]
        test_files = image_files[n_train+n_val:]

        # dictionary for subsets
        subsets = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        # copy each subset into its folder
        for subset, files in subsets.items():
            subset_class_path = join(output_folder, subset, class_name)
            makedirs(subset_class_path, exist_ok=True)

            for idx, f in enumerate(files, 1):
                src_path = join(class_path, f)
                dst_path = join(subset_class_path, f)
                copy2(src_path, dst_path)

                # progress
                base_string = f"copying {subset} image #INDEX# of #TOTAL# for class {class_name}"
                print_progress_message(base_string=base_string,
                                       index=idx,
                                       total=len(files))

    print("Stratified split complete!")
    return

######################################################################
# argument parsing related functions

def get_args_dict() -> dict:
    """
    Perses the arguments and returns a dictionary of the arguments.
    """
    description = 'stratified split module'
    parser = ArgumentParser(description=description)

    # input folder param
    parser.add_argument('-i','--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing class-organized images')

    # class list
    parser.add_argument('-cl', '--class-list',
                        nargs='+',
                        dest='class_list',
                        required=True,
                        help='list of class folder names')

    #images extension
    parser.add_argument('-x','--images-extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jgp) of images in input folders')
    # output folder
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output base folder (train/validation/test will be created inside)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())
    return args_dict

######################################################################
# defining main function

def main():
    """Runs main code."""
    args_dict = get_args_dict()

    images_folder = args_dict['images_folder']
    class_list = args_dict['class_list']
    extension = args_dict['images_extension']
    output_folder = args_dict['output_folder']

    print_execution_parameters(params_dict=args_dict)

    stratified_split(class_folders=class_list,
                     input_folder=images_folder,
                     output_folder=output_folder,
                     train_ratio=0.7,
                     val_ratio=0.2,
                     test_ratio=0.1)


######################################################################
# running main function

if __name__ == '__main__':
    main()

######################################################################
# end of current module