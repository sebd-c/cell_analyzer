# generate segmentation dfs module
print('initializing...')  # noqa

# Code destined to generating
# segmentation mask based df.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from pandas import DataFrame
from pandas import concat
from pandas import read_pickle
from scipy.spatial import cKDTree
from numpy import min
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters


print('all required libraries successfully imported.')  # noqa


#####################################################################
def make_img_class_counts(image_df: DataFrame) -> DataFrame:
    """

    """

    # since there are repeated cytoplasms, delete duplicates to make a unique df
    unique_df = image_df.drop_duplicates(subset=['image_name', 'cyto_id'], keep='first')

    # establish lists to hold distances between classes
    # n = normal; q = quiescent; fs = fully-senescent; sl = senescent-like

    # normal to others
    nn_dist_list = []
    nq_dist_list = []
    nfs_dist_list = []
    nsl_dist_list = []

    # quiescent to others
    qq_dist_list = []
    qfs_dist_list = []
    qsl_dist_list = []

    # fully senescent to others
    fsfs_dist_list = []
    fssl_dist_list = []

    # senescent like to others
    slsl_dist_list = []

    for index_1, row_1 in unique_df.iterrows():

        # get contour of 1st object
        contour_1 = row_1['cyto_contour']

        # get its list of coordinates
        coords_1 = contour_1.reshape(-1, 2)

        # get df to loop on (which is all of it except the row already in contour)
        subtracted_df = unique_df.drop(index_1)

        # get row 1 class
        class_1 = row_1['label']
        # loop to get all other objects
        for index_2, row_2 in subtracted_df.iterrows():

            # get row 2 class
            class_2 = row_2['label']

            # get contour of 2nd object
            contour_2 = row_2['cyto_contour']

            # get its list of coordinates
            coords_2 = contour_2.reshape(-1, 2)

            # Build KDTree for the second contour
            tree = cKDTree(coords_2)

            # Query nearest neighbor for each point in contour1
            distances, _ = tree.query(coords_1, k=1)

            # Smallest distance between contours
            min_distance = min(distances)

            # assign to which list append the distance:
            # n x n
            if class_1 == 0 and class_2 == 0:
                nn_dist_list.append(min_distance)
            # n x q
            elif class_1 == 0 and class_2 == 1:
                nq_dist_list.append(min_distance)
            # n x fs
            elif class_1 == 0 and class_2 == 2:
                nfs_dist_list.append(min_distance)
            # n x sl
            elif class_1 == 0 and class_2 == 3:
                nsl_dist_list.append(min_distance)
            # q x q
            elif class_1 == 1 and class_2 == 1:
                qq_dist_list.append(min_distance)
            # q x fs
            elif class_1 == 1 and class_2 == 2:
                qfs_dist_list.append(min_distance)
            # q x sl
            elif class_1 == 1 and class_2 == 3:
                qsl_dist_list.append(min_distance)
            # fs x fs
            elif class_1 == 2 and class_2 == 2:
                fsfs_dist_list.append(min_distance)
            # fs x sl
            elif class_1 == 2 and class_2 == 3:
                fssl_dist_list.append(min_distance)
            # sl x sl
            elif class_1 == 3 and class_2 == 3:
                slsl_dist_list.append(min_distance)

    # organize in dict format
    dist_classes_dict = {'nn_dist': nn_dist_list,
                         'nq_dist': nq_dist_list,
                         'nfs_dist': nfs_dist_list,
                         'nsl_dist': nsl_dist_list,
                         'qq_dist': qq_dist_list,
                         'qfs_dist': qfs_dist_list,
                         'qsl_dist': qsl_dist_list,
                         'fsfs_dist': fsfs_dist_list,
                         'fssl_dist': fssl_dist_list,
                         'slsl_dist': slsl_dist_list,
                        }

    # assembling contour df, i.e, making a row
    dist_classes_df = DataFrame(dist_classes_dict, index=[0])  # noqa
    # returns single row of dataframe
    return dist_classes_df


def make_folder_class_counts(folder_df: DataFrame) -> DataFrame:
    """
    """
    # get list of image names to loop
    image_names = folder_df.groupby(['image_name'])

    # create an empty list to hold the single imgs df
    images_dist_df_list = []

    # loop through all images
    for image_name in image_names:
        # separate df by img
        image_df = folder_df[folder_df['image_name'] == image_name]

        # get distances dict per img
        image_dists_df = make_img_class_counts(image_df=image_df)

        # appending current df to dfs list
        images_dist_df_list.append(image_dists_df)

    # concat the list of dfs into a single df
    concat_images_dist_df = concat(images_dist_df_list, ignore_index=True)


    return concat_images_dist_df


# module specific aux functions
def plot_folder_class_hist(contours_pickle_path: str,
                           output_folder: str
                           ) -> None:
    """

    """

    # read pickle with object infos
    contours_df = read_pickle(contours_pickle_path)

    # convert it to the accurate colors
    images_dist_df = make_folder_class_counts(folder_df=contours_df)

    # getting pairs of histograms to be plotted
    pairs_list = images_dist_df.columns

    # plotting histogram
    for pair in pairs_list:



    return


def make_folder_contours_df(masks_input_folder: str,
                            og_img_folder: str,
                            phase_img_folder: str,
                            masks_img_extension: str,
                            csv_output_folder: str,
                            overlays_output_folder: str,
                            ) -> None:
    """
    Given a path to a folder containing
    cytoplasm's masks, generates a df
    containing the wanted information,
    and saving the results
    in the output folder.
    """
    # getting masks files in respective input folder
    mask_files = get_files_in_folder(path_to_folder=masks_input_folder,
                                     extension=masks_img_extension)

    # analogous to og imgs
    og_files = get_files_in_folder(path_to_folder=og_img_folder,
                                   extension=masks_img_extension)

    # analogous to phase imgs
    phase_files = get_files_in_folder(path_to_folder=phase_img_folder,
                                      extension=masks_img_extension)

    masks_files_num = len(mask_files)

    # create empty list to hold the dfs
    dfs_list = []

    # iterating over mask files
    for file_index, mask_file in enumerate(mask_files, 1):
        # printing progress message
        base_string = 'generating segmentation df #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=masks_files_num)

        # getting respective og img
        og_img_file = og_files[file_index - 1]

        # getting respective phase img
        phase_file = phase_files[file_index - 1]

        # getting current image mask input/output paths
        mask_input_path = join(masks_input_folder,
                               mask_file)

        # analogous getting current og image input/output paths
        og_img_input_path = join(og_img_folder,
                                 og_img_file
                                 )

        # analogous getting current og image input/output paths
        phase_img_input_path = join(phase_img_folder,
                                    phase_file
                                    )
        # get image contour df and respective overlay-ed imgs
        image_df = make_image_contours_df(mask_name=mask_file,
                                          mask_path=mask_input_path,
                                          og_img_path=og_img_input_path,
                                          p_img_path=phase_img_input_path,
                                          overlays_output_folder=overlays_output_folder
                                          )

        # append curr img df to dir df
        dfs_list.append(image_df)

    # concatenating "dfs" from dfs lists into
    # a pandas dataframe
    contour_df = concat(dfs_list, ignore_index=True)

    # create the path to save the output path
    output_path = join(csv_output_folder,
                       'contours_df.pickle')

    # saving df
    contour_df.to_pickle(output_path)

    # printing execution message
    print(f'output saved to {csv_output_folder}')
    print('analysis complete!')

    return


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'generate segmentation df module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # grayscale input folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing original images')
    # phase input folder param
    parser.add_argument('-p', '--phase-images-folder',
                        dest='phase_folder',
                        required=True,
                        help='defines path to folder containing phase images')

    # input masks folder param
    parser.add_argument('-m', '--masks-folder',
                        dest='masks_folder',
                        required=True,
                        help='defines path to folder containing cellpose masks outputs')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # csv output folder param
    parser.add_argument('-co', '--csv_output_folder',
                        dest='csv_output_folder',
                        required=True,
                        help='defines path to output folder (csv)')

    # overlays output folder param
    parser.add_argument('-oo', '--overlays_output_folder',
                        dest='overlays_output_folder',
                        required=True,
                        help='defines path to output folder (overlays)')

    # # overlays output folder param
    # parser.add_argument('-st', '--segmentation_type',
    #                     dest='segmentation_type',
    #                     required=True,
    #                     help='defines which type of segmentation it is. "nuc" or "cyto"')

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

    # getting masks folder
    masks_folder = args_dict['masks_folder']

    # getting images folder
    images_folder = args_dict['images_folder']

    # phase images folder
    phase_folder = args_dict['phase_folder']

    # getting images extension
    images_extension = args_dict['images_extension']

    # getting csv output path
    csv_output_folder = args_dict['csv_output_folder']

    # getting overlays output path
    overlays_output_folder = args_dict['overlays_output_folder']

    # # getting type of segmentation
    # segmentation_type = args_dict['segmentation_type']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to get the segmentation df for a folder
    make_folder_contours_df(masks_input_folder=masks_folder,
                            og_img_folder=images_folder,
                            phase_img_folder=phase_folder,
                            masks_img_extension=images_extension,
                            csv_output_folder=csv_output_folder,
                            overlays_output_folder=overlays_output_folder,
                            )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
