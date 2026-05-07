# Supplementary Code — HetSenCellDB

Reproducibility package for the paper
"HetSenCellDB: A Dataset of Morphologically Heterogeneous Cell Populations for Segmentation and Classification of Senescent Phenotypes"
Submitted to NeurIPS 2026.

## Requirements 

Install the environment for requirements with:

`conda env create -f environment.yml`

This installs Python 3.11 with all required packages (numpy, pandas, open-cv, skimage,...) as needed.

## Step by step data handling

### Image Processing

To generate Normalized and Equalized images, run:

`python3 -m src.imaging.enhance_img -i <FOLDER_INPUT_PATH> -o <FOLDER_OUTPUT_PATH>`

where the input path being the path to the folder with images to be normalized. This was ran twice, once for cytoplasm and one for nuclei grayscale channels.

### Generation of Tabular Data 

To generate the table of features, run:

`python3 -m src.tabular.generate_segmentation_df -i <GRAYSCALE_CHANNEL_FOLDER_INPUT_PATH> -p <RGB_BRIGHTFIELD_FOLDER_INPUT_PATH> -m <MASKS_FOLDER_INPUT_PATH> -co <TABLE_FOLDER_OUTPUT_PATH> -oo <OVERLAYS_FOLDER_OUTPUT_PATH>`

This needs to be ran for cytoplasm and nuclei separately.

You can either stop at having 2 separate datasets or run:

`python3 -m src.tabular.generate_cytnuc_df -cf <GRAYSCALE_CYTOPLASM_OVERLAYS_FOLDER_INPUT_PATH> -nf <GRAYSCALE_NUCLEI_OVERLAYS_FOLDER_INPUT_PATH> -cic <CYTOPLASM_TABLE_INPUT_PATH> -nic <NUCLEI_TABLE_INPUT_PATH> -co <TABLE_FOLDER_OUTPUT_PATH> -oo <OVERLAYS_FOLDER_OUTPUT_PATH>`

Then, you'll have a merged overlay of the cytoplasm and nuclei, and a joined table of the structures. It should be notated that multinucleation occurs and there might be repeated cytoplasm structures. The only instances to be maintained in this dataset are those who have both nucleus and cytoplasm information, thus, no N/A exists by cytoplasm/nucleus mismatch.

## Label attribution

For the labeling process, you should run:

`python3 -m src.tabular.filter_df_by_df_w_label -i <DATAFRAME_RESULT_PATH> -l <HUMAN_LABELS_CSV_PATH> -o <DATAFRAME_OUTPUT_PATH>`

Specific settings of accepted label columns are directly inside the code, please check the column names of labels when running this script.

### Generation of crops

To generate the crops, run:

`python3 -m src.imaging.generate_crops -c <GRAYSCALE_CYTOPLASM_FOLDER_INPUT_PATH> -n <GRAYSCALE_NUCLEI_FOLDER_INPUT_PATH> -p <RGB_BRIGHTFIELD_FOLDER_INPUT_PATH> -d <DATAFRAME_RESULT_PATH> -co <GRAYSCALE_CYTOPLASM_CROPS_OUTPUT_FOLDER_PATH> -no <GRAYSCALE_NUCLEI_CROPS_OUTPUT_FOLDER_PATH> -po <RGB_BRIGHTFIELD_CROPS_OUTPUT_FOLDER_PATH>`

Lastly, to move the crops to folders containing label in directory names, run:

`python3 -m src.imaging.sort_crop_by_class -i <CROPS_FOLDER_INPUT_PATH> -d <DATAFRAME_RESULT_PATH> -cl <STR_OF_CLASS_LIST> -o <OUTPUT_FOLDER_PATH>`

