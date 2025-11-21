import os
import sys
import yaml
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.img_processing.enhance_img import enhance_dir_imgs
from src.img_processing.histogram_matching import histogram_matching
from src.img_processing.obj_connector import unbinarize_dir_imgs as obj_connector
from src.generators.generate_binary_masks import generate_binary_masks
from src.generators.generate_segmentation_df import make_folder_contours_df as generate_segmentation_dataframe
from src.utils.join_tx_dfs import make_tx_output

class CustomFormatter(logging.Formatter):
    green = "\033[32m"
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

logger = logging.getLogger()
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
logger.setLevel(logging.INFO)

def load_config(config_path: str) -> dict:
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    logger.info("Getting configuration for the project...")
    config = load_config(config_path='config.yaml')
    logger.info("Configuration loaded successfully.")

    base_input_folder = config['base_input_folder']
    base_output_folder = config['base_output_folder']

    logger.info("Starting image enhancement process...")

    try:
        params = config['enhance_img']

        if params['run']: 
            input_folder = os.path.join(base_input_folder, params['input_folder'])
            output_folder = os.path.join(base_output_folder, params['output_folder'])

            os.makedirs(output_folder, exist_ok=True)

            logger.info("Enhancing images from %s and saving to %s", base_input_folder, output_folder)

            enhance_dir_imgs(input_folder=input_folder,
                            output_folder=output_folder,
                            img_extension=params['images_extension'],
                            clip_limit=params['clip_limit'],
                            tile_size=params['tile_size'])

            logger.info("Image enhancement process completed.")
        
        else:
            logger.error("Skipped Image enhancement process")
        
    except Exception as e:
        logger.info("An error occurred: %s", e)
    
    logger.info("Starting histogram matching process...")
    try:
        params = config['histogram_matching']
        if params['run']: 
            input_folder = os.path.join(base_input_folder, params['input_folder'])
            output_folder = os.path.join(base_output_folder, params['output_folder'])
            reference_path = params['reference_image'] +  params['images_extension']
            os.makedirs(output_folder, exist_ok=True)

            logger.info("Matching histograms of images from %s to reference image %s and saving to %s", input_folder, reference_path, output_folder)

            histogram_matching(folder=input_folder,
                            reference_path=reference_path,
                            output_folder=output_folder)

            logger.info("Histogram matching process completed.")
        
        else:
            logger.info("Skipped Histogram matching process")
    
    except Exception as e:
        logger.error("An error occurred: %s", e)
    
    logger.info("Starting binary mask generation process...")

    try:
        params = config['binary_masks']
        if params['run']: 
            input_folder = os.path.join(base_output_folder, params['input_folder'])
            output_folder = os.path.join(base_output_folder, params['output_folder'])
            os.makedirs(output_folder, exist_ok=True)

            logger.info("Generating binary masks from images in %s and saving to %s", input_folder, output_folder)

            generate_binary_masks(input_folder=input_folder,
                                images_extension=params['images_extension'],
                                lower_threshold=params['lower_threshold'],
                                upper_threshold=params['upper_threshold'],
                                min_size=params['min_size'],
                                max_size=params['max_size'],
                                kernel_size=params['kernel_size'],
                                output_folder=output_folder
                                )

            logger.info("Binary mask generation process completed.")
        
        else:
            logger.info("Skipped Binary mask generation process")
    
    except Exception as e:
        logger.error("An error occurred: %s", e)
    
    logger.info("Starting object connection process...")
    try:
        params = config['object_connector']
        if params['run']: 
            input_folder = os.path.join(base_output_folder, params['input_folder'])
            output_folder = os.path.join(base_output_folder, params['output_folder'])
            os.makedirs(output_folder, exist_ok=True)

            logger.info("Connecting objects in images from %s and saving to %s", input_folder, output_folder)

            obj_connector(input_folder=input_folder,
                        output_folder=output_folder,
                        img_extension=params['images_extension'])

            logger.info("Object connection process completed.")
        
        else:
            logger.info("Skipped Object connection process")
    
    except Exception as e:
        logger.error("An error occurred: %s", e)
    
    logger.info("Starting segmentation dataframe generation process...")

    try:
        params = config['segmentation']
        if params['run']: 
            images_folder = os.path.join(base_output_folder, params['images_folder'])
            phase_images_folder = os.path.join(base_output_folder, params['phase_images_folder'])
            masks_folder = os.path.join(base_output_folder, params['masks_folder'])
            output_csv = os.path.join(base_output_folder, params['output_csv'])
            overlay_folder = os.path.join(base_output_folder, params['overlay_folder'])
            os.makedirs(output_csv, exist_ok=True)
            os.makedirs(overlay_folder, exist_ok=True)

            logger.info("Generating segmentation dataframe from images in %s, phase images in %s, and masks in %s. Saving CSV to %s and overlays to %s", images_folder, phase_images_folder, masks_folder, output_csv, overlay_folder)

            generate_segmentation_dataframe(og_img_folder=images_folder,
                                            phase_img_folder=phase_images_folder,
                                            masks_input_folder=masks_folder,
                                            masks_img_extension=params['images_extension'],
                                            csv_output_folder=output_csv,
                                            overlays_output_folder=overlay_folder)

            logger.info("Segmentation dataframe generation process completed.")
        
        else:
            logger.info("Skipped Segmentation dataframe generation process")
        
    except Exception as e:
        logger.error("An error occurred: %s", e)
    
    logger.info("Starting transaction dataframe joining process...")

    try:
        params = config['join_tx_df']
        if params['run']: 
            drug_df_path = params['drug_df_path']
            ctr_df_path = params['ctr_df_path']
            drug_label = params['drug_label']
            ctr_label = params['ctr_label']
            output_folder = os.path.join(base_output_folder, params['output_folder'])
            os.makedirs(output_folder, exist_ok=True)

            logger.info("Joining transaction dataframes from %s and %s, labeling drug as %s and control as %s. Saving to %s", drug_df_path, ctr_df_path, drug_label, ctr_label, output_folder)

            make_tx_output(drug_csv_input_path=drug_df_path,
                           ctr_csv_input_path=ctr_df_path,
                           label_01=drug_label,
                           label_02=ctr_label,
                           output_folder=output_folder
                           )

            logger.info("Transaction dataframe joining process completed.")
        
        else:
            logger.info("Skipped Transaction dataframe joining process")
    
    except Exception as e:
        logger.error("An error occurred: %s", e)
    
    logger.info("All processes completed.")

if __name__ == "__main__":
    main()