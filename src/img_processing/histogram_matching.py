import os
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
import tifffile as tiff
from skimage import io

def histogram_matching(folder, reference_path, output_folder) -> None:
    os.makedirs(output_folder, exist_ok=True)

    reference = io.imread(reference_path)

    for image_name in os.listdir(folder):
        image = io.imread(os.path.join(folder, image_name))
        matched = match_histograms(image, reference, channel_axis=-1)
        output_img = os.path.join(output_folder, image_name)

        tiff.imwrite(output_img, matched.astype(image.dtype))

if __name__ == "__main__":
    histogram_matching()