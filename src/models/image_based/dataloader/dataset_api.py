# imports
import os
import random
from collections import defaultdict
import tensorflow as tf
import tifffile
import numpy as np
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import get_files_in_folder
from src.utils.aux_funcs import print_execution_parameters
###############################################################################

# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'generate data loader module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # grayscale input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help='Root directory with class subfolders')

    # grayscale input folder param
    parser.add_argument('-b', '--batch-size',
                        dest='batch_size',
                        type=int,
                        default=32,
                        required=True,
                        help='defines batch size to train model')

    # grayscale input folder param
    parser.add_argument('-s', '--image-size',
                        dest='image_size',
                        required=True,
                        help='defines size of image to be resized at (image-size, image-size)')

    # segmentation dataframe input folder param
    parser.add_argument('-c', '--num-channels',
                        dest='num_channels',
                        default=1,
                        type=int,
                        help='Expected number of channels (e.g. 1 or 3)')

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

##################################################################################
# data loader functions

def collect_files(data_dir:str):

    # iterate through class folders names
    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    # getting folder name, i.e, class names
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # create lists to store file paths and respective labels
    file_paths = []
    labels = []

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for file_path in os.listdir(class_dir):
            if file_path.lower().endswith(".tif"):
                file_paths.append(os.path.join(class_dir, file_path))
                labels.append(class_to_idx[class_name])

    return file_paths, labels, class_names


def load_tiff_py(path):
    img = tifffile.imread(path.decode("utf-8"))
    img = img.astype(np.float32)

    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    return img


def load_tiff_tf(path, label):
    img = tf.py_function(
        func=load_tiff_py,
        inp=[path],
        Tout=tf.float32
    )

    img.set_shape([None, None, None])  # flexible, fix later if needed
    return img, label


def preprocess(img, label, img_size, num_channels):
    # Ensure channel dim
    if tf.rank(img) == 2:
        img = tf.expand_dims(img, -1)

    # Enforce channel count if needed
    if num_channels is not None:
        img = img[..., :num_channels]

    if img_size is not None:
        img = tf.image.resize(img, (img_size, img_size))

    return img, label


def build_dataset(args):
    file_paths, labels, class_names = collect_files(args.data_dir)

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # Load + normalize TIFFs
    ds = ds.map(
        load_tiff_tf,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shape, channels, resize
    ds = ds.map(
        lambda x, y: preprocess(x, y, args.img_size, args.num_channels),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.shuffle(args.shuffle_buffer)
    ds = ds.batch(args.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, class_names


def split_samples(
    samples,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
):
    assert train_ratio + val_ratio < 1.0

    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    return train, val, test


def stratified_split(
    samples,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
):
    """
    Deterministic stratified split into train / val / test.
    """
    assert train_ratio + val_ratio < 1.0

    rng = random.Random(seed)

    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    train, val, test = [], [], []

    for cls, cls_samples in by_class.items():
        rng.shuffle(cls_samples)

        n = len(cls_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(cls_samples[:n_train])
        val.extend(cls_samples[n_train:n_train + n_val])
        test.extend(cls_samples[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def stratified_kfold_split(
    samples,
    k=5,
    seed=42
):
    """
    Returns a list of (train_samples, val_samples) for each fold.
    """
    rng = random.Random(seed)

    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    # shuffle within each class
    for cls_samples in by_class.values():
        rng.shuffle(cls_samples)

    # create k buckets per class
    class_folds = defaultdict(list)
    for cls, cls_samples in by_class.items():
        for i, sample in enumerate(cls_samples):
            class_folds[i % k].append(sample)

    folds = []
    for i in range(k):
        val = class_folds[i]
        train = []

        for j in range(k):
            if j != i:
                train.extend(class_folds[j])

        folds.append((train, val))

    return folds


def make_tf_dataset(
    samples,
    batch_size,
    img_size=None,
    num_channels=None,
    shuffle=False
):
    paths, labels = zip(*samples)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    ds = ds.map(load_tiff_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda x, y: preprocess(x, y, img_size, num_channels),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        ds = ds.shuffle(len(samples))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_benchmark_datasets(
    data_dir,
    batch_size,
    img_size=None,
    num_channels=None,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
):
    samples, class_names = collect_files(data_dir)

    train_s, val_s, test_s = split_samples(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )

    train_ds = make_tf_dataset(
        train_s,
        batch_size,
        img_size,
        num_channels,
        shuffle=True
    )

    val_ds = make_tf_dataset(
        val_s,
        batch_size,
        img_size,
        num_channels,
        shuffle=False
    )

    test_ds = make_tf_dataset(
        test_s,
        batch_size,
        img_size,
        num_channels,
        shuffle=False
    )

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "class_names": class_names,
        "num_classes": len(class_names)
    }


def build_stratified_benchmark(
    data_dir,
    batch_size,
    img_size=None,
    num_channels=None,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
):
    samples, class_names = collect_files(data_dir)

    train_s, val_s, test_s = stratified_split(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )

    return {
        "train": make_tf_dataset(train_s, batch_size, img_size, num_channels, shuffle=True),
        "val": make_tf_dataset(val_s, batch_size, img_size, num_channels),
        "test": make_tf_dataset(test_s, batch_size, img_size, num_channels),
        "class_names": class_names,
        "num_classes": len(class_names)
    }


def build_stratified_kfold_benchmark(
    data_dir,
    batch_size,
    k=5,
    img_size=None,
    num_channels=None,
    seed=42
):
    samples, class_names = collect_files(data_dir)

    folds = stratified_kfold_split(samples, k=k, seed=seed)

    tf_folds = []
    for fold_idx, (train_s, val_s) in enumerate(folds):
        tf_folds.append({
            "fold": fold_idx,
            "train": make_tf_dataset(train_s, batch_size, img_size, num_channels, shuffle=True),
            "val": make_tf_dataset(val_s, batch_size, img_size, num_channels),
        })

    return {
        "folds": tf_folds,
        "class_names": class_names,
        "num_classes": len(class_names)
    }

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
    args = parse_args()

    dataset, class_names = build_dataset(args)

    print("Classes:", class_names)

    # sanity check
    for imgs, labels in dataset.take(1):
        print("Batch shape:", imgs.shape)
        print("Labels:", labels.numpy())


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module