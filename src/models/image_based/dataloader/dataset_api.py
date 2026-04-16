# imports
import os
import random
from collections import defaultdict
import tensorflow as tf
import tifffile
import numpy as np
from argparse import ArgumentParser
from src._execution_formatting import enter_to_continue
from src._execution_formatting import print_progress_message
from src._execution_formatting import get_files_in_folder
from src._execution_formatting import print_execution_parameters
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

def collect_files(data_dir: str, images_extension: str = ".tif"):

    # iterate through class folders names
    class_names = sorted([d for d in os.listdir(data_dir)
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
            if file_path.lower().endswith(images_extension.lower()):
                file_paths.append(os.path.join(class_dir, file_path))
                labels.append(class_to_idx[class_name])

    return file_paths, labels, class_names


def collect_paired_files(data_dir_a: str,
                         data_dir_b: str,
                         images_extension: str = ".tif"):
    class_names_a = {d for d in os.listdir(data_dir_a)
                     if os.path.isdir(os.path.join(data_dir_a, d))}
    class_names_b = {d for d in os.listdir(data_dir_b)
                     if os.path.isdir(os.path.join(data_dir_b, d))}

    class_names = sorted(class_names_a & class_names_b)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    file_paths_a = []
    file_paths_b = []
    labels = []

    for class_name in class_names:
        class_dir_a = os.path.join(data_dir_a, class_name)
        class_dir_b = os.path.join(data_dir_b, class_name)

        files_b = {
            f.lower(): f for f in os.listdir(class_dir_b)
            if f.lower().endswith(images_extension.lower())
        }
        for file_name in os.listdir(class_dir_a):
            if not file_name.lower().endswith(images_extension.lower()):
                continue
            key = file_name.lower()
            if key not in files_b:
                continue
            file_paths_a.append(os.path.join(class_dir_a, file_name))
            file_paths_b.append(os.path.join(class_dir_b, files_b[key]))
            labels.append(class_to_idx[class_name])

    return file_paths_a, file_paths_b, labels, class_names


def check_paired_consistency(data_dir_a: str,
                             data_dir_b: str,
                             images_extension: str = ".tif"):
    class_names_a = {d for d in os.listdir(data_dir_a)
                     if os.path.isdir(os.path.join(data_dir_a, d))}
    class_names_b = {d for d in os.listdir(data_dir_b)
                     if os.path.isdir(os.path.join(data_dir_b, d))}

    only_in_a = sorted(class_names_a - class_names_b)
    only_in_b = sorted(class_names_b - class_names_a)
    common = sorted(class_names_a & class_names_b)

    missing_in_b = {}
    missing_in_a = {}
    for class_name in common:
        class_dir_a = os.path.join(data_dir_a, class_name)
        class_dir_b = os.path.join(data_dir_b, class_name)

        files_a = {f.lower() for f in os.listdir(class_dir_a)
                   if f.lower().endswith(images_extension.lower())}
        files_b = {f.lower() for f in os.listdir(class_dir_b)
                   if f.lower().endswith(images_extension.lower())}

        missing_b = sorted(files_a - files_b)
        missing_a = sorted(files_b - files_a)
        if missing_b:
            missing_in_b[class_name] = missing_b
        if missing_a:
            missing_in_a[class_name] = missing_a

    return {
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "missing_in_b": missing_in_b,
        "missing_in_a": missing_in_a,
    }


def load_tiff_py(path):
    path_str = path.numpy().decode("utf-8")
    img = tifffile.imread(path_str)
    img = img.astype(np.float32)

    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    return img


def load_tiff_pair_py(path_a, path_b):
    path_a_str = path_a.numpy().decode("utf-8")
    path_b_str = path_b.numpy().decode("utf-8")
    img_a = tifffile.imread(path_a_str).astype(np.float32)
    img_b = tifffile.imread(path_b_str).astype(np.float32)

    img_a = (img_a - img_a.min()) / (img_a.max() - img_a.min() + 1e-6)
    img_b = (img_b - img_b.min()) / (img_b.max() - img_b.min() + 1e-6)

    if img_a.ndim == 2:
        img_a = img_a[..., None]
    if img_b.ndim == 2:
        img_b = img_b[..., None]

    img = np.concatenate([img_a, img_b], axis=-1)
    return img.astype(np.float32)


def load_tiff_tf(path, label):
    img = tf.py_function(func=load_tiff_py,
                         inp=[path],
                         Tout=tf.float32
                         )

    img.set_shape([None, None, None])  # flexible, fix later if needed
    return img, label


def load_tiff_pair_tf(path_a, path_b, label):
    img = tf.py_function(func=load_tiff_pair_py,
                         inp=[path_a, path_b],
                         Tout=tf.float32
                         )
    img.set_shape([None, None, None])
    return img, label


def preprocess(img, label, img_size, num_channels):
    # Ensure channel dim
    if tf.rank(img) == 2:
        img = tf.expand_dims(img, -1)
    img = tf.ensure_shape(img, [None, None, None])

    # Enforce channel count if needed
    if num_channels is not None:
        channels = tf.shape(img)[-1]
        if num_channels == 3:
            img = tf.cond(
                tf.equal(channels, 1),
                lambda: tf.repeat(img, 3, axis=2),
                lambda: tf.cond(
                    tf.equal(channels, 2),
                    lambda: tf.concat([img, img[..., :1]], axis=2),
                    lambda: img,
                ),
            )
        elif num_channels > 1:
            img = tf.cond(
                tf.equal(channels, 1),
                lambda: tf.repeat(img, num_channels, axis=2),
                lambda: img,
            )
        img = img[..., :num_channels]

    if img_size is not None:
        if isinstance(img_size, (tuple, list)):
            size = tuple(img_size)
        else:
            size = (img_size, img_size)
        img = tf.image.resize(img, size)

    return img, label


def _get_data_dir(args):
    return getattr(args, "data_dir", None) or getattr(args, "input_folder", None)


def build_dataset(args):
    data_dir = _get_data_dir(args)
    data_dir2 = getattr(args, "data_dir2", None) or getattr(args, "input_folder2", None)
    images_extension = getattr(args, "images_extension", ".tif")
    if data_dir2:
        file_paths_a, file_paths_b, labels, class_names = collect_paired_files(
            data_dir, data_dir2, images_extension=images_extension
        )
        samples = list(zip(file_paths_a, file_paths_b, labels))
        ds = tf.data.Dataset.from_tensor_slices((file_paths_a, file_paths_b, labels))
        ds = ds.map(load_tiff_pair_tf, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        file_paths, labels, class_names = collect_files(
            data_dir, images_extension=images_extension
        )
        samples = list(zip(file_paths, labels))

        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        ds = ds.map(load_tiff_tf, num_parallel_calls=tf.data.AUTOTUNE)

    # Shape, channels, resize
    img_size = getattr(args, "img_size", None) or getattr(args, "image_size", None)
    ds = ds.map(lambda x, y: preprocess(x, y, img_size, args.num_channels),
                num_parallel_calls=tf.data.AUTOTUNE,
                )

    shuffle_buffer = getattr(args, "shuffle_buffer", len(samples))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(args.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, class_names


def split_samples(samples,
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


def _get_sample_label(sample):
    return sample[-1]


def stratified_split(samples,
                     train_ratio:float = 0.7,
                     val_ratio:float = 0.15,
                     seed:int = 42
                     ):
    """
    Deterministic stratified split into train / val / test.
    """
    assert train_ratio + val_ratio < 1.0

    rng = random.Random(seed)

    by_class = defaultdict(list)
    for sample in samples:
        by_class[_get_sample_label(sample)].append(sample)

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


def stratified_kfold_split(samples,
                           k:int = 5,
                           seed:int = 42
                           ):
    """
    Returns a list of (train_samples, val_samples) for each fold.
    """
    rng = random.Random(seed)

    by_class = defaultdict(list)
    for sample in samples:
        by_class[_get_sample_label(sample)].append(sample)

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


def make_tf_dataset(samples,
                    batch_size,
                    img_size=None,
                    num_channels=None,
                    shuffle=False
                    ):
    if len(samples[0]) == 3:
        paths_a, paths_b, labels = zip(*samples)
        ds = tf.data.Dataset.from_tensor_slices(
            (list(paths_a), list(paths_b), list(labels))
        )
        ds = ds.map(load_tiff_pair_tf, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        paths, labels = zip(*samples)
        ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
        ds = ds.map(load_tiff_tf, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(
        lambda x, y: preprocess(x, y, img_size, num_channels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        ds = ds.shuffle(len(samples))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_benchmark_datasets(data_dir: str,
                             batch_size: int,
                             img_size=None,
                             num_channels=None,
                             train_ratio=0.7,
                             val_ratio=0.15,
                             data_dir2=None,
                             images_extension=".tif",
                             seed=42
                             ):
    if data_dir2:
        file_paths_a, file_paths_b, labels, class_names = collect_paired_files(
            data_dir, data_dir2, images_extension=images_extension
        )
        samples = list(zip(file_paths_a, file_paths_b, labels))
    else:
        file_paths, labels, class_names = collect_files(
            data_dir, images_extension=images_extension
        )
        samples = list(zip(file_paths, labels))

    train_s, val_s, test_s = split_samples(samples,
                                           train_ratio=train_ratio,
                                           val_ratio=val_ratio,
                                           seed=seed
                                           )

    train_ds = make_tf_dataset(train_s,
                               batch_size=batch_size,
                               img_size=img_size,
                               num_channels=num_channels,
                               shuffle=True
                               )

    val_ds = make_tf_dataset(val_s,
                             batch_size=batch_size,
                             img_size=img_size,
                             num_channels=num_channels,
                             shuffle=True
                             )

    test_ds = make_tf_dataset(test_s,
                              batch_size=batch_size,
                              img_size=img_size,
                              num_channels=num_channels,
                              shuffle=True
                              )

    return {"train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "class_names": class_names,
            "num_classes": len(class_names)
            }


def build_stratified_benchmark(data_dir,
                               batch_size,
                               img_size=None,
                               num_channels=None,
                               train_ratio=0.7,
                               val_ratio=0.15,
                               data_dir2=None,
                               images_extension=".tif",
                               seed=42
                               ):
    if data_dir2:
        file_paths_a, file_paths_b, labels, class_names = collect_paired_files(
            data_dir, data_dir2, images_extension=images_extension
        )
        samples = list(zip(file_paths_a, file_paths_b, labels))
    else:
        file_paths, labels, class_names = collect_files(
            data_dir, images_extension=images_extension
        )
        samples = list(zip(file_paths, labels))

    train_s, val_s, test_s = stratified_split(samples,
                                              train_ratio=train_ratio,
                                              val_ratio=val_ratio,
                                              seed=seed
                                              )

    return {"train": make_tf_dataset(train_s,
                                     batch_size,
                                     img_size,
                                     num_channels,
                                     shuffle=True),
            "val": make_tf_dataset(val_s,
                                   batch_size,
                                   img_size,
                                   num_channels),
            "test": make_tf_dataset(test_s,
                                    batch_size,
                                    img_size,
                                    num_channels),
            "train_samples": train_s,
            "val_samples": val_s,
            "test_samples": test_s,
            "class_names": class_names,
            "num_classes": len(class_names)
            }


def build_stratified_kfold_benchmark(data_dir,
                                     batch_size,
                                     k=5,
                                     img_size=None,
                                     num_channels=None,
                                     data_dir2=None,
                                     images_extension=".tif",
                                     seed=42
                                     ):
    if data_dir2:
        file_paths_a, file_paths_b, labels, class_names = collect_paired_files(
            data_dir, data_dir2, images_extension=images_extension
        )
        samples = list(zip(file_paths_a, file_paths_b, labels))
    else:
        file_paths, labels, class_names = collect_files(
            data_dir, images_extension=images_extension
        )
        samples = list(zip(file_paths, labels))

    folds = stratified_kfold_split(samples, k=k, seed=seed)

    tf_folds = []
    for fold_idx, (train_s, val_s) in enumerate(folds):
        tf_folds.append({"fold": fold_idx,
                         "train": make_tf_dataset(train_s,
                                                  batch_size=batch_size,
                                                  img_size=img_size,
                                                  num_channels=num_channels,
                                                  shuffle=True
                                                  ),
                         "val": make_tf_dataset(val_s,
                                                batch_size=batch_size,
                                                img_size=img_size,
                                                num_channels=num_channels,
                                                shuffle=True
                                                ),
                         "train_samples": train_s,
                         "val_samples": val_s,
                         })

    return {"folds": tf_folds,
            "class_names": class_names,
            "num_classes": len(class_names)
            }

######################################################################
