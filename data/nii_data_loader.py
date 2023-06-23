import os
import os.path

import SimpleITK as sitk
import numpy as np


IMG_EXTENSIONS = ['.nii.gz']


def is_nii_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()  # change upper case to lower case
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    photoClasses = [d for d in os.listdir(
        dir) if os.path.isfile(os.path.join(dir, d))]
    photoClasses.sort()
    photo_class_to_idx = {photoClasses[i]: i for i in range(len(photoClasses))}
    return photoClasses, photo_class_to_idx


def make_dataset(dir, photo_class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isfile(d):
            continue
        path = d
        item = (path, photo_class_to_idx[target])
        images.append(item)

    return images


def collect_nii_path(path):
    # walk all the .nii files in path
    all_file_list = []
    gci(path, all_file_list)
    all_file_list.append(path)

    return all_file_list


def gci(filepath, all_file_list):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            all_file_list.append(fi_d)
            gci(fi_d, all_file_list)


def nii_slides_loader(nii_file_path, num, transform=None):
    item = sitk.ReadImage(nii_file_path)
    nii_slides = sitk.GetArrayFromImage(item)
    if transform is not None:
        nii_slides = transform(nii_slides)
    return nii_slides[num, :, :]


def matrix_resize(filein, sacle_size, crop_size, random_crop_para):
    # TODO:
    temp = np.reshape(filein, [sacle_size, sacle_size])


def normalize_nii(mrnp):
    matLPET = mrnp / mrnp.max() * 2.0 - 1

    return matLPET


def load_set(path):
    classes, class_to_idx = find_classes(path)
    loaded_set = make_dataset(path, class_to_idx)
    if len(loaded_set) == 0:
        raise (RuntimeError("Found 0 images in subfolders of: " + path + "\n"
                                                                         "Supported image extensions are: " + ",".join(
            IMG_EXTENSIONS)))
    return loaded_set


def seg_transform(seg):
    seg = np.where(seg > 0, 1, np.finfo(float).eps)
    return seg