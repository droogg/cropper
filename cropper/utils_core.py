import numpy as np
from skimage.io import imread
import cv2
import os
from os.path import normpath, join, dirname, basename
import random
import torch
from PIL import Image
import math
from cropper.utils_img import *
Image.MAX_IMAGE_PIXELS = None

def img_tile_parameters(path_img: str, req_size: tuple, overlap: int) -> tuple:
    img_init = imread(path_img)
    img_height = img_init.shape[0]
    img_width = img_init.shape[1]
    # compute number of tiles
    num_img_vertical = math.ceil((math.ceil(img_height / req_size[0]) * overlap + img_height) / req_size[0])
    num_img_horizontal = math.ceil((math.ceil(img_width / req_size[1]) * overlap + img_width) / req_size[1])
    # compute total number of tiles
    num_tiles = num_img_vertical * num_img_horizontal
    return img_init, img_height, img_width, num_img_vertical, num_img_horizontal, num_tiles


def create_img_fullpath(path_img: str, destination_path: str) -> tuple:
    if destination_path is None or not isinstance(destination_path, str):
        img_fullpath = create_crop_folder(path_img).split('.')[0]
        if not isinstance(destination_path, str) and destination_path is not None:
            print(f'A value passed as a destination directory {destination_path} is not a directory. ' \
                  f'The destination path has been changed to the {dirname(img_fullpath)}.')
        destination_path = dirname(img_fullpath)
    else:
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        img_fullpath = normpath(join(destination_path, basename(path_img).split('.')[0]))
    return img_fullpath, destination_path


def create_crop_folder(path_img: str) -> str:
    ''' create a folder for cut images and return a new image address '''
    dir_name = dirname(path_img)
    img_name = basename(path_img)
    crop_folder_name = join(dir_name, 'crops')
    if not os.path.exists(crop_folder_name):
        os.makedirs(crop_folder_name)
        print(f'New folder "{crop_folder_name}" created for tiles image stores')
    img_fullpath = norm_path(crop_folder_name) + img_name
    return img_fullpath


def folder_creater(path, annotation_path, motive = 'annotation'):
    if annotation_path is None or not isinstance(annotation_path, str):
        pr = normpath(join(path, motive))
        if not os.path.exists(pr):
            os.makedirs(pr)
            print(f'New folder "{pr}" created for {motive} txt file stores')
        if not isinstance(annotation_path, str) and annotation_path is not None:
            print(f'A value passed as a {motive} directory {annotation_path} is not a directory. ' \
                  f'The destination path has been changed to the "{pr}".')
    #         else: print(f'New folder "{pr}" created for annotation txt file stores')
    else:
        if not os.path.exists(annotation_path):
            os.makedirs(annotation_path)
        pr = annotation_path
    return pr


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def xywh2xyxy(box):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    if isinstance(box, torch.Tensor):
        x, y, w, h = box.t()
        return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).t()
    else:  # numpy
        x, y, w, h = box.T
        return np.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).T