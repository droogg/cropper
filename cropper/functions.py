import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2
import os
from os.path import normpath, join, dirname, basename
import random
import torch
from PIL import Image
import math
from cropper.utils import *
Image.MAX_IMAGE_PIXELS = None


def tilinig_image(path_img: str, req_size: tuple, overlap: int, destination_path: str = None,
                  annotation_prefix: str = None) -> str:
    '''
    Cut the image with set size in req_size. Cutting only one image pointed to in {path_img}.
    To process all images in dirname(path_img) run to loop on get_sorted_img_path(path, extention) from utils.py:

        path_img_list = get_sorted_img_path(path, extention)
        for path_img in path_img_list:
            path_crop = tilinig_image(path_img, req_size, overlap, annotation_prefix)

    '''
    # check is an annotation image, set init parameters
    annotation_img, init_point_x, i_save = False, 0, 1
    if not annotation_prefix is None:
        annotation_img = annotation_prefix in path_img
    img_init, img_height, img_width, num_img_vertical, num_img_horizontal, num_tiles = img_tile_parameters(path_img,
                                                                                                           req_size, overlap)
    # build path for save crop image
    img_fullpath, destination_path = create_img_fullpath(path_img, destination_path)
    # 2 loop: up-level loop on x-axes (horizontal), nested loop on y-axes (vertical)
    for i_tmp1 in range(num_img_horizontal):
        init_point_y = 0  # set init parameter for 2nd loop
        if init_point_x >= overlap:
            init_point_x -= overlap  # make overlap in crop image on y-axes
        for i_tmp2 in range(num_img_vertical):
            if init_point_y >= overlap:
                init_point_y -= overlap  # make overlap in crop image on y-axes
            img = img_init[init_point_y:(init_point_y + req_size[0]), init_point_x:(init_point_x + req_size[1])]
            # add padding if img shape less then 1024x1024x3
            if np.delete(img.shape, 2).tolist() != np.array(req_size).tolist():
                img_height, img_width = img.shape[0], img.shape[1]
                img0 = np.zeros((req_size[0], req_size[1], img_init.shape[2]))
                img0[0:img_height, 0:img_width, :] = img
                img = img0
            if annotation_img:
                cv2.imwrite(img_fullpath.split(annotation_prefix)[0] + f'_CROP_{i_save}' + annotation_prefix + '.png',
                            img)
            else:
                cv2.imwrite(img_fullpath + f'_CROP_{i_save}.png', img)
            init_point_y += req_size[0]  # slide initial point on y-axes for next image
            i_save += 1
        init_point_x += req_size[1]  # slide initial point on y-axes for next image
    assert (i_save - 1) == num_tiles, f'The actual number of tiles in the image: {i_save - 1}, ' \
                                      f'does not match the calculated number of tiles that ' \
                                      f'should have been the result: {num_tiles}'
    return destination_path


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
        img_fullpath = os.path.normpath(os.path.join(destination_path, basename(path_img).split('.')[0]))
    return img_fullpath, destination_path


def create_crop_folder(path_img: str) -> str:
    dir_name = dirname(path_img)
    img_name = basename(path_img)
    crop_folder_name = join(dir_name, 'crops')
    if not os.path.exists(crop_folder_name):
        os.makedirs(crop_folder_name)
        print(f'New folder "{crop_folder_name}" created for tiles image stores')
    img_fullpath = norm_path(crop_folder_name) + img_name
    return img_fullpath


def create_yolo_annotation(img_couple_list: list, path: str, annotation_path: str = None, plot_boundingbox: int = 0):
    '''
    Create yolo format annotation for all image in {path}
    '''
    path = norm_path(path)
    # crate folder if folder don't exist
    annotation_path = folder_creater(path, annotation_path)

    i_plot = 1
    for img_name, annotation_name in img_couple_list:
        img = imread(path + img_name)
        annotation = imread(path + annotation_name)
        Yp, Xp = np.nonzero(annotation[:, :, 3])

        img_height = img.shape[0]  # height img
        img_width = img.shape[1]  # width img

        x = Xp.reshape(-1, 1) / img_width
        y = Yp.reshape(-1, 1) / img_height
        width = np.array([35] * len(Xp)).reshape(-1, 1) / img_width
        height = np.array([35] * len(Xp)).reshape(-1, 1) / img_height
        y_for_annotation = np.hstack((x, y, width, height))
        y_for_vizual = np.hstack((Xp.reshape(-1, 1), Yp.reshape(-1, 1), np.array([35] * len(Xp)).reshape(-1, 1),
                                  np.array([35] * len(Xp)).reshape(-1, 1)))
        y_viz = xywh2xyxy(y_for_vizual)

        if plot_boundingbox > i_plot:
            img_l = img.copy()
            for i in range(len(Xp)):
                plot_one_box(y_viz[i], img_l)
            plt.figure(i_plot)
            plt.imshow(img_l)
            plt.scatter(Xp, Yp, 5, 'r')
            plt.pause(0.0005)
            plt.show()
            i_plot += 1
        y_list = y_for_annotation.tolist()
        for i in range(len(y_list)):
            y_list[i].insert(0, 0)
        y_str = str(y_list).strip('[]').replace(',', '').replace('[', '')
        y_list_of_str = y_str.split('] ')
        img_name_s = img_name.split('.')[0]
        with open(normpath(join(annotation_path, '{}.txt'.format(img_name_s))), 'w') as filehandle:
            for listitem in y_list_of_str:
                filehandle.write('%s\n' % listitem)


def folder_creater(path, annotation_path):
    if annotation_path is None or not isinstance(annotation_path, str):
        pr = normpath(join(path, 'annotation'))
        if not os.path.exists(pr):
            os.makedirs(pr)
            print(f'New folder "{pr}" created for annotation txt file stores')
        if not isinstance(annotation_path, str) and annotation_path is not None:
            print(f'A value passed as a annotation directory {annotation_path} is not a directory. ' \
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