import numpy as np
import random
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2
from os.path import normpath, join, dirname, basename
import os
from PIL import Image
from cropper.utils_img import *
from cropper.utils_core import *
Image.MAX_IMAGE_PIXELS = None


def tilinig_image(path_img: str, req_size: tuple, overlap: int, destination_path: str = None,
                  annotation_prefix: str = None) -> str:
    '''
    Cut the image with set size in req_size. Cutting only one image pointed to in {path_img}.
    To process all images in dirname(path_img) run to loop on get_sorted_img_path(path, extension) from utils_img.py:

        path_img_list = get_sorted_img_path(path, extension)
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
    img_name = basename(path_img)
    img_dir = dirname(path_img)
    new_img_dir = folder_creater(img_dir, destination_path, motive = 'crops')
    new_img_dir_crop = folder_creater(new_img_dir, norm_path(new_img_dir) + 'crops_img', motive='crops_img')

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
                new_img_dir_annotation = folder_creater(new_img_dir, norm_path(new_img_dir)+'annotation', motive = 'annotation')
                img_path_4_save = norm_path(new_img_dir_annotation) + img_name.split(annotation_prefix)[0]+ \
                    f'_CROP_{i_save}' + annotation_prefix + '.png'
                cv2.imwrite(img_path_4_save, img)
            else:
                img_path_4_save = norm_path(new_img_dir_crop) + img_name.split('.')[0]+ f'_CROP_{i_save}.png'
                cv2.imwrite(img_path_4_save, img)
            init_point_y += req_size[0]  # slide initial point on y-axes for next image
            i_save += 1
        init_point_x += req_size[1]  # slide initial point on y-axes for next image
    assert (i_save - 1) == num_tiles, f'The actual number of tiles in the image: {i_save - 1}, ' \
                                      f'does not match the calculated number of tiles that ' \
                                      f'should have been the result: {num_tiles}'
    return new_img_dir, new_img_dir_crop


def create_yolo_annotation(img_couple_list: list, path: str, annotation_path: str = None, plot_boundingbox: int = 0):
    '''
    Create yolo format annotation for all image in {path}
    '''
    path = norm_path(path)
    # crate folder if folder don't exist
    annotation_path = folder_creater(path, annotation_path, motive = 'annotation_txt')

    i_plot = 1
    for img_name, annotation_name in img_couple_list:
        img = imread(join(path,'crops_img', img_name))
        annotation = imread(join(path, 'annotation' ,annotation_name))
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


def split_dataset(path: str, extension: 'list of file extensions', train_part = 0.8, shuffle: bool = False,
                  seed: 'int or None' = None, save_result = True, save_path: str = None, out_prefix_txt: str = 'custom_dataset'):
    '''
    Split dataset and get two text file with image path listing (test and train subdataset)
    :param path: str - path to image folder
    :param extension: list - list of file extension, example: ['png', 'jpg']
    :param train_part: float - length of train subdataset
    :param shuffle: bool - take data in sorted order or shuffle
    :param seed: int or None - mix in a predefined way (seed of random module)
    :param save_result: should text files with image links be saved
    :param out_prefix_txt: str - txt file title prefix
    :return: None
    '''
    path = norm_path(path)
    if train_part > 1.0 or train_part < 0:
        print("Train dataset cannot be more than 100% of the data or be negative. Set train_part values in range [0., "
              "1.]")
        print("Length of train part changed on 80% of data")
        train_part = 0.8
    path_img_list = get_sorted_img_path(path, extension)
    if shuffle:
        random.seed(seed)
        random.shuffle(path_img_list)
        random.seed()
    list_train, list_val = path_img_list[:(int(train_part * len(path_img_list)))], path_img_list[(int(train_part * len(path_img_list))):]
    print('количество изображений в train: {}, количество изображений в validation: {}'.format(len(list_train),
                                                                                               len(list_val)))
    list_train.sort(key= natural_keys)
    list_val.sort(key= natural_keys)
    assert len(path_img_list) == (len(list_train) + len(
        list_val)), 'Количество исходных изображений не соответствует сумме разбитых на две части изображений'
    if save_result:
        path = os.path.abspath(os.path.join(os.path.dirname(path),".."))
        save_path = folder_creater(path, save_path, motive='img_link_list')
        for postfix, list_data in zip(['train', 'test'], [list_train, list_val]):
            with open(normpath(join(save_path, f'{out_prefix_txt}_{postfix}.txt')), 'w') as filehandle:
                for listitem in list_data:
                    filehandle.write('%s\n' % listitem)