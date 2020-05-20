import numpy as np
import random
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2
from random import sample
from shutil import copyfile
from pathlib import PurePath, Path
from os.path import normpath, join, dirname, basename
import os
from PIL import Image
from cropper.utils_img import *
from cropper.utils_core import *
Image.MAX_IMAGE_PIXELS = None


def tilinig_image_and_img_annotation(dir_path: str, path_img: str, req_size: tuple, overlap: int,
                                     scale_crop: bool = True, save_img_trashold: float = 0.3,
                                     destination_path: str = None, annotation_prefix: str = None) -> tuple:
    '''
    Cut the image in accordance with the specified size and the specified overlap. If the cut-off part of the original
    image is smaller than the required size, fill the rest of the image with zeros.
    If scale_crop = true, then if the cropped image is small, then scale it on the larger side to the required size of
    this side. Save only those sliced images that occupy the area from the required size of at least save_img_trashold.

    :param dir_path: str - directory where images and annotations are located
    :param path_img: str - folder where the images are located (may coincide with the dir_path)
    :param req_size: typle of int (height, width) - required crop image size
    :param overlap: int - how much the cropped image overlaps the previous cropped image
    :param scale_crop: bool - whether it is worth enlarging the cropped image along the long side to the required size
                    if the cropped image on both sides is smaller than the required size. The proportion is maintained.
    :param save_img_trashold: float [0,1] - the fraction of the area of the cut out image from the required, less than
                    which the image is not saved
    :param destination_path: str - destination path where clipped images and annotations will be placed
    :param annotation_prefix: str - your annotations should have the same name as the images, but at the end of the
                    name should be added a suffix that is characteristic of all annotations.
    :return: new_img_dir: str - new directory with folders with images and annotations
             new_img_dir_crop: str - folder with cut images
             new_img_dir_annotation: str - folder with cut annotation
             scale_dict: dict - a dictionary containing the names of the cut and scaled images and the values by how much
                            they are scaled.

    Cut the image with set size in req_size. Cutting only one image pointed to in {path_img}.
    To process all images in dirname(path_img) run to loop on get_sorted_img_path(path, extension) from utils_img.py:

        path_img_list = get_sorted_img_path(path, extension)
        for path_img in path_img_list:
            path_crop_dir,_,_,_ = tilinig_image(path_img, req_size, overlap, annotation_prefix)

    '''
    # check is an annotation image, set init parameters
    annotation_img, init_point_x, i_save = False, 0, 1
    if not annotation_prefix is None:
        annotation_img = annotation_prefix in path_img
    img_init, img_height, img_width, num_img_vertical, num_img_horizontal, num_tiles = img_tile_parameters(path_img,
                                                                                                           req_size, overlap)
    # build path for save crop image
    img_name = basename(path_img)
    # img_dir = dirname(path_img)

    new_img_dir = folder_creater(dir_path, destination_path, motive = 'crops')
    new_img_dir_crop = folder_creater(new_img_dir, norm_path(new_img_dir) + 'crops_img', motive='crops_img')
    new_img_dir_annotation = folder_creater(new_img_dir, norm_path(new_img_dir) + 'annotation', motive='annotation')
    # 2 loop: up-level loop on x-axes (horizontal), nested loop on y-axes (vertical)
    scale_dict = {}
    for i_tmp1 in range(num_img_horizontal):
        init_point_y = 0  # set init parameter for 2nd loop
        if init_point_x >= overlap:
            init_point_x -= overlap  # make overlap in crop image on y-axes
        for i_tmp2 in range(num_img_vertical):
            if init_point_y >= overlap:
                init_point_y -= overlap  # make overlap in crop image on y-axes
            img = img_init[init_point_y:(init_point_y + req_size[0]), init_point_x:(init_point_x + req_size[1])]
            # add padding if img shape less then 1024x1024x3
            img_height, img_width = img.shape[0], img.shape[1]
            img_big_s = img_height if img_height > img_width else img_width
            req_s = req_size[0] if img_height > img_width else req_size[1]
            if scale_crop: scale = req_s / img_big_s
            else: scale = 1
            if np.delete(img.shape, 2).tolist() != np.array(req_size).tolist():

                img_area = img_height * img_width * scale * scale
                img_req_area = req_size[0] * req_size[1]
                img_part_area = img_area / img_req_area
                if img_part_area < save_img_trashold:
                    continue

                img_height, img_width = img.shape[0], img.shape[1]
                img0 = np.zeros((req_size[0], req_size[1], img_init.shape[2]))
                # img scale ######################
                if scale_crop:
                    if not annotation_img:
                        in_image_scaled = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                        ys, xs = in_image_scaled.shape[0], in_image_scaled.shape[1]
                        ################
                        img0[0:ys, 0:xs, :] = in_image_scaled
                    else:
                        Yp, Xp = np.nonzero(img[:, :, 3])
                        Yp = (Yp*scale).astype(int)
                        Xp = (Xp*scale).astype(int)
                        img0[Yp,Xp,0] = 255
                        img0[Yp,Xp,3] = 255
                        ann_name = img_name.split(annotation_prefix)[0]+f'_CROP_{i_save}' + annotation_prefix + '.png'
                        scale_dict[ann_name] = scale
                else: img0[0:img_height, 0:img_width, :] = img
                img = img0
            if annotation_img:
                img_path_4_save = norm_path(new_img_dir_annotation) + img_name.split(annotation_prefix)[0]+ \
                    f'_CROP_{i_save}' + annotation_prefix + '.png'
                cv2.imwrite(img_path_4_save, to_BRG(img))
            else:
                img_path_4_save = norm_path(new_img_dir_crop) + img_name.split('.')[0]+ f'_CROP_{i_save}.png'
                cv2.imwrite(img_path_4_save, to_BRG(img))
            init_point_y += req_size[0]  # slide initial point on y-axes for next image
            i_save += 1
        init_point_x += req_size[1]  # slide initial point on y-axes for next image
    # assert (i_save - 1) == num_tiles, f'The actual number of tiles in the image: {i_save - 1}, ' \
    #                                   f'does not match the calculated number of tiles that ' \
    #                                   f'should have been the result: {num_tiles}'
    return new_img_dir, new_img_dir_crop, new_img_dir_annotation, scale_dict


def tilinig_image_and_txt_annotation(dir_path: str, path_img: str, req_size: tuple, overlap: int,
                                     destination_path: str = None, annotations: 'ndarray' = None,
                                     box_trashold: float = 0.3, save_img_trashold: float = 0.3) -> tuple:
    '''
    Cut the image and, in accordance with the cut images, cut the text annotations. Save only those bounding boxes
    whose area from the original area of the bounding boxes is greater than the box_threshold.
    If the cropped image is small, then scale it on the larger side to the required size of
    this side. Save only those sliced images that occupy the area from the required size of at least save_img_trashold.

    :param dir_path: str - directory where images and annotations are located
    :param path_img: str - folder where the images are located (may coincide with the dir_path)
    :param req_size: typle of int (height, width) - required crop image size
    :param overlap: int - how much the cropped image overlaps the previous cropped image
    :param destination_path: str - destination path where clipped images and annotations will be placed
    :param annotations: ndarray - array of annotations [x1,y1, x2, y2] - top left and bottom right angle
    :param box_trashold: float [0,1] - the fraction of the area of the cut out bounding box from the initial, less than
                    which the bounding box is not saved
    :param save_img_trashold: float [0,1] - the fraction of the area of the cut out image from the required, less than
                    which the image is not saved
    :return: new_img_dir: str - new directory with folders with images and annotations
             new_img_dir_crop: str - folder with cut images
             new_ann_dir_crop: str - folder with cut annotations
    '''
    # check is an annotation image, set init parameters
    init_point_x, i_save = 0, 1
    img_init, img_height, img_width, num_img_vertical, num_img_horizontal, num_tiles = img_tile_parameters(path_img,
                                                                                                           req_size,
                                                                                                           overlap)
    # build path for save crop image
    img_name = basename(path_img)
    # img_dir = dirname(path_img)
    new_img_dir = folder_creater(dir_path, destination_path, motive='crops')


    new_img_dir_crop = folder_creater(new_img_dir, norm_path(new_img_dir) + 'crops_img', motive='crops_img')
    new_ann_dir_crop = folder_creater(new_img_dir, norm_path(new_img_dir) + 'txt_annotation', motive='txt_annotation')

    # 2 loop: up-level loop on x-axes (horizontal), nested loop on y-axes (vertical)
    for i_tmp1 in range(num_img_horizontal):
        init_point_y = 0  # set init parameter for 2nd loop
        if init_point_x >= overlap:
            init_point_x -= overlap  # make overlap in crop image on y-axes
        for i_tmp2 in range(num_img_vertical):
            ann = []
            if init_point_y >= overlap:
                init_point_y -= overlap  # make overlap in crop image on y-axes
            img = img_init[init_point_y:(init_point_y + req_size[0]), init_point_x:(init_point_x + req_size[1])]
            cropcoord = (init_point_y, init_point_y + req_size[0], init_point_x,
                         init_point_x + req_size[1])  # (ymin, ymax, xmin, xmax)

            img_height, img_width = img.shape[0], img.shape[1]
            img_big_s = img_height if img_height > img_width else img_width
            req_s = req_size[0] if img_height > img_width else req_size[1]
            # add padding if img shape less then 1024x1024x3
            scale = req_s / img_big_s
            if np.delete(img.shape, 2).tolist() != np.array(req_size).tolist():
                # проверяем какую часть требуемого размера заполняет полученное изображение с учетом его увеличения
                img_area = img_height*img_width*scale*scale
                img_req_area = req_size[0]*req_size[1]
                img_part_area = img_area/img_req_area
                # print(img_area, img_req_area, img_part_area)
                if img_part_area < save_img_trashold:
                    continue
                img0 = np.zeros((req_size[0], req_size[1], img_init.shape[2]))
                # img scale ######################
                in_image_scaled = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                ys, xs = in_image_scaled.shape[0], in_image_scaled.shape[1]
                ################
                img0[0:ys, 0:xs, :] = in_image_scaled
                img = img0
            if annotations.shape[0]:
                for annotation in annotations:
                    x1, y1, x2, y2 = annotation
                    origin_box_area = abs(x1 - x2) * abs(y1 - y2) * scale*scale
                    ymin, ymax, xmin, xmax = cropcoord
                    box_agle_in_crop = {}
                    box_angle_coord = {'top_left': [x1, y1],
                                       'bottom_left': [x1, y2],
                                       'top_right': [x2, y1],
                                       'bottom_right': [x2, y2]}
                    for key in box_angle_coord.keys():
                        if cropcoord[2] < box_angle_coord[key][0] < cropcoord[3] and cropcoord[0] < \
                                box_angle_coord[key][1] < cropcoord[1]:
                            box_agle_in_crop[key] = True
                        else:
                            box_agle_in_crop[key] = False
                    if True in box_agle_in_crop.values():  # если хоть один угол бокса попал на изображение
                        if x1 < xmin: x1 = xmin
                        if x2 > xmax: x2 = xmax
                        if y1 < ymin: y1 = ymin
                        if y2 > ymax: y2 = ymax
                        crop_box_area = abs(x1 - x2) * abs(y1 - y2) * scale*scale
                        box_part = crop_box_area / origin_box_area
                        if box_part >= box_trashold:
                            box_save = True
                            ann_stack = np.array([x1 - xmin, y1 - ymin, x2 - xmin, y2 - ymin]) * scale
                            ann_stack = xyxy2xywh(ann_stack)
                            xnn, ynn, wnn, hnn = ann_stack
                            x = xnn / req_size[1]
                            y = ynn / req_size[0]
                            w = wnn / req_size[1]
                            h = hnn / req_size[0]
                            ann_stack = np.hstack((x, y, w, h))
                            ann.append(ann_stack.tolist())
            else:
                ann = annotations
            img_path_4_save_img = norm_path(new_img_dir_crop) + img_name.split('.')[0] + f'_CROP_{i_save}.png'
            img_path_4_save_ann = norm_path(new_ann_dir_crop) + img_name.split('.')[0] + f'_CROP_{i_save}.txt'
            cv2.imwrite(img_path_4_save_img, to_BRG(img))

            for i in range(len(ann)):
                ann[i].insert(0, 0)
            ann_str = str(ann).strip('[]').replace(',', '').replace('[', '')
            ann_list_of_str = ann_str.split('] ')
            with open(img_path_4_save_ann, 'w') as filehandle:
                for listitem in ann_list_of_str:
                    # filehandle.write('%s\n' % listitem)
                    if listitem == ann_list_of_str[-1]:
                        filehandle.write('%s' % listitem)
                    else:
                        filehandle.write('%s\n' % listitem)

            init_point_y += req_size[0]  # slide initial point on y-axes for next image
            i_save += 1
        init_point_x += req_size[1]  # slide initial point on y-axes for next image
    # assert (i_save - 1) == num_tiles, f'The actual number of tiles in the image: {i_save - 1}, ' \
    #                                   f'does not match the calculated number of tiles that ' \
    #                                   f'should have been the result: {num_tiles}'
    return new_img_dir, new_img_dir_crop, new_ann_dir_crop


def create_yolo_annotation_from_img_annotation(img_couple_list: list, dir_path: str, path_ann: str, path_img: str,
                                               annotation_path: str = None, scale_dict = {}, plot_boundingbox: int = 0):
    '''
    Create yolo format annotation for all image in {path}

    :param img_couple_list: list - list of all image / image-annotation pairs
    :param dir_path: str - directory where images and annotations are located
    :param path_ann: str - folder where the cut image-annotations are located (may coincide with the dir_path)
    :param path_img: str - folder where the cut images are located (may coincide with the dir_path)
    :param annotation_path: str - folder in which text annotations of the yolo format should be saved
    :param scale_dict: dict - a dictionary containing the names of the cut and scaled images and the values by how much
                            they are scaled.
    :param plot_boundingbox: int - number of figures to display - display image with boxes
    :return: annotation_path: str - the folder in which the text annotations of the yolo format were saved
    '''
    dir_path = norm_path(dir_path)
    # crate folder if folder don't exist
    annotation_path = folder_creater(dir_path, annotation_path, motive = 'annotation_txt')
    bb_size = (35,35) # width, hight
    i_plot = 1
    for img_name, annotation_name in img_couple_list:
        if annotation_name in scale_dict:
            scale = scale_dict[annotation_name]
            # print(annotation_name+f' in scale dict : {scale_dict[annotation_name]}')
        else: scale = 1
        img = imread(join(path_img, img_name))
        annotation = imread(join(path_ann, annotation_name))
        Yp, Xp = np.nonzero(annotation[:, :, 3])

        img_height = img.shape[0]  # height img
        img_width = img.shape[1]  # width img

        x = Xp.reshape(-1, 1)/ img_width
        y = Yp.reshape(-1, 1)/ img_height
        width = np.array([int(bb_size[0]*scale)] * len(Xp)).reshape(-1, 1) / img_width
        height = np.array([int(bb_size[1]*scale)] * len(Xp)).reshape(-1, 1) / img_height
        y_for_annotation = np.hstack((x, y, width, height))
        y_for_vizual = np.hstack((Xp.reshape(-1, 1), Yp.reshape(-1, 1), np.array([int(bb_size[0]*scale)] * len(Xp)).reshape(-1, 1),
                                  np.array([int(bb_size[1]*scale)] * len(Xp)).reshape(-1, 1)))
        y_viz = xywh2xyxy(y_for_vizual)

        if plot_boundingbox >= i_plot:
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
                # filehandle.write('%s\n' % listitem)
                if listitem == y_list_of_str[-1]:
                    filehandle.write('%s' % listitem)
                else:
                    filehandle.write('%s\n' % listitem)
    return annotation_path


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


def create_dataset_with_set_empty_samples(path_crop_dir, path_crop_img, path_crop_txtann, extansion,
                                          empty_part, mod = 'silent'):
    '''
    Consider all text annotations in terms of finding the object in the frame. Divide all images into those that have
    objects and no objects. Create a new dataset that will contain the specified percentage of images without objects
    from the number of images with objects.

    :param path_crop_dir: str - directory where images and annotations are located
    :param path_crop_img: str - folder where the images are located
    :param path_crop_txtann: str - folder where the txt annotation are located
    :param extansion: list - list of file extension, example: ['png', 'jpg']
    :param empty_part: float [0,1] - what proportion of images with objects will be images without objects
    :param mod: 'copy' - copy images and annotations to a new directory
                'del' - delete unnecessary files (images and annotations without objects) in the directory where they
                        are located
                'silent' - only display statistical information
    :return: dst_img - the location of the image in a new data set
             dst_ann - the location of the txt annotation in a new data set
    '''
    img_txt_couple = get_img_couple_imgtxt(path_crop_img, path_crop_txtann, extansion)
    dst_img, dst_ann = path_crop_img, path_crop_txtann
    empty_file = []
    not_empty_file = []
    for i in range(len(img_txt_couple)):
        path_img, path_ann = img_txt_couple[i]
        file_size = os.stat(path_ann).st_size
        if not file_size: empty_file.append(img_txt_couple[i])
        else: not_empty_file.append(img_txt_couple[i])
    not_empty_part = int(len(not_empty_file)*empty_part)
    if not_empty_part > len(empty_file): not_empty_part = len(empty_file)
    if mod == 'copy':
        empty_file_req_part = sample(empty_file, not_empty_part)
        dataset = not_empty_file.copy()
        dataset.extend(empty_file_req_part)
        # copy result dataset in new dir
        dst_base = PurePath(path_crop_dir).joinpath('dataset_with_set_emptyfile')
        dst_img = PurePath(dst_base).joinpath('images')
        dst_ann = PurePath(dst_base).joinpath('labels')
        for path_exs in [dst_img, dst_ann]:
            if not Path(path_exs).exists():
                Path(path_exs).mkdir(parents=True, exist_ok=True)
                print(f'{path_exs} is created')
        for path_img,path_ann in dataset:
            dst_img_n = PurePath(dst_img).joinpath(PurePath(path_img).name)
            dst_ann_n = PurePath(dst_ann).joinpath(PurePath(path_ann).name)
            copyfile(path_img, dst_img_n)
            copyfile(path_ann, dst_ann_n)
    elif mod == 'del':
        # remove excess file
        empty_file_for_del = [empty_file.pop(random.randrange(len(empty_file))) for _ in range(len(empty_file)-not_empty_part)]
        for sample_del in empty_file_for_del:
            path_img, path_ann = sample_del
            Path(path_img).unlink()
            Path(path_ann).unlink()
    elif mod =='silent':
        print(f'Number of images with objects: {len(not_empty_file)}\n'
             f'The number of images without objects: {len(empty_file)}\n'
             f'The number of images without objects that will be left in the data set: {not_empty_part}')
    return dst_img, dst_ann


def plot_img_and_boxes(path_img, path_annotation, plot_random = True, seed = None, max_plot = 30):
    '''
    Display a number of images with bounding boxes

    :param path_img: str - folder where the cut images are located
    :param path_annotation: str - folder where the cut txt annotation are located
    :param plot_random: bool - draw random images (if True) or in sorted order (if False)
    :param seed: int - random seed for random plot
    :param max_plot: int - max number of plot figure
    :return: None
    '''
    list_a = get_img_couple_imgtxt(path_img= path_img, path_ann=path_annotation,
                                   extension=['png', 'jpg', 'txt'])
    if plot_random:
        random.seed(seed)
        random.shuffle(list_a)
        random.seed()
    i_plot = 1
    for path_img, path_ann in list_a:
        img = imread(path_img)
        ann_txt = np.loadtxt(path_ann)
        yviz = ann_txt
        if ann_txt.shape[0]:
            if len(ann_txt.shape) == 1: ann_txt = ann_txt[None]
            ann_txt = np.delete(ann_txt, 0, 1)
            ann_list =[]
            for ann in ann_txt:
                xnn, ynn, wnn, hnn = ann
                x = xnn * img.shape[1]
                y = ynn * img.shape[0]
                w = wnn * img.shape[1]
                h = hnn * img.shape[0]
                ann_stack = np.hstack((x, y, w, h))
                ann_list.append(ann_stack.tolist())
            ann_txt = np.array(ann_list)
            yviz = xywh2xyxy(np.array(ann_txt))
            yviz = yviz.astype(int)
        img_l = img.copy()
        for i in range(len(yviz)):
            label = 'car'
            plot_one_box(yviz[i], img_l, label=label, line_thickness=1)
        plt.figure(i_plot)
        plt.imshow(img_l)
        plt.text(10, -10, f'count of object on image: {len(ann_txt)}', fontdict = {'color': 'black', 'size': 16})
        plt.show()
        i_plot += 1
        if i_plot > max_plot: break