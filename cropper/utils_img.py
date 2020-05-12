import os
from os.path import join
from glob import glob
import re

__all__ = ['norm_path', 'get_sorted_img_path', 'get_sorted_img_names', 'get_img_couple']


def norm_path(path: str) -> str:
    return ''.join([os.path.normpath(path), '/']) if os.path.isdir(path) else path


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)',text)]


def img_listextend (path, extension):
    img_list = []
    for ext in extension:
        img_list.extend(glob(join(path, '*.'+ ext)))
    return img_list


def get_sorted_img_path(path: str, extension: list) -> list:
    path = norm_path(path)
    img_list = img_listextend(path, extension)
    img_list.sort(key=natural_keys)
    return img_list


def get_sorted_img_names(path: str, extension: list) -> list:
    path, img_fnl = norm_path(path), []
    img_list = get_sorted_img_path(path, extension)
    for i in range(len(img_list)):
        img_fnl.append(os.path.basename(img_list[i]))
    return img_fnl


def get_img_couple(path, extension):
    path = norm_path(path)
    img_fnl, img_couple_list = get_sorted_img_names(path, extension), []
    for i in range(len(img_fnl) - 1):
        if len(img_fnl) % 2 != 0:
            raise Exception('Количество изображений и аннотаций не совпадает')
        elif img_fnl[i].split('.')[0] + '_Annotated_Cars.png' == img_fnl[i + 1]:
            img_couple_list.append([img_fnl[i], img_fnl[i + 1]])
    assert len(img_couple_list) == len(img_fnl) / 2, \
        'get_img_couple: Список неверно отсортирован или количество аннотаций-изображений несоответствует'
    return img_couple_list