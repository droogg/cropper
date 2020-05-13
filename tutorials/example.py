import sys
sys.path.append('../')
from cropper.utils import get_sorted_img_path, get_img_couple
from cropper.functions import tilinig_image, create_yolo_annotation

path = '/home/vic/PycharmProjects/DS_neuro/Task/detection/MY/cowc_dataset_1'

req_size = (1024,1024)
overlap = 35 # in pixels

path_img_list = get_sorted_img_path(path, 'png')
for path_img in path_img_list:
    path_crop, path_crop_img = tilinig_image(path_img, req_size, overlap, annotation_prefix= '_Annotated_Cars')

img_couple_list = get_img_couple(path_crop, ['png'])
print('Number of couple image/annotation: ', len(img_couple_list))
create_yolo_annotation(img_couple_list, path_crop, plot_boundingbox = 10)

split_dataset(path_crop_img, ['png'], 0.8, shuffle=True)
