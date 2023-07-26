import numpy as np
import os
import os.path as osp
import shutil
from tqdm import tqdm

img_list = np.genfromtxt('mat2txt.txt', dtype=str) # [num, img, cls, istest]
class_mappings = np.genfromtxt('label_map.txt', dtype=str)
class_mappings = {a[0]: a[1] for a in class_mappings}

for item in tqdm(img_list):
    if bool(int(item[-1])):
        cls_folder = osp.join('cars196', 'train', class_mappings[item[2]])
        if not os.path.exists(cls_folder):
            os.mkdir(cls_folder)
        shutil.copy(item[1], osp.join(cls_folder, item[1].split('/')[-1]))
    else:
        cls_folder = osp.join('cars196', 'val', class_mappings[item[2]])
        if not os.path.exists(cls_folder):
            os.mkdir(cls_folder)
        shutil.copy(item[1], osp.join(cls_folder, item[1].split('/')[-1]))
