import numpy as np
import os
import os.path as osp
import shutil
from tqdm import tqdm

train_val_list = np.genfromtxt('train_test_split.txt', dtype='str')
img_list = np.genfromtxt('images.txt', dtype='str')

img_id_mapping = {a[0]: a[1] for a in img_list}
for img, is_train in tqdm(train_val_list):
    if bool(int(is_train)):
        # print(osp.join('CUB200', 'val', img_id_mapping[img]))
        os.remove(osp.join('CUB200', 'val', img_id_mapping[img]))
    else:
        os.remove(osp.join('CUB200', 'train', img_id_mapping[img]))
