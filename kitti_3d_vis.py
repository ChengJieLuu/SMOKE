# kitti_3d_vis.py
 
 
from __future__ import print_function
 
import os
import sys
import cv2
import random
import os.path
import shutil
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
from kitti_util import *
 
def visualization():
    dataset = kitti_object(r'./datasets/kitti/')
 
    path = r'./tools/logs/inference/kitti_test/data'
    # path = r'label/'
    # Save_Path = r'./outimg/'
    Save_fold = r'temp_3d_vis'
    if not os.path.exists(Save_fold):
        os.makedirs(Save_fold)
    files = os.listdir(path)
    for file in files:
        name = file.split('.')[0]
        # if not name.endswith('camera0'):
        #     continue
        save_path = os.path.join(Save_fold, name.split('_')[0] + '.png')
        # data_idx = int(name[:6])
 
        img_path = os.path.join("./datasets/kitti/testing/image_2",file.split('.')[0]+".png")
        # img_path = os.path.join("image",file.split('.')[0]+".png")
        # label_path = os.path.join("label",file.split('.')[0]+".txt")
        label_path = os.path.join("./tools/logs/inference/kitti_test/data",file.split('.')[0]+".txt")

        # Load data from dataset
        # objects = dataset.get_label_objects(data_idx)

        lines = [line.rstrip() for line in open(label_path)]
        objects = [Object3d(line) for line in lines]

        # img = dataset.get_image(data_idx)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # calib = dataset.get_calibration(data_idx)
        calib = dataset.get_calibration(name)
        print(' ------------ save image with 3D bounding box ------- ')
        print('name:', name)
        show_image_with_boxes(img, objects, calib, save_path, True)
        
 
if __name__=='__main__':
    visualization()