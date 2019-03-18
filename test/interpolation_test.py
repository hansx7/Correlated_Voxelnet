import os
import sys
sys.path.append('..')
from dataLoader.kitti_detection_object import kitti_detection_object
from visualization.viz import *
from core.proj_utils import get_lidar_in_image_fov
from transform.interpolation import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


root_dir = ROOT_DIR + '/datasets/kitti/detection'
datasets = kitti_detection_object(root_dir)

test_idx = 4
lidar = datasets.get_lidar(test_idx)
image = datasets.get_image(test_idx)
calib = datasets.get_calibration(test_idx)
label = datasets.get_label_objects(test_idx)

img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_height, img_width, img_channel = img.shape
print(('Image shape: ', img.shape))
pc_velo = lidar[:,0:3]

lidar_fov, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo, calib, 0, 0, img_width, img_height, True)
imgfov_pts_2d = pts_2d[fov_inds,:]
imgfov_pc_rect = calib.project_velo_to_rect(lidar_fov)
print(lidar_fov.shape)

lidar_img, inds = generate_lidar_img(lidar_fov, calib, img_height, img_width)
cv2.imshow('lidar_img', lidar_img)
cv2.waitKey()

den_lidar = densitify_point_cloud(lidar_fov,lidar_img, inds, 3, window_size=3)
print(den_lidar.shape)

den_lidar_img, den_inds = generate_lidar_img(den_lidar, calib, img_height, img_width)
cv2.imshow('den_lidar_img', den_lidar_img)
cv2.waitKey()