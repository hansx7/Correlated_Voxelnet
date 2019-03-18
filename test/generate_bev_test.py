import sys
sys.path.append('..')
from dataLoader.kitti_tracking_object import *
from core.proj_utils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

root_dir = ROOT_DIR + '/datasets/kitti/tracking'

BEV_WIDTH = 416
BEV_HEIGHT = 416
BEV_VOXEL_SIZE = 0.2

for idx in range(2):
    dataset = kitti_tracking_object(root_dir,idx)
    frames_num = len(dataset)
    calib_obj = dataset.get_calibration()

    for frame_idx in range(frames_num):
        print(frame_idx)
        image = dataset.get_image(frame_idx)
        lidar = dataset.get_lidar(frame_idx)[:, 0:3] # NX3
        label_obj = dataset.get_label_objects(frame_idx)

        img_h, img_w, img_c = image.shape
        velo_img_fov = get_lidar_in_image_fov(lidar, calib_obj, 0, 0, img_w, img_h, False) # Mx3 (M <= N)
        bev_img, _ = generate_bev_image(velo_img_fov, label_obj, calib_obj, BEV_WIDTH, BEV_HEIGHT, BEV_VOXEL_SIZE, True)

        cv2.imshow('bev', bev_img)
        cv2.waitKey()