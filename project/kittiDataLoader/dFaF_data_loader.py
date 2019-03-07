import os
import cv2
import sys
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

sys.path.append('../../')
import dataLoader.kitti_utils as utils
from project.utils.utils import get_filtered_lidar
from project.config.dFaF_config import config as cfg
from project.utils.create_voxel import voxelization_v2
from project.utils.targets_build import cal_target
from project.utils.data_aug import aug_data
from core.proj_utils import get_lidar_in_image_fov


class KittiDetectionDataset(Dataset):
    def __init__(self, data_root, set_root='/kitti/', set='train',
                 train_type=cfg.train_type, using_img=False):

        self.set = set
        self.using_img = using_img
        self.set_root = cfg.ROOT_DIR + set_root

        if train_type == 'Car':
            self.train_type = ['Car', 'Van']
        else:
            self.train_type = [train_type]

        self.data_root = os.path.join(data_root, 'training')
        self.lidar_path = os.path.join(self.data_root, 'velodyne')
        self.image_path = os.path.join(self.data_root, 'image_2')
        self.label_path = os.path.join(self.data_root, 'label_2')
        self.calib_path = os.path.join(self.data_root, 'calib')
        
        with open(os.path.join(self.set_root, '%s.txt' % self.set.split('_')[-1])) as f:
            self.file_list = f.read().splitlines()
            if self.set == 'mini_val':
                self.file_list = random.sample(self.file_list, cfg.mini_val_size)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # get data path
        lidar_file = self.lidar_path + '/' + self.file_list[idx] + '.bin'
        calib_file = self.calib_path + '/' + self.file_list[idx] + '.txt'
        label_file = self.label_path + '/' + self.file_list[idx] + '.txt'

        # get raw data
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        calib = utils.Calibration(calib_file)
        labels = [line.rstrip() for line in open(label_file)]
        labels = [utils.Object3d(line, datasets='detection') for line in labels]

        # get filtered lidar
        pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(lidar[:, :3], calib, 0, 0, 
                                                           cfg.IMG_WIDTH, cfg.IMG_HEIGHT, True)
        lidar = lidar[fov_inds]

        # bounding-box encoding
        # (B, 8, 3), (B, 7)
        boxes3d, bblists = self.label_preprocess(labels, calib)

        if self.set == 'train':
            # data aug
            np.random.seed()
            choice = np.random.randint(1, 10)
            if choice > 5 and len(boxes3d) > 0:
                lidar, boxes3d = aug_data(lidar, boxes3d)
        
        lidar, boxes3d = get_filtered_lidar(cfg, lidar, boxes3d)
        
        # voxelization
        voxel_features = voxelization_v2(cfg, lidar)
        
        if boxes3d.ndim == 3:
            pos_equal_one, neg_equal_one, targets = cal_target(boxes3d, cfg)
        else:
            feature_map_size = (int(cfg.H * cfg.feature_map_rate),
                                int(cfg.W * cfg.feature_map_rate))
            pos_equal_one = np.zeros((*feature_map_size, cfg.anchors_per_position))
            neg_equal_one = np.zeros((*feature_map_size, cfg.anchors_per_position))
            targets = np.zeros((*feature_map_size, 7 * cfg.anchors_per_position))

        # copy n_frame
        voxel_features = np.concatenate([[voxel_features] * cfg.n_frame], axis=0)
        pos_equal_one = np.concatenate([pos_equal_one] * cfg.n_frame, axis=2)
        neg_equal_one = np.concatenate([neg_equal_one] * cfg.n_frame, axis=2)
        targets = np.concatenate([targets] * cfg.n_frame, axis=2)

        if self.using_img:
            image_file = self.image_path + '/' + self.file_list[idx] + '.png'
            image = cv2.imread(image_file)
            image = cv2.resize(image, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
            return self.file_list[idx], voxel_features, pos_equal_one, neg_equal_one, targets, image
        else:
            return self.file_list[idx], voxel_features, pos_equal_one, neg_equal_one, targets

    def label_preprocess(self, label, calib):
        boxes3d = []
        bblists = []
        for obj in label:
            if obj.type in self.train_type:
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                box3d = calib.project_rect_to_velo(box3d_pts_3d)
                boxes3d.append(box3d)

                bblist = obj.to_xyzhmlr()
                bblists.append(bblist)

        boxes3d = np.concatenate([boxes3d], axis=0)
        bblists = np.concatenate([bblists], axis=0)
        return boxes3d, bblists


if __name__ == '__main__':
    datasets = KittiDetectionDataset(cfg.KITTI_DETECTION_DATASET_ROOT, set='train',train_type='Car')
    dataloader = DataLoader(datasets, batch_size=16, num_workers=16)
    print(len(dataloader))
    for i_batch, (ids, voxel_features, pos_equal_one, neg_equal_one, targets) in enumerate(dataloader, 0):
        print(ids)
        print(i_batch, voxel_features.size(), pos_equal_one.size(), neg_equal_one.size(), targets.size())