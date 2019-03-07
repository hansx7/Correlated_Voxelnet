import sys
import cv2
import copy
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.append('../../')
import dataLoader.kitti_utils as utils
from project.config.pointS3D_config import config as cfg
from project.utils.utils import get_filtered_seq
from project.utils.targets_build import cal_batch_targets
from project.utils.create_voxel import seq_voxelization
from core.proj_utils import get_lidar_in_image_fov
from dataLoader.kitti_tracking_object import kitti_tracking_object


class KittiTrackingDataset(Dataset):
    def __init__(self, data_root, set_root='/kitti/', seq_len=cfg.seq_len, stride=cfg.stride,
                 train_type=cfg.train_type, set='train', using_img=False):
        ''' kitti tracking datasets loader
            data_root:      tracking datasets root path
            seq_len:        frame sequence length
            stride:         sliding stride for sample sequences
            train_type:     object type for training, including 'Car', 'Pedestrian', 'Cyclist', 'All'
            split:          'training' or 'val'
        '''
        self.set = set
        if self.set == 'train':
            self.videos = cfg.TRAIN_SPLIT
        elif self.set == 'val' or self.set == 'mini_val':
            self.videos = cfg.VAL_SPLIT
        elif self.set == 'val_test':
            self.videos = cfg.VAL_TEST_SPLIT

        if train_type == 'Car':
            self.obj_type = ['Car', 'Van']
        else:
            self.obj_type = [train_type]

        self.seq_len = seq_len
        self.stride = stride
        self.data_root = data_root
        self.train_type = train_type
        self.using_img = using_img
        self.set_root = cfg.ROOT_DIR + set_root
        self.train_val = open(self.set_root + 'tracking_trainval.txt').readlines()

        self.data_list = self.split_train_val()
        if self.set == 'mini_val':
            self.data_list = random.sample(self.data_list, cfg.mini_val_size)

        self.length = len(self.data_list)


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        data_list = self.data_list[idx]

        lidar_seq = self.get_enhanced_lidar(data_list)
        # get a specific range of data
        lidar_seq = get_filtered_seq(cfg, lidar_seq)
        # [n_frames, 4, D, H, W]
        voxel_features_seq = seq_voxelization(cfg, lidar_seq, method='v3')

        if self.set == 'val_test':
            image_seq = self.get_images(data_list)
            return data_list, voxel_features_seq, image_seq

        label_seq = self.get_labels(data_list)
        # bounding-box encoding
        pos_equal_one, neg_equal_one, targets = cal_batch_targets(label_seq, cfg)

        if self.using_img:
            image_seq = self.get_images(data_list)
            return data_list, voxel_features_seq, pos_equal_one, neg_equal_one, targets, image_seq
        else:
            return data_list, voxel_features_seq, pos_equal_one, neg_equal_one, targets


    def get_images(self, data):
        video_id = int(data[0])
        datasets = kitti_tracking_object(self.data_root, video_id, split=self.set)
        imgs = []
        for frame_id in data[1:]:
            img = datasets.get_image(frame_id)
            img = cv2.resize(img, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
            imgs.append(img)
        return np.concatenate([imgs], axis=0)


    def get_enhanced_lidar(self, data):
        '''
        get lidar with RGB information from image
        :param data: ['0000', 0, 1, 2, 3, 4]
        :return: enhanced_lidars  [N, 7], (x, y, z, d, b, g, r)
        '''
        video_id = int(data[0])
        datasets = kitti_tracking_object(self.data_root, video_id, split=self.set)
        calib = datasets.get_calibration()
        ref_idx = data[-1]
        ref_lidar = datasets.get_lidar(ref_idx)
        ref_img = cv2.resize(datasets.get_image(ref_idx), (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))

        ref_pc_velo, ref_pts_2d, ref_fov_inds = get_lidar_in_image_fov(ref_lidar[:, :3], calib, 0, 0,
                                                               cfg.IMG_WIDTH, cfg.IMG_HEIGHT, True)
        ref_lidar = ref_lidar[ref_fov_inds, :]
        ref_pts_2d = ref_pts_2d[ref_fov_inds, :].astype(np.uint8)
        ref_rgb_info = ref_img[ref_pts_2d[:, 0], ref_pts_2d[:, 1]] / 255
        ref_elidar = np.concatenate([ref_lidar, ref_rgb_info], axis=1)

        enhanced_lidars = []
        for idx in data[1:-1]:
            trans, rotation = datasets.get_transform(ref_idx, idx)
            pre_lidar = datasets.get_lidar(idx)
            pre_img = cv2.resize(datasets.get_image(idx), (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))

            pre_pv_velo, pre_pts_2d, pre_fov_inds = get_lidar_in_image_fov(pre_lidar[:, :3], calib, 0, 0,
                                                                           cfg.IMG_WIDTH, cfg.IMG_HEIGHT, True)
            pre_lidar = pre_lidar[pre_fov_inds, :]
            pre_lidar[:, :3] = (pre_lidar[:, :3] - trans) @ rotation

            pre_pts_2d = pre_pts_2d[pre_fov_inds, :].astype(np.uint8)
            pre_rgb_info = pre_img[pre_pts_2d[:, 0], pre_pts_2d[:, 1]] / 255
            pre_elidar = np.concatenate([pre_lidar, pre_rgb_info], axis=1)

            enhanced_lidars.append(pre_elidar)

        enhanced_lidars.append(ref_elidar)
        return enhanced_lidars


    def get_labels(self, data_list):
        ''' change labels format, from 'Object3d' list to dict of 'boxes3d' and 'ories3d', also transform
        pre frames 3d bounding boxes and 3d orientation vector to current frame vehicle coordinate system.

        input:
            data_list:      batch data list, example: ['0000', 0, 1, 2, 3, 4], give video id and frame id
        output:
            labels:         dict, usually {'boxes3d': [ndarray] B x 8 x 3, 'ories3d': [ndarray] B x 2 x 3}
                            if self.train_type == 'All', then it has element 'type'additionally:
                            [ndarray] B x 3 (one-hot vector)
                            if using_img, then it has element 'box2d' additionally: [ndarray] B x 4
        '''
        video_id = int(data_list[0])
        datasets = kitti_tracking_object(self.data_root, video_id)
        calib = datasets.get_calibration()
        ref_idx = data_list[-1]
        labels = []
        for idx in range(len(data_list) - 1):
            raw_label = datasets.get_label_objects(idx)
            # discard object we don't use
            label = []
            for obj in raw_label:
                if obj.type in self.obj_type:
                    label.append(obj)

            if len(label) == 0:
                labels.append({})
                continue
            new_label = {}
            boxes3d = []
            ories3d = []
            boxes2d = []
            trans, rotation = datasets.get_transform(ref_idx, data_list[1:][idx])
            for obj in label:
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                box3d = calib.project_rect_to_velo(box3d_pts_3d)
                # Draw heading arrow
                ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
                ori3d = calib.project_rect_to_velo(ori3d_pts_3d)

                # transform 3d bounding box and 3d orientation to current frame coordinate system
                box3d = (box3d - trans) @ rotation
                ori3d = (ori3d - trans) @ rotation

                boxes3d.append(box3d)
                ories3d.append(ori3d)
                boxes2d.append(obj.box2d)

            new_label['boxes3d'] = np.concatenate([boxes3d], axis=0)
            new_label['ories3d'] = np.concatenate([ories3d], axis=0)
            if self.using_img:
                new_label['box2d'] = np.concatenate([ories3d], axis=0)

            labels.append(new_label)
        return labels

    def split_train_val(self):
        '''split frames into batches, each batch contains 'seq_len' (default is 5)
        continuous frames, with sliding step of 'stride'
        return:
            dataList: batch list, each item in list likes this: ['0000', 0, 1, 2, 3, 4],
                      first string is video id, and the rest are four continuous frames id.
        '''
        train_val = self.train_val
        data_list = []
        # ['0000',0,1,2,3,4]
        for i in range(0, len(train_val)-self.seq_len, self.stride):
            temp_stream = []
            for j in range(i, i+self.seq_len):
                line = train_val[j]
                # empty line '\n'
                if len(line) == 1:
                    if len(temp_stream) == self.seq_len + 1:
                        data_list.append(copy.copy(temp_stream))
                    break

                video_id = line.split('.')[0].split('/')[2]
                if len(temp_stream) == 0:
                    temp_stream.append(video_id)

                frame_id = line.split('.')[0].split('/')[3]
                if video_id in self.videos:
                    frame_id = int(frame_id)
                    if len(temp_stream) == self.seq_len + 1:
                        data_list.append(copy.copy(temp_stream))
                        temp_stream = [video_id]
                    temp_stream.append(frame_id)

            if len(temp_stream) == self.seq_len + 1:
                data_list.append(copy.copy(temp_stream))

        return data_list



if __name__ == '__main__':

    data_root = cfg.KITTI_TRACKING_DATASET_ROOT

    datasets = KittiTrackingDataset(data_root, set_root='/kitti/', seq_len=5, stride=1,
                                    train_type='Car', set='train')
    print(len(datasets))
    dataloader = DataLoader(datasets, batch_size=2, num_workers=4)
    for i_batch, (data_list, voxel_features_seq, pos_equal_one, neg_equal_one, targets) \
            in enumerate(dataloader, 0):
        print(i_batch, data_list, voxel_features_seq.size(), targets.size())


