import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('../../')
import dataLoader.kitti_utils as utils
from project.config.dt_config import config as cfg
from project.utils.create_voxel import voxelization_v1
from project.utils.targets_build import cal_target
from project.utils.data_aug import aug_data
from project.utils.utils import get_filtered_lidar
from core.proj_utils import get_lidar_in_image_fov
from dataLoader.kitti_tracking_object import kitti_tracking_object
from core.bev_generators.bev_slices import BevSlices


class KittiDTDataset(Dataset):
    def __init__(self, data_root, set_root='/kitti/', stride=cfg.stride,
                 train_type=cfg.train_type, set='train', using_img=False):
        ''' kitti tracking datasets loader
            data_root:      tracking datasets root path
            seq_len:        frame sequence length
            stride:         sliding stride for sample sequences
            train_type:     object type for training, including 'Car', 'Pedestrian', 'Cyclist', 'All'
            split:          'train' or 'val'
        '''
        self.set = set
        if self.set == 'train':
            self.videos = cfg.TRAIN_SPLIT
        elif self.set == 'val' or self.set == 'mini_val':
            self.videos = cfg.VAL_SPLIT
        elif self.set == 'val_test':
            self.videos = cfg.VAL_TEST_SPLIT

        self.stride = stride
        self.data_root = data_root

        if train_type == 'Car':
            self.train_type =['Car', 'Van']
        else:
            self.train_type = [train_type]

        self.using_img = using_img
        self.set_root = cfg.ROOT_DIR + set_root
        self.train_val = self.set_root + 'tracking_trainval.txt'

        self.data_list = self.split_train_val()
        self.length = len(self.data_list)


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        idx_info = self.data_list[idx]
        video_id = int(idx_info[0].split('_')[0])
        pre_idx = int(idx_info[0].split('_')[1])
        next_idx = int(idx_info[1].split('_')[1])

        datasets = kitti_tracking_object(self.data_root, video_id, split=self.set)
        calib = datasets.get_calibration()

        lidars = self.get_lidar(pre_idx, next_idx, datasets, calib)

        # bev_maps = self.get_bev_map(lidars)

        images = self.get_image(pre_idx, next_idx, datasets)

        gt_boxes3d, gt_boxes2d, _, tracking_id = \
            self.get_labels(pre_idx, next_idx, datasets, calib)

        num_boxes = np.array([len(gt_boxes3d[0]), len(gt_boxes3d[1])])

        if num_boxes[0] == 0:
            gt_boxes3d[0] = np.zeros((8,3))
            gt_boxes2d[0] = np.zeros((1,4))
            tracking_id[0] = -1

        if num_boxes[1] == 0:
            gt_boxes3d[1] = np.zeros((8,3))
            gt_boxes2d[1] = np.zeros((1,4))
            tracking_id[1] = -1

        labels = {
            'gt_boxes3d': gt_boxes3d,
            'gt_boxes2d': gt_boxes2d,
            'tracking_id': tracking_id
        }

        # get filtered lidar
        _, _, fov_inds0 = get_lidar_in_image_fov(lidars[0][:, :3], calib, 0, 0,
                                                 cfg.IMG_WIDTH, cfg.IMG_HEIGHT, return_more=True)
        aug_lidar0 = lidars[0][fov_inds0]
        _, _, fov_inds1 = get_lidar_in_image_fov(lidars[1][:, :3], calib, 0, 0,
                                                 cfg.IMG_WIDTH, cfg.IMG_HEIGHT, return_more=True)
        aug_lidar1 = lidars[1][fov_inds1]

        # data augment
        if self.set == 'train':
            np.random.seed()
            choice = np.random.randint(1, 10)
            if choice > 5:
                if len(labels['gt_boxes3d'][0]) > 0:
                    lidars[0], labels['gt_boxes3d'][0] = aug_data(lidars[0], labels['gt_boxes3d'][0])
                if len(labels['gt_boxes3d'][1]) > 0:
                    lidars[0], labels['gt_boxes3d'][1] = aug_data(lidars[1], labels['gt_boxes3d'][1])

        lidars[0], labels['gt_boxes3d'][0] = get_filtered_lidar(cfg, aug_lidar0, labels['gt_boxes3d'][0])
        lidars[1], labels['gt_boxes3d'][1] = get_filtered_lidar(cfg, aug_lidar1, labels['gt_boxes3d'][1])

        voxel_features, voxel_coords = self.get_voxel_features_and_coords(cfg, lidars)
        voxel_mask = voxel_features[0].shape[0]
        voxel_features = np.concatenate((voxel_features[0], voxel_features[1]), axis=0)
        voxel_coords = np.concatenate((voxel_coords[0], voxel_coords[1]), axis=0)

        pos_equal_one0, neg_equal_one0, targets0 = cal_target(labels['gt_boxes3d'][0], cfg)
        pos_equal_one1, neg_equal_one1, targets1 = cal_target(labels['gt_boxes3d'][1], cfg)
        pos_equal_one = np.concatenate((pos_equal_one0, pos_equal_one1), axis=2)
        neg_equal_one = np.concatenate((neg_equal_one0, neg_equal_one1), axis=2)
        targets = np.concatenate((targets0, targets1), axis=2)
        # targets_mask = targets0.shape[2]

        return idx_info, lidars, images, labels, num_boxes, voxel_features, voxel_coords, voxel_mask, \
               pos_equal_one, neg_equal_one, targets#, targets_mask


    def split_train_val(self):
        '''split frames into batches, each batch contains 'seq_len' (default is 2)
        continuous frames, with sliding step of 'stride' (default is 1)

        return:
            dataList: batch list, each item in list likes this: ['0000_0', '0000_1'],
                      first substring is video id, and the rest are four continuous frames id.
                      the last frame will be duplicated if necessary: ['0000_153', '0000_153']
        '''
        def extract_id(string):
            video_id = string.split('.')[0].split('/')[2]
            frame_id = string.split('.')[0].split('/')[3]
            frame_id = str(int(frame_id))
            return video_id + '_' + frame_id


        data_list = []

        train_val = open(self.train_val).read().split('\n\n')
        for item in train_val:
            item = item.split('\n')
            item = list(map(extract_id, item))
            for i in range(len(item)):
                cur = item[i]
                if i+self.stride < len(item):
                    next = item[i+self.stride]
                else:
                    next = item[-1]
                data_list.append([cur, next])

        return data_list


    def get_lidar(self, pre_idx, next_idx, datasets, calib):
        lidars = []

        pre_lidar = datasets.get_lidar(pre_idx)
        pre_pc_velo, pre_pts_2d, pre_fov_inds = get_lidar_in_image_fov(pre_lidar[:, :3], calib, 0, 0,
                                                               cfg.IMG_WIDTH, cfg.IMG_HEIGHT, True)
        pre_lidar = pre_lidar[pre_fov_inds, :]
        lidars.append(pre_lidar)

        trans, rotation = datasets.get_transform(pre_idx, next_idx)
        next_lidar = datasets.get_lidar(next_idx)

        next_pv_velo, next_pst_2d, next_fov_inds = get_lidar_in_image_fov(next_lidar[:, :3], calib, 0, 0,
                                                                           cfg.IMG_WIDTH, cfg.IMG_HEIGHT, True)
        next_lidar = next_lidar[next_fov_inds, :]
        next_lidar[:, :3] = (next_lidar[:, :3] - trans) @ rotation
        lidars.append(next_lidar)

        return lidars

    def get_bev_map(self, lidars):
        bev_maps = []
        generater = BevSlices(cfg)
        for lidar in lidars:
            lidar = lidar[:, :3]
            # (3,N)
            lidar = lidar.transpose()
            bev_images = generater.generate_bev('lidar', lidar, None,
                                                   cfg.area_extents, cfg.voxel_size)
            height_maps = bev_images.get('height_maps')
            density_map = bev_images.get('density_map')
            bev_input = np.dstack((*height_maps, density_map))
            bev_maps.append(bev_input)
        return bev_maps

    def get_image(self, pre_idx, next_idx, datasets):
        images = []
        pre_image = datasets.get_image(pre_idx)
        next_image = datasets.get_image(next_idx)
        pre_image = cv2.resize(pre_image, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
        # TODO: image rectify among frames
        next_image = cv2.resize(next_image, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
        images.append(pre_image)
        images.append(next_image)
        return images

    def get_labels(self, pre_idx, next_idx, datasets, calib):
        gt_boxes3d = [[], []]
        gt_boxes2d = [[], []]
        gt_ories3d = [[] ,[]]
        tracking_id = [[], []]

        trans, rotation = datasets.get_transform(pre_idx, next_idx)
        pre_label = datasets.get_label_objects(pre_idx)
        next_label = datasets.get_label_objects(next_idx)
        raw_labels = [pre_label, next_label]

        for i in range(len(raw_labels)):
            temp_label = raw_labels[i]
            for obj in temp_label:
                if not obj.type in self.train_type:
                    continue
                else:
                    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                    box3d = calib.project_rect_to_velo(box3d_pts_3d)
                    # Draw heading arrow
                    ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
                    ori3d = calib.project_rect_to_velo(ori3d_pts_3d)

                    # TODO: boxes2d coordinate rectify among frames
                    box2d = obj.box2d

                    obj_id = obj.obj_id

                    if i == 1:
                        # transform 3d bounding box and 3d orientation to current frame coordinate system
                        box3d = (box3d - trans) @ rotation
                        ori3d = (ori3d - trans) @ rotation

                    gt_boxes3d[i].append(box3d)
                    gt_boxes2d[i].append(box2d)
                    gt_ories3d[i].append(ori3d)
                    tracking_id[i].append(obj_id)

            gt_boxes3d[i] = np.array(gt_boxes3d[i])
            gt_boxes2d[i] = np.array(gt_boxes2d[i])
            gt_ories3d[i] = np.array(gt_ories3d[i])
            tracking_id[i] = np.array(tracking_id[i])


        return gt_boxes3d, gt_boxes2d, gt_ories3d, tracking_id

    def get_voxel_features_and_coords(self, cfg, lidars):
        voxel_feature0, voxel_coord0 = voxelization_v1(cfg, lidars[0])
        voxel_feature1, voxel_coord1 = voxelization_v1(cfg, lidars[1])
        return [voxel_feature0, voxel_feature1], [voxel_coord0, voxel_coord1]


if __name__ == '__main__':

    data_root = cfg.KITTI_TRACKING_DATASET_ROOT

    datasets = KittiDTDataset(data_root, set_root='/kitti/', stride=1, train_type='Car', set='train')
    print(len(datasets))
    dataloader = DataLoader(datasets, batch_size=1, num_workers=4)
    for i_batch, (idx_info, lidars, images, labels, num_boxes, voxel_features, voxel_coords, voxel_mask, \
                  pos_equal_one, neg_equal_one, targets) in enumerate(dataloader, 0):
        print(i_batch, idx_info,
              # ('lidar size', lidars[0].size(), lidars[1].size()),
              # ('image size', images[0].size(), images[1].size()),
              # # (bev_maps[0].size(), bev_maps[1].size()),
              # '\n',
              # ('labels', labels['gt_boxes3d'][0], '\n', labels['gt_boxes3d'][1]), '\n',
              # (labels['gt_boxes2d'][0].size(), labels['gt_boxes2d'][1].size()),
              # (labels['tracking_id'][0], labels['tracking_id'][1]),
              # ('num_boxes', num_boxes[0]),
              # ('voxel_features', voxel_features[0].shape, voxel_features[1].shape, \
              #  torch.cat((voxel_features[0], voxel_features[1]), 1).shape),
              # ('voxel_coords', voxel_coords[0].shape, voxel_coords[1].shape, \
              #  torch.cat((voxel_coords[0], voxel_coords[1]), 1).shape), \
              '\nvoxel_features', voxel_features.shape, \
              '\nvoxel_coords', voxel_coords.shape, voxel_mask, \
              '\npos_equal_one', pos_equal_one.shape, \
              '\nneg_equal_one', neg_equal_one.shape, \
              '\ntargets', targets.shape#, targets_mask
              # '\npos_equal_one', (pos_equal_one[0].shape, pos_equal_one[1].shape, \
              #  torch.cat((pos_equal_one[0], pos_equal_one[1]), 3).shape), \
              # '\nneg_equal_one', (neg_equal_one[0].shape, neg_equal_one[1].shape, \
              #  torch.cat((neg_equal_one[0], neg_equal_one[1]), 3).shape), \
              # '\ntargets', (targets[0].shape, targets[1].shape, \
              #  torch.cat((targets[0], targets[1]), 3).shape)
              )

