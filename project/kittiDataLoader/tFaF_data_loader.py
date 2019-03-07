import sys
import os
import cv2
import copy
import pickle
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.append('../../')
import dataLoader.kitti_utils as utils
from project.config.tFaF_config import config as cfg
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

        self.seq_len = seq_len
        self.stride = stride
        self.data_root = data_root
        self.train_type = train_type
        self.using_img = using_img
        self.set_root = cfg.ROOT_DIR + set_root
        self.train_val = open(self.set_root + 'tracking_trainval.txt').readlines()

        if self.set == 'val_test':
            self.data_list = self.split_train_val()
        else:
            self.cache_path = os.path.join(self.set_root, 'tracking_datalist_cache')
            self.file_list = self.split_train_val()

            if self.set == 'mini_val':
                self.file_list = random.sample(self.file_list, cfg.mini_val_size)
            # reconstruction 'self.data_list' according to train_type
            # it would be cost a lot of time if generating data_list and labels_list every time,
            # so we store the data in the first time when generated and  load them in next time.
            os.makedirs(self.cache_path, exist_ok=True)
            self.data_file_name = self.cache_path + '/%d_%d_%s_%s_data.json' \
                             % (self.seq_len, self.stride, self.train_type, self.set)

            self.label_file_name = self.cache_path + '/%d_%d_%s_%s_label.json' \
                             % (self.seq_len, self.stride, self.train_type, self.set)

            if os.path.exists(self.data_file_name) and os.path.exists(self.label_file_name):
                self.data_list = self.load_data(self.data_file_name)
                self.label_list = self.load_data(self.label_file_name)
            else:
                self.data_list, self.label_list = self.datasets_filter()

            assert (len(self.data_list) == len(self.label_list))

        self.length = len(self.data_list)


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        data_list = self.data_list[idx]
        lidar_seq = self.get_enhanced_lidar(data_list)
        # get a specific range of data
        lidar_seq = get_filtered_seq(cfg, lidar_seq)
        voxel_features_seq = seq_voxelization(cfg, lidar_seq)

        if self.set == 'val_test':
            image_seq = self.get_images(data_list)
            return data_list, voxel_features_seq, image_seq

        raw_label = self.label_list[idx]
        label_seq = self.get_labels(data_list, raw_label)
        # bounding-box encoding
        pos_equal_one, neg_equal_one, targets = cal_batch_targets(label_seq, cfg)

        if self.using_img:
            image_seq = self.get_images(data_list)
            return data_list, voxel_features_seq, pos_equal_one, neg_equal_one, targets, image_seq
        else:
            return data_list, voxel_features_seq, pos_equal_one, neg_equal_one, targets


    def load_data(self, path):
        with open(path, 'rb') as f:
            out = pickle.load(f)
        return out


    def get_images(self, data):
        video_id = int(data[0])
        datasets = kitti_tracking_object(self.data_root, video_id, split=self.set)
        imgs = []
        for frame_id in data[1:]:
            img = datasets.get_image(frame_id)
            img = cv2.resize(img, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
            imgs.append(img)
        return np.concatenate([imgs], axis=0)


    def get_lidar(self, data):
        ''' get transformed point cloud. every frame was represent in current frame(data[-1])
            vehicle ccoordinate system.
        '''
        video_id = int(data[0])
        datasets = kitti_tracking_object(self.data_root, video_id, split=self.set)
        calib = datasets.get_calibration()
        ref_idx = data[-1]
        ref_lidar = datasets.get_lidar(ref_idx)
        ref_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(ref_lidar[:, :3], calib, 0, 0,
                                                            cfg.IMG_WIDTH, cfg.IMG_HEIGHT, True)
        ref_lidar = ref_lidar[fov_inds, :]
        lidars = []
        for idx in data[1:-1]:
            trans, rotation = datasets.get_transform(ref_idx, idx)
            pre_lidar = datasets.get_lidar(idx)
            pre_pv_velo, pre_pst_2d, pre_fov_inds = get_lidar_in_image_fov(pre_lidar[:, :3], calib, 0, 0,
                                               cfg.IMG_WIDTH, cfg.IMG_HEIGHT, True)
            pre_lidar = pre_lidar[pre_fov_inds, :]
            pre_lidar[:, :3] = (pre_lidar[:, :3] - trans) @ rotation
            lidars.append(pre_lidar)
        lidars.append(ref_lidar)
        return lidars


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


    def get_labels(self, data, raw_labels):
        ''' change labels format, from 'Object3d' list to dict of 'boxes3d' and 'ories3d', also transform
        pre frames 3d bounding boxes and 3d orientation vector to current frame vehicle coordinate system.

        input:
            data:           batch data list, example: ['0000', 0, 1, 2, 3, 4], give video id and frame id
            raw_labels:     batch labels after 'datasets_filter()', example: [[Object3d(0), ...],
                            [Object3d(2),...],...]
        output:
            labels:         dict, usually {'boxes3d': [ndarray] B x 8 x 3, 'ories3d': [ndarray] B x 2 x 3}
                            if self.train_type == 'All', then it has element 'type'additionally:
                            [ndarray] B x 3 (one-hot vector)

                            if using_img, then it has element 'box2d' additionally: [ndarray] B x 4
        '''
        video_id = int(data[0])
        datasets = kitti_tracking_object(self.data_root, video_id)
        calib = datasets.get_calibration()
        ref_idx = data[-1]
        labels = []
        for idx in range(len(data[1:])):
            label = raw_labels[idx]
            if len(label) == 0:
                labels.append({})
                continue
            new_label = {}
            boxes3d = []
            ories3d = []
            trans, rotation = datasets.get_transform(ref_idx, data[1:][idx])
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

            new_label['boxes3d'] = np.concatenate([boxes3d], axis=0)
            new_label['ories3d'] = np.concatenate([ories3d], axis=0)

            if self.train_type == 'All':
                new_label['type'] = self.combine_labels(label, 'type')

            if self.using_img:
                new_label['box2d'] = self.combine_labels(label, 'box2d')

            labels.append(new_label)
        return labels


    def combine_labels(self, label, name='box2d'):
        output = []
        for obj in label:
            if name == 'box2d':
                output.append(obj.box2d)
            if name == 'type':
                if obj.type in ['Car', 'Van']:
                    dtype = np.array([1, 0, 0])
                elif obj.type == 'Pedestrian':
                    dtype = np.array([0, 1, 0])
                elif obj.type == 'Cyclist':
                    dtype = np.array([0, 0, 1])
                else:
                    print('None type occurs!')
                    return
                output.append(dtype)
        output = np.concatenate([output], axis=1)
        return output


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


    def datasets_filter(self):
        '''reconstruction datasets according to 'train_type'.
        If train_type == 'All', then only discard  the batch that only contains the object in
                                ['Person_sitting', 'Tram', 'DontCare'] that we don't care
        If train_type == 'Car', then only remain the batch that contains the object in
                                ['Car', 'Van', 'Truck', 'Misc']
        If train_type == 'Pedestrian', then only remain the batch that contains 'Pedestrian' object
        If train_type == 'Cyclist', then only remain the batch that contains 'Cyclist' object

        meanwhile, we also discard the label object in each frame that not in train_type

        output:
            datalist:       data batches, a list with element likes ['0000', 0, 1, 2, 3, 4]
            labelslist:     a list corresponds to 'datalist',  each element is a list that contains a batch of label,
                            likes [[LabelObject(0)...], [LabelObject(1),...], ...], each LabelObject is an instance
                            of class ‘Object3d’ in kitti_utils.py

            ##Attention##:  element likes [[LabelObject(0),...], [], [LabelObject(2),...], ...]
                            will also be contained in labelslist
        '''
        data_list = self.file_list
        train_type = self.train_type
        # store filtered total labels
        label_list = []
        # store discard batch
        discard_batch = []
        for data in data_list:
            # store filtered batch labels, likes [[LabelObject(0),...], [], [LabelObject(2),...], ...]
            label_batch = []
            video_id = int(data[0])
            kitti_obj = kitti_tracking_object(self.data_root, video_id)
            # whether to discard current batch
            is_discard = True
            for frame_id in data[1:]:
                # frame objects labels
                labels = kitti_obj.get_label_objects(frame_id)
                # new object labels in one frame, [LabelObject(0),...]
                new_labels = []
                # loop for labels
                for label in labels:
                    obj_type = label.type
                    # if one object in frame meet 'train_type', then the whole batch is available
                    if train_type == 'All' and obj_type in ['Car', 'Van', 'Pedestrian', 'Cyclist']:
                        is_discard = False
                        new_labels.append(label)
                    elif train_type == 'Car' and obj_type in ['Car', 'Van']:
                        is_discard = False
                        new_labels.append(label)
                    elif train_type == obj_type:
                        is_discard = False
                        new_labels.append(label)

                label_batch.append(new_labels)
            # frames sequence has no object that meet 'train_type', then the batch will be discarded
            if is_discard:
                discard_batch.append(data)
            else:
                label_list.append(label_batch)

        for batch in discard_batch:
             data_list.remove(batch)

        # write data_list, labels_list into json file
        with open(self.data_file_name, 'wb') as df:
            pickle.dump(data_list, df)

        with open(self.label_file_name, 'wb') as lf:
            pickle.dump(label_list, lf)

        return data_list, label_list


if __name__ == '__main__':

    data_root = cfg.KITTI_TRACKING_DATASET_ROOT

    datasets = KittiTrackingDataset(data_root, set_root='/kitti/', seq_len=2, stride=1,
                                    train_type='Car', set='train')
    print(len(datasets))
    dataloader = DataLoader(datasets, batch_size=2, num_workers=4)
    for i_batch, (data_list, voxel_features_seq, pos_equal_one, neg_equal_one, targets) \
            in enumerate(dataloader, 0):
        print(i_batch, data_list, voxel_features_seq.size(), targets.size())


