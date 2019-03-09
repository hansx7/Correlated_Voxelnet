from __future__ import print_function

import os
import sys
import numpy as np
import cv2

import dataLoader.kitti_utils as utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class kitti_tracking_object(object):
    # Load and parse object data into a usable format.
    def __init__(self, root_dir, video_id, split='train'):
        # root_dir contains training and testing folders
        self.root_dir = root_dir
        self.video_id = video_id
        if split in ['train', 'val', 'mini_val', 'val_test']:
            self.split = 'training'
        else:
            self.split = 'testing'

        self.split_dir = os.path.join(root_dir, self.split)

        self.image_dir = os.path.join(self.split_dir, 'image_2', '%04d' %self.video_id)
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne', '%04d' % self.video_id)
        self.label_dir = os.path.join(self.split_dir, 'label_2')
        self.oxts_dir  = os.path.join(self.split_dir, 'oxts')
        self.plane_dir = os.path.join(self.split_dir, 'planes')

        if self.video_id < len(os.listdir(self.calib_dir)):
            self.num_samples = len(os.listdir(self.lidar_dir))
        else:
            print('Unknown frame: %04d' % self.video_id)
            exit(-1)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png' %idx)
        return cv2.imread(img_filename)

    def get_lidar(self, idx):
        assert (idx < self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' %idx)
        lidar = np.fromfile(lidar_filename, dtype=np.float32)
        lidar = lidar.reshape((-1, 4))
        return lidar

    def get_planes(self, idx):
        assert (idx < self.num_samples)
        name = '%02d%04d.txt' %(self.video_id, idx)
        plane_dir = os.path.join(self.plane_dir, name)
        plane = open(plane_dir).readlines()[3].split()
        plane = [float(i) for i in plane]
        return plane

    def get_calibration(self):
        calib_filename = os.path.join(self.calib_dir, '%04d.txt' %self.video_id)
        return utils.Calibration(calib_filename)

    def get_oxts(self, idx):
        oxts_filename = os.path.join(self.oxts_dir, '%04d.txt' %self.video_id)
        lines = [line.rstrip() for line in open(oxts_filename).readlines()]
        oxts_line = lines[idx]
        return utils.Oxts(oxts_line)

    def get_label_objects(self, idx):
        assert (self.video_id < self.num_samples and self.split == 'training')
        label_filename = os.path.join(self.label_dir, '%04d.txt' %self.video_id)
        lines = [line.rstrip() for line in open(label_filename).readlines()]
        labels = []
        for line in lines:
            l = line.split(" ")
            if l[2] == 'DontCare': continue
            if int(l[0]) == idx:
                labels.append(utils.Object3d(line, datasets='tracking'))
            if int(l[0]) > idx:
                break
        return labels

    def get_transform(self, idx_ref, idx):
        '''get translation vector and rotation matrix, in order to represent previous frame
            in current frame's coordinate.

        input:
            idx_ref:    current frame id
            idx:        pre frame id

        output:
            distance:   translation vector      1 x 3
            matrix:     rotation matrix         3 x 3
        '''
        oxts_ref = self.get_oxts(idx_ref)
        oxts = self.get_oxts(idx)
        distance = oxts_ref.displacement(oxts)
        Rz = oxts_ref.get_rotate_matrix(oxts, 'z')
        Rx = oxts_ref.get_rotate_matrix(oxts, 'y')
        Ry = oxts_ref.get_rotate_matrix(oxts, 'x')
        matrix = Rz @ Rx @ Ry
        return distance, matrix

if __name__=='__main__':
    root_dir = ROOT_DIR + '/datasets/kitti/tracking'
    datasets = kitti_tracking_object(root_dir, 0)
    calib = datasets.get_calibration()
    calib.print_object()
    for i in range(len(datasets)):
        labels = datasets.get_label_objects(i)
        print("====== data[" + str(i) + "] =====")
        for label in labels:
            print("===== object =====")
            label.print_object()