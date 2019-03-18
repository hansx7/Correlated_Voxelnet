from __future__ import print_function

import os
import cv2
import sys
sys.path.append('..')
import numpy as np
import dataLoader.kitti_utils as utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class kitti_detection_object(object):
    '''Load and parse kitti detection data into a usable format.'''

    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            # self.num_samples = 7481
            self.num_samples = 3224
        elif split == 'testing':
            # self.num_samples = 7518
            self.num_samples = 11095
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')
        self.oxts_dir = os.path.join(self.split_dir, 'oxts')
        self.plane_dir = os.path.join(self.split_dir, 'planes')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
        return cv2.imread(img_filename)

    def get_lidar(self, idx):
        assert (idx < self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))
        lidar = np.fromfile(lidar_filename, dtype=np.float32)
        lidar = lidar.reshape((-1, 4))
        return lidar

    def get_planes(self, idx):
        assert (idx < self.num_samples)
        plane_dir = os.path.join(self.plane_dir, '%06d.txt' %(idx))
        plane = open(plane_dir).readlines()[3].split()
        plane = [float(i) for i in plane]
        return plane

    def get_calibration(self, idx):
        assert (idx < self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert (idx < self.num_samples and self.split == 'training')
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        lines = [line.rstrip() for line in open(label_filename)]
        labels = [utils.Object3d(line) for line in lines]
        return labels

    def get_oxts(self, idx):
        video_id = idx / 10000
        oxts_filename = os.path.join(self.oxts_dir, '%04d.txt' %video_id)
        lines = [line.rstrip() for line in open(oxts_filename).readlines()]
        oxts_line = lines[idx]
        return utils.Oxts(oxts_line)

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
    root_dir = ROOT_DIR + '/datasets/kitti/detection'
    datasets = kitti_detection_object(root_dir)
    for i in range(len(datasets)):
        labels = datasets.get_label_objects(i)
        print("====== data[" + str(i) + "] =====")
        for label in labels:
            print("===== object =====")
            label.print_object()

        calib = datasets.get_calibration(i)
        calib.print_object()

