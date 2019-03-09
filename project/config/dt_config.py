import sys
import os

class config(object):
    # dir
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    KITTI_DETECTION_DATASET_ROOT = '/data/hk1/hansx/datasets/kitti/detection/'  # on server
    # KITTI_DETECTION_DATASET_ROOT = '/media/mooyu/Guoxs_Data/Datasets/Kitti/detection/'  # on pc

    KITTI_TRACKING_DATASET_ROOT = '/data/hk1/hansx/datasets/kitti/tracking/'  # on server
    # KITTI_TRACKING_DATASET_ROOT = '/media/mooyu/Guoxs_Data/Datasets/3D_Object_Tracking_Evaluation_2012/'

    EVAL_SCRIPT_DIR = ROOT_DIR + '/kitti_eval/evaluate_object_3d_offline'

    # project name
    PROJECT_NAME = 'dt'

    # datasets spliting
    # split 22 videos into training set and validation set
    TRAIN_SPLIT = ['0000', '0002', '0004', '0006',
                   '0008', '0010', '0012', '0014',
                   '0016', '0018', '0020', '0001',
                   '0003', '0005', '0007', '0009', '0011']

    VAL_SPLIT = ['0013', '0015', '0017', '0019', '0021']

    VAL_TEST_SPLIT = ['0019']

    # training setting
    train_type = 'Car'

    # training setting
    stride = 1

    # image size
    IMG_HEIGHT = 375
    IMG_WIDTH = 1242

    # points cloud range
    area_extents = [[-40, 40], [-1, 2.5], [0, 70]]
    num_slices = 7
    voxel_size = 0.1
