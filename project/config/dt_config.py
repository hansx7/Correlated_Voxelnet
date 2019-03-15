import sys
import os
import math

class config(object):
    # dir
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    KITTI_DETECTION_DATASET_ROOT = '/data/hk1/hansx/datasets/kitti/detection/'  # on server
    # KITTI_DETECTION_DATASET_ROOT = '/media/mooyu/Guoxs_Data/Datasets/Kitti/detection/'  # on pc

    KITTI_TRACKING_DATASET_ROOT = '/data/hk1/hansx/datasets/kitti/tracking/'  # on server
    # KITTI_TRACKING_DATASET_ROOT = '/media/mooyu/Guoxs_Data/Datasets/3D_Object_Tracking_Evaluation_2012/'

    EVAL_SCRIPT_DIR = ROOT_DIR + '/kitti_eval/evaluate_object_3d_offline'

    # project name
    PROJECT_NAME = 'voxelnet'

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

    # points cloud range
    xrange = [0, 70.4]
    yrange = [-40, 40]
    zrange = [-2, 2]

    # anchor size
    ANCHOR_L = 3.9
    ANHCOR_W = 1.6
    ANCHOR_H = 1.56

    # voxel size
    vd = 0.4
    vh = 0.2
    vw = 0.2

    # maximum number of points per voxel
    T = 35

    # voxel grid
    W = math.ceil((xrange[1] - xrange[0]) / vw)
    H = math.ceil((yrange[1] - yrange[0]) / vh)
    D = math.ceil((zrange[1] - zrange[0]) / vd)

    # feature map size / input size, input_size: 400x352
    feature_map_rate = 1 / 2

    # non-maximum suppression
    nms_threshold = 0.01
    score_threshold = 0.7
    anchors_per_position = 2

    # iou threshold
    pos_threshold = 0.6
    neg_threshold = 0.4

    cal_iou_method = 'from_avod'

    # training setting
    checkpoints_interval = 5  # epoch
    summary_interval = 5  # batch
    val_interval = 100  # batch
    test_interval = 20  # epoch
    viz_interval = 20  # batch
    mini_val_size = 100

    # for tensorboard birdview
    BV_LOG_FACTOR = 4

    # GPU and CPU
    batch_size_per_gpu = 1
    device_ids = [0]
    n_cpus = 16

    # training setting
    batch_size = batch_size_per_gpu * len(device_ids)
    epoches = 200

    # loss
    alpha = 3
    beta = 1
    gamma = 5
    neg_ratio = 3

    # optimizer
    lr = 0.01
    lr_scheduler_step = 150
    lr_scheduler_min = 4e-8
