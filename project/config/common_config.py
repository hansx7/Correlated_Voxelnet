import math
import os

class Config:
    # dir
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    KITTI_DETECTION_DATASET_ROOT = '/data/hk1/datasets/kitti/detection/'  # on server
    # KITTI_DETECTION_DATASET_ROOT = '/media/mooyu/Guoxs_Data/Datasets/Kitti/detection/'  # on pc

    KITTI_TRACKING_DATASET_ROOT = '/data/hk1/datasets/kitti/tracking/' # on server
    # KITTI_TRACKING_DATASET_ROOT = '/media/mooyu/Guoxs_Data/Datasets/3D_Object_Tracking_Evaluation_2012/'

    EVAL_SCRIPT_DIR = ROOT_DIR + '/kitti_eval/evaluate_object_3d_offline'

    # training setting
    train_type = 'Car'
    checkpoints_interval = 5  # epoch
    summary_interval = 5      # batch
    val_interval = 100        # batch
    test_interval = 20        # epoch
    viz_interval = 20         # batch
    mini_val_size = 100

    # points cloud range
    xrange = [0, 70.4]
    yrange = [-38.4, 38.4]
    zrange = [-2, 2]

    # anchor size
    ANCHOR_L = 3.9
    ANHCOR_W = 1.6
    ANCHOR_H = 1.56

    # voxel size
    vd = 0.2
    vh = 0.2
    vw = 0.2

    # voxel grid
    W = math.ceil((xrange[1] - xrange[0]) / vw)
    H = math.ceil((yrange[1] - yrange[0]) / vh)
    D = math.ceil((zrange[1] - zrange[0]) / vd)

    # maxiumum number of points per voxel
    T = 35

    anchors_per_position = 2

    # iou threshold
    pos_threshold = 0.6
    neg_threshold = 0.4

    # non-maximum suppression
    nms_threshold = 0.3
    score_threshold = 0.5

    # image size
    IMG_HEIGHT = 375
    IMG_WIDTH = 1242

    # for tensorboard birdview
    BV_LOG_FACTOR = 4

    cal_iou_method = 'from_avod'