import sys
import math
sys.path.append('../../')
from project.config.common_config import Config

class config(Config):
    # project name
    PROJECT_NAME = 'voxelnet'

    # GPU and CPU
    batch_size_per_gpu = 1
    device_ids = [2]
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


    # points cloud range
    xrange = [0, 70.4]
    yrange = [-40, 40]
    zrange = [-2, 2]
    
    # voxel size
    vd = 0.4
    vh = 0.2
    vw = 0.2

    # voxel grid
    W = math.ceil((xrange[1] - xrange[0]) / vw)
    H = math.ceil((yrange[1] - yrange[0]) / vh)
    D = math.ceil((zrange[1] - zrange[0]) / vd)

    # feature map size / input size, input_size: 400x352
    feature_map_rate = 1 / 2

    # non-maximum suppression
    nms_threshold = 0.01
    score_threshold = 0.7

    cal_iou_method = 'from_avod'

