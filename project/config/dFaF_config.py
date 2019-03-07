import sys
sys.path.append('../../')
from project.config.common_config import Config

class config(Config):
    # project name
    PROJECT_NAME = 'dFaF'

    # GPU and CPU
    device_ids = [2,3]
    n_cpu = 16

    #training
    n_frame = 5
    feature_channels = 2

    batch_size = 4
    epoches = 100

    #loss
    alpha = 3
    beta = 1
    gamma = 5
    neg_ratio = 3

    # optimizer
    lr = 0.01
    lr_scheduler_step = 10
    lr_scheduler_min = 3e-8

    # feature map size / input size, input_size: 400x352
    feature_map_rate = 1 / 8


