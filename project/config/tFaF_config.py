import sys
sys.path.append('../../')
from project.config.common_config import Config

class config(Config):
    # project name
    PROJECT_NAME = 'tFaF'

    # datasets spliting
    # split 22 videos into training set and validation set
    TRAIN_SPLIT = ['0000', '0002', '0004', '0006',
                   '0008', '0010', '0012', '0014',
                   '0016', '0018', '0020', '0001',
                   '0003', '0005', '0007', '0009', '0011']

    VAL_SPLIT = ['0013', '0015', '0017', '0019', '0021']

    VAL_TEST_SPLIT = ['0019']

    # GPU and CPU
    device_ids = [0, 1, 2, 3]
    n_cpu = 16

    # training setting
    n_frame = 5
    seq_len = 5
    stride = 1

    batch_size = 2
    epoches = 100

    # loss
    alpha = 1
    beta = 1
    gamma = 1
    neg_ratio = 5

    # optimizer
    lr = 3e-4
    lr_scheduler_step = 5
    lr_scheduler_min = 4e-8

    # feature map size / input size, input_size: 400x352
    feature_map_rate = 1 / 8