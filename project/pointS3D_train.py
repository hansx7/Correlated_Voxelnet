import os
import sys
import time
import shutil
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

sys.path.append('../')
from project.config.pointS3D_config import config as cfg
from project.models.pointS3D import PointS3D
from project.models.loss import VoxelLoss, LRMloss, LRMloss_v2
from project.utils.eval_utils import generate_tracking_pre_label
from project.utils.test_utils import generate_summary_img
from project.kittiDataLoader.pointS3D_data_loader import KittiTrackingDataset


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.01)


def train_iter(dataloader_train, model, optimizer, criterion,
               epoch, eval_index, writer, log, is_summary=True):
    epoch_size = len(dataloader_train)
    conf_loss = 0.0
    reg_loss = 0.0
    for i_batch, (data_list, voxel_features_seq, pos_equal_one, neg_equal_one, targets) \
            in enumerate(dataloader_train, 0):
        t0 = time.time()
        # wrapper to variable
        voxel_features = Variable(voxel_features_seq).float().cuda()
        pos_equal_one = Variable(pos_equal_one).float().cuda()
        neg_equal_one = Variable(neg_equal_one).float().cuda()
        targets = Variable(targets).float().cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        psm, rm = model(voxel_features)

        # calculate loss
        conf_loss, reg_loss, cls_pos_loss, cls_neg_loss = \
            criterion(rm, psm, pos_equal_one, neg_equal_one, targets)
        loss = conf_loss + reg_loss

        # backward
        loss.backward()
        optimizer.step()

        t1 = time.time()

        if is_summary:
            if (i_batch + 1) % cfg.summary_interval == 0:
                writer.add_scalar('train/conf_loss', conf_loss, epoch * epoch_size + i_batch)
                writer.add_scalar('train/reg_loss', reg_loss, epoch * epoch_size + i_batch)
                writer.add_scalar('train/cls_pos_loss', cls_pos_loss, epoch * epoch_size + i_batch)
                writer.add_scalar('train/cls_neg_loss', cls_neg_loss, epoch * epoch_size + i_batch)
                writer.add_scalar('train/total_loss', loss, epoch * epoch_size + i_batch)

        info = 'epoch ' + str(epoch) + '/' + str(cfg.epoches) + ' || iter ' + repr(i_batch) + '/' + str(epoch_size) + \
               ' || Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f || cls_pos_loss: %.4f || cls_neg_loss: %.4f' \
               ' || Timer: %.4f sec.' % (loss.data, conf_loss.data, reg_loss.data, cls_pos_loss.data,
                                         cls_neg_loss.data, (t1 - t0))
        print(info)
        log.write(info + '\n')

        # evalution step
        if (i_batch + 1) % cfg.val_interval == 0:
            print('========= evalution step start ========')
            eval_iter(model, criterion, eval_index, writer)
            eval_index += 1
            print('========= evalution step end ========')

    return model, conf_loss, reg_loss, eval_index


def eval_iter(model, criterion, eval_index, writer, is_summary=True):
    tconf_loss = []
    treg_loss = []
    tcls_pos_loss = []
    tcls_neg_loss = []
    tloss = []
    data_root = cfg.KITTI_TRACKING_DATASET_ROOT

    datasets_val = KittiTrackingDataset(data_root, seq_len=cfg.seq_len, stride=cfg.stride,
                                        set='mini_val', train_type=cfg.train_type)
    dataloader_val = DataLoader(datasets_val, batch_size=cfg.batch_size, num_workers=cfg.n_cpu,
                                shuffle=True, pin_memory=True, drop_last=True)

    for i_batch, (data_list, voxel_features_seq, pos_equal_one, neg_equal_one, targets,) \
            in enumerate(dataloader_val, 0):
        # wrapper to variable
        voxel_features = Variable(voxel_features_seq).float().cuda()
        pos_equal_one = Variable(pos_equal_one).float().cuda()
        neg_equal_one = Variable(neg_equal_one).float().cuda()
        targets = Variable(targets).float().cuda()

        # forward
        t0 = time.time()
        with torch.no_grad():
            psm, rm = model(voxel_features)

        # calculate loss
        conf_loss, reg_loss, cls_pos_loss, cls_neg_loss = \
            criterion(rm, psm, pos_equal_one, neg_equal_one, targets)
        loss = conf_loss + reg_loss

        t1 = time.time()

        tconf_loss.append(conf_loss.data)
        treg_loss.append(reg_loss.data)
        tcls_pos_loss.append(cls_pos_loss.data)
        tcls_neg_loss.append(cls_neg_loss.data)
        tloss.append(loss.data)

        print('Evalution step: ' + '|| Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f ||' \
                                   'cls_pos_loss: %.4f || cls_neg_loss: %.4f || Timer: %.4f sec.'
              % (loss.data, conf_loss.data, reg_loss.data, cls_pos_loss.data, cls_neg_loss.data, (t1 - t0)))

    # summary
    if is_summary:
        writer.add_scalar('val/conf_loss', np.mean(tconf_loss), eval_index)
        writer.add_scalar('val/reg_loss', np.mean(treg_loss), eval_index)
        writer.add_scalar('val/cls_pos_loss', np.mean(tcls_pos_loss), eval_index)
        writer.add_scalar('val/cls_neg_loss', np.mean(tcls_neg_loss), eval_index)
        writer.add_scalar('val/total_loss', np.mean(tloss), eval_index)


def test_iter(dataloader, model, eval_path, data_root, epoch, writer, is_summary=True):
    eval_root = eval_path + str(epoch) + '/data/'
    os.makedirs(eval_root, exist_ok=True)
    for i_batch, (data_list, voxel_features_seq, pos_equal_one, neg_equal_one, targets, images) \
            in enumerate(dataloader, 0):
        # wrapper to variable
        voxel_features = Variable(voxel_features_seq).float().cuda()

        # forward
        with torch.no_grad():
            psm, rm = model(voxel_features)

        prob = torch.sigmoid(psm.permute(0, 2, 3, 1))  # (batch_size, h, w, 2*5)
        reg_head = rm.permute(0, 2, 3, 1).contiguous()  # (batch_size, h, w, 10*7)
        reg_head = reg_head.view(reg_head.size(0),
                                 reg_head.size(1),
                                 reg_head.size(2), -1, 7)  # (batch_size, h, w, 10, 7)

        for i in range(prob.size()[0]):
            i_prob = prob[i].cpu().numpy()
            i_reg_head = reg_head[i].cpu().numpy()
            i_data_list = [item[i] for item in data_list]
            i_data_list[1:] = [int(i) for i in i_data_list[1:]]
            # (type, 0, 0, 0, xmin, ymin, xmax, ymax, h, w, l, xc, yc, zc, ry, score)
            names, pre_labels, confs = generate_tracking_pre_label(i_prob, i_reg_head, i_data_list,
                                                                   data_root, cfg)

            # only summary first one in a batch
            if is_summary and (i_batch + 1) % cfg.viz_interval == 0:
                if (i + 1) == prob.size()[0]:
                    image = images[i].numpy()
                    h_prob = i_prob[:, :, :2]
                    print(names[0], image.shape, h_prob.shape, len(pre_labels), len(confs))
                    front_image, bird_view, heatmap = \
                        generate_summary_img(names[0], image[0], h_prob, pre_labels[0],
                                             confs[0], data_root, cfg, datasets='tracking')

                    writer.add_image('predict_' + str(epoch) + '/front_image/', front_image, epoch)
                    writer.add_image('predict_' + str(epoch) + '/bird_view/', bird_view, epoch)
                    writer.add_image('predict_' + str(epoch) + '/heatmap/', heatmap, epoch)

            # store pre_labels
            for (name, pre_label) in zip(names, pre_labels):
                if not os.path.exists(eval_root + name + '.txt'):
                    with open(eval_root + name + '.txt', 'w+') as f:
                        f.write(pre_label)

        # run eval code on whole val datasets
    GT_DIR = os.path.join(cfg.ROOT_DIR, 'prediction/tFaF/label/')
    PRED_DIR = eval_path + str(epoch) + '/'
    OUTPUT_DIR = eval_path + str(epoch) + '/output.txt'
    code = 'nohup %s %s %s > %s 2>&1 &' % (cfg.EVAL_SCRIPT_DIR, GT_DIR, PRED_DIR, OUTPUT_DIR)
    os.system(code)


def train():
    print('Start training...')
    # get dataset
    data_root = cfg.KITTI_TRACKING_DATASET_ROOT
    datasets_train = KittiTrackingDataset(data_root, seq_len=cfg.seq_len, stride=cfg.stride,
                                          set='train', train_type=cfg.train_type)
    dataloader_train = DataLoader(datasets_train, batch_size=cfg.batch_size, num_workers=cfg.n_cpu,
                                  shuffle=True, pin_memory=True, drop_last=True)

    datasets_val = KittiTrackingDataset(data_root, seq_len=cfg.seq_len, stride=cfg.stride,
                                        set='val', train_type=cfg.train_type, using_img=True)
    dataloader_val = DataLoader(datasets_val, batch_size=cfg.batch_size, num_workers=cfg.n_cpu,
                                shuffle=True, pin_memory=True, drop_last=True)

    # network
    model = PointS3D(cfg)

    # add multiple GPUs
    main_gpu = cfg.device_ids[0]
    torch.cuda.set_device(main_gpu)
    device = torch.device("cuda:%d" % main_gpu if torch.cuda.is_available() else "cpu")
    n_cuda = len(cfg.device_ids)
    if n_cuda > 1:
        print("Using GPUs: ", cfg.device_ids)
        model = nn.DataParallel(model, cfg.device_ids)

    model.to(device)
    model.train()

    # print total parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total Trainable params: ', params)

    # initialization
    print('Initializing weights...')
    model.apply(weights_init)
    # model.module.load_state_dict(torch.load('./checkpoints/'+ cfg.PROJECT_NAME +'/2018-12-22_22/4s.pkl'))

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    schedular = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_scheduler_step, gamma=0.1)
    # schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.lr_scheduler_step,
    #                                                cfg.lr_scheduler_min)

    # define loss function
    # criterion = VoxelLoss(cfg)
    criterion = LRMloss(cfg)

    now_time = datetime.datetime.now()
    date_dir = datetime.datetime.strftime(now_time, '%Y-%m-%d_%H')
    log_dir = './log_files/' + cfg.PROJECT_NAME
    os.makedirs(log_dir, exist_ok=True)
    logname = log_dir + '/log_%s.txt' % date_dir
    logfile = open(logname, 'w')

    # tensorboard writer
    tensorboard_root = './TB_logs/' + cfg.PROJECT_NAME + '/' + date_dir
    if os.path.exists(tensorboard_root):
        for file in os.listdir(tensorboard_root):
            os.remove(os.path.join(tensorboard_root,file))
    writer = SummaryWriter(tensorboard_root)

    eval_index = 0
    for i in range(cfg.epoches):
        schedular.step()
        model, conf_loss, reg_loss, eval_index = train_iter(dataloader_train, model, optimizer,
                                                            criterion, i, eval_index, writer, logfile)

        # save model
        if (i + 1) % cfg.checkpoints_interval == 0:
            print("Saving models...")
            if n_cuda > 1:
                model_dict = model.module.state_dict()
            else:
                model_dict = model.state_dict()

            location = "checkpoints/%s/%s" % (cfg.PROJECT_NAME, date_dir)
            os.makedirs(location, exist_ok=True)
            torch.save(model_dict, location + '/%ds.pkl' % i)

        # testing
        if (i + 1) % cfg.test_interval == 0:
            eval_root = './prediction/' + cfg.PROJECT_NAME + '/' + date_dir + '/'
            os.makedirs(eval_root, exist_ok=True)
            # copy *_config.py and train.py to eval_root
            shutil.copy('./config/' + cfg.PROJECT_NAME + '_config.py',
                        eval_root + cfg.PROJECT_NAME + '_config.py')
            shutil.copy('./' + cfg.PROJECT_NAME + '_train.py',
                        eval_root + cfg.PROJECT_NAME + '_train.py')
            print('====== Start Testing ======')
            test_iter(dataloader_val, model, eval_root, data_root, i, writer)
            print('====== End Testing ======')

    writer.close()
    logfile.close()


if __name__ == '__main__':
    train()