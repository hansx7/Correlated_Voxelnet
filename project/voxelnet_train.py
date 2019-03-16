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
from project.utils.utils import run_eval
from project.models.voxelnet import VoxelNet
from project.models.loss import VoxelLoss, LRMloss
from project.config.dt_config import config as cfg
from project.utils.eval_utils import generate_pre_label
from project.utils.test_utils import generate_summary_img
# from project.kittiDataLoader.voxelnet_data_loader import KittiDetectionDataset
from project.kittiDataLoader.dt_data_loader import KittiDTDataset

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.01)

        
def detection_collate(batch):
    voxel_features = []
    voxel_coords = []
    pos_equal_one = []
    neg_equal_one = []
    targets = []

    image = []
    name = []
    lens = [item[1].shape[0] for item in batch]
    max_len = np.max(lens)
    for i, sample in enumerate(batch):
        name.append(sample[0])
        N = len(sample[1])
        voxel_features.append(
            np.pad(sample[1], ((0, max_len-N),(0, 0),(0, 0)),
                mode='constant', constant_values=0))
        voxel_coords.append(
            np.pad(sample[2], ((0, max_len-N), (0, 0)),
                mode='constant', constant_values=-1))

        pos_equal_one.append(sample[3])
        neg_equal_one.append(sample[4])
        targets.append(sample[5])
        image.append(sample[6]) 

    return name, np.concatenate([voxel_features], axis=0), \
           np.concatenate([voxel_coords], axis=0),  \
           np.array(pos_equal_one), np.array(neg_equal_one), \
           np.array(targets), np.array(image)
    
    
def train_iter(dataloader_train, model, optimizer, criterion, epoch, eval_index,
               writer, log, is_summary=True):
    epoch_size = len(dataloader_train)
    conf_loss = 0
    reg_loss = 0
    for i_batch, (names, lidars, images, labels, num_boxes, voxel_features, voxel_coords, voxel_mask, \
                  pos_equal_one, neg_equal_one, targets, targets_diff) in enumerate(dataloader_train, 0):
        t0 = time.time()
        # wrapper to variable
        # voxel_features = Variable(torch.from_numpy(voxel_features)).float().cuda()
        # voxel_coords = Variable(torch.from_numpy(voxel_coords)).long().cuda()
        # pos_equal_one = Variable(torch.from_numpy(pos_equal_one)).float().cuda()
        # neg_equal_one = Variable(torch.from_numpy(neg_equal_one)).float().cuda()
        # targets = Variable(torch.from_numpy(targets)).float().cuda()
        voxel_features = Variable(voxel_features).float().cuda()
        voxel_coords = Variable(voxel_coords).long().cuda()
        pos_equal_one = Variable(pos_equal_one).float().cuda()
        neg_equal_one = Variable(neg_equal_one).float().cuda()
        targets = Variable(targets).float().cuda()
        targets_diff = Variable(targets_diff).float().cuda()
        
        # zero the parameter gradient
        optimizer.zero_grad()

        # forward
        psm0, rm0, psm1, rm1, corr = model(voxel_features, voxel_coords, voxel_mask)

        corr = Variable(corr).float().cuda()

        # calculate loss
        conf_loss0, reg_loss0, cls_pos_loss0, cls_neg_loss0, corr_loss0 = \
            criterion(rm0, psm0, pos_equal_one[:, :, :, :2], neg_equal_one[:, :, :, :2], targets[:, :, :, :14], \
                      corr[:, :4, :, :], targets_diff[:, :, :, :7])
        conf_loss1, reg_loss1, cls_pos_loss1, cls_neg_loss1, corr_loss1 = \
            criterion(rm1, psm1, pos_equal_one[:, :, :, 2:], neg_equal_one[:, :, :, 2:], targets[:, :, :, 14:], \
                      corr[:, 4:, :, :], targets_diff[:, :, :, 7:])
        conf_loss = conf_loss0 + conf_loss1
        reg_loss = reg_loss0 + reg_loss1
        cls_pos_loss = cls_pos_loss0 + cls_pos_loss1
        cls_neg_loss = cls_neg_loss0 + cls_neg_loss1
        corr_loss = corr_loss0 + corr_loss1
        loss = conf_loss + reg_loss + corr_loss

        # backward
        loss.backward()
        optimizer.step()

        t1 = time.time()
        
        if is_summary:
            if (i_batch + 1) % cfg.summary_interval == 0:
                writer.add_scalar('data/conf_loss',     conf_loss,    epoch * epoch_size + i_batch)
                writer.add_scalar('data/reg_loss',      reg_loss,     epoch * epoch_size + i_batch)
                writer.add_scalar('data/corr_loss',     corr_loss,    epoch * epoch_size + i_batch)
                writer.add_scalar('data/cls_pos_loss',  cls_pos_loss, epoch * epoch_size + i_batch)
                writer.add_scalar('data/cls_neg_loss',  cls_neg_loss, epoch * epoch_size + i_batch)
                writer.add_scalar('data/total_loss',    loss,         epoch * epoch_size + i_batch)

        info = 'epoch '+str(epoch)+'/'+str(cfg.epoches)+' || iter '+repr(i_batch)+'/'+str(epoch_size)+ \
               ' || Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f || corr_loss : %.4f || cls_pos_loss: %.4f || ' \
               'cls_neg_loss: %.4f || Timer: %.4f sec.' % (loss.data, conf_loss.data, reg_loss.data, corr_loss,
                                                           cls_pos_loss.data, cls_neg_loss.data, (t1 - t0))
        print(info)
        log.write(info+'\n')
        
        # evalution step
        if (i_batch + 1) % cfg.val_interval == 0:
            print('========= evalution step start ========')
            eval_iter(model, criterion, eval_index, writer)
            eval_index += 1
            print('========= evalution step end =========')
        
    return model, conf_loss, reg_loss, eval_index


def eval_iter(model, criterion, eval_index, writer, is_summary=True):
    tconf_loss = []
    treg_loss = []
    tcls_pos_loss = []
    tcls_neg_loss = []
    tloss = []
    data_root = cfg.KITTI_DETECTION_DATASET_ROOT

    # datasets_val = KittiDetectionDataset(data_root, set='mini_val', train_type=cfg.train_type, using_img=True)
    datasets_val = KittiDTDataset(data_root, set='mini_val', train_type=cfg.train_type, using_img=True)
    dataloader_val = DataLoader(datasets_val, batch_size=cfg.batch_size, num_workers=cfg.n_cpus,
                                collate_fn=detection_collate, shuffle=False, pin_memory=False, drop_last=True)

    for i_batch, (names, voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, image) \
            in enumerate(dataloader_val, 0):

        # wrapper to variable
        voxel_features = Variable(torch.from_numpy(voxel_features)).float().cuda()
        voxel_coords = Variable(torch.from_numpy(voxel_coords)).long().cuda()
        pos_equal_one = Variable(torch.from_numpy(pos_equal_one)).float().cuda()
        neg_equal_one = Variable(torch.from_numpy(neg_equal_one)).float().cuda()
        targets = Variable(torch.from_numpy(targets)).float().cuda()
        
        # forward
        t0 = time.time()
        with torch.no_grad():
            psm, rm = model(voxel_features, voxel_coords)
            
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
        
        print('Evalution step: '+ '|| Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f ||' \
              'cls_pos_loss: %.4f || cls_neg_loss: %.4f || Timer: %.4f sec.' % (loss.data,
                conf_loss.data, reg_loss.data, cls_pos_loss.data, cls_neg_loss.data, (t1 - t0)))
            
    # summary
    if is_summary:
        writer.add_scalar('val/conf_loss',    np.mean(tconf_loss),    eval_index)
        writer.add_scalar('val/reg_loss',     np.mean(treg_loss),     eval_index)
        writer.add_scalar('val/cls_pos_loss', np.mean(tcls_pos_loss), eval_index)
        writer.add_scalar('val/cls_neg_loss', np.mean(tcls_neg_loss), eval_index)
        writer.add_scalar('val/total_loss',   np.mean(tloss),         eval_index)
    

def test_iter(dataloader, model, eval_path, data_root, epoch, writer, is_summary=True):
    calib_root = cfg.KITTI_DETECTION_DATASET_ROOT + 'training/calib/'
    eval_root = eval_path + str(epoch) + '/data/'
    os.makedirs(eval_root, exist_ok=True)
    for i_batch, (names, voxel_features, voxel_coords, pos_equal_one, neg_equal_one,targets, images) \
            in enumerate(dataloader, 0):
        # wrapper to variable
        voxel_features = Variable(torch.from_numpy(voxel_features)).float().cuda()
        voxel_coords = Variable(torch.from_numpy(voxel_coords)).long().cuda()
        
        with torch.no_grad():
            psm, rm = model(voxel_features, voxel_coords)
        
        # generate predicted label and stored
        prob = torch.sigmoid(psm.permute(0, 2, 3, 1))      # (batch_size, h, w, 2)
        reg_head = rm.permute(0, 2, 3, 1).contiguous()     # (batch_size, h, w, 2*7)
        reg_head = reg_head.view(reg_head.size(0),
                                 reg_head.size(1),
                                 reg_head.size(2), -1, 7)

        # (type, 0, 0, 0, xmin, ymin, xmax, ymax, h, w, l, xc, yc, zc, ry, score)
        pre_labels, scores = generate_pre_label(prob, reg_head, names, calib_root, cfg)

        for i in range(len(names)):
            name = names[i].zfill(6)+'.txt'
            pre_label = pre_labels[i]
            with open(eval_root + name, 'w+') as f:
                   f.write(pre_label)
        
        # only summry 1 in a batch
        if is_summary and (i_batch+1) % cfg.viz_interval == 0:
            prob = prob.cpu().numpy()
            front_image, bird_view, heatmap = \
                generate_summary_img(names[0], images[0], prob[0],
                                     pre_labels[0], scores[0], data_root, cfg, datasets='detection')
            writer.add_image('predict_'+str(epoch)+'/front_image/', front_image,  epoch)
            writer.add_image('predict_'+str(epoch)+'/bird_view/',   bird_view,    epoch)
            writer.add_image('predict_'+str(epoch)+'/heatmap/',     heatmap,      epoch)

    # run eval code on whole val datasets
    run_eval(data_root, eval_path, epoch, cfg)


def train():
    print('Start training...')
    # get dataset
    data_root = cfg.KITTI_TRACKING_DATASET_ROOT
    # datasets_train = KittiDetectionDataset(data_root, set='train', train_type=cfg.train_type, using_img=True)
    datasets_train = KittiDTDataset(data_root, set='train', train_type=cfg.train_type, using_img=True)
    dataloader_train = DataLoader(datasets_train, batch_size=cfg.batch_size, num_workers=cfg.n_cpus,
                                  shuffle=False, pin_memory=True, drop_last=True)
    
    # datasets_val = KittiDetectionDataset(data_root, set='val', train_type=cfg.train_type, using_img=True)
    datasets_val = KittiDTDataset(data_root, set='val', train_type=cfg.train_type, using_img=True)
    dataloader_val = DataLoader(datasets_val, batch_size=cfg.batch_size, num_workers=cfg.n_cpus,
                                shuffle=False, pin_memory=False, drop_last=True)
    
    # model
    model = VoxelNet(cfg)

    # add multiple GPUs
    main_gpu = cfg.device_ids[0]
    torch.cuda.set_device(main_gpu)
    device = torch.device("cuda:%d" % main_gpu if torch.cuda.is_available() else "cpu")
    if len(cfg.device_ids) > 1:
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
    # model.module.load_state_dict(torch.load('./checkpoints/'+cfg.PROJECT_NAME+'/2019-01-08_15/9s.pkl'))

    
    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr)
    # optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    schedular = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_scheduler_step, gamma=0.1)
    # schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, 2, 4e-8)

    # define loss function
    criterion = VoxelLoss(cfg)
    # criterion = LRMloss(cfg)

    now_time = datetime.datetime.now()
    date_dir = datetime.datetime.strftime(now_time,'%Y-%m-%d_%H')
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
        # training step
        model, conf_loss, reg_loss, eval_index = train_iter(dataloader_train, model, optimizer,
                                                            criterion, i, eval_index, writer, logfile)
            
        # save model
        if (i+1) % cfg.checkpoints_interval == 0:
            print("====== Start Saving models ======")
            if len(cfg.device_ids) > 1:
                model_dict = model.module.state_dict()
            else:
                model_dict = model.state_dict()

            location = "checkpoints/%s/%s" % (cfg.PROJECT_NAME, date_dir)
            os.makedirs(location, exist_ok=True)
            torch.save(model_dict, location + '/%ds.pkl' % i)
            print("====== End Saving models ======")
        
        # testing step
        if (i+1) % cfg.test_interval == 0:
            eval_root = './prediction/' + cfg.PROJECT_NAME + '/' + date_dir + '/'
            os.makedirs(eval_root, exist_ok=True)
            # copy voxelnet_config.py to eval_root
            shutil.copy('./config/'+cfg.PROJECT_NAME+'_config.py',
                        eval_root + cfg.PROJECT_NAME+'_config.py')
            shutil.copy('./' + cfg.PROJECT_NAME + '_train.py',
                        eval_root + cfg.PROJECT_NAME + '_train.py')
            print('====== Start Testing ======')
            test_iter(dataloader_val, model, eval_root, data_root, i, writer)
            print('====== End Testing ======')

    writer.close()
    logfile.close()


if __name__ == '__main__':
    train()