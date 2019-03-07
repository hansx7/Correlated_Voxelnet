import sys
import numpy as np
sys.path.append('../../')
import dataLoader.kitti_utils as utils
from project.utils.targets_build import cal_target_to_label
from core.proj_utils import project_lidar_to_image
from dataLoader.kitti_tracking_object import kitti_tracking_object
from project.utils.utils import non_max_suppression, center_to_corner_box3d,\
    box3d_corner_to_center_batch


def generate_pre_label(scores, reg_head, names, calib_root, cfg):
    '''
    scores:    (batch_size, h, w, 2)
    reg_head:  (batch_size, h, w, 2*7)
    name:      (batch_size,)
    '''
    batch_size = scores.size()[0]
    batch_labels = []
    confs = []
    for i in range(batch_size):
        b_scores = scores[i].cpu().numpy()
        b_reg_head = reg_head[i].cpu().numpy()
        name = names[i]
        
        conf = b_scores.reshape(-1)
        target = b_reg_head.reshape(-1, 7)
        target = cal_target_to_label(target, cfg)
        # nms
        target, conf = non_max_suppression(target, conf, cfg)
        confs.append(conf)
        label_str = ''
        if len(target) != 0:
            boxes3d = center_to_corner_box3d(target)
            boxes3d = boxes3d[:, :, [1, 2, 0]]
            boxes3d[:, :, :2] = -boxes3d[:, :, :2]
            n_box = target.shape[0]
            for i in range(n_box):
                score = conf[i]
                box3d = boxes3d[i]
                label_str += '%s %.4f %.4f %.4f ' % (cfg.train_type, 0.0, 0.0, 0.0)
                
                # project the 3d bounding box into the image plane
                calib = utils.Calibration(calib_root+name.zfill(6)+'.txt')
                box2d = project_lidar_to_image(box3d, calib.P)  # (8, 2)
                xmin = np.min(box2d[:, 0])
                xmax = np.max(box2d[:, 0])
                ymin = np.min(box2d[:, 1])
                ymax = np.max(box2d[:, 1])
                label_str += '%.4f %.4f %.4f %.4f ' % (xmin, ymin, xmax, ymax)
                label_str += '%.4f %.4f %.4f ' % (target[i, 3], target[i, 4], target[i, 5])
                label_str += '%.4f %.4f %.4f %.4f ' %(-target[i, 1], -target[i, 2], target[i, 0], np.pi/2-target[i, -1])
                label_str += '%.4f\n' %(score)
        batch_labels.append(label_str)
    return batch_labels, confs


def generate_tracking_pre_label(prob, reg_head, data_list, data_root, cfg):
    '''
        generate tracking prediction result

        :param prob:                ndarray[h, w, cfg.anchors_per_position * cfg.n_frames]
        :param reg_head:            ndarray[h, w, cfg.anchors_per_position * cfg.n_frames, 7]
        :param data_list:           data batches, a list with element likes ['0000', 0, 1, 2, 3, 4]
        :param data_root:           path to datasets
        :param cfg:                 config

        :return:
        names:      String list
        pre_labels: String list
        confs:      float list
    '''
    frame_count = len(data_list) - 1
    datasets = kitti_tracking_object(data_root, int(data_list[0]))
    calib = datasets.get_calibration()
    names = []
    pre_labels = []
    confs = []
    for i in range(frame_count):
        frame_id = data_list[i + 1]
        frame_name = data_list[0]+ str(frame_id).zfill(4)
        names.append(frame_name[2:])   # '00180011 ==> 180011'

        anchor_n = cfg.anchors_per_position
        frame_reg_head = reg_head[:, :, anchor_n * i:anchor_n * (i + 1), :]   # (h, w, anchor_n, 7)
        frame_score = prob[:, :, anchor_n * i:anchor_n * (i + 1)]             # (h, w, anchor_n)
        target = frame_reg_head.reshape(-1, 7)                                # (h*w*anchor_n, 7)
        conf = frame_score.reshape(-1)
        target = cal_target_to_label(target, cfg)
        # nms
        target, conf = non_max_suppression(target, conf, cfg)
        confs.append(conf)
        label_str = ''
        if len(target) != 0:
            # transfer current frame coordinate to it own
            trans, rotation = datasets.get_transform(data_list[-1], frame_id)
            boxes3d = center_to_corner_box3d(target)
            boxes3d = boxes3d @ np.linalg.inv(rotation) + trans
            target = box3d_corner_to_center_batch(boxes3d)

            boxes3d = boxes3d[:, :, [1, 2, 0]]
            boxes3d[:, :, :2] = -boxes3d[:, :, :2]
            n_box = target.shape[0]
            for i in range(n_box):
                score = conf[i]
                box3d = boxes3d[i]
                label_str += '%s %.4f %.4f %.4f ' % (cfg.train_type, 0.0, 0.0, 0.0)

                # project the 3d bounding box into the image plane
                box2d = project_lidar_to_image(box3d, calib.P)  # (8, 2)
                xmin = np.min(box2d[:, 0])
                xmax = np.max(box2d[:, 0])
                ymin = np.min(box2d[:, 1])
                ymax = np.max(box2d[:, 1])
                label_str += '%.4f %.4f %.4f %.4f ' % (xmin, ymin, xmax, ymax)
                label_str += '%.4f %.4f %.4f ' % (target[i, 3], target[i, 4], target[i, 5])
                label_str += '%.4f %.4f %.4f %.4f ' % \
                             (-target[i, 1], -target[i, 2], target[i, 0], np.pi / 2 - target[i, -1])
                label_str += '%.4f\n' % (score)
        pre_labels.append(label_str)

    return names, pre_labels, confs