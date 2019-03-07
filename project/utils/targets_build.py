import sys
import numpy as np
sys.path.append('..')
from project.utils.utils import box3d_corner_to_center_batch, cal_iou


def cal_batch_targets(labels, cfg):
    '''calculate a batch of target
    input:
        labels:     seq_len*{'boxes3d': (B, 8, 3), 'ories3d': (B, 2, 3)}
    output:
        pos_equal_one:      (w, l, anchors_per_position*seq_len)
        neg_equal_one:      (w, l, anchors_per_position*seq_len)
        targets:            (w, l, anchors_per_position*seq_len * 7)
    '''
    batch_pos_equal_one = []
    batch_neg_equal_one = []
    batch_targets = []
    feature_map_size = (int(cfg.H * cfg.feature_map_rate), int(cfg.W * cfg.feature_map_rate))
    for label in labels:
        if label == {}:
            pos_equal_one = np.zeros((*feature_map_size, cfg.anchors_per_position))
            neg_equal_one = np.zeros((*feature_map_size, cfg.anchors_per_position))
            targets = np.zeros((*feature_map_size, 7 * cfg.anchors_per_position))
        else:
            box3d = label['boxes3d']
            #ori3d = label['ories3d']
            pos_equal_one, neg_equal_one, targets = cal_target(box3d, cfg)
        batch_pos_equal_one.append(pos_equal_one)
        batch_neg_equal_one.append(neg_equal_one)
        batch_targets.append(targets)
    batch_pos_equal_one = np.concatenate(batch_pos_equal_one, axis=-1)
    batch_neg_equal_one = np.concatenate(batch_neg_equal_one, axis=-1)
    batch_targets = np.concatenate(batch_targets, axis=-1)
    return batch_pos_equal_one, batch_neg_equal_one, batch_targets


def cal_target(gt_box3d, cfg):
    ''' calculate target
    input:
        gt_box3d:           (B, 8, 3)
        *default_anchors:   (w, l, anchors_per_position, 7) (w, l, anchors_per_position, 7)
    output:
        pos_equal_one:      (w, l, 2)
        neg_equal_one:      (w, l, 2)
        targets:            (w, l, 2*7)

    attention: IoU is calculate on birdview
    '''
    default_anchors = compute_default_anchors(cfg).reshape(-1, 7)
    anchors_d = np.sqrt(default_anchors[:, 4] ** 2 + default_anchors[:, 5] ** 2)

    # feature map shape, default is (50, 44)
    feature_map_shape = (int(cfg.H * cfg.feature_map_rate), int(cfg.W * cfg.feature_map_rate))
    # 2 pre defined bounding box
    pos_equal_one = np.zeros((*feature_map_shape, cfg.anchors_per_position))
    neg_equal_one = np.zeros((*feature_map_shape, cfg.anchors_per_position))
    targets = np.zeros((*feature_map_shape, cfg.anchors_per_position * 7))

    # (N, 8, 3) ——> (N, 7)
    gt_xyzhwlr = box3d_corner_to_center_batch(gt_box3d)

    iou = cal_iou(default_anchors, gt_xyzhwlr, cfg)

    # get maximum gt_box3d anchor's id
    id_highest = np.argmax(iou.T, axis=1)
    id_highest_gt = np.arange(iou.T.shape[0])
    mask = iou.T[id_highest_gt, id_highest] > 0
    # remove negative iou
    id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]
    # find anchor iou > cfg.pos_threshold
    id_pos, id_pos_gt = np.where(iou > cfg.pos_threshold)
    # find anchor iou < cfg.neg_threshold
    id_neg = np.where(np.sum(iou < cfg.neg_threshold, axis=1) == iou.shape[1])[0]

    id_pos = np.concatenate([id_pos, id_highest])
    id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])

    id_pos, index = np.unique(id_pos, return_index=True)
    id_pos_gt = id_pos_gt[index]
    id_neg.sort()
    # cal the target and set the equal one
    index_x, index_y, index_z = np.unravel_index(
        id_pos, (*feature_map_shape, cfg.anchors_per_position))

    pos_equal_one[index_x, index_y, index_z] = 1

    # ATTENTION: index_z should be np.array
    targets[index_x, index_y, np.array(index_z) * 7] = \
        (gt_xyzhwlr[id_pos_gt, 0] - default_anchors[id_pos, 0]) / anchors_d[id_pos]
    targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
        (gt_xyzhwlr[id_pos_gt, 1] - default_anchors[id_pos, 1]) / anchors_d[id_pos]
    targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
        (gt_xyzhwlr[id_pos_gt, 2] - default_anchors[id_pos, 2]) / default_anchors[id_pos, 3]
    targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
        gt_xyzhwlr[id_pos_gt, 3] / default_anchors[id_pos, 3])
    targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
        gt_xyzhwlr[id_pos_gt, 4] / default_anchors[id_pos, 4])
    targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
        gt_xyzhwlr[id_pos_gt, 5] / default_anchors[id_pos, 5])
    targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
            gt_xyzhwlr[id_pos_gt, 6] - default_anchors[id_pos, 6])

    index_x, index_y, index_z = np.unravel_index(
        id_neg, (*feature_map_shape, cfg.anchors_per_position))
    neg_equal_one[index_x, index_y, index_z] = 1
    # to avoid a box be pos/neg in the same time
    index_x, index_y, index_z = np.unravel_index(
        id_highest, (*feature_map_shape, cfg.anchors_per_position))
    neg_equal_one[index_x, index_y, index_z] = 0

    return pos_equal_one, neg_equal_one, targets


def compute_default_anchors(cfg):
    #   anchors: (w, l, cfg.anchors_per_position, 7) x y z h w l r
    x = np.linspace(cfg.xrange[0] + cfg.vw, cfg.xrange[1] - cfg.vw, cfg.W * cfg.feature_map_rate)
    y = np.linspace(cfg.yrange[0] + cfg.vh, cfg.yrange[1] - cfg.vh, cfg.H * cfg.feature_map_rate)
    cx, cy = np.meshgrid(x, y)
    # all is (w, l, cfg.anchors_per_position)
    cx = np.tile(cx[..., np.newaxis], cfg.anchors_per_position)
    cy = np.tile(cy[..., np.newaxis], cfg.anchors_per_position)
    cz = np.ones_like(cx)
    w = np.ones_like(cx) * cfg.ANHCOR_W
    l = np.ones_like(cx) * cfg.ANCHOR_L
    h = np.ones_like(cx) * cfg.ANCHOR_H
    r = np.ones_like(cx)
    if cfg.anchors_per_position == 2:
        cz = cz * -1.8
        r[..., 0] = 0
        r[..., 1] = np.pi / 2
    else:   # cfg.anchors_per_position == 4
        cz[..., :2] = -0.2
        cz[..., 2:] = -1.8
        r[..., [0,2]] = 0
        r[..., [1,3]] = np.pi / 2
    anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1)
    return anchors


def cal_target_to_label(targets, cfg, batch=False):
    '''
    xg = xt * da + xd, yg = yt * da + yt, zg = zt * hd + zt
    hg = e^(ht) * hd, wg = e^(wt) * wd, lg = e^(lt) * ld
    theta_g = theta_t + theta_d

    input:
        targets:    (N, 7)
    '''
    # (w*l*anchors_per_position, 7)
    default_anchors = compute_default_anchors(cfg).reshape(-1, 7)

    if batch:
        n = targets.shape[0] // default_anchors.shape[0]
        default_anchors = [default_anchors]*n
        default_anchors = np.concatenate(default_anchors, 0)
        default_anchors = default_anchors.reshape(-1, 7)

    anchors_d = np.sqrt(default_anchors[:, 4] ** 2 + default_anchors[:, 5] ** 2)
    new_targets = np.zeros_like(targets)

    new_targets[:, 0] = targets[:, 0] * anchors_d + default_anchors[:, 0]
    new_targets[:, 1] = targets[:, 1] * anchors_d + default_anchors[:, 1]
    new_targets[:, 2] = targets[:, 2] * default_anchors[:, 3] + default_anchors[:, 2]

    new_targets[:, 3] = np.exp(np.minimum(targets[:, 3], 50)) * default_anchors[:, 3]
    new_targets[:, 4] = np.exp(np.minimum(targets[:, 4], 50)) * default_anchors[:, 4]
    new_targets[:, 5] = np.exp(np.minimum(targets[:, 5], 50)) * default_anchors[:, 5]

    new_targets[:, 6] = targets[:, 6] + default_anchors[:, 6]
    return new_targets