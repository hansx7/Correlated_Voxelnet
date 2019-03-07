import numpy as np


def get_filtered_seq(cfg, lidars, boxes3ds=None):
    '''get specific data in a range
    input:
        lidars: lidar data list, seq_len * (N, 3)
        boxes3ds: label data list, seq_len * (B, 8, 3)
    '''
    for idx in range(len(lidars)):
        lidar = lidars[idx]
        if boxes3ds is not None:
            boxes3d = boxes3ds[idx]
            lidar, boxes3d = get_filtered_lidar(cfg, lidar, boxes3d)
            lidars[idx] = lidar
            boxes3ds[idx] = boxes3d
        else:
            lidar = get_filtered_lidar(cfg, lidar)
            lidars[idx] = lidar

    if boxes3ds is not None:
        return lidars, boxes3ds
    else:
        return lidars


def get_filtered_lidar(cfg, lidar, boxes3d=None):
    '''get specific data in a range
    input:
        lidars: lidar data, (N, 3)
        boxes3d: label data, (B, 8, 3)
    '''
    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]

    filter_x = np.where((pxs >= cfg.xrange[0]) & (pxs < cfg.xrange[1]))[0]
    filter_y = np.where((pys >= cfg.yrange[0]) & (pys < cfg.yrange[1]))[0]
    filter_z = np.where((pzs >= cfg.zrange[0]) & (pzs < cfg.zrange[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)
    filter_xyz = np.intersect1d(filter_xy, filter_z)

    if boxes3d is not None:
        if len(boxes3d) > 0:
            box_x = (boxes3d[:, :, 0] >= cfg.xrange[0]) & (boxes3d[:, :, 0] < cfg.xrange[1])
            box_y = (boxes3d[:, :, 1] >= cfg.yrange[0]) & (boxes3d[:, :, 1] < cfg.yrange[1])
            box_z = (boxes3d[:, :, 2] >= cfg.zrange[0]) & (boxes3d[:, :, 2] < cfg.zrange[1])
            box_xyz = np.sum(box_x & box_y & box_z, axis=1)
            return lidar[filter_xyz], boxes3d[box_xyz > 0]
        else:
            return lidar[filter_xyz], boxes3d

    return lidar[filter_xyz]


def box3d_corner_to_center_batch(box3d_corner):
    '''(B, 8, 3) ——> (B, 7) [xc, yc, zc, h, w, l, r]

    qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''

    assert box3d_corner.ndim == 3
    batch_size = box3d_corner.shape[0]
    # cal xc, yc, zc
    xyz = np.mean(box3d_corner[:, :4, :], axis=1)

    h = abs(np.mean(box3d_corner[:, 4:, 2] - box3d_corner[:, :4, 2], axis=1, keepdims=True))
    w = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 2, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 5, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 6, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    l = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 1, [0, 1]] - box3d_corner[:, 2, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 5, [0, 1]] - box3d_corner[:, 6, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    theta = (np.arctan2(abs(box3d_corner[:, 2, 1] - box3d_corner[:, 1, 1]),
                        abs(box3d_corner[:, 2, 0] - box3d_corner[:, 1, 0])) +
             np.arctan2(abs(box3d_corner[:, 3, 1] - box3d_corner[:, 0, 1]),
                        abs(box3d_corner[:, 3, 0] - box3d_corner[:, 0, 0])) +
             np.arctan2(abs(box3d_corner[:, 2, 0] - box3d_corner[:, 3, 0]),
                        abs(box3d_corner[:, 3, 1] - box3d_corner[:, 2, 1])) +
             np.arctan2(abs(box3d_corner[:, 1, 0] - box3d_corner[:, 0, 0]),
                        abs(box3d_corner[:, 0, 1] - box3d_corner[:, 1, 1])))[:, np.newaxis] / 4

    return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(batch_size, 7)


def anchors_center_to_corner(anchors):
    '''generate birdview box2d in lidar coordinate system
    input:
        anchors:        (N, 7) [xc, yc, zc, h, w, l, r]
    output:
        anchor_corner:  (N, 4, 2)
    '''
    N = anchors.shape[0]
    anchor_corner = np.zeros((N, 4, 2))
    for i in range(N):
        anchor = anchors[i]
        translation = anchor[0:3]
        h, w, l = anchor[3:6]
        rz = anchor[-1]
        Box = np.array([
            [-l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2]])
        # re-create 3D bounding box in velodyne coordinate system
        rotMat = np.array([
            [np.cos(rz), -np.sin(rz)],
            [np.sin(rz), np.cos(rz)]])
        velo_box = np.dot(rotMat, Box)
        cornerPosInVelo = velo_box + np.tile(translation[:2], (4, 1)).T
        box2d = cornerPosInVelo.transpose()
        anchor_corner[i] = box2d
    return anchor_corner


def center_to_corner_box3d(boxes_center):
    '''generate birdview box2d in lidar coordinate system
        input:
            boxes_center:    (N, 7) [xc, yc, zc, h, w, l, r]
        output:
            boxes3d:         (N, 8, 3)
        '''
    N = boxes_center.shape[0]
    translation = boxes_center[:, :3]  # (N,3)
    w = boxes_center[:, 4]  # (N,)
    l = boxes_center[:, 5]
    h = boxes_center[:, 3]
    rz = boxes_center[:, -1]
    zeros = np.zeros((N,))
    trackletBox = np.array([
        # in velodyne coordinates around zero point and without orientation yet
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [zeros, zeros, zeros, zeros, h, h, h, h]
    ])
    trackletBox = trackletBox.transpose(2, 0, 1) # (N, 3, 8)

    rotMat = np.zeros((N, 3, 3))
    rotMat[:, 0, 0] = np.cos(rz)
    rotMat[:, 0, 1] = -np.sin(rz)
    rotMat[:, 1, 0] = np.sin(rz)
    rotMat[:, 1, 1] = np.cos(rz)
    rotMat[:, 2, 2] = 1

    cornerPosInVelo = rotMat @ trackletBox + np.tile(translation, (8, 1, 1)).transpose(1, 2, 0)
    boxes3d = cornerPosInVelo.transpose(0, 2, 1)

    return boxes3d


def anchors_center_to_corner_fast(anchors):
    '''A more fast way to generate birdview box2d in lidar coordinate system
        input:
            anchors:        (N, 7) [xc, yc, zc, h, w, l, r]
        output:
            anchor_corner:  (N, 4, 2)
        '''
    translation = anchors[:, :3]    # (N,3)
    w = anchors[:, 4]   # (N,)
    l = anchors[:, 5]
    rz = anchors[:, -1]

    Box = np.array([
            [-l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2]])

    Box = Box.transpose(2, 0, 1)  # (N,2,4)

    rotMat = np.array([
        [np.cos(rz), -np.sin(rz)],
        [np.sin(rz), np.cos(rz)]])

    rotMat = rotMat.transpose(2, 0, 1)    # (N,2,2)

    velo_box = rotMat @ Box     # (N, 2, 4)
    cornerPosInVelo = velo_box + np.tile(translation[:, :2], (4, 1, 1)).transpose(1, 2, 0)  # (N, 2, 4)
    anchor_corner = cornerPosInVelo.transpose(0, 2, 1)
    return anchor_corner


def corner_to_standup_box2d_batch(boxes_corner):
    # (N, 8, 3) -> (N, 4) x_min, y_min, x_max, y_max
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)
    return standup_boxes2d


def cal_iou(boxes1, boxes2, cfg):
    '''

    :param boxes1: ndarray
               (M, 7) shaped array with bboxes [xc, yc, zc, h, w, l, theta]
    :param boxes2: ndarray
               (N, 7) shaped array with bboxes [xc, yc, zc, h, w, l, theta]
    :param cfg:  config
    :return:
    '''
    method = cfg.cal_iou_method
    iou = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    if method == 'from_avod':
        from project.boxes_utils.from_avod.boxes_overlaps import cal_iou_3d
        iou = cal_iou_3d(boxes1, boxes2)
    if method == 'from_stackoverflow':
        from project.boxes_utils.from_stackoverflow.boxes_overlaps import boxes3d_iou
        iou = boxes3d_iou(boxes1, boxes2, bev_only=True)
    if method == 'from_sx':
        from project.boxes_utils.from_sx.boxes_overlaps import cal_ious_3d
        iou = cal_ious_3d(boxes1, boxes2)
    if method == 'from_numba':
        from project.boxes_utils.from_numba.rotate_iou import rotate_iou_gpu_eval
        boxes = boxes1[:, [0, 1, 3, 4, 6]]
        query_boxes = boxes2[:, [0, 1, 3, 4, 6]]
        boxes = boxes.numpy()
        query_boxes = query_boxes.numpy()
        iou = rotate_iou_gpu_eval(boxes, query_boxes, device_id=0)
    if method == 'from_voxelnet':
        from project.boxes_utils.from_voxelnet.box_overlaps import bbox_overlaps
        # (N, 7) ——> (N, 4, 2)
        anchors_corner = anchors_center_to_corner_fast(boxes1)
        # (N, 4, 2) ——> (N, 4) x_min, y_min, x_max, y_max
        anchors_standup_2d = corner_to_standup_box2d_batch(anchors_corner)
        # (N, 7) ——> (N, 8, 3)
        gt_corner = center_to_corner_box3d(boxes2)
        # (N, 8, 3) ——> (N, 4) x_min, y_min, x_max, y_max
        gt_standup_2d = corner_to_standup_box2d_batch(gt_corner)
        iou = bbox_overlaps(
             np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
             np.ascontiguousarray(gt_standup_2d).astype(np.float32),
         )
    return iou


def non_max_suppression(prediction, scores, cfg):
    '''Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Parameters
    ----------
    prediction: ndarray
        (N, 7)
    scores : ndarray
        (N,)
    Returns
    -------
    filtered boxes
        (N', 7)
    '''
    # filter out confidence scores below threshold
    mask_idx = scores > cfg.score_threshold
    scores = scores[mask_idx]
    boxes3d = prediction[mask_idx]
    print(len(boxes3d))
    if len(scores) == 0:
        return [], []

    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        # every time the first is the biggst, and add it directly
        i = index[0]
        keep.append(i)
        box3d = np.expand_dims(boxes3d[i], axis=0)
        ious = cal_iou(boxes3d[index[1:]], box3d, cfg)
        idx = np.where(abs(ious) <= cfg.nms_threshold)[0]
        index = index[idx+1]
    return boxes3d[keep], scores[keep]


def run_eval(data_root, eval_path, epoch, cfg):
    import os
    GT_DIR = data_root + '/training/label_2'
    PRED_DIR = eval_path + str(epoch) + '/'
    OUTPUT_DIR = eval_path + str(epoch) + '/output.txt'
    code = 'nohup %s %s %s > %s 2>&1 &' % (cfg.EVAL_SCRIPT_DIR, GT_DIR, PRED_DIR, OUTPUT_DIR)
    os.system(code)