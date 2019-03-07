import sys
import cv2
import numpy as np
sys.path.append('../../')
import dataLoader.kitti_utils as utils
from project.utils.colorize import colorize
from dataLoader.kitti_utils import compute_box_3d


def generate_summary_img(tag, image, prob, pre_labels, scores, data_root, cfg, datasets='detection'):
    if datasets == 'detection':
        name = tag.zfill(6)
        lidar_file = data_root + '/training/velodyne/' + name + '.bin'
        calib_file = data_root + '/training/calib/'    + name + '.txt'
        label_file = data_root + '/training/label_2/'  + name + '.txt'
    else:
        name = tag[:2].zfill(4) + '/' + tag[2:].zfill(6)
        lname = tag[:2] + str(tag[2:]).zfill(4)
        lidar_file = data_root + '/training/velodyne/' + name + '.bin'
        label_file = data_root + '/training/label/'  + lname + '.txt'
        calib_file = data_root + '/training/calib/'    + tag[:2].zfill(4) + '.txt'
    
    labels = [line.rstrip() for line in open(label_file)]
    gt_labels_src = [utils.Object3d(line) for line in labels]

    calib = utils.Calibration(calib_file)
    lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    
    # remove labels not for 'train_type'
    gt_labels = []
    for label in gt_labels_src:
        if label.type in ['Car', 'Van']:
            gt_labels.append(label)
    
    pre_labels = pre_labels.split('\n')[:-1]
    pre_labels = [line[:-6].rstrip() for line in pre_labels]
    pre_labels = [utils.Object3d(line) for line in pre_labels]
    
    front_image = draw_lidar_box3d_on_image(image, pre_labels, scores, gt_labels, calib)
    bird_view = lidar_to_bird_view_img(lidar, cfg)
    bird_view = draw_lidar_box3d_on_birdview(bird_view, pre_labels, scores, gt_labels, calib, cfg)
    heatmap = colorize(prob, cfg.BV_LOG_FACTOR)
    return front_image.transpose(2,0,1), bird_view.transpose(2,0,1), heatmap.transpose(2,0,1)


def draw_lidar_box3d_on_image(img, pre_labels, scores, gt_labels, calib):
    img_2d_pre, img_3d_pre = show_image_with_boxes(img, gt_labels, calib, color=(255, 0, 255))
    img_2d, _ = show_image_with_boxes(img_2d_pre, pre_labels, calib, scores=scores, color=(0, 255, 255))
    _, img_3d = show_image_with_boxes(img_3d_pre, pre_labels, calib, color=(0, 255, 255))
    
    img = np.concatenate([img, img_2d, img_3d], axis=0)
    
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def show_image_with_boxes(img, objects, calib, scores=None, color=(255,255,255)):
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img) # for 2d bbox
    img2 = np.copy(img) # for 3d bbox
    obj_len = len(objects)
    for i in range(0, obj_len):
        obj = objects[i]
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)), color, 2)
        # put text
        if scores is not None:
            score = '%.4f' %scores[i]
            # put text
            img1 = cv2.putText(img1, score, (int(obj.xmax), int(obj.ymin)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)

        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=color)
    return img1, img2


def lidar_to_bird_view_img(lidar, cfg):
    # Input:
    #   lidar: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    factor = cfg.BV_LOG_FACTOR
    birdview = np.zeros((cfg.H * factor, cfg.W * factor, 1))
    for point in lidar:
        x, y = point[0:2]
        if cfg.xrange[0] < x < cfg.xrange[1] and cfg.yrange[0] < y < cfg.yrange[1]:
            x, y = int((x - cfg.xrange[0]) / cfg.vw * factor), int((y - cfg.yrange[0]) / cfg.vh * factor)
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    # TODO: adjust this factor
    birdview = np.clip((birdview / divisor * 255) *
                       5 * factor, a_min=0, a_max=255)
    birdview = np.tile(birdview, 3).astype(np.uint8)
    return birdview


def draw_lidar_box3d_on_birdview(birdview, pre_labels, scores, gt_labels, calib, cfg, color=(0, 255, 255),
                                 gt_color=(255, 0, 255), thickness=1):
    # Input:
    #   birdview: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = birdview.copy()
    boxes3d = np.array([i.boxes3d for i in pre_labels])
    boxes3d = camera_to_lidar_box(boxes3d, calib)
    
    gt_boxes3d = np.array([i.boxes3d for i in gt_labels])
    gt_boxes3d = camera_to_lidar_box(gt_boxes3d, calib)
    
    corner_boxes3d = center_to_corner_box3d(boxes3d)
    corner_gt_boxes3d = center_to_corner_box3d(gt_boxes3d)
    # draw gt
    for box in corner_gt_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], cfg)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], cfg)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], cfg)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], cfg)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 gt_color, thickness, cv2.LINE_AA)

    # draw detections
    assert len(corner_boxes3d) == len(scores), print('length of scores not match!')
    for (box, score) in zip(corner_boxes3d, scores):

        x0, y0 = lidar_to_bird_view(*box[0, 0:2], cfg)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], cfg)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], cfg)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], cfg)
        
        #put text
        score = '%.4f' % score
        img = cv2.putText(img, score, (int(x1), int(y1)),
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        #put text
        score = '%.4f' % score
        img = cv2.putText(img, score, (int(x1), int(y1)),
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 color, thickness, cv2.LINE_AA)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
    return image


def lidar_to_bird_view(x, y, cfg):
    # using the cfg.INPUT_XXX
    factor = cfg.BV_LOG_FACTOR
    a = (x - cfg.xrange[0]) / cfg.vw * factor
    b = (y - cfg.yrange[0]) / cfg.vh * factor
    a = np.clip(a, a_max=(cfg.xrange[1] - cfg.xrange[0]) / cfg.vw * factor, a_min=0)
    b = np.clip(b, a_max=(cfg.yrange[1] - cfg.yrange[0]) / cfg.vh * factor, a_min=0)
    return a, b


def lidar_box3d_to_camera_box(boxes3d, calib, cal_projection=False):
    # (N, 7) -> (N, 4)/(N, 8, 2)  x,y,z,h,w,l,rz -> x1,y1,x2,y2/8*(x, y)
    num = len(boxes3d)
    boxes2d = np.zeros((num, 4), dtype=np.int32)
    projections = np.zeros((num, 8, 2), dtype=np.float32)

    lidar_boxes3d_corner = center_to_corner_box3d(boxes3d)

    for n in range(num):
        box3d = lidar_boxes3d_corner[n]
        box3d = lidar_to_camera_point(box3d, calib)
        points = np.hstack((box3d, np.ones((8, 1)))).T  # (8, 4) -> (4, 8)
        P = np.concatenate((calib.P, np.array([[0,0,0,0]])), 0)
        points = np.matmul(P, points).T
        points[:, 0] /= points[:, 2]
        points[:, 1] /= points[:, 2]

        projections[n] = points[:, 0:2]
        minx = int(np.min(points[:, 0]))
        maxx = int(np.max(points[:, 0]))
        miny = int(np.min(points[:, 1]))
        maxy = int(np.max(points[:, 1]))

        boxes2d[n, :] = minx, miny, maxx, maxy

    return projections if cal_projection else boxes2d


def center_to_corner_box3d(boxes_center):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([
            # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
            np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    return ret


def camera_to_lidar_box(boxes, calib):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(
            x, y, z, calib), h, w, l, -ry - np.pi / 2
        rz = angle_in_limit(rz)
        ret.append([x, y, z, h, w, l, rz])
    return np.array(ret).reshape(-1, 7)


def lidar_to_camera_point(points, calib):
    # (N, 3) -> (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))]).T
    V2C = np.concatenate([calib.V2C, np.array([0,0,0,1]).reshape(1,4)], 0)
    R0 = np.eye(4)
    R0[:3, :3] = calib.R0
    points = np.matmul(V2C, points)
    points = np.matmul(R0, points).T
    points = points[:, 0:3]
    return points.reshape(-1, 3)


def camera_to_lidar(x, y, z, calib):
    p = np.array([x, y, z, 1])
    V2C = np.concatenate([calib.V2C, np.array([0,0,0,1]).reshape(1,4)], 0)
    R0 = np.eye(4)
    R0[:3, :3] = calib.R0
    p = np.matmul(np.linalg.inv(R0), p)
    p = np.matmul(np.linalg.inv(V2C), p)
    p = p[0:3]
    return tuple(p)


def lidar_to_camera(x, y, z, calib):
    p = np.array([x, y, z, 1])
    V2C = np.concatenate([calib.V2C, np.array([0,0,0,1]).reshape(1,4)], 0)
    R0 = np.eye(4)
    R0[:3, :3] = calib.R0
    p = np.matmul(V2C, p)
    p = np.matmul(R0, p)
    p = p[0:3]
    return tuple(p)


def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle