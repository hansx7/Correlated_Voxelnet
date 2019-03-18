from __future__ import print_function
import torch
import numpy as np
from scipy.spatial import ConvexHull


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def boxes2d_iou(boxes1, boxes2):
    """Computes IoU between bounding boxes (two axis-aligned bounding boxes).

    Parameters
    ----------
    boxes1 : ndarray
        (N, 4) shaped array with bboxes (x_min, y_min, x_max, y_max)
    boxes2 : ndarray
        (M, 4) shaped array with bboxes (x_min, y_min, x_max, y_max)
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs, float in [0, 1]
    """
    assert (boxes1[:, 0] < boxes1[:, 1]).all()
    assert (boxes1[:, 2] < boxes1[:, 3]).all()
    assert (boxes2[:, 0] < boxes2[:, 1]).all()
    assert (boxes2[:, 2] < boxes2[:, 3]).all()

    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    iw = np.minimum(np.expand_dims(boxes1[:, 2], axis=1), boxes2[:, 2]) - \
         np.maximum(np.expand_dims(boxes1[:, 0], axis=1), boxes2[:, 0])

    ih = np.minimum(np.expand_dims(boxes1[:, 3], axis=1), boxes2[:, 3]) - \
         np.maximum(np.expand_dims(boxes1[:, 1], axis=1), boxes2[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    intersection = iw * ih

    ua = np.expand_dims((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]), axis=1) + area2 - intersection

    ua = np.maximum(ua, np.finfo(float).eps)

    iou = intersection / ua
    assert iou >= 0.0
    assert iou <= 1.0

    return iou


def boxes3d_iou(boxes1, boxes2, bev_only=True):
    """Computes IoU between 3D bounding boxes.
    qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        Parameters
        ----------
        boxes1 : ndarray
            (N, 8, 3) shaped array with bboxes
        boxes2 : ndarray
            (M, 8, 3) shaped array with bboxes
        bev_only: bool
        Returns
        -------
        IoU:    ndarray
            (N, M) shaped array with 3D IoUs, float in [0, 1]
        IoU_2d: ndarray
            (N, M) shaped array with 2D BEV IoUs, float in [0, 1]
        """
    # transform corner points in counter clockwise order
    if not bev_only:
        rect1 = boxes1[:, :4, :2]  # (N, 4, 2)
        rect2 = boxes2[:, :4, :2]  # (M, 4, 2)
    else:
        rect1 = boxes1
        rect2 = boxes2
    area1 = poly_area(rect1[:, :, 0], rect1[:, :, 1])  # (N,)
    area2 = poly_area(rect2[:, :, 0], rect2[:, :, 1])  # (M,)

    inter_areas = cal_intersection(rect1, rect2)    # (N, M)
    ua = np.expand_dims(area1, axis=1) + area2 - inter_areas
    IoU_2d = inter_areas / ua

    if not bev_only:
        zmax = np.minimum(np.expand_dims(boxes1[:, 0, 2],axis=1), boxes2[:, 0, 2])  # (N,M)
        zmin = np.maximum(np.expand_dims(boxes1[:, 4, 2],axis=1), boxes2[:, 4, 2])  # (N,M)
        inter_vol = inter_areas * np.maximum(0.0, zmax - zmin) # (N,M)
        vol1 = boxes3d_vol(boxes1)  # (N,)
        vol2 = boxes3d_vol(boxes2)  # (M,)
        ua_3d = np.expand_dims(vol1, axis=1) + vol2 - inter_vol
        IoU = inter_vol / ua_3d
        return IoU, IoU_2d

    return IoU_2d


def poly_area(x, y):
    '''Computes areas of boxes BEV.
    Parameters:
    ----------
    x: ndarray
        (N, 4) four vertices x coordinate (x0,x1,x2,x3)
    y: ndarray
        (N, 4) four vertices y coordinate (y0,y1,y2,y3)
    Returns
    ------
    areas: ndarray
        (N,)  areas of N boxes BEV
    '''
    x_roll = np.roll(x, 1, axis=1)      # (x3,x0,x1,x2)
    y_roll = np.roll(y, 1, axis=1)      # (y3,y0,y1,y2)
    areas = 0.5 * np.abs(np.sum(x * y_roll, axis=1) - np.sum(y * x_roll, axis=1))   # (N,)
    return areas


def cal_intersection(rect1, rect2):
    '''Computes intersection of rect1 and rect2
    Parameters
    ----------
    rect1 : ndarray
        (N, 4, 2)
    rect2 : ndarray
        (M, 4, 2)
    Returns
    -------
    intersections: ndarray
        (N, M) shaped array of intersection areas
    TODO: rewrite this function using numpy operation
    '''
    N,M = rect1.shape[0], rect2.shape[0]
    intersections = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            p1 = [(rect1[i][r, 0], rect1[i][r, 1]) for r in range(3, -1, -1)]
            p2 = [(rect2[j][r, 0], rect2[j][r, 1]) for r in range(3, -1, -1)]
            inter_p = polygon_clip(p1, p2)
            if inter_p is not None:
                hull_inter = ConvexHull(inter_p)
                intersections[i, j] = hull_inter.volume
    return intersections


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def compute_intersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(compute_intersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(compute_intersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)


def boxes3d_vol(boxes):
    ''' boxes: (N,8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum(boxes[:, 0, :] - boxes[:, 1, :], axis=1) ** 2)   # (N,)
    b = np.sqrt(np.sum(boxes[:, 1, :] - boxes[:, 2, :], axis=1) ** 2)   # (N,)
    c = np.sqrt(np.sum(boxes[:, 0, :] - boxes[:, 4, :], axis=1) ** 2)   # (N,)
    return a*b*c



