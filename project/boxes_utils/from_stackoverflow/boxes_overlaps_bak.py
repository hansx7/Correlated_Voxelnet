import numpy as np
import torch
from math import pi, cos, sin


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x*v.y - self.y*v.x


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a*p.x + self.b*p.y + self.c

    def intersection(self, other):
        # See e.g. https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a*other.b - self.b*other.a
        return Vector(
            (self.b*other.c - self.c*other.b)/w,
            (self.c*other.a - self.a*other.c)/w
        )


def rectangle_vertices(cx, cy, w, h, r):
    angle = pi*r/180
    dx = w/2
    dy = h/2
    dxcos = dx*cos(angle)
    dxsin = dx*sin(angle)
    dycos = dy*cos(angle)
    dysin = dy*sin(angle)
    return (
        Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos - -dysin,  dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos -  dysin,  dxsin +  dycos),
        Vector(cx, cy) + Vector(-dxcos -  dysin, -dxsin +  dycos)
    )


def intersection_area(r1, r2):
    # r1 and r2 are in (center, width, height, rotation) representation
    # First convert these into a sequence of vertices

    rect1 = rectangle_vertices(*r1)
    rect2 = rectangle_vertices(*r2)

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1],
            line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p.x*q.y - p.y*q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))


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
    '''
    Computes IoU between 3D bounding boxes.
    Parameters
        ----------
        boxes1 : ndarray
            (N, 7) shaped array with bboxes [xc, yc, zc, h, w, l, theta]
        boxes2 : ndarray
            (M, 7) shaped array with bboxes [xc, yc, zc, h, w, l, theta]
        bev_only: bool
        Returns
        -------
        IoU:    ndarray
            (N, M) shaped array with 3D IoUs, float in [0, 1]
        IoU_2d: ndarray
            (N, M) shaped array with 2D BEV IoUs, float in [0, 1]
    '''
    N, M = boxes1.shape[0], boxes2.shape[0]
    intersection = np.zeros((N, M))

    for i in range(N):
        cx_i, cy_i = boxes1[i][:2]
        w_i, l_i = boxes1[i][4:6]
        theta_i = boxes1[i][-1] * 180 / np.pi
        r1 = (cx_i, cy_i, w_i, l_i, theta_i)
        for j in range(M):
            cx_j, cy_j = boxes2[j][:2]
            w_j, l_j = boxes2[j][4:6]
            theta_j = boxes2[j][-1] * 180 / np.pi
            r2 = (cx_j, cy_j, w_j, l_j, theta_j)
            temp_inter = intersection_area(r1, r2)
            intersection[i, j] = temp_inter
    areas1 = boxes1[:, 4] * boxes1[:, 5]  # (N,)
    areas2 = boxes2[:, 4] * boxes2[:, 5]  # (M,)
    union = np.expand_dims(areas1, axis=1) + areas2 - intersection
    iou_2d = intersection / union

    if not bev_only:
        areas1 = areas1 * boxes1[:, 3]
        areas2 = areas2 * boxes2[:, 3]
        zmax = np.minimum(np.expand_dims(boxes1[:, 2]+boxes1[:, 3]/2, axis=1), boxes2[:, 2]+boxes2[:, 3]/2)  # (N,M)
        zmin = np.maximum(np.expand_dims(boxes1[:, 2]-boxes1[:, 3]/2, axis=1), boxes2[:, 2]-boxes2[:, 3]/2)  # (N,M)
        inter_vol = intersection * np.maximum(0.0, zmax - zmin)  # (N,M)
        inter = inter_vol.T
        union_3d = np.expand_dims(areas1, axis=1) + areas2 - inter_vol
        iou_3d = inter_vol / union_3d
        return iou_3d

    return iou_2d