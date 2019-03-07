import numpy as np
from libc.math cimport pi, cos, sin

class Vector:
    def __init__(self, double x, double y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        cdef double x = self.x + v.x
        cdef double y = self.y + v.y
        return Vector(x, y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        cdef double x = self.x - v.x
        cdef double y = self.y - v.y
        return Vector(x, y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        cdef double x = self.x * v.y
        cdef double y = self.y * v.x
        cdef double cross = x - y
        return cross


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        cdef out = self.a*p.x + self.b*p.y + self.c
        return out

    def intersection(self, other):
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a*other.b - self.b*other.a
        return Vector(
            (self.b*other.c - self.c*other.b)/w,
            (self.c*other.a - self.a*other.c)/w
        )


def rectangle_vertices(double cx, double cy, double w, double h, double r):
    cdef double angle = pi*r/180
    cdef double dx = w/2
    cdef double dy = h/2
    cdef double dxcos = dx*cos(angle)
    cdef double dxsin = dx*sin(angle)
    cdef double dycos = dy*cos(angle)
    cdef double dysin = dy*sin(angle)

    cdef double v1x = -dxcos - -dysin
    cdef double v1y = -dxsin + -dycos
    cdef double v2x = dxcos - -dysin
    cdef double v2y = dxsin + -dycos
    cdef double v3x = dxcos -  dysin
    cdef double v3y = dxsin +  dycos
    cdef double v4x = -dxcos -  dysin
    cdef double v4y = -dxsin +  dycos
    return (
        Vector(cx, cy) + Vector(v1x, v1y),
        Vector(cx, cy) + Vector(v2x, v2y),
        Vector(cx, cy) + Vector(v3x, v3y),
        Vector(cx, cy) + Vector(v4x, v4y)
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



def boxes3d_iou(double[:, :] boxes1, double[:, :] boxes2, bev_only=True):
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

    cdef int N = boxes1.shape[0]
    cdef int M =  boxes2.shape[0]

    cdef double[:, :] intersection = np.zeros((N, M))

    cdef int i, j
    cdef double cx_i, cy_i, w_i, l_i, theta_i
    cdef double cx_j, cy_j, w_j, l_j, theta_j
    for i in range(N):
        cx_i = boxes1[i][0]
        cy_i = boxes1[i][1]
        w_i = boxes1[i][4]
        l_i = boxes1[i][5]
        theta_i = boxes1[i][6] * 180 / np.pi
        r1 = (cx_i, cy_i, w_i, l_i, theta_i)
        for j in range(M):
            cx_j = boxes2[j][0]
            cy_j = boxes2[j][1]
            w_j = boxes2[j][4]
            l_j = boxes2[j][5]
            theta_j = boxes2[j][6] * 180 / np.pi
            r2 = (cx_j, cy_j, w_j, l_j, theta_j)
            intersection[i, j] = intersection_area(r1, r2)
    areas1 = np.multiply(boxes1[:, 4], boxes1[:, 5])  # (N,)
    areas2 = np.multiply(boxes2[:, 4], boxes2[:, 5])  # (M,)
    union = np.expand_dims(areas1, axis=1) + areas2 - intersection
    iou_2d = intersection / union

    if not bev_only:
        areas1 = np.multiply(areas1, boxes1[:, 3])
        areas2 = np.multiply(areas2, boxes2[:, 3])
        zmax = np.minimum(np.expand_dims(boxes1[:,2]+np.dot(0.5,boxes1[:,3]), axis=1), boxes2[:,2]+ np.dot(0.5,boxes2[:,3]))  # (N,M)
        zmin = np.maximum(np.expand_dims(boxes1[:,2]-np.dot(0.5,boxes1[:,3]), axis=1), boxes2[:,2]-np.dot(0.5,boxes2[:,3]))  # (N,M)
        inter_vol = intersection * np.maximum(0.0, zmax - zmin)  # (N,M)
        union_3d = np.expand_dims(areas1, axis=1) + areas2 - inter_vol
        iou_3d = inter_vol / union_3d
        return iou_3d

    return iou_2d