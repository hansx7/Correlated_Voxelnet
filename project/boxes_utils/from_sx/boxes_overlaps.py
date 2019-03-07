import ctypes
import os
from ctypes import *
import numpy as np

def init():
    class Matrix(Structure):
        _fields_ = [('size_x', c_int), ('size_y', c_int), ('arr', c_double * 500000)]
    cur_path = os.getcwd() + '/boxes_utils/from_sx/'
    IoU = ctypes.cdll.LoadLibrary(cur_path+'lib_iou2d.so')
    IoU.IoU_interface3.restype = Matrix
    return IoU

def cal_ious_2d(boxes1, boxes2):
    '''
    calculate 2d ious in bev
    Parameters
    ----------
        boxes1: (N, 7) [xc, yc, zc, h, w, l, r]
        boxes2: (M, 7) [xc, yc, zc, h, w, l, r]
    Returns
    -------
    iou_2ds: (N,M) float in [0,1]
    '''
    N, M = boxes1.shape[0], boxes2.shape[0]
    boxes1_2d = boxes1[:, [0, 1, 4, 5, 6]]  # (N, 5)
    boxes1_2d[:,-1] = boxes1_2d[:, -1] * 180 / np.pi # rad to angle
    boxes2_2d = boxes2[:, [0, 1, 4, 5, 6]]  # (M, 5)
    boxes2_2d[:, -1] = boxes2_2d[:, -1] * 180 / np.pi

    # calculate 2d boxes area
    areas1 = boxes1_2d[:, 2] * boxes1_2d[:, 3]  # (N,)
    areas2 = boxes2_2d[:, 2] * boxes2_2d[:, 3]  # (M,)

    # calculate 2d boxes overlaps
    cboxes1 = [(ctypes.c_double * 5)(*boxes1_2d[i]) for i in range(N)]
    cboxes1 = ((ctypes.c_double * 5) * N)(*cboxes1)
    cboxes2 = [(ctypes.c_double * 5)(*boxes2_2d[i]) for i in range(M)]
    cboxes2 = ((ctypes.c_double * 5) * M)(*cboxes2)

    IoU = init()
    inter = IoU.IoU_interface3(cboxes1, cboxes2, N, M)
    intersection = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            intersection[i,j] = inter.arr[i * inter.size_y + j]

    union = np.expand_dims(areas1, axis=1) + areas2 - intersection
    iou_2ds = intersection / union
    return iou_2ds


def cal_ious_3d(boxes1, boxes2):
    '''
    calculate 2d ious in bev
    Parameters
    ----------
        boxes1: (N, 7) [xc, yc, zc, h, w, l, r]
        boxes2: (M, 7) [xc, yc, zc, h, w, l, r]
    Returns
    -------
    iou_3ds: (N,M) float in [0,1]
    '''
    N, M = boxes1.shape[0], boxes2.shape[0]
    #calculate delta height
    zmax = np.minimum(np.expand_dims(boxes1[:, 2] + boxes1[:, 3] / 2, axis=1), boxes2[:, 2] + boxes2[:, 3] / 2)  # (N,M)
    zmin = np.maximum(np.expand_dims(boxes1[:, 2] - boxes1[:, 3] / 2, axis=1), boxes2[:, 2] - boxes2[:, 3] / 2)  # (N,M)

    # calculate 3d boxes area
    areas1 = boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]  # (N,)
    areas2 = boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]  # (M,)

    # calculate 2d overlaps
    boxes1_2d = boxes1[:, [0, 1, 4, 5, 6]]  # (N, 5)
    boxes1_2d[:,-1] = boxes1_2d[:, -1] * 180 / np.pi # rad to angle
    boxes2_2d = boxes2[:, [0, 1, 4, 5, 6]]  # (M, 5)
    boxes2_2d[:, -1] = boxes2_2d[:, -1] * 180 / np.pi

    # calculate 2d boxes overlaps
    cboxes1 = [(ctypes.c_double*5)(*boxes1_2d[i]) for i in range(N)]
    cboxes1 = ((ctypes.c_double * 5) * N)(*cboxes1)
    cboxes2 = [(ctypes.c_double * 5)(*boxes2_2d[i]) for i in range(M)]
    cboxes2 = ((ctypes.c_double * 5) * M)(*cboxes2)

    IoU = init()
    inter = IoU.IoU_interface3(cboxes1, cboxes2, N, M)
    intersection = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            intersection[i,j] = inter.arr[i * inter.size_y + j]

    intersection_3d = intersection * np.maximum(0.0, zmax - zmin)  # (N,M)
    union = np.expand_dims(areas1, axis=1) + areas2 - intersection_3d
    iou_3ds = intersection / union
    return iou_3ds


def non_max_suppression(prediction, scores, conf_thres, nms_thres):
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
    mask_idx = scores > conf_thres
    scores = scores[mask_idx]
    boxes3d = prediction[mask_idx]

    if len(scores) == 0:
        return None

    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        # every time the first is the biggst, and add it directly
        i = index[0]
        keep.append(i)
        box3d = np.expand_dims(boxes3d[i], axis=0)
        ious = cal_ious_2d(boxes3d[index[1:]], box3d)
        idx = np.where(abs(ious) <= nms_thres)[0]
        index = index[idx+1]
    return boxes3d[keep], scores[keep]


if __name__ == '__main__':
    rnd_data = np.array([[77.93, 69.15, 0.0, 10.0, 27.77, 08.86, 053.83],
                         [14.21, 66.49, 0.0, 10.0, 04.92, 53.86, 143.35],
                         [77.63, 00.59, 0.0, 10.0, 00.01, 00.01, 123.62],
                         [57.36, 91.72, 0.0, 10.0, 34.26, 05.40, 059.26],
                         [57.82, 64.29, 0.0, 10.0, 25.67, 53.68, 132.11],
                         [31.35, 40.67, 0.0, 10.0, 51.23, 28.62, 095.30],
                         [30.69, 30.58, 0.0, 10.0, 40.22, 98.02, 039.29],
                         [80.42, 50.11, 0.0, 10.0, 84.56, 13.93, 161.67],
                         [37.84, 49.19, 0.0, 10.0, 44.21, 73.73, 022.29],
                         [43.70, 83.15, 0.0, 10.0, 43.24, 51.98, 165.37]])

    ious = cal_ious_2d(rnd_data, rnd_data)
    print(ious.shape)
    #print(ious)



