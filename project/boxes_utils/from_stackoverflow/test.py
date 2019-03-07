import numpy as np
import time
from boxes_overlaps import boxes3d_iou

N,M = 10000, 10
boxes1 = np.zeros((N, 7))
boxes2 = np.zeros((M,7))
XYZ1 = np.random.rand(N, 3) * 30
XYZ2 = np.random.rand(M, 3) * 30
theta1 = np.random.rand(N) * np.pi
theta2 = np.random.rand(M) * np.pi
boxes1[:,:3] = XYZ1
boxes1[:, -1] = theta1
boxes1[:, 3:6] = [10,15,20]

boxes2[:, :3] = XYZ2
boxes2[:, -1] = theta2
boxes2[:, 3:6] = [15, 10, 20]

start = time.time()
iou2d = boxes3d_iou(boxes1, boxes2)
end = time.time()
print(end-start)