import shapely.geometry
import shapely.affinity
import numpy as np


class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle * 180 / np.pi

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())




from matplotlib import pyplot
from descartes import PolygonPatch

fig = pyplot.figure(1, figsize=(10, 4))
ax = fig.add_subplot(121)
ax.set_xlim(20, 40)
ax.set_ylim(-20, 20)

# [3.67603805, 3.16718838, 1.78338605, 3.68533091, 1.567041]
# [44.42013097, 8.63965821, 1.74515417, 4.23434444, -1.459617]
# [28.94961407, -17.46514908, 1.5438987, 4.09564139, -0.10274765]

# [29.50232558, -17.05714286, 1.6 3.9, 0.]
# [3.45581395, 2.43673469, 1.6, 3.9, 1.57079633]
# [44.15348837, 8.93469388, 1.6, 3.9, 1.57079633]

iou = [0.15503276, 0.54602476, 0.35440958] # [0.36626517 0.61445984 0.4228988 ]

r1 = RotatedRect(28.94961407, -17.46514908, 1.5438987, 4.09564139, -0.10274765)
r2 = RotatedRect(29.50232558, -17.05714286, 1.6, 3.9, 0.)

inter = r1.intersection(r2).area
area1 = r1.get_contour().area
area2 = r2.get_contour().area

iou = inter / (area1 + area2 - inter)
print(iou)

ax.add_patch(PolygonPatch(r1.get_contour(), fc='#990000', alpha=0.7))
ax.add_patch(PolygonPatch(r2.get_contour(), fc='#000099', alpha=0.7))
ax.add_patch(PolygonPatch(r1.intersection(r2), fc='#009900', alpha=1))

pyplot.show()