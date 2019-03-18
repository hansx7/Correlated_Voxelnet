import math
import numpy as np

class config:
    # maxiumum number of points per voxel
    Max_num = 35

    # Minumum number of poitns per occupied voxel
    Min_num = 1

    #voxel size
    voxel_size = [0.2, 0.2, 0.2]

    # points cloud range
    xrange = [0, 70.4]
    yrange = [-40, 40]
    zrange = [-2, 2]

    # voxel grid
    W = math.ceil((xrange[1] - xrange[0]) / voxel_size[0])
    H = math.ceil((yrange[1] - yrange[0]) / voxel_size[1])
    D = math.ceil((zrange[1] - zrange[0]) / voxel_size[2])

    # image size
    IMG_HEIGHT = 375
    IMG_WIDTH = 1242

    # anchors: (200, 176, 2, 7) x y z h w l r
    x = np.linspace(xrange[0] + voxel_size[0], xrange[1] - voxel_size[0], W / 2)
    y = np.linspace(yrange[0] + voxel_size[1], yrange[1] - voxel_size[1], H / 2)
    cx, cy = np.meshgrid(x, y)
