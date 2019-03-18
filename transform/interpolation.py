import numpy as np
import matplotlib.pyplot as plt

def interpolation_lidar_image(lidar, image, inds, window_size=5):
    lidar_size = lidar.shape[0]
    interpolate_points = []
    step = (window_size - 1) // 2
    inds_padding = np.pad(inds, ((0,0),(step,step)), 'constant', constant_values=(0,0))
    h,w,c = image.shape
    for i in range(h):
        for j in range(step, w+step):
            if inds_padding[i][j] != -1:
                continue
            # get its neighbour
            neighbours = inds_padding[i, j-step:j+step]
            # compute average
            points = []
            for idx in neighbours:
                if idx == -1:
                    continue
                point = lidar[idx]
                points.append(point)
            if len(points) == 0:
                continue
            new_point = np.mean(points,axis=0)
            interpolate_points.append(new_point)
            inds[i][j-step] = lidar_size
            lidar_size += 1
    if len(interpolate_points) == 0:
        print('no points interpolation!')
        return lidar, image, inds
    new_lidar = np.concatenate((lidar, interpolate_points),axis=0)
    return new_lidar, image, inds


def densitify_point_cloud(lidar, lidar2image, inds, iter, window_size=5):
    for _ in range(iter):
        lidar, lidar2image, inds = interpolation_lidar_image(lidar, lidar2image, inds, window_size)
    return lidar


def generate_lidar_img(lidar, calib, img_h, img_w):
    lidar_img = np.zeros((img_h,img_w,3),dtype=np.uint8)
    img2lidar_inds = np.ones((img_h, img_w), dtype=np.int32) * (-1)
    imgfov_pts_2d = calib.project_velo_to_image(lidar)
    imgfov_pc_rect = calib.project_velo_to_rect(lidar)

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0 / depth), :]
        color = np.asarray(color, dtype=np.uint8)
        dw, dh = int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))
        dw = dw if dw < img_w else img_w - 1
        dh = dh if dh < img_h else img_h - 1
        img2lidar_inds[dh][dw] = i
        lidar_img[dh][dw] = color

    return lidar_img, img2lidar_inds
