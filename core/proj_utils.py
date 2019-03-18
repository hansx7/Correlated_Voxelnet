import numpy as np
import cv2

from dataLoader import kitti_utils as utils


def project_lidar_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    #print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=0.5):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def generate_bev_image(velo_img_fov, label_obj, calib_obj, width, height, voxel_size, with_box2d = False):
    '''
    generate birth eye view image of lidar
    :param velo_img_fov: a subset of lidar that can be projected to image
    :param label_obj: label object of lidar
    :param calib_obj: calibration object
    :param width:  image width
    :param height:  image height
    :param voxel_size: voxel size, 0.2 x 0.2 for a pixel default
    :param with_box2d: whether return with bounding box
    '''
    velo_bev = velo_img_fov[:,:2].copy()
    # adjust position of view, make sure the main part near center. '20' and '100' can be change for better view
    y_shift = 20 - 0.5 * (-np.min(velo_bev[:,0]) / voxel_size)
    x_shift = 100 - 0.5 * (-np.min(velo_bev[:, 1]) / voxel_size)

    velo_bev[:, 0] = (velo_bev[:, 0] - np.min(velo_bev[:, 0])) / voxel_size + y_shift
    velo_bev[:, 1] = (velo_bev[:, 1] - np.min(velo_bev[:, 1])) / voxel_size + x_shift

    bev = np.asarray(velo_bev, dtype=np.uint16)
    # get height data of fov
    bev_h = velo_img_fov[:, 2]
    # map height data to [0,255]
    bev_h = (bev_h - np.min(bev_h)) / (np.max(bev_h) - np.min(bev_h)) * 255
    # init bev image
    bev_img = np.zeros((height, width), dtype = np.uint8)
    # record the count of point that map to each pixel, used in height average calculation
    bev_count = np.zeros((height, width), dtype = np.uint8)
    h,w = bev.shape
    for i in range(h):
        # ignore the point that out of bev image
        if bev[i][0] > height-1 or bev[i][1] > width-1:
            continue

        # accumulate height of each pixel
        bev_img[bev[i][0], bev[i][1]] += bev_h[i]
        # count += 1
        bev_count[bev[i][0], bev[i][1]] += 1

    # set height average as pixel value
    bev_img = bev_img / bev_count

    bev_img = bev_img.astype(np.uint8)
    # whether draw with boxes
    if with_box2d:
        bev_img, bev_box2d = get_bev_with_box2d(bev_img, velo_img_fov, label_obj,
                                            calib_obj, y_shift, x_shift, voxel_size, True)
        return bev_img, bev_box2d
    else:
        return bev_img


def get_bev_with_box2d(image, lidar, label_obj, calib_obj, y_offset, x_offset, voxel_size, more = False):
    '''
    draw 2d boxes in bev image
    '''
    y_shift = np.min(lidar[:,0])
    x_shift = np.min(lidar[:,1])
    bev_box2d = []
    for obj in label_obj:
        # compute box3d and project to velodyne coordinate
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib_obj.P)
        box3d_pts_3d_velo = calib_obj.project_rect_to_velo(box3d_pts_3d)
        box2d = []
        # for more detail ,turn to kitti_util.draw_projected_box3d
        for k in range(0,4):
            i,j = k, (k+1)%4
            y1, x1, z1 = box3d_pts_3d_velo[i, :]
            y2, x2, z2 = box3d_pts_3d_velo[j, :]
            # make the same operation as points
            y1 = int((y1 - y_shift) / voxel_size + y_offset)
            x1 = int((x1 - x_shift) / voxel_size + x_offset)
            y2 = int((y2 - y_shift) / voxel_size + y_offset)
            x2 = int((x2 - x_shift) / voxel_size + x_offset)
            box2d.append([x1,y1])
            #draw line in image
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)
        bev_box2d.append(box2d)
    if more:
        return image, bev_box2d
    else:
        return image