import sys
import time
sys.path.append('..')
from voxelization.config import config as cfg


def get_filtered_lidar(lidar):
    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]

    filter_x = np.where((pxs >= cfg.xrange[0]) & (pxs < cfg.xrange[1]))[0]
    filter_y = np.where((pys >= cfg.yrange[0]) & (pys < cfg.yrange[1]))[0]
    filter_z = np.where((pzs >= cfg.zrange[0]) & (pzs < cfg.zrange[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)
    filter_xyz = np.intersect1d(filter_xy, filter_z)

    return lidar[filter_xyz]


def get_feature(lidar_cluster, method='mean'):
    if method == 'max':
        return np.max(lidar_cluster[:, 2])
    if method == 'min':
        return np.min(lidar_cluster[:, 2])
    if method == 'mean':
        return np.mean(lidar_cluster[:, 2])


def create_voxel(lidar):
    # shuffling the points
    np.random.shuffle(lidar)

    voxel_coords = ((lidar[:, :3] - np.array([cfg.xrange[0], cfg.yrange[0], cfg.zrange[0]])) /
                   (cfg.voxel_size[0], cfg.voxel_size[1], cfg.voxel_size[2])).astype(np.int32)

    # convert to (D, H, W)
    voxel_coords = voxel_coords[:, [2, 1, 0]]
    voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)

    voxel_feature = np.zeros((cfg.D, cfg.H, cfg.W), dtype=np.float32)
    voxel_mask = np.zeros_like(voxel_feature)
    for i in range(len(voxel_coords)):
        idx = tuple(voxel_coords[i, :])
        # points in voxel i
        pts = lidar[inv_ind == i]
        # remove points that out of limit
        if voxel_counts[i] > cfg.Max_num:
            pts = pts[:cfg.Max_num, :]
            voxel_counts[i] = cfg.Max_num
        # set voxel_feature[idx] value
        voxel_feature[idx] = get_feature(pts)
        voxel_mask[idx] = 1

    return voxel_feature, voxel_mask


def voxelization(lidar):
    '''get the voxel of lidar data
    input:
        lidar:  (N, 3)
    output:
        voxel_features: N' x cfg.T x 7, N' is the number of non-empty voxel, cfg.T is the maximum number of points to
                                        cal features for the voxel, points out of cfg.T will be discard.
                                        features: (x, y, z, x_mean, y_mean, z_mean, r)
        voxel_coords:   N' x 3          the index of voxel in 3D grid.

    '''
    # shuffling the points
    np.random.shuffle(lidar)

    voxel_coords = ((lidar[:, :3] - np.array([cfg.xrange[0], cfg.yrange[0], cfg.zrange[0]])) /
                    (cfg.vw, cfg.vh, cfg.vd)).astype(np.int32)

    # convert to  (D, H, W)
    voxel_coords = voxel_coords[:, [2, 1, 0]]
    voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)

    voxel_features = []

    for i in range(len(voxel_coords)):
        voxel = np.zeros((cfg.T, 7), dtype=np.float32)
        pts = lidar[inv_ind == i]
        if voxel_counts[i] > cfg.T:
            pts = pts[:cfg.T, :]
            voxel_counts[i] = cfg.T
        # augment the points
        voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
        voxel_features.append(voxel)

    return np.array(voxel_features), voxel_coords


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from dataLoader.kitti_tracking_object import kitti_tracking_object
    from core.proj_utils import get_lidar_in_image_fov
    from visualization.viz_utils import *

    root_dir = '/media/mooyu/Guoxs_Data/Datasets/3D_Object_Tracking_Evaluation_2012/'
    datasets = kitti_tracking_object(root_dir, 6)
    lidar = datasets.get_lidar(140)[:, :3]
    calib = datasets.get_calibration()
    labels = datasets.get_label_objects(140)

    lidar = get_lidar_in_image_fov(lidar, calib, 0, 0, cfg.IMG_WIDTH, cfg.IMG_HEIGHT)

    filtered_lidar = get_filtered_lidar(lidar)

    box3d = [0, 70.4, -40, 40, -2, 2]

    # fig = show_lidar_with_boxes(filtered_lidar, labels, calib)
    # draw_bounding_box(box3d, fig)
    # input()

    begin = time.time()
    voxel_feature, voxel_mask = create_voxel(filtered_lidar)
    end = time.time()
    print('create voxel cost time: %.4f' %(end-begin))

    # print(voxel_feature.shape)
    # zeros = voxel_feature == 0
    # print(np.sum(zeros) / (cfg.D * cfg.H * cfg.W))

    '''
    plt.figure(1)
    for i in range(8):
        img_i = voxel_feature[i]
        plt.subplot(2, 4, i+1)
        plt.imshow(img_i, cmap='gray')

    plt.show()

    img_all = np.sum(voxel_feature, axis=0)
    plt.imshow(img_all, cmap='gray')
    plt.show()
    '''

