import numpy as np

def seq_voxelization(cfg, lidars, method='v2'):
    '''get the voxel of a list of lidar data'''
    seq_features = []
    for lidar in lidars:
        if method == 'v3':
            voxel_features = voxelization_v3(cfg, lidar)
        else:
            voxel_features = voxelization_v2(cfg, lidar)
        seq_features.append(voxel_features)

    batch_features = np.concatenate([seq_features], axis=0)
    return batch_features


def voxelization_v1(cfg, point_cloud):
    '''get the voxel of lidar data
    input:
        point_cloud:  (N, 4)
    output:
        voxel_features: N' x cfg.T x 7, N' is the number of non-empty voxel, cfg.T is the maximum number of points to
                                        cal features for the voxel, points out of cfg.T will be discard.
                                        features: (x, y, z, x_mean, y_mean, z_mean, r)
        voxel_coords:   N' x 3          the index of voxel in 3D grid.

    '''
    # shuffling the points
    np.random.shuffle(point_cloud)

    # print('point cloud range')
    # print('x range', np.max(point_cloud[:, 0]), np.min(point_cloud[:, 0]))
    # print('y range', np.max(point_cloud[:, 1]), np.min(point_cloud[:, 1]))
    # print('z range', np.max(point_cloud[:, 2]), np.min(point_cloud[:, 2]))

    voxel_index = ((point_cloud[:, :3] - np.array([cfg.xrange[0], cfg.yrange[0], cfg.zrange[0]])) /
                    (cfg.vw, cfg.vh, cfg.vd)).astype(np.int32)

    # convert to  (D, H, W)
    voxel_index = voxel_index[:, [2, 1, 0]]
    
    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)
    
    K = len(coordinate_buffer)

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=(K), dtype=np.int64)

    # [K, T, 7] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(K, cfg.T, 7), dtype=np.float32)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < cfg.T:
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1

    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis=1, keepdims=True)/number_buffer.reshape(K, 1, 1)
    
    return feature_buffer, coordinate_buffer


def voxelization_v2(cfg, lidar):
    '''get the voxel of lidar data
        input:
            lidar:  (N, 4)
        output:
            voxel_feature:      (H / vd, H/ vH, W / vw), default (20, 400, 352), the whole 3d cube. the value of each
                                voxel is hand-crafted height statistic, such as 'mean height', 'max height'... default
                                is 'mean height'
        '''
    # shuffling the points
    np.random.shuffle(lidar)

    voxel_coords = ((lidar[:, :3] - np.array([cfg.xrange[0], cfg.yrange[0], cfg.zrange[0]])) /
                    (cfg.vw, cfg.vh, cfg.vd)).astype(np.int32)

    # convert to (D, H, W)
    voxel_coords = voxel_coords[:, [2, 1, 0]]
    voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0,
                                                    return_inverse=True, return_counts=True)

    voxel_feature = np.zeros((cfg.D, cfg.H, cfg.W, 2), dtype=np.float32)

    for i in range(len(voxel_coords)):
        idx = tuple(voxel_coords[i, :])
        # points in voxel i
        pts = lidar[inv_ind == i]
        # remove points that out of limit
        # if voxel_counts[i] > cfg.T:
        #    pts = pts[:cfg.T, :]
        #    voxel_counts[i] = cfg.T
        # set voxel_feature[idx] value
        voxel_feature[idx] = get_feature(pts, method='mean')

    return voxel_feature.transpose(3, 0, 1, 2)


def get_feature(lidar_cluster, method='mean'):
    features = np.zeros((2,), dtype=np.float32)
    if method == 'max':
        features[0] = np.max(lidar_cluster[:, 2])
    if method == 'min':
        features[0] = np.min(lidar_cluster[:, 2])
    if method == 'mean':
        features[0] = np.mean(lidar_cluster[:, 2])
    features[1] = np.mean(lidar_cluster[:, 3])
    return features

def voxelization_v3(cfg, en_lidar):
    '''get the voxel of lidar data
        input:
            lidar:  (N, 7)  [x, y, z, d, b, g, r]
        output:
            voxel_feature:      (H / vd, H/ vH, W / vw, 4), default (20, 400, 352), the whole 3d cube. the value of each
                                voxel is hand-crafted height statistic, such as 'mean height', 'max height'... default
                                is 'mean height'
        '''
    # shuffling the points
    np.random.shuffle(en_lidar)

    voxel_coords = ((en_lidar[:, :3] - np.array([cfg.xrange[0], cfg.yrange[0], cfg.zrange[0]])) /
                    (cfg.vw, cfg.vh, cfg.vd)).astype(np.int32)

    # convert to (D, H, W)
    voxel_coords = voxel_coords[:, [2, 1, 0]]
    voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0,
                                                    return_inverse=True, return_counts=True)

    voxel_feature = np.zeros((cfg.D, cfg.H, cfg.W, 2), dtype=np.float32)

    for i in range(len(voxel_coords)):
        idx = tuple(voxel_coords[i, :])
        # points in voxel i
        pts = en_lidar[inv_ind == i]
        # set voxel_feature[idx] value
        voxel_feature[idx] = get_enhanced_feature(pts)

    return voxel_feature.transpose(3, 0, 1, 2)


def get_enhanced_feature(en_lidar):
    '''
    :param en_lidar: enhanced_lidar, (N, 7) [x, y, z, d, b, g, r]
    :return: features, (5,), [np.min(z), np.max(d), np.mean(b, g, r)]
    '''
    features = np.zeros((2,), dtype=np.float32)
    # get the height
    features[0] = np.mean(en_lidar[:, 2])
    # get the max reflection intensity
    features[1] = np.max(en_lidar[:, 3])
    # get the mean of rgb
    # features[2:] = np.mean(en_lidar[:, 4:], axis=0)
    return features