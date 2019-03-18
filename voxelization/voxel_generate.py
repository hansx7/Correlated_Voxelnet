import numpy as np
import itertools

def voxel_split(pc, cls='Car'):
    '''

    :param pc: point cloud Nx4
    :param scence_size:  scence size, C X H X W
    :param voxel_size: voxel size, c x h x w
    :return: voxel_dict
    '''
    if cls == 'Car':
        scene_size = np.array([4, 80, 70.4], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 400, 352], dtype=np.int64)
        lidar_coord = np.array([0, 40, 3], dtype=np.float32)
        max_point_number = 35
    else:
        scene_size = np.array([4, 40, 48], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 200, 240], dtype=np.int64)
        lidar_coord = np.array([0, 20, 3], dtype=np.float32)
        max_point_number = 45

    shifted_coord = pc[:,:3] + lidar_coord
    # reverse pc coordinate (X,Y,Z) -> (Z,Y,X)
    shifted_coord = shifted_coord[:,::-1]

    voxel_index = np.floor(shifted_coord / voxel_size).astype(np.int)

    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    point_cloud = pc[bound_box]
    voxel_index = voxel_index[bound_box]

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)

    K = len(coordinate_buffer)
    T = max_point_number

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=(K), dtype=np.int64)

    # [K, T, 7] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(K, T, 7), dtype=np.float32)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, pc):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1

    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
                                feature_buffer[:, :, :3].sum(axis=1, keepdims=True) / number_buffer.reshape(K, 1, 1)

    voxel_dict = {'feature_buffer': feature_buffer,
                  'coordinate_buffer': coordinate_buffer,
                  'number_buffer': number_buffer}
    return voxel_dict


def genenrate_voxel_coords(scence_size, voxel_size):
    axis_len = [scence_size[2*i+1] - scence_size[2*i] for i in range(3)]
    Xmin, Xmax, Ymin, Ymax, Zmin, Zmax = scence_size
    voxel_x, voxel_y, voxel_z = [int(i / j) for i,j in zip(axis_len, voxel_size)]
    print(voxel_x,voxel_y,voxel_z)
    voxel_grids = []
    x_grid = [Xmin + voxel_size[0] * i for i in range(voxel_x)]
    y_grid = [Ymin + voxel_size[1] * i for i in range(voxel_y)]
    z_grid = [Zmin + voxel_size[2] * i for i in range(voxel_z)]
    # apply Cartesian product to generate grid
    voxel_coords = itertools.product(x_grid,y_grid,z_grid)
    for i in voxel_coords:
        voxel_grids.append([i[0],i[0]+voxel_size[0],
                            i[1],i[1]+voxel_size[1],
                            i[2],i[2]+voxel_size[2]])
    return voxel_grids

if __name__=='__main__':
    import sys
    import os
    sys.path.append('..')
    from dataLoader.kitti_detection_object import kitti_detection_object
    from visualization.viz_utils import *

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)

    root_dir = ROOT_DIR + '/datasets/kitti/detection'
    datasets = kitti_detection_object(root_dir)
    lidar = datasets.get_lidar(5)
    scence_size = [-40,40, -30,50, -4,4]
    voxel_size = [8, 8, 4]
    bound_x = np.logical_and(
        lidar[:, 0] >= scence_size[0], lidar[:, 0] < scence_size[1])
    bound_y = np.logical_and(
        lidar[:, 1] >= scence_size[2], lidar[:, 1] < scence_size[3])
    bound_z = np.logical_and(
        lidar[:, 2] >= scence_size[4], lidar[:, 2] < scence_size[5])
    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
    lidar_fixed = lidar[bound_box][:,:3]

    fig = draw_lidar_simple(lidar_fixed)
    fig = draw_bounding_box(scence_size,fig)
    voxel_grids = genenrate_voxel_coords(scence_size, voxel_size)
    i = 1
    for voxel in voxel_grids:
        print('draw box: %d / %d' %(i, len(voxel_grids)))
        fig = draw_bounding_box(voxel, fig)
        i += 1

    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    mlab.show()
    input()