import numpy as np


def aug_data(lidar, boxes3d):
    '''
    data augmentation
    input:
        lidars: Ni x 4
        boxes3d
    return:
        lidars and labels after augmentation
    '''
    aug_lidar = np.copy(lidar)
    aug_boxes3d = np.copy(boxes3d)
    np.random.seed()
    choice = np.random.randint(1, 10)
    if choice >=7:
        # print('apply perturbation')
        # apply perturbation independently to each gt_box3d and points in box
        aug_lidar, aug_boxes3d = perturbation(aug_lidar, aug_boxes3d)
    elif choice < 7 and choice >=4:
        # print('apply global rotation')
        # global rotation
        aug_lidar, aug_boxes3d = global_rotation(aug_lidar, aug_boxes3d)
    else:
        # print('apply global scaling')
        # global scaling
        aug_lidar, aug_boxes3d = global_scaling(aug_lidar, aug_boxes3d)
            
    return aug_lidar, aug_boxes3d


def perturbation(lidar, boxes3d):
    N = len(boxes3d)
    scale_factor = 1.05
    for i in range(N):
        box3d = boxes3d[i]
        # collision flag
        flag = True
        reset_count = 0
        while flag and reset_count < 100:
            angle = np.random.uniform(-np.pi / 10, np.pi / 10)
            trans = np.random.normal(0.0, 0.5, size=(2,))
            # scale box3d
            box3d = boxwise_scale(box3d, scale_factor)
            points, inds = extract_pc_in_box3d(lidar[:, :3], box3d)
            # check collision
            axis = np.mean(box3d, axis=0)
            t_box3d = rotate_with_axis(box3d, axis, trans[0], trans[1], 0, angle)
            if not is_collision(i, boxes3d, scale_factor):
                flag = False
                # print(angle, trans)
                t_points = rotate_with_axis(points, axis, trans[0], trans[1], 0, angle)
                lidar[inds, :3] = t_points
                boxes3d[i] = t_box3d
            else:
                flag = True
                reset_count += 1

    return lidar, boxes3d


def global_rotation(lidar, boxes3d):
    angle = np.random.uniform(-np.pi / 4, np.pi / 4)
    lidar[:, :3] = point_transform(lidar[:, :3], 0, 0, 0, rz=angle)
    gr_boxes3d = box_transform(boxes3d, 0, 0, 0, r=angle)
    return lidar, gr_boxes3d


def global_scaling(lidar, boxes3d):
    factor = np.random.uniform(0.95, 1.05)
    # print(factor)
    lidar[:, :3] = lidar[:, :3] * factor
    boxes3d = boxes3d * factor
    return lidar, boxes3d

def in_hull(p, hull):
    """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    # scale box3d
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def is_collision(idx, boxes3d, factor):
    box3d = boxes3d[idx]
    box3d = boxwise_scale(box3d, factor)
    for i in range(len(boxes3d)):
        if i == idx:
            continue
        else:
            box3d_i = boxwise_scale(boxes3d[i], factor)
            inds = in_hull(box3d, box3d_i)
            if inds.any():
                return True
    return False


def boxwise_scale(box3d, factor):
    mat = np.zeros((4, 4))
    mat[0, 0] = factor
    mat[1, 1] = factor
    mat[2, 2] = factor
    mat[3, 3] = 1
    mean_x, mean_y, mean_z = np.mean(box3d, axis=0)
    o_box3d = - box3d + [mean_x, mean_y, mean_z]
    mat[:, 3] = mean_x, mean_y, mean_z, 1
    scaled_box3d = np.hstack([o_box3d, np.ones((8, 1))])
    scaled_box3d = np.matmul(scaled_box3d, mat)
    scaled_box3d = scaled_box3d[:, :3] + [mean_x, mean_y, mean_z]
    return scaled_box3d


def rotate_with_axis(points, axis, tx, ty, tz, angle):
    o_points = points - axis
    r_points = point_transform(o_points, tx, ty, tz, rz=angle)
    out_points = r_points + axis
    return out_points


def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
    '''
    Input:
        points:     Nx3
        tx/ty/tz:   translation
        rx/ry/rz:   rotation, (rad)
    output:
        points:     Nx3
    '''
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))])
    mat1 = np.eye(4)
    mat1[3, 0:3] = tx, ty, tz
    points = np.matmul(points, mat1)
    if rx != 0:
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        mat[3, 3] = 1
        mat[1, 1] = np.cos(rx)
        mat[1, 2] = -np.sin(rx)
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)
    if ry != 0:
        mat = np.zeros((4, 4))
        mat[1, 1] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(ry)
        mat[0, 2] = np.sin(ry)
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)
    if rz != 0:
        mat = np.zeros((4, 4))
        mat[2, 2] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(rz)
        mat[0, 1] = -np.sin(rz)
        mat[1, 0] = np.sin(rz)
        mat[1, 1] = np.cos(rz)
        points = np.matmul(points, mat)
    return points[:, 0:3]


def box_transform(boxes_corner, tx, ty, tz, r=0):
    # boxes3d (B, 8, 3)
    for idx in range(len(boxes_corner)):
        boxes_corner[idx] = point_transform(boxes_corner[idx], tx, ty, tz, rz=r)
    return boxes_corner



if __name__ == '__main__':
    import sys
    import mayavi.mlab as mlab
    sys.path.append('../../')
    import dataLoader.kitti_utils as utils
    from project.utils.utils import get_filtered_lidar
    from project.config.dFaF_config import config as cfg
    from dataLoader.kitti_detection_object import kitti_detection_object
    from core.proj_utils import get_lidar_in_image_fov
    from visualization.viz import draw_lidar, draw_gt_boxes3d, draw_bounding_box

    datasets = kitti_detection_object(cfg.KITTI_DETECTION_DATASET_ROOT)
    idx = 1951
    lidar = datasets.get_lidar(idx)
    calib = datasets.get_calibration(idx)
    label = datasets.get_label_objects(idx)

    # get filtered lidar
    pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(lidar[:, :3], calib, 0, 0,
                                                       cfg.IMG_WIDTH, cfg.IMG_HEIGHT, True)
    lidar = lidar[fov_inds]

    # label to boxes3d
    boxes3d = []
    for obj in label:
        if obj.type in ['Car', 'Van']:
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d = calib.project_rect_to_velo(box3d_pts_3d)
            boxes3d.append(box3d)

    boxes3d = np.concatenate([boxes3d], axis=0)
    # data aug
    lidar_aug, boxes3d_aug = aug_data(lidar, boxes3d)

    lidar_aug, boxes3d_aug = get_filtered_lidar(lidar_aug, boxes3d_aug)

    fig1 = draw_lidar(lidar_aug)
    fig1 = draw_gt_boxes3d(boxes3d, fig1, box_id=0, color=(1, 1, 1))
    fig1 = draw_gt_boxes3d(boxes3d_aug, fig1, box_id=1, color=(1, 0, 0))
    box3d = [0, 70.4, -40, 40, -2, 2]
    fig1 = draw_bounding_box(box3d, fig1)
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig1)

    mlab.show()
    input()


