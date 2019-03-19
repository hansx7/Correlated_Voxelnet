from PIL import Image
import sys
sys.path.append('..')
from visualization.viz_utils import *
from dataLoader.kitti_utils import compute_box_3d, compute_orientation_3d
from core.proj_utils import get_lidar_in_image_fov


def show_image_with_boxes(img, objects, calib, color=(0, 255, 0), show3d=True):
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img) # for 2d bbox
    img2 = np.copy(img) # for 3d bbox
    for obj in objects:
        if obj.type=='DontCare':continue
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)), color, 2)
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        img2 = draw_projected_box3d(img2, box3d_pts_2d, color=color)
    Image.fromarray(img1).show()
    if show3d:
        Image.fromarray(img2).show()
        return img1, img2
    return img1


def show_lidar_with_boxes(pc_velo, objects, calib):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''

    print(('All point num: ', pc_velo.shape[0]))
    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(pc_velo, fig=fig)
    box_id = 0
    for obj in objects:
        if obj.type=='DontCare':continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1,y1,z1 = ori3d_pts_3d_velo[0,:]
        x2,y2,z2 = ori3d_pts_3d_velo[1,:]
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, box_id=box_id)
        box_id += 1
        mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5, 0.5, 0.5),
            tube_radius=None, line_width=1, figure=fig)
    mlab.show(1)
    return fig


def show_lidar_with_box3d_ori3d(lidar, boxes3d, ories3d, fig=None, draw_id=True):
    '''Show all LiDAR points
        Draw 3d bounding box and orientation vector in LiDAR point cloud (in velo coord system)
    '''
    if fig == None:
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(lidar,fig=fig,draw_rigion=False)
    box_id = 0
    for (box, ori) in zip(boxes3d, ories3d):
        draw_gt_boxes3d([box], fig=fig, box_id=box_id, draw_text=draw_id)
        box_id += 1
        x1, y1, z1 = ori[0, :]
        x2, y2, z2 = ori[1, :]
        mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.5, 0.5, 0.5),
                    tube_radius=None, line_width=1, figure=fig)
    mlab.show(1)
    return fig


def show_multi_lidar_with_box3d_ori3d(lidars, labels):
    '''Show all LiDAR points in continuous frames after convert to the same coordinate
        Draw 3d bounding box and orientation vector in LiDAR point cloud (in velo coord system)
    '''
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    print('lidars', lidars)
    print('labels', labels['gt_boxes3d'])
    # for (lidar, label) in zip(lidars, labels):
    #     if label == {}:
    #         continue
    #     print('lidar', lidar.shape)
    #     print('label', label)
    #     box3d = labels['gt_boxes3d']
    #     ori3d = labels['gt_ories3d']
    #     fig = show_lidar_with_box3d_ori3d(lidar, box3d, ori3d, fig, draw_id=False)
    for i in range(2):
        box3d = labels['gt_boxes3d'][i]
        ori3d = labels['gt_ories3d'][i]
        print('go into utils')
        fig = show_lidar_with_box3d_ori3d(lidars[i], box3d, ori3d, fig, draw_id=False)
    mlab.show(1)
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3]*255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i,2]
        color = cmap[int(640.0/depth),:]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i,0])),
            int(np.round(imgfov_pts_2d[i, 1]))),
            1, color=tuple(color), thickness=-1)
    Image.fromarray(img).show()
    return img
