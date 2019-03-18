import os
import sys
sys.path.append('..')
from dataLoader.kitti_tracking_object import kitti_tracking_object
from dataLoader.kitti_detection_object import kitti_detection_object
from visualization.viz import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

root_dir = ROOT_DIR + '/datasets/kitti/detection'
datasets = kitti_detection_object(root_dir)

test_idx = 3
lidar = datasets.get_lidar(test_idx)
image = datasets.get_image(test_idx)
calib = datasets.get_calibration(test_idx)
label = datasets.get_label_objects(test_idx)
# plane = datasets.get_planes(test_idx)
# print(plane)

img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_height, img_width, img_channel = img.shape
print(('Image shape: ', img.shape))
pc_velo = lidar[:,0:3]

# def filter_lidar(lidar, plane):
#     params = np.array([plane[0], plane[2], plane[1]])
#     params = params[:, np.newaxis]
#     out = lidar @ params
#     out = out.reshape(-1)
#     idx = (out <= plane[3])
#     return lidar[idx]

for i in range(3):
    print('label{}'.format(i), label[i].__dict__)
pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0, img_width, img_height)

# Draw 2d and 3d boxes on image
show_image_with_boxes(img, label, calib, show3d=True)
input()

# Show all LiDAR points. Draw 3d box in LiDAR point cloud
# print(pc_velo.shape)
# pc_velo = filter_lidar(pc_velo, plane)
print(pc_velo.shape)
show_lidar_with_boxes(pc_velo, label, calib)
mlab.show()
input()

# Show lidar on image
# show_lidar_on_image(pc_velo, img, calib, img_width, img_height)
# input()
