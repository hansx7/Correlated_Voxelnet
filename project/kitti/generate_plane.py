import sys
import numpy as np
import os
sys.path.append('../../')
from pyntcloud import PyntCloud
from dataLoader.kitti_tracking_object import kitti_tracking_object
from core.proj_utils import get_lidar_in_image_fov


root_dir = '/media/mooyu/Guoxs_Data/Datasets/3D_Object_Tracking_Evaluation_2012/'

def generate_plane(in_path, out_path):
    lidar = np.fromfile(in_path, dtype=np.float32)
    lidar = lidar.reshape((-1, 4))
    lidar = lidar[:,:3]
    np.savetxt('./temp.txt', lidar)
    cloud = PyntCloud.from_file('./temp.txt',
                                sep=" ",
                                header=0,
                                names=["x", "y", "z"])

    is_floor = cloud.add_scalar_field("plane_fit", n_inliers_to_stop=len(cloud.points) / 30, max_dist=0.001)
    cloud.points = cloud.points[cloud.points["is_plane"] > 0]
    three_points = cloud.get_sample("points_random", n=3, as_PyntCloud=False)
    three_points_np = []
    for i in range(len(three_points)):
        three_points_np.append(np.array([three_points["x"][i], three_points["z"][i], three_points["y"][i] + 1.65]))

    vector_one = three_points_np[1] - three_points_np[0]
    vector_two = three_points_np[2] - three_points_np[0]
    normal = np.cross(vector_one, vector_two)
    normal_normalized = normal / np.linalg.norm(normal)

    fwriter = open(out_path, 'w+')
    fwriter.write('# Plane\nWidth 4\nHeight 1\n')
    fwriter.write('%.6e %.6e %.6e %.6e' %(normal_normalized[0],
                                          normal_normalized[1],
                                          normal_normalized[2],
                                          1.65))
    fwriter.close()




lidar_root = root_dir+'training/velodyne/'
for i in range(22):
    videos_id = os.listdir(lidar_root)
    for id in videos_id:
        out_dir = root_dir + 'training/planes/' + id
        os.makedirs(out_dir, exist_ok=True)
        lidar_names = os.listdir(lidar_root+id)
        for name in lidar_names:
            in_path = lidar_root+ id + '/' + name
            out_path = out_dir + '/' + name.split('.')[0] + '.txt'
            generate_plane(in_path, out_path)





