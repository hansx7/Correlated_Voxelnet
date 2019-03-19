# Introduction

These are some common operations for point cloud writing in Python, including:

- Reading data from Kitti Datasets
- Visualization 
- Voxelization
- ...

# Dependencies

- Python 3.5+
- Mayavi
- numpy
- pytorch
- opencv-python
- pillow
- cython


Install Mayavi (scientific data visualization and plotting in Python) on Ubuntu
Ref: http://docs.enthought.com/mayavi/mayavi/installation.html
```bash
sudo apt-get install python-vtk python-qt4 python-qt4-gl python-setuptools python-numpy python-configobj
sudo pip install mayavi
pip install PyQt5
```


# Kitti

the `kitti` folder including two kind of datasets samples currently, `detection` 
([3D Object Detection Evaluation 2017](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d))
and `tracking` ([Object Tracking Evaluation 2012](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)).

The folders should look something like the following:

    kitti
        detection
            testing
            training
                calib         # calibration files
                image_2       # left images, 10 pngs
                label_2       # kitti label files
                velodyne      # point cloud files, 10 bins
        tracking
            testing           # 1 video, has 10 frames
            training          # 2 videos, each has 10 frames
                calib
                image_2
                label_2
                oxts          # GPS/IMU data
                velodyne


*The full project can be seen on my lab's server.*
