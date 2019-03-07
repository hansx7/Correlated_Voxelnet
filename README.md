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


# How to Use the Server

using ssh:

```
ssh hk1@222.200.180.181
password: hk1

The server have four Nvidia Tesla P100.

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 396.26                 Driver Version: 396.26                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-SXM2...  Off  | 00000000:04:00.0 Off |                    0 |
| N/A   57C    P0   161W / 300W |  15143MiB / 16280MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla P100-SXM2...  Off  | 00000000:06:00.0 Off |                    0 |
| N/A   51C    P0   200W / 300W |  12087MiB / 16280MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla P100-SXM2...  Off  | 00000000:07:00.0 Off |                    0 |
| N/A   48C    P0   178W / 300W |  12075MiB / 16280MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla P100-SXM2...  Off  | 00000000:08:00.0 Off |                    0 |
| N/A   56C    P0   152W / 300W |  12075MiB / 16280MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
```

The server has installed `anaconda` , and the environment `pytorch04` is used for
the project.
```
conda activate pytorch04
# look the package installed in anaconda
codna list
# look all codna environments
conda env list
# create new env
conda create -n [env_name] python=[3.6/2.7]
```

Use server as `jupyter notebook` server:

using this cmd connect to server
```
ssh hk1@222.200.180.181 -L 127.0.0.1:[local_port]:127.0.0.1:[server_notebook_port]
```
`local_port` is unused port on your computer, such as `1234` `1235` etc. `server_notebook_port`
is the server notebook port, default is `9999`, if more than one jupyter notebook is hold,
then the port is `10000`, `10001`... 

> The `kitti_tracking_datasets` and `kitti_detection_datasets` are stored in `/data/hk1/datastes/kitti`.


