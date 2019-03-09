from __future__ import print_function

from core.proj_utils import *


class Object3d(object):
    ''' 3d detection object label '''
    def __init__(self, label_file_line, datasets='detection'):
        data = label_file_line.split(' ')
        self.datasets = datasets
        if datasets == 'detection':
            data[1:] = [float(x) for x in data[1:]]
            # extract label, truncation, occlusion
            self.type = data[0]  # 'Car', 'Pedestrian', ...

        elif datasets == 'tracking':
            data[3:] = [float(x) for x in data[3:]]
            # extract frame, object id
            self.frame = int(data[0])
            self.obj_id = int(data[1])

            # extract label, truncation, occlusion
            self.type = data[2]  # 'Car', 'Pedestrian', ...

        self.truncation = data[-14]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[-13])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[-12]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[-11]  # left
        self.ymin = data[-10]  # top
        self.xmax = data[-9]  # right
        self.ymax = data[-8]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[-7]  # box height
        self.w = data[-6]  # box width
        self.l = data[-5]  # box length (in meters)
        self.t = (data[-4], data[-3], data[-2])  # location (x,y,z) in camera coord.
        self.ry = data[-1]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        
        self.boxes3d = self.to_xyzhmlr()

    def update_by_oxts(self, center_coord, ry):
        self.t = (center_coord[0], center_coord[1], center_coord[2])
        self.ry = ry
    
    def to_xyzhmlr(self):
        return [self.t[0], self.t[1], self.t[2], self.h, self.w, self.l, self.ry]

    def print_object(self):
        if self.datasets == 'tracking':
            print('Frame_id, object_id: %d, %d' % (self.frame, self.obj_id))

        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rigid transform from IMU coord to Velodyne coord
        self.I2V = calibs['Tr_imu_to_velo']
        self.I2V = np.reshape(self.I2V, [3,4])
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def print_object(self):
        print('P, V2C, C2V, R0: ')
        print(self.P)
        print(self.V2C)
        print(self.C2V)
        print(self.R0)
        print('c_u, c_v, f_u, f_v, b_x, b_y: ')
        print(self.c_u, self.c_v, self.f_u, self.f_v, self.b_x, self.b_y)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


class Oxts(object):
    '''
    GPS/IMU information, written for each synchronized frame, each text file contains 30 values
    '''

    def __init__(self, oxts_lines):
        data = oxts_lines.split()
        self.latitude   = float(data[0])       # latitude of the oxts-unit (deg)
        self.longitude  = float(data[1])       # longitude of the oxts-unit (deg)
        self.altitude   = float(data[2])       # altitude of the oxts-unit (m)
        self.roll       = float(data[3])       # roll angle (rad),  0 = level, positive = left side up (-pi..pi)
        self.pitch      = float(data[4])       # pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
        self.yaw        = float(data[5])       # heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)

    def distance(self, object):
        '''
        calculate the diatance of two point using (latitude, longitude)

        L = 2R * arcsin(sqrt(sin^2((lat1-lat2)/2) + cos(lon1) * cos(lon2) * sin^2((lon1-lon2)/2)))
        '''
        lat1, lon1 = rad(self.latitude), rad(self.longitude)
        lat2, lon2 = rad(object.latitude), rad(object.longitude)
        R = 6378137.0   # radius of earth (m)
        a = lat1 - lat2
        b = lon1 - lon2
        dis = 2 * R * np.arcsin(
                            np.sqrt(np.power(np.sin(a/2), 2) +
                            np.cos(lat1) * np.cos(lat2) * np.power(np.sin(b/2),2))
                    )
        return dis

    def displacement(self, object):
        d = self.distance(object)
        delta_yaw = self.yaw - object.yaw
        delta_pitch = self.pitch - object.pitch
        delta_x = d * np.cos(delta_yaw)
        delta_y = d * np.sin(delta_yaw)
        delta_z = d * np.sin(delta_pitch)
        return np.array([delta_x, delta_y, delta_z])

    def get_rotate_matrix(self, object, axis='z'):
        if axis == 'x':
            delta_pitch = self.pitch - object.pitch
            return rotx(delta_pitch)
        if axis == 'y':
            delta_roll = self.roll - object.roll
            return roty(delta_roll)
        elif axis == 'z':
            delta_yaw = self.yaw - object.yaw
            return rotz(delta_yaw)

    def get_whole_rotate_matrix(self):
        R_pitch = rotx(self.pitch)
        R_roll = roty(self.roll)
        R_yaw = rotz(self.yaw)
        return R_yaw @ R_pitch @ R_roll


def rad(deg):
    '''Convert degree to rad'''
    return deg * np.pi / 180.00


def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_lidar_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orientation_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    '''

    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l], [0, 0], [0, 0]])

    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0, :] = orientation_3d[0, :] + obj.t[0]
    orientation_3d[1, :] = orientation_3d[1, :] + obj.t[1]
    orientation_3d[2, :] = orientation_3d[2, :] + obj.t[2]

    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = None
        return orientation_2d, np.transpose(orientation_3d)

    # project orientation into the image plane
    orientation_2d = project_lidar_to_image(np.transpose(orientation_3d), P)
    return orientation_2d, np.transpose(orientation_3d)


def get_box3d_ori3d(objects, calib):
    boxes3d = []
    ories3d = []
    for obj in objects:
        _, box3d_pts_3d = compute_box_3d(obj, calib.P)
        box3d = calib.project_rect_to_velo(box3d_pts_3d)

        _, ori3d_pts_3d = compute_orientation_3d(obj, calib.P)
        ori3d = calib.project_rect_to_velo(ori3d_pts_3d)

        boxes3d.append(box3d)
        ories3d.append(ori3d)

    return boxes3d, ories3d