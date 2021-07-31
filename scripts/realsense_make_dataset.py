#!/usr/bin/env python
# coding: latin-1


import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud, Image

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2 as msg_PointCloud2

from geometry_msgs.msg import Point32
import std_msgs.msg
from jsk_rviz_plugins.msg import *
from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_recognition_msgs.msg import BoundingBox
#from tf.transformations import *
from scipy.spatial.transform import Rotation as R
import ctypes

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import ros_numpy


##### 3DOD
import pickle       
import numpy as np
import math
import cv2
import time
##### 3DOD

from cv_bridge import CvBridge, CvBridgeError

import sys

#


#  ███████╗    ██╗   ██╗    ███╗   ██╗     ██████╗    ████████╗    ██╗     ██████╗     ███╗   ██╗    ███████╗
#  ██╔════╝    ██║   ██║    ████╗  ██║    ██╔════╝    ╚══██╔══╝    ██║    ██╔═══██╗    ████╗  ██║    ██╔════╝
#  █████╗      ██║   ██║    ██╔██╗ ██║    ██║            ██║       ██║    ██║   ██║    ██╔██╗ ██║    ███████╗
#  ██╔══╝      ██║   ██║    ██║╚██╗██║    ██║            ██║       ██║    ██║   ██║    ██║╚██╗██║    ╚════██║
#  ██║         ╚██████╔╝    ██║ ╚████║    ╚██████╗       ██║       ██║    ╚██████╔╝    ██║ ╚████║    ███████║
#  ╚═╝          ╚═════╝     ╚═╝  ╚═══╝     ╚═════╝       ╚═╝       ╚═╝     ╚═════╝     ╚═╝  ╚═══╝    ╚══════╝

# sends a 3D boxes to a ros publisher 
# =========================================
def send_3d_bbox_anno(publisher, center, dim, angle, header):
    """
    param center: x,y,z
    dims: h, w, l
    angles: angle
    publisher: ros publisher
    header: ros header
    """
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'camera_color_frame'
    box_arr = BoundingBoxArray()
    box_arr.header = header

    # ------------------------------------------------------------------------------------------------------
    # set variables
    # ------------------------------------------------------------------------------------------------------

    box_arr = BoundingBoxArray()
    box_arr.header = header

    pred_x = center[0]
    pred_y = center[1]
    pred_z = center[2]
    pred_h = dim[0]
    pred_w = dim[1]
    pred_l = dim[2]
    pred_r_y = angle

    # ------------------------------------------------------------------------------------------------------
    # cerate 3D box
    # ------------------------------------------------------------------------------------------------------

    box_pred = BoundingBox()  
    box_pred.header.stamp = rospy.Time.now()
    box_pred.header.frame_id = "camera_color_frame"
    box_pred.pose.position.x = pred_x
    box_pred.pose.position.y = pred_y
    box_pred.pose.position.z = pred_z
    # possible modofications 
    #print("Location :",location_array[i][0],location_array[i][1],location_array[i][2])
    #rotation_pred = quaternion_about_axis(pred_r_y+(np.pi/2),(0,1,0))
    #rotation_pred = quaternion_about_axis(-np.pi/2,(0,0,1))
    #qy_pred = quaternion_about_axis(pred_r_y+(np.pi/2),(0,1,0))
    #qz_pred = quaternion_about_axis(-np.pi/2,(0,0,1))
    #rotation_pred = quaternion_multiply(qy_pred, qz_pred)
    rotation_pred = R.from_euler('z', -pred_r_y, degrees=False).as_quat()
    #[0.         0.         0.09410831 0.99556196]
    box_pred.pose.orientation.x = rotation_pred[0]
    box_pred.pose.orientation.y = rotation_pred[1]
    box_pred.pose.orientation.z = rotation_pred[2]
    box_pred.pose.orientation.w = rotation_pred[3]
    box_pred.dimensions.x = pred_h#location_array[i][3]
    box_pred.dimensions.y = pred_w#location_array[i][4]
    box_pred.dimensions.z = pred_l#location_array[i][5]
    #box_pred.value = (counter % 100) / 100.0        
    box_arr.boxes.append(box_pred)

    # ------------------------------------------------------------------------------------------------------
    # publish teh box to rviz
    # ------------------------------------------------------------------------------------------------------

    publisher.publish(box_arr)

# get the corners of a 3D bounding box
# =========================================
def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 
    
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
    
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners

# rotates a list of an arbitrary number of points (x,y,z) around an axis
# =========================================
def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)




def pc2_to_xyzrgb(point):
	# Thanks to Panos for his code used in this function.
    x, y, z = point[:3]
    rgb = point[3]

    # cast float32 to int so that bitwise operations are possible
    s = struct.pack('>f', rgb)
    i = struct.unpack('>l', s)[0]
    # you can get back the float value by the inverse operations
    pack = ctypes.c_uint32(i).value
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)
    return x, y, z#, r, g, b



#  ███╗   ███╗     █████╗     ██╗    ███╗   ██╗
#  ████╗ ████║    ██╔══██╗    ██║    ████╗  ██║
#  ██╔████╔██║    ███████║    ██║    ██╔██╗ ██║
#  ██║╚██╔╝██║    ██╔══██║    ██║    ██║╚██╗██║
#  ██║ ╚═╝ ██║    ██║  ██║    ██║    ██║ ╚████║
#  ╚═╝     ╚═╝    ╚═╝  ╚═╝    ╚═╝    ╚═╝  ╚═══╝


# this function creates a ros listener which wait wait for messages from realsense sensor to save 
# them with ground truth into a folder
# =========================================
class listener(): 

    def __init__(self, dataset_root_path, bb_rotation, start_idx, end_idx, train_or_test, empty_annos, save_every_x_step):

        # ------------------------------------------------------------------------------------------------------
        # set variables
        # ------------------------------------------------------------------------------------------------------

        # set the ros publisher
        rospy.init_node('listener', anonymous=True)
        self.pub_pc2 = rospy.Publisher('pc_duplicate2', PointCloud2)    
        self.bb_ground_truth_pub = rospy.Publisher("bb_ground_truth", BoundingBoxArray) 

        # other variables
        self.image_counter = 0
        self.start_idx = int(start_idx) # index for the point clouds # also effects the naming
        #self.image_counter = 700
        self.end_idx = int(end_idx)#2400 # index where self.image_counter have to stop
        #self.end_idx = 750#750
        self.dataset_root_path = dataset_root_path # where the data should be stored
        self.mode = train_or_test # option:test/train / are we doing recording for training or testing?
        self.data = [] # here we cache point clouds 
        self.angle = [float(bb_rotation)] # annotation angle / needs to be set according to current annotation 

        # set the header which is used for the ros messages
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_color_frame'
        self.header = header

        # Dummy for ros subscriber
        self.subscriber = None

        # 
        self.empty_annos = empty_annos # if True, there will be saved an empty annos file

        self.save_every_x_step = save_every_x_step

    # this listener buffers point clouds without calib or annotations
    # =========================================
    def callback_no_annos(self, data):

        print("Current ID: {}".format(str(self.image_counter)))

        # ------------------------------------------------------------------------------------------------------
        # fill up the buffer chache for the point clouds
        # ------------------------------------------------------------------------------------------------------

        if self.image_counter >= self.start_idx and self.image_counter < self.end_idx:
            self.data.append(data)

        # increase image_counter
        self.image_counter += 1

        # debug # send dummy bb to rviz (for no certain reason)
        # dimensions = np.array([1.8,0.4,0.6])
        # location = np.array([-0.1,0.0,2.0])
        # angle = 0 # this is set to zero always for dummy reasons
        # send_3d_bbox_anno(self.bb_ground_truth_pub, location[[2,0,1]],dimensions[[1,2,0]],angle, self.header)


    # if we have image_counter_min elements in the buffer store them
    # =========================================
    def save(self):
        # set up the second counter responsible for naming the point clouds 
        image_counter = 0 

        # iterate ovr cache
        for data in self.data:

            # ------------------------------------------------------------------------------------------------------
            # convert pountcloud format form sensor to array
            # ------------------------------------------------------------------------------------------------------

            #pc = np.array([pp[:3] for pp in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))])
            pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
            #pcx = ros_numpy.point_cloud2.merge_rgb_fields(data)

            # ------------------------------------------------------------------------------------------------------
            # take every 4 points to reduce data
            # ------------------------------------------------------------------------------------------------------

            pc = pc[1::10]
            print(pc.size)

            # ------------------------------------------------------------------------------------------------------
            # Rotate to get to Lidar Kitti coords
            # ------------------------------------------------------------------------------------------------------

            # TRANSFORM to lidar coords (x,y,z)
            # - the output of the realsense are image coords
            # - rviz already shows the direct output of the realsense in lidar coords because of the topic "CameraInfo"

            r = R.from_euler('y', -90, degrees=True).as_dcm()
            r2 = R.from_euler('x', 90, degrees=True).as_dcm()
            pc = np.dot(pc,r)
            pc = np.dot(pc,r2)
            pc = np.array(pc) + [0.0,0.0,1.0]

            # ------------------------------------------------------------------------------------------------------
            # save point cloud
            # ------------------------------------------------------------------------------------------------------


            image_counter_leading_zeros = "%06d" % (int(image_counter),) # we create leading zeros of the index to match the filenames of the dataset
            image_counter += 1
            filepath_pc = self.dataset_root_path + "testing_live/" + "velodyne/" + str(image_counter_leading_zeros) + ".pkl"
            # lift z axis to match the ground at 0
            with open(filepath_pc, 'wb') as file:
                pickle.dump(np.array(pc), file, 2)

            # ------------------------------------------------------------------------------------------------------
            # Publish 3D bbox for debug 
            # ------------------------------------------------------------------------------------------------------

            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                ]
            # https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
            pc_2 = point_cloud2.create_cloud(self.header, fields, pc)
            self.pub_pc2.publish(pc_2)

            # ------------------------------------------------------------------------------------------------------
            # exit program
            # ------------------------------------------------------------------------------------------------------

        import sys
        print("DONE")
        sys.exit()


    # this listener saves point clouds with calib and annotations
    # =========================================
    def callback(self, data):

        print("Current ID: {}".format(str(self.image_counter)))
         # increase image_counter
        self.image_counter += 1

        # ------------------------------------------------------------------------------------------------------
        # if we reached the number of point clouds we wanted to save: exit
        # ------------------------------------------------------------------------------------------------------

        # if self.start_idx > self.end_idx:
        #     import sys
        #     print("DONE")
        #     sys.exit()

        # ------------------------------------------------------------------------------------------------------
        # if we did not reach the image ID when we want to start saving: skip
        # ------------------------------------------------------------------------------------------------------

        # if self.image_counter <= self.start_idx:
        #     return None

        if self.image_counter % save_every_x_step == 0:
            
            # ------------------------------------------------------------------------------------------------------
            # print how many point clouds we have saved already
            # ------------------------------------------------------------------------------------------------------

            
            print("Current ID Naming: {}".format(str(self.start_idx)))

            # ------------------------------------------------------------------------------------------------------
            # convert pountcloud format form sensor to array
            # ------------------------------------------------------------------------------------------------------

            #pc = np.array([pc2_to_xyzrgb(pp) for pp in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z","rgb"))])
            pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)

            # ------------------------------------------------------------------------------------------------------
            #take every 4 points to reduce data
            # ------------------------------------------------------------------------------------------------------

            #pc = pc[1::4]
            pc = pc[1::4] # the rosbag has more points for some reason

            # ------------------------------------------------------------------------------------------------------
            # Rotate to get to Kitti coords
            # ------------------------------------------------------------------------------------------------------

            r = R.from_euler('y', -90, degrees=True).as_dcm()
            r2 = R.from_euler('x', 90, degrees=True).as_dcm()
            pc = np.dot(pc,r)
            pc = np.dot(pc,r2)

            # ------------------------------------------------------------------------------------------------------
            # save point cloud
            # ------------------------------------------------------------------------------------------------------

            # we create leading zeros of the index to match the filenames of the dataset
            image_counter_leading_zeros = "%06d" % (int(self.start_idx),) 
            self.start_idx += 1
            if self.mode == "train":
                filepath_pc = self.dataset_root_path + "training/" + "velodyne/" + str(image_counter_leading_zeros) + ".pkl"
            elif self.mode == "test":
                filepath_pc = self.dataset_root_path + "testing/" + "velodyne/" + str(image_counter_leading_zeros) + ".pkl"
            # lift z axis to match the ground at 0 (realsense sensor was 1 meter above the ground while recording, so ground is at -1)
            pc = np.array(pc) + [0.0,0.0,1.0]
            with open(filepath_pc, 'wb') as file:
                pickle.dump(np.array(pc), file, 2)

            # ------------------------------------------------------------------------------------------------------
            # save Calibration
            # ------------------------------------------------------------------------------------------------------

            # everything but R0_rect and Tr_velo_to_cam are palceholder
            # the placeholder are needed that the data work within the pipeline made for kitti data orginally
            calib = """P0: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 0.000000000000e+00 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.797842000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 4.575831000000e+01 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 -3.454157000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.981016000000e-03
P3: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.341081000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 2.330660000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.201153000000e-03
R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0
Tr_velo_to_cam: 0.0 -1.0 0.0 0.0 0.0 0.0 -1.0 0.0 1.0 0.0 0.0 0.0
Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01"""
            if self.mode == "train":
                filepath_calibration = self.dataset_root_path + "training/" + "calib/" + str(image_counter_leading_zeros) + ".txt"
            elif self.mode == "test":
                filepath_calibration = self.dataset_root_path + "testing/" + "calib/" + str(image_counter_leading_zeros) + ".txt"
            with open(filepath_calibration, 'w') as file:
                file.write(calib)
            
            # ------------------------------------------------------------------------------------------------------
            # Save Annotations
            # ------------------------------------------------------------------------------------------------------
            
            name = "Pedestrian"
            truncated = [0.0]
            occluded = [0]
            alpha = [0.0]
            bbox = [700.0, 100.00, 800.00, 300.00] # just set that the height is > 100 because this effect the difficulty assessment in "create data"
            dimensions = [1.8,0.4,0.6] # height,width,length (kitti format)
            #location = [2.0,0.0,-0.1] lidar coords: 
            #location = [-0.1,0.0,2.0] # image coords: we need to store in kitti image coords 
            location = [1.0,-0.05,3.0] # image coords: we need to store in kitti image coords z(-x)(-y)
            angle = self.angle
            annoations = truncated+occluded+alpha+bbox+dimensions+location+angle
            annoations = name + " " + ' '.join(str(e) for e in annoations)
            if self.mode == "train":
                filepath_annoation = self.dataset_root_path + "training/" + "label_2/" + str(image_counter_leading_zeros) + ".txt"
            elif self.mode == "test":
                filepath_annoation = self.dataset_root_path + "testing/" + "label_2/" + str(image_counter_leading_zeros) + ".txt"
            if self.empty_annos == False:
                with open(filepath_annoation, 'w') as file:
                    file.write(annoations)
                    
                
                # ------------------------------------------------------------------------------------------------------
                # Publish 3D bbox for debug 
                # ------------------------------------------------------------------------------------------------------

                # also transform location and dimensions back to my rviz coords
                send_3d_bbox_anno(self.bb_ground_truth_pub, np.array(location)[[2,0,1]],np.array(dimensions)[[1,2,0]],angle[0], self.header)

            else:
                with open(filepath_annoation, 'w') as file:
                    file.write("")
                    
                    
            

    # Just helper function to run this class
    # has a countdown and sets up the ros subscriber
    # =========================================
    def run(self,live_mode):

        # ------------------------------------------------------------------------------------------------------
        # Countdown
        # ------------------------------------------------------------------------------------------------------

        def countdown(t):
            while t > 0:
                print(t)
                t -= 1
                time.sleep(1)
            print("START NOW")
        countdown(1)

        # ------------------------------------------------------------------------------------------------------
        # Setting up the ROS Listener
        # ------------------------------------------------------------------------------------------------------

        # listener without annotations (live)
        if live_mode == "live_mode_on":
            self.subscriber = rospy.Subscriber("/camera/depth/color/points", msg_PointCloud2, self.callback_no_annos) # source: realsense directly
            #self.subscriber = rospy.Subscriber("/camera/depth/points", msg_PointCloud2, self.callback_no_annos) # source: to capture from depth_image_proc/point_cloud_xyz package

            # we check regularly if we have enough images (end_idx) and than unregister the subscriber and save the points
            while True:
                time.sleep(0.1)
                if self.image_counter >= self.end_idx:
                    self.subscriber.unregister()
                    self.save()
        
        # listener with annotations
        if live_mode == "live_mode_off":
            #self.subscriber = rospy.Subscriber("/camera/depth/color/points", msg_PointCloud2, self.callback) # source: realsense directly
            self.subscriber = rospy.Subscriber("/camera/depth/color/points", msg_PointCloud2, self.callback) # source: to capture from depth_image_proc/point_cloud_xyz package

            # spin() simply keeps python from exiting until this node is stopped
            # rospy.spin()
            
            # we check regularly if we have enough images (end_idx) and than unregister the subscriber and save the points
            while True:
                time.sleep(0.1)
                if self.start_idx > self.end_idx:
                    self.subscriber.unregister()
                    print("DONE")
                    sys.exit()
        

        
if __name__ == '__main__':
    
    # python scripts/realsense_make_dataset.py live_mode_off /home/makr/Documents/data/pedestrian_3d_own/1/object/ -3.14 0 150 test True 750

    try:
        live_mode = sys.argv[1]
        if live_mode == "live_mode_on":
            dataset_root_path = sys.argv[2]
            bb_rotation = 0.0
            start_idx = 1000
            end_idx = 6000
            train_or_test = "0"
            empty_annos = False
            image_counter_naming_start = None
            save_every_x_step = None
        if live_mode == "live_mode_off":
            dataset_root_path = sys.argv[2]
            bb_rotation = float(sys.argv[3])
            start_idx = int(sys.argv[4])
            end_idx = int(sys.argv[5])
            train_or_test = sys.argv[6]
            empty_annos = bool(sys.argv[7])
            save_every_x_step = int(sys.argv[8])
        listener = listener(dataset_root_path=dataset_root_path, 
                            bb_rotation=bb_rotation, 
                            start_idx=start_idx, 
                            end_idx=end_idx,
                            train_or_test=train_or_test,
                            empty_annos=empty_annos,
                            save_every_x_step=save_every_x_step)
        listener.run(live_mode=live_mode)
   
    except rospy.ROSInterruptException:
        pass
