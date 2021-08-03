#!/usr/bin/env python3
# coding: latin-1

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud, Image
from geometry_msgs.msg import Point32
import std_msgs.msg
from jsk_rviz_plugins.msg import *
from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_recognition_msgs.msg import BoundingBox
from scipy.spatial.transform import Rotation as R

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


##### 3DOD
import pickle
import numpy as np
import math
import cv2
##### 3DOD

from cv_bridge import CvBridge, CvBridgeError



#  ███████╗    ██╗   ██╗    ███╗   ██╗     ██████╗    ████████╗    ██╗     ██████╗     ███╗   ██╗    ███████╗
#  ██╔════╝    ██║   ██║    ████╗  ██║    ██╔════╝    ╚══██╔══╝    ██║    ██╔═══██╗    ████╗  ██║    ██╔════╝
#  █████╗      ██║   ██║    ██╔██╗ ██║    ██║            ██║       ██║    ██║   ██║    ██╔██╗ ██║    ███████╗
#  ██╔══╝      ██║   ██║    ██║╚██╗██║    ██║            ██║       ██║    ██║   ██║    ██║╚██╗██║    ╚════██║
#  ██║         ╚██████╔╝    ██║ ╚████║    ╚██████╗       ██║       ██║    ╚██████╔╝    ██║ ╚████║    ███████║
#  ╚═╝          ╚═════╝     ╚═╝  ╚═══╝     ╚═════╝       ╚═╝       ╚═╝     ╚═════╝     ╚═╝  ╚═══╝    ╚══════╝


# sends a list of 3D boxes to a ros publisher 
# =========================================
def send_3d_bbox(centers, dims, angles, publisher, header):
    """
    param center: list of x,y,z
    dims: list of h, w, l
    angles: list of angles
    publisher: ros publisher
    header: ros header
    """

    # ------------------------------------------------------------------------------------------------------
    # set variables
    # ------------------------------------------------------------------------------------------------------

    box_arr = BoundingBoxArray()
    box_arr.header = header

    # ------------------------------------------------------------------------------------------------------
    # iterate over boxes
    # ------------------------------------------------------------------------------------------------------

    for i in range(len(centers)):

        center = centers[i]
        dim = dims[i]
        angle = angles[i]

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

# removes annos with low score
# =========================================
def remove_low_score(image_anno, thresh):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, s in enumerate(image_anno['score']) if s >= thresh
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations

# get the corners of a 3D bounding box
# =========================================
def kitti_anno_to_corners(info, annos=None):
    rect = info['R0_rect']
    Tr_velo_to_cam = info['Tr_velo_to_cam']

    if annos is None:
        annos = info['annos']
    dims = annos['dimensions']
    loc = annos['location']
    rots = annos['rotation_y']
    scores = None
    if 'score' in annos:
        scores = annos['score']
    boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    boxes_lidar = box_camera_to_lidar(boxes_camera, rect,
                                                 Tr_velo_to_cam)

    return (boxes_lidar[:, :3], boxes_lidar[:, 3:6], boxes_lidar[:, 6])       



    boxes_corners = center_to_corner_box3d(
        boxes_lidar[:, :3],
        boxes_lidar[:, 3:6],
        boxes_lidar[:, 6],
        origin=[0.5, 0.5, 0.5],
        axis=2)
    return boxes_corners, scores, boxes_lidar

# transforms the center style 3D bounding box to corner coords
# =========================================
def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=[0.5, 1.0, 0.5],
                           axis=1):
    """convert kitti locations, dimensions and angles to corners
    
    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners

# convert camera to lidar coords
# =========================================
def box_camera_to_lidar(data, r_rect, velo2cam):
    xyz = data[:, 0:3]
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)

# convert camera to lidar coords
# =========================================
def camera_to_lidar(points, r_rect, velo2cam):
    points_shape = list(points.shape[0:-1])
    # creating homogeneous coords
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)

    r_rect_hom = np.eye(4)
    r_rect_hom[0:3, 0:3] = r_rect

    velo2cam_hom = np.eye(4)
    velo2cam_hom[0:3, :] = velo2cam
    # invert rectification and velo2cam and apply it on points
    lidar_points = np.dot(points, np.linalg.inv(np.dot(r_rect_hom,velo2cam_hom).T))
    # return only not homo coords
    return lidar_points[..., :3]



#  ███╗   ███╗     █████╗     ██╗    ███╗   ██╗
#  ████╗ ████║    ██╔══██╗    ██║    ████╗  ██║
#  ██╔████╔██║    ███████║    ██║    ██╔██╗ ██║
#  ██║╚██╔╝██║    ██╔══██║    ██║    ██║╚██╗██║
#  ██║ ╚═╝ ██║    ██║  ██║    ██║    ██║ ╚████║
#  ╚═╝     ╚═╝    ╚═╝  ╚═╝    ╚═╝    ╚═╝  ╚═══╝
                                        

# This function laods pointclouds, their ground truth and predictions(annos) and sends them to rviz
# =========================================
def sendToRVIZ():

    # ------------------------------------------------------------------------------------------------------
    # set variables
    # ------------------------------------------------------------------------------------------------------
    
    point_cloud_pub = rospy.Publisher('point_cloud', PointCloud2)
    location_debug_pub = rospy.Publisher('location_debug', PointCloud)
    bbox_debug_pub = rospy.Publisher('bbox_debug', PointCloud)
    bb_ground_truth_pub = rospy.Publisher("bb_ground_truth", BoundingBoxArray)
    bb_pred_guess_1_pub = rospy.Publisher("bb_pred_guess_1", BoundingBoxArray)
    bb_pred_guess_2_pub = rospy.Publisher("bb_pred_guess_2", BoundingBoxArray)
    pub_image = rospy.Publisher('image', Image)
    rospy.init_node('talker', anonymous=True)
    #giving some time for the publisher to register
    rospy.sleep(0.1)
    counter = 0
    counter_ini = counter
    prediction_min_score = 0.3

    # ------------------------------------------------------------------------------------------------------
    # set parameter
    # ------------------------------------------------------------------------------------------------------

    show_annotations = True
    show_ground_truth = True
    mode = "testing_sampled" # Options: live, training_unsampled, testing_sampled, testing_unsampled, gt_database_val, gt_database
    # training_sampled is not here since the sampling is done during training, to see that you have to use the 
    # "save" param in "load_data.py" and "send_to_rviz.py"

    # ------------------------------------------------------------------------------------------------------
    # Load Pointcloud ground truths
    # ------------------------------------------------------------------------------------------------------

    if mode == "testing_unsampled":
        # load gt
        with open("/home/makr/Documents/data/pedestrian_3d_own/1/object/kitti_infos_val.pkl", "rb") as file: 
            kitti_infos = pickle.load(file)
        # load annos
        
        # load point clouds
        point_cloud_path = "/home/makr/Documents/data/pedestrian_3d_own/1/object/testing/velodyne" # TESTING SAMPLED/UNSAMPLED DATA (depending on what you copy into the folder)
        show_annotations = False
    if mode == "testing_sampled":
        # load gt
        with open("/home/makr/Documents/data/pedestrian_3d_own/1/object/kitti_infos_val_sampled.pkl", "rb") as file: 
            kitti_infos = pickle.load(file)
        # load annos
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_377/out_dir_eval_results/result_epoch_22.pkl", "rb") as file: 
        #    thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_399/out_dir_eval_results/result_epoch_8.pkl", "rb") as file: 
        #    thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_411/out_dir_eval_results/result_epoch_16.pkl", "rb") as file: 
        #    thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_411/out_dir_eval_results/result.pkl", "rb") as file: 
        #    thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_496/out_dir_eval_results/result.pkl", "rb") as file: 
        #    thesis_eval_dict = pickle.load(file)
        with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_6/out_dir_eval_results/result.pkl", "rb") as file: 
           thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_27/out_dir_eval_results/result_epoch_21.pkl", "rb") as file: 
        #    thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_29/out_dir_eval_results/result_epoch_12.pkl", "rb") as file: 
        #    thesis_eval_dict = pickle.load(file)
        with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_79/out_dir_eval_results/result_epoch_0.pkl", "rb") as file: 
           thesis_eval_dict = pickle.load(file)
        with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_103/out_dir_eval_results/result_epoch_0.pkl", "rb") as file: 
           thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_105/out_dir_eval_results/result_epoch_0.pkl", "rb") as file: 
        #    thesis_eval_dict = pickle.load(file)
        with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_108/out_dir_eval_results/result_epoch_3.pkl", "rb") as file: 
           thesis_eval_dict = pickle.load(file)
        # load point clouds
        point_cloud_path = "/home/makr/Documents/data/pedestrian_3d_own/1/object/testing/velodyne" # TESTING SAMPLED/UNSAMPLED DATA (depending on what you copy into the folder)
        show_annotations = True
        show_ground_truth = True
    if mode == "training_unsampled":
        # load gt
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_136/out_dir_eval_results/result_epoch_4.pkl", "rb") as file: 
        #     thesis_eval_dict = pickle.load(file)
        with open("/home/makr/Documents/data/pedestrian_3d_own/1/object/kitti_infos_train.pkl", "rb") as file: 
            kitti_infos = pickle.load(file)
        # load annos
        
        # load point clouds
        point_cloud_path = "/home/makr/Documents/data/pedestrian_3d_own/1/object/training/velodyne"
        show_annotations = False
        show_ground_truth = True
    if mode == "live":
        # load gt
        # load annos
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_127/out_dir_eval_results/result_epoch_15_velodyne_2.pkl", "rb") as file: 
        #     thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_368/out_dir_eval_results/result_e22_ve2.pkl", "rb") as file: 
        #     thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_377/out_dir_eval_results/result_e33_ve1.pkl", "rb") as file:
        #     thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_400/out_dir_eval_results/result_epoch_0.pkl", "rb") as file:
        #     thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_411/out_dir_eval_results/result_e16_ve2.pkl", "rb") as file:
        #     thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_448/out_dir_eval_results/result_epoch_9.pkl", "rb") as file:
        #     thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_465/out_dir_eval_results/result.pkl", "rb") as file:
        #     thesis_eval_dict = pickle.load(file)
        #with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_488/out_dir_eval_results/result_epoch_40.pkl", "rb") as file:
        #    thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_496/out_dir_eval_results/result_epoch_80.pkl", "rb") as file:
        #     thesis_eval_dict = pickle.load(file)
        # gut sind: 21,40(0.6),42(0.6)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_496/out_dir_eval_results/result_e21_ve5_2.pkl", "rb") as file:
        #     thesis_eval_dict = pickle.load(file)
        with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_496/out_dir_eval_results/result.pkl", "rb") as file:
            thesis_eval_dict = pickle.load(file)
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_496/out_dir_eval_results/result_e21_ve5_2.pkl", "rb") as file:
        #     thesis_eval_dict = pickle.load(file)
        # # load point clouds
        point_cloud_path = "/home/makr/Documents/data/pedestrian_3d_own/1/object/testing_live/velodyne"
        show_ground_truth = False # to false since we dont have annos for live data
    if mode == "gt_database_val":
        # load gt

        # load annosz

        # load point clouds
        point_cloud_path = "/home/makr/Documents/data/pedestrian_3d_own/1/object/gt_database_val"
        show_annotations = False
        show_ground_truth = False


    if mode == "gt_database":
        
        # load gt

        # load annos

        # load point clouds
        point_cloud_path = "/home/makr/Desktop/temp/pedestrian_3d_own/1/object/gt_database"
        show_annotations = False
        show_ground_truth = False

    # ------------------------------------------------------------------------------------------------------
    # load pointclouds from folder and order numerically
    # ------------------------------------------------------------------------------------------------------

    import os
    velodyne_data = []
    for root, dirs, files in os.walk(point_cloud_path, topdown=False):
        files = [x[:-4] for x in files] # cut the .png
        for name in sorted(files): # order the list // !!!!! every 10s elements is out of order since 
            name = name + ".pkl" # append .png again
            file = os.path.join(root, name)
            with open(file, 'rb') as file:
                velodyne_data.append(pickle.load(file, encoding="latin1"))

    # ------------------------------------------------------------------------------------------------------
    # iterate over the pointcloud files
    # ------------------------------------------------------------------------------------------------------

    
    print(counter)
    assert(len(velodyne_data)>0)
    while not rospy.is_shutdown() and counter < len(velodyne_data):

        rospy.sleep(0.0)

        try:
            input("Press Enter to continue...")
            
            print(counter)
            point_cloud = velodyne_data[counter]

            # ------------------------------------------------------------------------------------------------------
            # use our custom calibtrations
            # ------------------------------------------------------------------------------------------------------

            calib = {"R0_rect" : np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3,3),
                    "Tr_velo_to_cam" : np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0]).reshape(3,4)}
            
            
            # ------------------------------------------------------------------------------------------------------
            # Create and Publish Point Cloud
            # ------------------------------------------------------------------------------------------------------

            # declaring pointcloud
            my_awesome_pointcloud = PointCloud()
            # filling pointcloud header
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'camera_color_frame'
            my_awesome_pointcloud.header = header
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    ]
            # https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
            pc2 = point_cloud2.create_cloud(header, fields, point_cloud)
            point_cloud_pub.publish(pc2)

            # ------------------------------------------------------------------------------------------------------
            # Create and Publish Ground Truth Boxes 
            # ------------------------------------------------------------------------------------------------------

            if show_ground_truth:
                annotation_gt = kitti_infos[counter]
                centers_gt,dims_gt,angles_gt= kitti_anno_to_corners(calib, annotation_gt["annos"])
                # centers_gt = centers_gt * [1.0,-1.0,-1.0]
                # print(angles_gt)
                if len(centers_gt)>0:
                    centers_gt = centers_gt + [0.0,0.0,dims_gt[0][2]/2]
                
                # if angles_gt[0] >= 0:
                #     angles_gt = angles_gt - np.pi
                # else:
                #     angles_gt = angles_gt + np.pi
                send_3d_bbox(centers_gt, dims_gt, angles_gt, bb_ground_truth_pub,header)

            # ------------------------------------------------------------------------------------------------------
            # Create and Publish Annos
            # ------------------------------------------------------------------------------------------------------
# 247
            if show_annotations:
                annotations = thesis_eval_dict[counter]
                detection_anno = remove_low_score(annotations, prediction_min_score)
                if len(detection_anno["score"]) > 0:
                    print(detection_anno["score"])
                centers,dims,angles= kitti_anno_to_corners(calib, detection_anno) # [a,b,c] -> [c,a,b] (camera to lidar coords)
                #centers = centers + [0.0,0.0,1.9]   # Epoch 2
                #centers = centers + [0.0,0.0,2.3]      #Epoch 3            
                #centers = centers + [0.0,0.0,2.4] # live
                centers = centers + [0.0,0.0,1.0] # live # TODO: müsste eigentlich um höhe/2 angehoben werden weil Pointpillars die z position am boden angibt der bbox und rviz die z mitte der höhe der bbox
                send_3d_bbox(centers, dims, angles, bb_pred_guess_1_pub, header)      

            if(counter%100==0):
                print(counter)

            counter = counter +1
            if counter >= len(kitti_infos)-1:
                counter = 0
            
        except IOError as e:
            pass

if __name__ == '__main__':
    try:
        sendToRVIZ()
    except rospy.ROSInterruptException:
        pass
