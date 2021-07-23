#import from general python libraries
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import math
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os
import datetime
import cv2
import fire
import shutil
import yaml
import json
import numba
from numba import cuda

# ros
import rospy
import std_msgs
from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_recognition_msgs.msg import BoundingBox
from scipy.spatial.transform import Rotation as R


from libraries.metrics import update_metrics, Accuracy, PrecisionRecall, Scalar
import second.data.kitti_common as kitti


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

# Convert Network Predictions to Kitti Annotation style.
# Also filters annos outside the image or pc range
# =========================================
def predict_kitti_to_anno(example,
                          class_names,
                          predictions_dicts,
                          center_limit_range,
                          lidar_input,
                          global_set=None):
    
    '''
    Args:
        example: [0:voxels, 1:num_points, 2:coordinates, 3:rect, 4:Trv2c, 5:P2, 6:anchors, 7:anchors_mask, 8:image_idx, 9:image_shape]

    Returns:
        annos:  List of Dictionieries with the keys: 
                [name, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score]
                - each Dictioniery represents one pointcloud in the batch
                - each keyvalue contains a list of n prediction
                - coordinates are in Image coords and NOT Lidar (Kitti style)
    '''

    # ------------------------------------------------------------------------------------------------------
    # set params
    # ------------------------------------------------------------------------------------------------------
    
    batch_image_shape = example[9]

    # ------------------------------------------------------------------------------------------------------
    # Create placeholder for returns
    # ------------------------------------------------------------------------------------------------------

    annos = []

    # ------------------------------------------------------------------------------------------------------
    # Iterate over each element in batch
    # ------------------------------------------------------------------------------------------------------

    for i, preds_dict in enumerate(predictions_dicts):

        image_shape = batch_image_shape[i]
        batch_idx = preds_dict["batch_idx"]

        if preds_dict["box3d_camera"] is not None:

            # ------------------------------------------------------------------------------------------------------
            # convert to numpy 
            # ------------------------------------------------------------------------------------------------------

            box_preds_2d = preds_dict["bbox"]
            box_preds = preds_dict["box3d_camera"]
            scores = preds_dict["scores"]
            box_preds_lidar = preds_dict["box3d_lidar"]
            label_preds = preds_dict["label_preds"]

            # ------------------------------------------------------------------------------------------------------
            # Create placeholder for anno
            # ------------------------------------------------------------------------------------------------------

            anno = get_start_result_anno()

            # t_full_sample: 86.0, t_preprocess: 0.0, t_network: 13.0, t_predict: 38.0, t_anno: 35.0, t_rviz: 0.0
            # t_full_sample: 79.0, t_preprocess: 0.0, t_network: 13.0, t_predict: 50.0, t_anno: 15.0, t_rviz: 0.0
            # t_full_sample: 62.0, t_preprocess: 1.0, t_network: 13.0, t_predict: 39.0, t_anno: 10.0, t_rviz: 0.0
            # t_full_sample: 53.84, t_preprocess: 0.47, t_network: 12.0, t_predict: 34.86, t_anno: 6.88, t_rviz: 0.02
            # t_full_sample: 43.23, t_preprocess: 0.58, t_network: 10.5, t_predict: 31.84, t_anno: 0.8, t_rviz: 0.0
            
            num_example = 0
            for box_2d, box, box_lidar, score, label in zip(
                    box_preds_2d, box_preds, box_preds_lidar, scores,
                    label_preds):
                
                # ------------------------------------------------------------------------------------------------------
                # If Bounding Box outside the pc range, break the for loop
                # ------------------------------------------------------------------------------------------------------

                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue


                # ------------------------------------------------------------------------------------------------------
                # Fill up the anno placeholder
                # ------------------------------------------------------------------------------------------------------
                
                anno["name"].append(class_names[int(label)]) 
                anno["bbox"].append(box_2d) 
                anno["truncated"].append(0.0) # Fill up with 0 since the prediction does not know if its truncated
                anno["occluded"].append(0) # Fill up with 0 since the prediction does not know if its occluded
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box[6]) # Dont know whats going on with the alpha here
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])

                if global_set is not None: # (here False)
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1

            # ------------------------------------------------------------------------------------------------------
            # If there are annos left, stack them and append to the return value
            # ------------------------------------------------------------------------------------------------------

            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["batch_idx"] = np.array(
            [batch_idx] * num_example, dtype=np.int64)

    # ------------------------------------------------------------------------------------------------------
    # return annos
    # ------------------------------------------------------------------------------------------------------
    return annos



def get_start_result_anno():
    annotations = {}
    annotations.update({
        'bbox': [],
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': [],
    })
    return annotations

def empty_result_anno():
    annotations = {}
    annotations.update({
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
        'score': np.array([]),
    })
    return annotations

# convers network outputs for location to lidar coords
# =========================================
#@tf.function #// seems to be slower :()
# def second_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
#     """box decode for VoxelNet in lidar
#     Args:
#         boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
#         anchors ([N, 7] Tensor): anchors
#     """

#     # ------------------------------------------------------------------------------------------------------
#     # split all 7 location features of predictions and anchors
#     # a = anchors
#     # t = predictions
#     # ------------------------------------------------------------------------------------------------------

#     # need to convert box_encodings to z-bottom format
#     xa, ya, za, wa, la, ha, ra = tf.split(anchors, 7, axis=-1)
#     if encode_angle_to_vector:
#         xt, yt, zt, wt, lt, ht, rtx, rty = tf.split(
#             box_encodings, 1, axis=-1)
#     else:
#         xt, yt, zt, wt, lt, ht, rt = tf.split(box_encodings, 7, axis=-1)

#     # ------------------------------------------------------------------------------------------------------
#     # Elevate anchors by its height/2 to to get to the middle of the 3D anchor
#     # ------------------------------------------------------------------------------------------------------

#     za = za + ha / 2

#     # ------------------------------------------------------------------------------------------------------
#     # Get the diagonal lenth of the anchors
#     # ------------------------------------------------------------------------------------------------------

#     diagonal = tf.math.sqrt(la**2 + wa**2) 

#     # ------------------------------------------------------------------------------------------------------
#     # Multiply x,y coords of preds with the diagonal lenth to denormalize them 
#     # and add the anchors postitions (lidar world) to get the lidar world coords of the preds
#     # ------------------------------------------------------------------------------------------------------

#     xg = xt * diagonal + xa
#     yg = yt * diagonal + ya


#     # ------------------------------------------------------------------------------------------------------
#     # Multiply z coords of preds with the anchor height to denormalize them 
#     # and add the anchors postitions (lidar world) to get the lidar world coords of the preds
#     # ------------------------------------------------------------------------------------------------------

#     zg = zt * ha + za

#     if smooth_dim: # (false here)
#         lg = (lt + 1) * la
#         wg = (wt + 1) * wa
#         hg = (ht + 1) * ha
#     else:

#         # ------------------------------------------------------------------------------------------------------
#         # Exp(w,l,h) and multi with anchors w,l,h to denormalize them 
#         # ------------------------------------------------------------------------------------------------------
        
#         lg = tf.math.exp(lt) * la
#         wg = tf.math.exp(wt) * wa
#         hg = tf.math.exp(ht) * ha
#     if encode_angle_to_vector: # (false here)
#         rax = tf.math.cos(ra)
#         ray = tf.math.sin(ra)
#         rgx = rtx + rax
#         rgy = rty + ray 
#         rg = tf.math.atan2(rgy, rgx)
#     else:

#         # ------------------------------------------------------------------------------------------------------
#         # Add the angle offset {0,pi} of the anchors to the angle predictions
#         # ------------------------------------------------------------------------------------------------------

#         rg = rt + ra
#     zg = zg - hg / 2
#     return tf.concat([xg, yg, zg, wg, lg, hg, rg], axis=-1)


def second_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """

    # ------------------------------------------------------------------------------------------------------
    # split all 7 location features of predictions and anchors
    # a = anchors
    # t = predictions
    # ------------------------------------------------------------------------------------------------------
    
    # need to convert box_encodings to z-bottom format
    xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
    if encode_angle_to_vector:
        xt, yt, zt, wt, lt, ht, rtx, rty = np.split(box_encodings, 8, axis=-1)
    else:
        xt, yt, zt, wt, lt, ht, rt = np.split(box_encodings, 7, axis=-1)
    
    # ------------------------------------------------------------------------------------------------------
    # Elevate anchors by its height/2 to to get to the middle of the 3D anchor
    # ------------------------------------------------------------------------------------------------------

    za = za + ha / 2

    # ------------------------------------------------------------------------------------------------------
    # Get the diagonal lenth of the anchors
    # ------------------------------------------------------------------------------------------------------

    diagonal = np.sqrt(la**2 + wa**2)

    # ------------------------------------------------------------------------------------------------------
    # Multiply x,y coords of preds with the diagonal lenth to denormalize them 
    # and add the anchors postitions (lidar world) to get the lidar world coords of the preds
    # ------------------------------------------------------------------------------------------------------

    xg = xt * diagonal + xa
    yg = yt * diagonal + ya

    # ------------------------------------------------------------------------------------------------------
    # Multiply z coords of preds with the anchor height to denormalize them 
    # and add the anchors postitions (lidar world) to get the lidar world coords of the preds
    # ------------------------------------------------------------------------------------------------------

    zg = zt * ha + za
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:

        # ------------------------------------------------------------------------------------------------------
        # Exp(w,l,h) and multi with anchors w,l,h to denormalize them 
        # ------------------------------------------------------------------------------------------------------

        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha
    if encode_angle_to_vector:
        rax = np.cos(ra)
        ray = np.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = np.arctan2(rgy, rgx)
    else:

        # ------------------------------------------------------------------------------------------------------
        # Add the angle offset {0,pi} of the anchors to the angle predictions
        # ------------------------------------------------------------------------------------------------------

        rg = rt + ra
    zg = zg - hg / 2
    return np.concatenate([xg, yg, zg, wg, lg, hg, rg], axis=-1)

def nms(bboxes,
        scores,
        pre_max_size=None,
        post_max_size=None,
        iou_threshold=0.5):

    # ------------------------------------------------------------------------------------------------------
    # keep only pre_max_size bb predcition with the highest scores
    # ------------------------------------------------------------------------------------------------------

    if pre_max_size is not None: #(here True)
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        indices = np.argpartition(scores, -pre_max_size)[-pre_max_size:]
        scores = scores[[indices]]
        bboxes = bboxes[[indices]]

    dets = np.concatenate([bboxes, np.expand_dims(scores,axis=-1)], axis=1)

    if len(dets) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(nms_gpu(dets, iou_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return None
    if pre_max_size is not None:
        return indices[[keep]]
    else:
        return keep

def nms_gpu(dets, nms_overlap_thresh, device_id=0):
    """nms in gpu. 
    
    Args:
        dets ([type]): [description]
        nms_overlap_thresh ([type]): [description]
        device_id ([type], optional): Defaults to 0. [description]
    
    Returns:
        [type]: [description]
    """

    boxes_num = dets.shape[0]
    keep_out = np.zeros([boxes_num], dtype=np.int32)
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)
    boxes_host = dets[order, :]

    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    mask_host = np.zeros((boxes_num * col_blocks, ), dtype=np.uint64)
    blockspergrid = (div_up(boxes_num, threadsPerBlock),
                     div_up(boxes_num, threadsPerBlock))
    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes_host.reshape([-1]), stream)
        mask_dev = cuda.to_device(mask_host, stream)
        nms_kernel[blockspergrid, threadsPerBlock, stream](
            boxes_num, nms_overlap_thresh, boxes_dev, mask_dev)
        mask_dev.copy_to_host(mask_host, stream=stream)
    # stream.synchronize()
    num_out = nms_postprocess(keep_out, mask_host, boxes_num)
    keep = keep_out[:num_out]
    return list(order[keep])

@numba.jit(nopython=True)
def nms_postprocess(keep_out, mask_host, boxes_num):
    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    remv = np.zeros((col_blocks), dtype=np.uint64)
    num_to_keep = 0
    for i in range(boxes_num):
        nblock = i // threadsPerBlock
        inblock = i % threadsPerBlock
        mask = np.array(1 << inblock, dtype=np.uint64)
        if not (remv[nblock] & mask):
            keep_out[num_to_keep] = i
            num_to_keep += 1
            # unsigned long long *p = &mask_host[0] + i * col_blocks;
            for j in range(nblock, col_blocks):
                remv[j] |= mask_host[i * col_blocks + j]
                # remv[j] |= p[j];
    return num_to_keep

@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def iou_device(a, b):
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    width = max(right - left + 1, 0.)
    height = max(bottom - top + 1, 0.)
    interS = width * height
    Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return interS / (Sa + Sb - interS)

# Matthes GPU
@cuda.jit('(int64, float32, float32[:], uint64[:])')
def nms_kernel(n_boxes, nms_overlap_thresh, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if (tx < col_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if (tx < row_size):
        cur_box_idx = threadsPerBlock * row_start + tx
        # cur_box = dev_boxes + cur_box_idx * 5;
        t = 0
        start = 0
        if (row_start == col_start):
            start = tx + 1
        for i in range(start, col_size):
            iou = iou_device(dev_boxes[cur_box_idx * 5:cur_box_idx * 5 + 4],
                             block_boxes[i * 5:i * 5 + 4])
            if (iou > nms_overlap_thresh):
                t |= 1 << i
        col_blocks = ((n_boxes) // (threadsPerBlock) + (
            (n_boxes) % (threadsPerBlock) > 0))
        dev_mask[cur_box_idx * col_blocks + col_start] = t



#@tf.function
#@tf.function(input_signature = [tf.TensorSpec(shape=[None,4,2], dtype=tf.float32)])
# def corner_to_standup_nd(boxes_corner):
#     ndim = boxes_corner.shape[2]
#     standup_boxes = []
#     for i in range(ndim):
#         standup_boxes.append(tf.math.reduce_min(boxes_corner[:, :, i], axis=1))
#     for i in range(ndim):
#         standup_boxes.append(tf.math.reduce_max(boxes_corner[:, :, i], axis=1))
#     return tf.stack(standup_boxes, axis=1)

@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result

def tf_to_np_dtype(ttype):
    type_map = {
        tf.float16: np.dtype(np.float16),
        tf.float32: np.dtype(np.float32),
        tf.float16: np.dtype(np.float64),
        tf.int32: np.dtype(np.int32),
        tf.int64: np.dtype(np.int64),
        tf.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners
    
    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
    
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += tf.reshape(centers,(-1, 1, 2))
    return corners

#@tf.function
@tf.function(input_signature = [tf.TensorSpec(shape=[None,4,2], dtype=tf.float32),tf.TensorSpec(shape=[None,], dtype=tf.float32)])
def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.
    
    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = tf.math.sin(angles)
    rot_cos = tf.math.cos(angles)
    rot_mat_T = tf.stack(
        [ tf.stack([rot_cos, -rot_sin]),
          tf.stack([rot_sin, rot_cos])])
    return tf.einsum('aij,jka->aik', points, rot_mat_T)

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 
    
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32 
    
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = tf_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = tf.convert_to_tensor(corners_norm, dtype=dims.dtype)
    corners = tf.reshape(dims,(-1, 1, ndim)) * tf.reshape(corners_norm,(1, 2**ndim, ndim))
    return corners

# @tf.function(input_signature = [tf.TensorSpec(shape=[None,7], dtype=tf.float32),tf.TensorSpec(shape=[4,4], dtype=tf.float32),tf.TensorSpec(shape=[4,4], dtype=tf.float32)])
# def box_lidar_to_camera(data, r_rect, velo2cam):
#     xyz_lidar = data[..., 0:3]
#     w, l, h = data[..., 3:4], data[..., 4:5], data[..., 5:6]
#     r = data[..., 6:7]
#     xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
#     return tf.concat([xyz, l, h, w, r], axis=-1)

# def lidar_to_camera(points, r_rect, velo2cam):
#     num_points = tf.shape(points)[0]
#     points = tf.concat(
#         [points, tf.ones((num_points, 1),dtype=points.dtype)], axis=-1)
#     camera_points = points @ tf.transpose((r_rect @ velo2cam))
#     return camera_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]

def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[:, 0:3]
    w, l, h = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return np.concatenate([xyz, l, h, w, r], axis=1)

# def center_to_corner_box3d(centers,
#                            dims,
#                            angles,
#                            origin=[0.5, 1.0, 0.5],
#                            axis=1):
#     """convert kitti locations, dimensions and angles to corners
    
#     Args:
#         centers (float array, shape=[N, 3]): locations in kitti label file.
#         dims (float array, shape=[N, 3]): dimensions in kitti label file.
#         angles (float array, shape=[N]): rotation_y in kitti label file.
#         origin (list or array or float): origin point relate to smallest point.
#             use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
#         axis (int): rotation axis. 1 for camera and 2 for lidar.
#     Returns:
#         [type]: [description]
#     """
#     # 'length' in kitti format is in x axis.
#     # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
#     # center in kitti format is [0.5, 1.0, 0.5] in xyz.
#     corners = corners_nd(dims, origin=origin)
#     # corners: [N, 8, 3]
#     corners = rotation_3d_in_axis(corners, angles, axis=axis)
#     corners += tf.reshape(centers,(-1, 1, 3))
#     return corners

def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    # angles: [N]
    rot_sin = tf.math.sin(angles)
    rot_cos = tf.math.cos(angles)
    ones = tf.ones_like(rot_cos)
    zeros = tf.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = tf.stack([
            tf.stack([rot_cos, zeros, -rot_sin]),
            tf.stack([zeros, ones, zeros]),
            tf.stack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = tf.stack([
            tf.stack([rot_cos, -rot_sin, zeros]),
            tf.stack([rot_sin, rot_cos, zeros]),
            tf.stack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = tf.stack([
            tf.stack([zeros, rot_cos, -rot_sin]),
            tf.stack([zeros, rot_sin, rot_cos]),
            tf.stack([ones, zeros, zeros])
        ])
    else:
        raise ValueError("axis should in range")

    return tf.einsum('aij,jka->aik', points, rot_mat_T)

# def project_to_image(points_3d, proj_mat):
#     points_num = list(points_3d.shape)[:-1]
#     points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
#     points_4 = tf.concat(
#         [points_3d, tf.zeros((points_shape[0],points_shape[1],points_shape[2]),points_3d.dtype)], axis=-1)
#     # point_2d = points_4 @ tf.transpose(proj_mat, [1, 0])
#     point_2d = tf.matmul(points_4, tf.transpose(proj_mat))
#     point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
#     return point_2d_res


def project_to_image(points_3d, proj_mat):
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.zeros(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')