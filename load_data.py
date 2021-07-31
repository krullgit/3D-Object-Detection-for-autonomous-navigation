import tensorflow as tf
#import IPython.display as display
#from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
import os
import pathlib
import pickle
import math
import cv2
import copy
import numba
import numpy.random as npr
print(tf.__version__)
tf.compat.v1.enable_eager_execution()
from collections import defaultdict
import json
from json import JSONEncoder
import random
import os, os.path
import yaml
import sys
import time

# ros
import ros_numpy
import rospy
from sensor_msgs.msg import PointCloud2 as msg_PointCloud2
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from sensor_msgs import point_cloud2
from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_recognition_msgs.msg import BoundingBox

from libraries.eval_helper_functions import send_3d_bbox
####
# TargetAssigner
####

def filter_gt_box_outside_range_by_center(gt_boxes, limit_range):
    """remove gtbox outside training range.
    this function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_box_centers = gt_boxes[:, :2]
    bounding_box = minmax_to_corner_2d(
        np.asarray(limit_range)[np.newaxis, ...])
    ret = points_in_convex_polygon_jit(gt_box_centers, bounding_box)
    return ret.reshape(-1)

def points_in_rbbox(points, rbbox, lidar=True):
    if lidar:
        h_axis = 2
        origin = [0.5, 0.5, 0]
    else:
        origin = [0.5, 1.0, 0.5]
        h_axis = 1

    #origin = [0.0, 0.0, 0.0]
    origin = [0.5, 0.5, 0]
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=2)
    surfaces = corner_to_surfaces_3d_jit(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices

def second_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for VoxelNet
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, l, w, h, r
        anchors ([N, 7] Tensor): anchors
    """

    # ------------------------------------------------------------------------------------------------------
    # split all 7 location features of gt and anchors
    # a = anchors
    # t = predictions
    # ------------------------------------------------------------------------------------------------------

    xa, ya, za, wa, la, ha, ra = tf.split(tf.cast(anchors,tf.float32), 7, axis=-1)
    xg, yg, zg, wg, lg, hg, rg = tf.split(tf.cast(boxes,tf.float32), 7, axis=-1)

    # ------------------------------------------------------------------------------------------------------
    # Elevate anchors by its height/2 to to get to the middle of the 3D anchor
    # ------------------------------------------------------------------------------------------------------

    za = za + ha / 2

    # ------------------------------------------------------------------------------------------------------
    # Elevate gt by its height/2 to to get to the middle of the 3D gt
    # ------------------------------------------------------------------------------------------------------

    zg = zg + hg / 2

    # ------------------------------------------------------------------------------------------------------
    # Get the diagonal lenth of the anchors
    # ------------------------------------------------------------------------------------------------------

    diagonal = tf.math.sqrt(la**2 + wa**2)

    # ------------------------------------------------------------------------------------------------------
    # Subtract x,y anchors postitions (lidar world) from gt coords to get coords relativ the anchor center
    # and divide by diagonal to normalize z 
    # ------------------------------------------------------------------------------------------------------

    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal

    # ------------------------------------------------------------------------------------------------------
    # Subtract z anchors postitions (lidar world) from gt coords to get coords relativ the 3D anchor center
    # and divide anchro height to normalize z 
    # ------------------------------------------------------------------------------------------------------

    zt = (zg - za) / ha

    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        
        # ------------------------------------------------------------------------------------------------------
        # (1) divide w,l,h from gt with anchors w,l,h to normalize them and (2) Log it
        # to 1.: The division results that w,l,h will be around 1.0
        # to 2.: The Log effect w,l,h subsequently that <1.0 becomes below 0 and >1.0 becomes above 0
        # So in the result can be seen as a divergence from the norm in positve and negative direction
        # ------------------------------------------------------------------------------------------------------

        lt = tf.math.log(lg / la)
        wt = tf.math.log(wg / wa)
        ht = tf.math.log(hg / ha)   
    if encode_angle_to_vector:
        rgx = tf.math.cos(rg)
        rgy = tf.math.sin(rg)
        rax = tf.math.cos(ra)
        ray = tf.math.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return tf.concat([xt, yt, zt, wt, lt, ht, rtx, rty], dim=-1)
    else:
        rt = rg - ra
        return tf.concat([xt, yt, zt, wt, lt, ht, rt], axis=-1)

    # rt = rg - ra
    # return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)


@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=0.0):
    """calculate box iou. note that jit version runs 2x faster than cython in 
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps) *
                        (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def nearest_iou_similarity(boxes1, boxes2):
    # """Class to compute similarity based on the squared distance metric.

    # This class computes pairwise similarity between two BoxLists based on the
    # negative squared distance metric.
    # """
    # """Compute matrix of (negated) sq distances.

    # Args:
    #   boxlist1: BoxList holding N boxes.
    #   boxlist2: BoxList holding M boxes.

    # Returns:
    #   A tensor with shape [N, M] representing negated pairwise squared distance.
    # """
    boxes1_bv = rbbox2d_to_near_bbox(boxes1)
    boxes2_bv = rbbox2d_to_near_bbox(boxes2)
    ret = iou_jit(boxes1_bv, boxes2_bv, eps=0.0)
    return ret

def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
            return nearest_iou_similarity(
                anchors_rbv, gt_boxes_rbv)

def box_encoding_fn(boxes, anchors):
    return second_box_encode(boxes, anchors)

def assign(anchors,
               gt_boxes,
               anchors_mask,
               gt_classes,
               matched_thresholds,
               unmatched_thresholds,
               config_target_assigner):
        if anchors_mask is not None:
            prune_anchor_fn = lambda _: np.where(anchors_mask)[0]
        else:
            prune_anchor_fn = None

        sample_positive_fraction = config_target_assigner["sample_positive_fraction"]
        if sample_positive_fraction == "None": sample_positive_fraction = None
        rpn_batch_size = config_target_assigner["rpn_batch_size"]

        return create_target_np(
            anchors,
            gt_boxes,
            prune_anchor_fn=prune_anchor_fn,
            gt_classes=gt_classes,
            matched_threshold=matched_thresholds,
            unmatched_threshold=unmatched_thresholds,
            positive_fraction=sample_positive_fraction,
            rpn_batch_size=rpn_batch_size,
            norm_by_num_examples=False,
            box_code_size=7)

def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of
    size count)"""
    if count == len(inds):
        return data

    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of
    size count)"""
    if count == len(inds):
        return data

    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret



# Match all anchors with the gt_boxes
# =========================================
def create_target_np(all_anchors,
                     gt_boxes,
                     prune_anchor_fn,
                     gt_classes,
                     matched_threshold,
                     unmatched_threshold,
                     positive_fraction,
                     rpn_batch_size,
                     norm_by_num_examples,
                     box_code_size,
                     bbox_inside_weight=None):

    """Modified from FAIR detectron.
    Args:
        all_anchors: [num_of_anchors, box_ndim] float tensor.
        gt_boxes: [num_gt_boxes, box_ndim] float tensor.
        similarity_fn: a function, accept anchors and gt_boxes, return
            similarity matrix(such as IoU).
        box_encoding_fn: a function, accept gt_boxes and anchors, return
            box encodings(offsets).
        prune_anchor_fn: a function, accept anchors, return indices that
            indicate valid anchors.
        gt_classes: [num_gt_boxes] int tensor. indicate gt classes, must
            start with 1.
        matched_threshold: float, iou greater than matched_threshold will
            be treated as positives.
        unmatched_threshold: float, iou smaller than unmatched_threshold will
            be treated as negatives.
        bbox_inside_weight: unused
        positive_fraction: [0-1] float or None. if not None, we will try to
            keep ratio of pos/neg equal to positive_fraction when sample.
            if there is not enough positives, it fills the rest with negatives
        rpn_batch_size: int. sample_size
        norm_by_num_examples: bool. norm box_weight by number of examples, but
            I recommend to do this outside.
    Returns:
        labels, bbox_targets, bbox_outside_weights

        For each anchor we are interested in:
        labels: which label it has (class eg "pedestrian" OR -1 for "no class" most of the time),
        bbox_targets: what is the gt_box it is assigned to ([x, y, z, xdim, ydim, zdim, rad] OR 0) ,
        bbox_outside_weights: what is the weight for this anchor (1.0 for assigned or 0.0 for not assigned in this case)  
    """
    total_anchors = all_anchors.shape[0]

    # ------------------------------------------------------------------------------------------------------
    # filter the anchors with the mask
    # ------------------------------------------------------------------------------------------------------

    if prune_anchor_fn is not None: # here: lambda _: np.where(anchors_mask)[0]
        inds_inside = prune_anchor_fn(all_anchors)
        anchors = all_anchors[inds_inside, :]
        if not isinstance(matched_threshold, float):
            matched_threshold = matched_threshold[inds_inside]
        if not isinstance(unmatched_threshold, float):
            unmatched_threshold = unmatched_threshold[inds_inside]
    else:
        anchors = all_anchors
        inds_inside = None
    num_inside = len(inds_inside) if inds_inside is not None else total_anchors
    box_ndim = all_anchors.shape[1]
    # print('total_anchors: {}'.format(total_anchors))
    # print('inds_inside: {}'.format(num_inside))
    # print('anchors.shape: {}'.format(anchors.shape))
    # logger.debug('total_anchors: {}'.format(total_anchors))
    # logger.debug('inds_inside: {}'.format(num_inside))
    # logger.debug('anchors.shape: {}'.format(anchors.shape))
    if gt_classes is None:
        gt_classes = np.ones([gt_boxes.shape[0]], dtype=np.int32)
    # Compute anchor labels:
    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = np.empty((num_inside, ), dtype=np.int32)
    gt_ids = np.empty((num_inside, ), dtype=np.int32)
    labels.fill(-1)
    gt_ids.fill(-1)

    # ------------------------------------------------------------------------------------------------------
    # if we have gt_boxes & anchors for the current point cloud
    # ------------------------------------------------------------------------------------------------------

    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        # Compute overlaps between the anchors and the gt boxes overlaps
        anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside),
                                                anchor_to_gt_argmax]  #
        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[
            gt_to_anchor_argmax,
            np.arange(anchor_by_gt_overlap.shape[1])]
        # must remove gt which doesn't match any anchor.
        empty_gt_mask = gt_to_anchor_max == 0
        gt_to_anchor_max[empty_gt_mask] = -1
        # Find all anchors that share the max overlap amount
        # (this includes many ties)
        anchors_with_max_overlap = np.where(
            anchor_by_gt_overlap == gt_to_anchor_max)[0]
        # Fg label: for each gt use anchors with highest overlap
        # (including ties)
        gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
        labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        gt_ids[anchors_with_max_overlap] = gt_inds_force
        # Fg label: above threshold IOU
        pos_inds = anchor_to_gt_max >= matched_threshold
        gt_inds = anchor_to_gt_argmax[pos_inds]
        labels[pos_inds] = gt_classes[gt_inds]
        gt_ids[pos_inds] = gt_inds
        bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
    else:
        # labels[:] = 0
        bg_inds = np.arange(num_inside)
    fg_inds = np.where(labels > 0)[0]
    fg_max_overlap = None
    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        fg_max_overlap = anchor_to_gt_max[fg_inds]
    gt_pos_ids = gt_ids[fg_inds]
    # bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
    # bg_inds = np.where(labels == 0)[0]
    # subsample positive labels if we have too many
    if positive_fraction is not None:
        num_fg = int(positive_fraction * rpn_batch_size)
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1
            fg_inds = np.where(labels > 0)[0]

        # subsample negative labels if we have too many
        # (samples with replacement, but since the set of bg inds is large most
        # samples will not have repeats)
        num_bg = rpn_batch_size - np.sum(labels > 0)
        # print(num_fg, num_bg, len(bg_inds) )
        if len(bg_inds) > num_bg:
            enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
            labels[enable_inds] = 0
        bg_inds = np.where(labels == 0)[0]
    else:
        if len(gt_boxes) == 0 or anchors.shape[0] == 0:
            labels[:] = 0
        else:
            labels[bg_inds] = 0
            # re-enable anchors_with_max_overlap
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
    bbox_targets = np.zeros(
        (num_inside, box_code_size), dtype=all_anchors.dtype)
    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        # print(anchors[fg_inds, :].shape, gt_boxes[anchor_to_gt_argmax[fg_inds], :].shape)
        # bbox_targets[fg_inds, :] = box_encoding_fn(
        #     anchors[fg_inds, :], gt_boxes[anchor_to_gt_argmax[fg_inds], :])
        bbox_targets[fg_inds, :] = box_encoding_fn(
            gt_boxes[anchor_to_gt_argmax[fg_inds], :], anchors[fg_inds, :])
    # Bbox regression loss has the form:
    #   loss(x) = weight_outside * L(weight_inside * x)
    # Inside weights allow us to set zero loss on an element-wise basis
    # Bbox regression is only trained on positive examples so we set their
    # weights to 1.0 (or otherwise if config is different) and 0 otherwise
    # NOTE: we don't need bbox_inside_weights, remove it.
    # bbox_inside_weights = np.zeros((num_inside, box_ndim), dtype=np.float32)
    # bbox_inside_weights[labels == 1, :] = [1.0] * box_ndim

    # The bbox regression loss only averages by the number of images in the
    # mini-batch, whereas we need to average by the total number of example
    # anchors selected
    # Outside weights are used to scale each element-wise loss so the final
    # average over the mini-batch is correct
    # bbox_outside_weights = np.zeros((num_inside, box_ndim), dtype=np.float32)
    bbox_outside_weights = np.zeros((num_inside, ), dtype=all_anchors.dtype)
    # uniform weighting of examples (given non-uniform sampling)
    if norm_by_num_examples:
        num_examples = np.sum(labels >= 0)  # neg + pos
        num_examples = np.maximum(1.0, num_examples)
        bbox_outside_weights[labels > 0] = 1.0 / num_examples
    else:
        bbox_outside_weights[labels > 0] = 1.0
    # bbox_outside_weights[labels == 0, :] = 1.0 / num_examples

    # Map up to original set of anchors
    if inds_inside is not None:
        labels = unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        # bbox_inside_weights = unmap(
        #     bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = unmap(
            bbox_outside_weights, total_anchors, inds_inside, fill=0)
    # return labels, bbox_targets, bbox_outside_weights                             
    ret = {
        "labels": labels, # [int] for each anchors (the labels are mapped from the gt_boxes to the anchors) # NOTE that multiple anchors can map to the same gt_box
        "bbox_targets": bbox_targets, # [x, y, z, xdim, ydim, zdim, rad] for each anchors (most zero) # it describes the gt bb which an anchor is assigned to 
        "bbox_outside_weights": bbox_outside_weights, # [int] # 0 or 1 # one int for each anchors # however, this does not seem to be very important since each assigned anchor has a 1.0
        "assigned_anchors_overlap": fg_max_overlap, # [float] # between 0 and 1 # for each ASSIGNED ANCHOR, how big is the overlap with a gt_box
        "positive_gt_id": gt_pos_ids, # [int] # for each ASSIGNED ANCHOR, which original gt_box was assined [id from gt_boxes]
    }

    if inds_inside is not None:
        ret["assigned_anchors_inds"] = inds_inside[fg_inds] # [int] # for each ASSIGNED ANCHOR which anchor-index is assigned
    else:
        ret["assigned_anchors_inds"] = fg_inds
    return ret


def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes
    
def center_to_minmax_2d_0_5(centers, dims):
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)

def center_to_minmax_2d(centers, dims, origin=0.5):
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])

@numba.jit(nopython=True)
def fused_get_anchors_area(dense_map, anchors_bv, stride, offset,
                           grid_size):
    anchor_coor = np.zeros(anchors_bv.shape[1:], dtype=np.int32)
    grid_size_x = grid_size[0] - 1
    grid_size_y = grid_size[1] - 1
    N = anchors_bv.shape[0]
    ret = np.zeros((N), dtype=dense_map.dtype)
    for i in range(N):
        anchor_coor[0] = np.floor(
            (anchors_bv[i, 0] - offset[0]) / stride[0])
        anchor_coor[1] = np.floor(
            (anchors_bv[i, 1] - offset[1]) / stride[1])
        anchor_coor[2] = np.floor(
            (anchors_bv[i, 2] - offset[0]) / stride[0])
        anchor_coor[3] = np.floor(
            (anchors_bv[i, 3] - offset[1]) / stride[1])
        anchor_coor[0] = max(anchor_coor[0], 0)
        anchor_coor[1] = max(anchor_coor[1], 0)
        anchor_coor[2] = min(anchor_coor[2], grid_size_x)
        anchor_coor[3] = min(anchor_coor[3], grid_size_y)
        ID = dense_map[anchor_coor[3], anchor_coor[2]]
        IA = dense_map[anchor_coor[1], anchor_coor[0]]
        IB = dense_map[anchor_coor[3], anchor_coor[0]]
        IC = dense_map[anchor_coor[1], anchor_coor[2]]
        ret[i] = ID - IB - IC + IA
    return ret

@numba.jit(nopython=True)
def sparse_sum_for_anchors_mask(coors, shape):
    ret = np.zeros(shape, dtype=np.float32)
    for i in range(coors.shape[0]):
        ret[coors[i, 1], coors[i, 2]] += 1
    return ret

@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points,
                                    max_voxels):
    
    
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num

@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points,
                            max_voxels):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(points,
                     voxel_size,
                     coors_range,
                     max_points,
                     reverse_index,
                     max_voxels):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index: # here true
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)

    # ------------------------------------------------------------------------------------------------------
    # creating the voxels and assign points
    # this will be done on variable references given to the function (voxels,coors,num_points_per_voxel)
    # ------------------------------------------------------------------------------------------------------

    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    else:
        voxel_num = _points_to_voxel_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    # cut off the placeholder with the amount of data actually created
    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num] 
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    # voxels[:, :, -3:] = voxels[:, :, :3] - \
    #     voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)

    # ------------------------------------------------------------------------------------------------------
    # returns:
    # voxels: [M, max_points, ndim] [voxels, points in that voxel [here: 100], coords of those points in world coords (x,y,z)] / this is [P(pillars),N(points),D(dimensions) from the PointPillars Paper]
    # coords: [M, 3] [voxels, coords of voxel centers in feature_map_size corrdinates [1, 248, 296]]
    # num_points_per_voxel: [M] [number of point per voxel] question: why are in voxels always 100 points per voxel and here also sometime under 100?
    # answer: because it functions like a mask for "max_points" in "voxels" where only "num_points_per_voxel" are not zero
    # ------------------------------------------------------------------------------------------------------

    return voxels, coors, num_points_per_voxel

def rotation_points_single_angle(points, angle, axis=0):
    # points: [N, 3]
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
            dtype=points.dtype)
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
            dtype=points.dtype)
    elif axis == 0:
        rot_mat_T = np.array(
            [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
            dtype=points.dtype)
    else:
        raise ValueError("axis should in range")

    return points @ rot_mat_T

def global_rotation(gt_boxes, points, rotation=np.pi / 4):
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    points[:, :3] = rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2)
    gt_boxes[:, :3] = rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2)
    gt_boxes[:, 6] += noise_rotation
    return gt_boxes, points

def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

@numba.jit
def points_in_convex_polygon_jit(points, polygon, clockwise=True):
    """check points is in 2d convex polygons. True when point in polygon
    Args:
        points: [num_points, 2] array.
        polygon: [num_polygon, num_points_of_polygon, 2] array.
        clockwise: bool. indicate polygon is clockwise.
    Returns:
        [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_points_of_polygon = polygon.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon.shape[0]
    if clockwise:
        vec1 = polygon - polygon[:, [num_points_of_polygon - 1] +
                                 list(range(num_points_of_polygon - 1)), :]
    else:
        vec1 = polygon[:, [num_points_of_polygon - 1] +
                       list(range(num_points_of_polygon - 1)), :] - polygon
    # vec1: [num_polygon, num_points_of_polygon, 2]
    ret = np.zeros((num_points, num_polygons), dtype=np.bool_)
    success = True
    cross = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            success = True
            for k in range(num_points_of_polygon):
                cross = vec1[j, k, 1] * (polygon[j, k, 0] - points[i, 0])
                cross -= vec1[j, k, 0] * (polygon[j, k, 1] - points[i, 1])
                if cross >= 0:
                    success = False
                    break
            ret[i, j] = success
    return ret

def filter_gt_box_outside_range(gt_boxes, limit_range):
    """remove gtbox outside training range.
    this function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_boxes_bv = center_to_corner_box2d(
        gt_boxes[:, [0, 1]], gt_boxes[:, [3, 3 + 1]], gt_boxes[:, 6])
    bounding_box = minmax_to_corner_2d(
        np.asarray(limit_range)[np.newaxis, ...])
    ret = points_in_convex_polygon_jit(
        gt_boxes_bv.reshape(-1, 2), bounding_box)
    return np.any(ret.reshape(-1, 4), axis=1)

def minmax_to_corner_2d(minmax_box):
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box2d(center, dims, origin=0.0)

def global_translate(gt_boxes, points, noise_translate_std):
    """
    Apply global translation to gt_boxes and points.
    """

    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array([noise_translate_std, noise_translate_std, noise_translate_std])

    noise_translate = np.array([np.random.normal(0, noise_translate_std[0], 1),
                                np.random.normal(0, noise_translate_std[1], 1),
                                np.random.normal(0, noise_translate_std[0], 1)]).T

    points[:, :3] += noise_translate
    gt_boxes[:, :3] += noise_translate

    return gt_boxes, points


def global_scaling_v2(gt_boxes, points, min_scale=0.95, max_scale=1.05):
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def random_flip(gt_boxes, points, probability=0.5):
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        
        # gt_boxes_temp = np.zeros_like(gt_boxes)
        # gt_boxes_temp[gt_boxes[:, 6] <= 0] += [0.0,0.0,0.0,0.0,0.0,0.0,np.pi]
        # gt_boxes_temp[gt_boxes[:, 6] > 0] += [0.0,0.0,0.0,0.0,0.0,0.0,-np.pi]
        # gt_boxes += gt_boxes_temp
        
        #gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi
        
        gt_boxes[:, 6] = gt_boxes[:, 6] * -1
        
        points[:, 1] = -points[:, 1]
    return gt_boxes, points






def noise_per_object_v3_(gt_boxes,
                         points=None,
                         valid_mask=None,
                         rotation_perturb=np.pi / 4,
                         center_noise_std=1.0,
                         global_random_rot_range=np.pi / 4,
                         num_try=100,
                         group_ids=None):
    """random rotate or remove each groundtrutn independently.
    use kitti viewer to test this function points_transform_

    Args:
        gt_boxes: [N, 7], gt box in lidar.points_transform_
        points: [M, 4], point cloud in lidar.
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [
            -global_random_rot_range, global_random_rot_range
        ]
    enable_grot = np.abs(global_random_rot_range[0] -
                         global_random_rot_range[1]) >= 1e-3
    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [
            center_noise_std, center_noise_std, center_noise_std
        ]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes, ), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(
        scale=center_noise_std, size=[num_boxes, num_try, 3])
    # loc_noises = np.random.uniform(
    #     -center_noise_std, center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try])
    if group_ids is not None: # TODO: DELETE? always none I guess
        if enable_grot:
            set_group_noise_same_v2_(loc_noises, rot_noises, global_rot_noises,
                                     group_ids)
        else:
            set_group_noise_same_(loc_noises, rot_noises, group_ids)
        group_centers, group_id_num_dict = get_group_center(
            gt_boxes[:, :3], group_ids)
        if enable_grot:
            group_transform_v2_(loc_noises, rot_noises, gt_boxes[:, :3],
                                gt_boxes[:, 6], group_centers,
                                global_rot_noises, valid_mask)
        else:
            group_transform_(loc_noises, rot_noises, gt_boxes[:, :3],
                             gt_boxes[:, 6], group_centers, valid_mask)
        group_nums = np.array(list(group_id_num_dict.values()), dtype=np.int64)

    origin = [0.5, 0.5, 0]
    gt_box_corners = center_to_corner_box3d(
        gt_boxes[:, :3],
        gt_boxes[:, 3:6],
        gt_boxes[:, 6],
        origin=origin,
        axis=2)
    if group_ids is not None: # TODO: DELETE? always none I guess
        if not enable_grot:
            selected_noise = noise_per_box_group(gt_boxes[:, [0, 1, 3, 4, 6]],
                                                 valid_mask, loc_noises,
                                                 rot_noises, group_nums)
        else:
            selected_noise = noise_per_box_group_v2_(
                gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises,
                rot_noises, group_nums, global_rot_noises)
    else:
        if not enable_grot:
            selected_noise = noise_per_box(gt_boxes[:, [0, 1, 3, 4, 6]],
                                           valid_mask, loc_noises, rot_noises)
        else:
            selected_noise = noise_per_box_v2_(gt_boxes[:, [0, 1, 3, 4, 6]],
                                               valid_mask, loc_noises,
                                               rot_noises, global_rot_noises)
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    surfaces = corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(points, gt_boxes[:, :3], point_masks, loc_transforms,
                          rot_transforms, valid_mask)

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)

@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]

@numba.njit
def points_transform_(points, centers, point_masks, loc_transform,
                      rot_transform, valid_mask):
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i:i + 1, :3] = points[i:i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform

@numba.njit
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos

#@numba.jit(nopython=False)
def surface_equ_3d_jit(polygon_surfaces):
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d

@numba.jit(nopython=False)
def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces, 
            max_num_points_of_surface, 3] 
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces 
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jit(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] \
                     + points[i, 1] * normal_vec[j, k, 1] \
                     + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret

def _select_transform(transform, indices):
    result = np.zeros(
        (transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result

@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners. 
    Returns:
        surfaces (float array, [N, 6, 4, 3]): 
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array([
        0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7
    ]).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces

@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    # print(valid_mask)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j],
                                     rot_mat_T)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                # print(coll_mat)
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask

@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_T):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = -rot_sin
    rot_mat_T[1, 0] = rot_sin
    rot_mat_T[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_T


@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes):
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(
        1, 4, 2)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners

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

@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack(
        (boxes, boxes[:, slices, :]), axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = corner_to_standup_nd_jit(boxes)
    qboxes_standup = corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (
                                C[1] - A[1]) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (
                                C[1] - B[1]) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret


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

# This Object is responsible for hold and deliver of object sample coordinates 
#============================================================
class BatchSampler:
    def __init__(self, sampled_list, name, config, epoch=None, shuffle=True, drop_reminder=False):
        self._sampled_list = sampled_list # list of all the object coordinates


        self.config = config
        self._indices = np.arange(len(sampled_list))
        if shuffle: # is True
            np.random.shuffle(self._indices) # randomize the samples
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder
        self.random_translate() # scatter the samples 

    # get n(num) sample indices
    #============================================================
    def _sample(self, num):
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset() 
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        if self._name is not None:
            print("reset BatchSampler", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices) # randomize the samples
        self._idx = 0

    # get n(num) samples
    #============================================================
    def sample(self, num):
        # get first the indices
        indices = self._sample(num) 
        # than get the object coordinates to the samples
        return [self._sampled_list[i] for i in indices]
        # return np.random.choice(self._sampled_list, num)

    # randomly translate the sampled bounding boxes that they do not collide with each other (at least less likely)
    # =========================================
    def random_translate(self):
        # set translation noise intervals
        noise_x= [0,0]
        noise_y= self.config["sampler_noise_y"]
        for i, sample in enumerate(self._sampled_list):
            # get the x distance to camera
            sample_x_distance = self._sampled_list[i]["box3d_lidar"][0]
            # if x distance is smaller than 2.5, translate rather todards the camera
            if sample_x_distance <  self.config["sampler_noise_x_point"]:
                noise_x=  self.config["sampler_noise_x_closer"]
            # if x distance is greater than 2.5, translate rather away from the camera
            if sample_x_distance >= self.config["sampler_noise_x_point"]:
                noise_x=  self.config["sampler_noise_x_farther"]
            loc_noises = np.zeros(len(sample["box3d_lidar"])) 
            loc_noises[0] = loc_noises[0] + random.uniform(noise_x[0], noise_x[1]) # x
            loc_noises[1] = loc_noises[1] + random.uniform(noise_y[0], noise_y[1]) # y
            self._sampled_list[i]["box3d_lidar"] = self._sampled_list[i]["box3d_lidar"] + loc_noises
            #self._sampled_list[i]["box3d_lidar"] = self._sampled_list[i]["box3d_lidar"] + [1,0,1,0,0,0,0]

class DataBaseSamplerV2:
    def __init__(self, sampler_info_path, config):

        # ------------------------------------------------------------------------------------------------------
        #  Open Data Annotations and Filter
        # ------------------------------------------------------------------------------------------------------

        with open(sampler_info_path, 'rb') as f:
            db_infos = pickle.load(f)
            #db_infos["Pedestrian"]= db_infos["Pedestrian"][0:350] # TODO

        for k, v in db_infos.items():
            print(f"load {len(v)} {k} database infos")

        # filter certain difficulties from db_infos 
        _removed_difficulties = [-1]
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info["difficulty"] not in _removed_difficulties
            ]
        db_infos = new_db_infos

        # filter minimum points
        _min_gt_point_dict = {"Cyclist" : 5}
        for name, min_num in _min_gt_point_dict.items():
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                # replace the samples in db_infos with the ones which have enough points
                db_infos[name] = filtered_infos

        
        print("After filter database:")
        for k, v in db_infos.items():
            print(f"load {len(v)} {k} database infos")

        self.db_infos = db_infos
        self._group_db_infos = self.db_infos  # just use db_infos
        groups= [{"Cyclist":8}]
        self._sample_classes = []
        self._sample_max_nums = []

        # just split keys and values of groups into _sample_classes and _sample_max_nums
        for group_info in groups:
                group_names = list(group_info.keys())
                self._sample_classes += group_names
                self._sample_max_nums += list(group_info.values())

        # this variable will hold for every class a BatchSampler object which generates samples
        self._sampler_dict = {}
        for k, v in self._group_db_infos.items():
            self._sampler_dict[k] = BatchSampler(v, k, config)
        


# We are removing all the annos from image_anno which are not in the list "desired_objects"
# =========================================
def remove_undesired_objects(image_anno, desired_objects):

    # Find all indices which are relevant
    desired_objects_indices = []
    for i, x in enumerate(image_anno['name']):
        if x in desired_objects:
            desired_objects_indices.append(i)

    # create new map with all the desired annos
    desired_objects_annos = {}
    for key in image_anno.keys():
        desired_objects_annos[key] = (
            image_anno[key][desired_objects_indices])
    return desired_objects_annos

def camera_to_lidar(points, r_rect, velo2cam):
    points_shape = list(points.shape[0:-1])
    # creating homogeneous coords
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    # invert rectification and velo2cam and apply it on points
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    # return only not homo coords
    return lidar_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    
    xyz = data[:, 0:3]
       
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    # project from camera space to lidar space
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    
    # return lidar coord position and convert lhw(camera coord) size to hwl(lidar coord) size
    # RECALL ANNOTATIONS: yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)

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

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)
    
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
    corners += centers.reshape([-1, 1, 2])
    return corners

def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.
    
    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum('aij,jka->aik', points, rot_mat_T)


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
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(
            dims.dtype)
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

    

def create_anchors_3d_stride(feature_size,
                             sizes,
                             anchor_strides,
                             anchor_offsets,
                             rotations,
                             dtype=np.float32):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """

    # almost 2x faster than v1
    x_stride, y_stride, z_stride = anchor_strides # [0.16, 0.16, 0.0] / in cm 
    x_offset, y_offset, z_offset = anchor_offsets
    z_centers = np.arange(feature_size[0], dtype=dtype) # z = 1 because we have pillars with (infinite) height
    y_centers = np.arange(feature_size[1], dtype=dtype) # y = 248 horizontal tiles
    x_centers = np.arange(feature_size[2], dtype=dtype) # x = 296 vertical tiles
    # create the tiles by multiplying the number of centers with the tile sizes and adding the offset to center it
    z_centers = z_centers * z_stride + z_offset # complete biv area gets tiled 
    y_centers = y_centers * y_stride + y_offset # complete biv area gets tiled 
    x_centers = x_centers * x_stride + x_offset # complete biv area gets tiled 
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3]) # sizes of anchors
    rotations = np.array(rotations, dtype=dtype) # rotations for the anchors (according to pointpillars paper 0 and 90 degrees)
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5 # = [1,1,1,1,1]
    tile_shape[-2] = int(sizes.shape[0]) # = 1
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5]) # [x, y, z, xdim, ydim, zdim, rad]


def generate_anchors(feature_map_size, config_anchor_generator):
    anchors_list = []
    match_list, unmatch_list = [], []

    #----------------------------------------
    # set variables
    #----------------------------------------

    sizes = config_anchor_generator["sizes"]
    strides = config_anchor_generator["strides"]
    offsets = config_anchor_generator["offsets"]
    rotations = config_anchor_generator["rotations"]
    matched_threshold = config_anchor_generator["matched_threshold"]
    unmatched_threshold = config_anchor_generator["unmatched_threshold"]

    # ------------------------------------------------------------------------------------------------------
    # create anchors 
    # ------------------------------------------------------------------------------------------------------

    anchors = create_anchors_3d_stride(feature_map_size,sizes,strides,offsets,rotations) # (1, 248, 296, 2, 7) [channels, height, width, rotations, features] # features: [x, y, z, xdim, ydim, zdim, rad]
    anchors = anchors.reshape([*anchors.shape[:3], -1, 7])
    anchors_list.append(anchors)

    # ------------------------------------------------------------------------------------------------------
    # create for every anchor an entry in match_list and unmatch_list (with the corresponding thresholds) 
    # ------------------------------------------------------------------------------------------------------

    num_anchors = np.prod(anchors.shape[:-1])
    match_list.append(
        np.full([num_anchors], matched_threshold, anchors.dtype))
    unmatch_list.append(
        np.full([num_anchors], unmatched_threshold, anchors.dtype))

    # ------------------------------------------------------------------------------------------------------
    # return anchors and tresholdss
    # ------------------------------------------------------------------------------------------------------

    anchors = np.concatenate(anchors_list, axis=-2)
    matched_thresholds = np.concatenate(match_list, axis=0)
    unmatched_thresholds = np.concatenate(unmatch_list, axis=0)
    return {
        "anchors": anchors,
        "matched_thresholds": matched_thresholds,
        "unmatched_thresholds": unmatched_thresholds
    }


# draw samples from sampler_class and check if they collide with gt_boxes
# ==================================================
def sample_all(sampler_class,
            root_path,
            gt_boxes,
            gt_names,
            num_point_features,
            custom_dataset,
            sample_classes,
            sample_max_nums,
            sampler_max_point_collision,
            sampler_min_point_collision,
            points,
            random_crop=False):


    # ------------------------------------------------------------------------------------------------------ 
    #  reduce the number of objects to sample by the number of objects of this class already present in the ground truth
    # ------------------------------------------------------------------------------------------------------

    _rate = 1.0 
    sampled_num_dict = {}
    sample_num_per_class = []
    for class_name, max_sample_num in zip(sample_classes,sample_max_nums):
            # the more objects of a kind we already have in the current point cloud the less additional samples we need here of this type
            sampled_num = int(max_sample_num -
                              np.sum([n == class_name for n in gt_names]))
            sampled_num = np.round(_rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)
    
    # ------------------------------------------------------------------------------------------------------
    # set variables 
    # ------------------------------------------------------------------------------------------------------

    sampled_groups = sample_classes
    sampled = []
    sampled_gt_boxes = []
    avoid_coll_boxes = gt_boxes # just a copy of the gt boxes

    # ------------------------------------------------------------------------------------------------------
    # for each sample group and corresponding requiered number of samples
    # ------------------------------------------------------------------------------------------------------

    for class_name, sampled_num in zip(sampled_groups, sample_num_per_class):
        
        # ------------------------------------------------------------------------------------------------------
        # take a random number of samples
        # ------------------------------------------------------------------------------------------------------
        
        # sampled_num = random.randint(0, sampled_num) 
               
        # ------------------------------------------------------------------------------------------------------
        # if in this sample group name more than zero, samples are req.
        # ------------------------------------------------------------------------------------------------------
        
        if sampled_num > 0:
            
            # ------------------------------------------------------------------------------------------------------
            # rename relevant variables
            # ------------------------------------------------------------------------------------------------------

            name, num, gt_boxes = class_name, sampled_num, avoid_coll_boxes

            # ------------------------------------------------------------------------------------------------------
            # get the sampled bounding boxes
            # ------------------------------------------------------------------------------------------------------
            
            sampled_sampler = sampler_class._sampler_dict[name].sample(num) 
            num_gt = gt_boxes.shape[0] 
            num_sampled = len(sampled_sampler)

            # ------------------------------------------------------------------------------------------------------
            # convert gt bb to 2d corners
            # ------------------------------------------------------------------------------------------------------
    
            gt_boxes_bv = center_to_corner_box2d(
                gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, 6])

            # ------------------------------------------------------------------------------------------------------
            # get sampled boxes and concatenate them with gt boxes
            # ------------------------------------------------------------------------------------------------------

            sp_boxes = np.stack([i["box3d_lidar"] for i in sampled_sampler], axis=0)
            valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
            valid_mask = np.concatenate(
                [valid_mask,
                np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0)
            boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
            sp_boxes_new = boxes[gt_boxes.shape[0]:]

            # ------------------------------------------------------------------------------------------------------
            # convert sampled bb to 2d corners
            # ------------------------------------------------------------------------------------------------------

            sp_boxes_bv = center_to_corner_box2d(sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

            # concatenate with gt boxes
            total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)

            # ------------------------------------------------------------------------------------------------------
            # collision test gt and sampled boxes
            # ------------------------------------------------------------------------------------------------------

            coll_mat = box_collision_test(total_bv, total_bv) # 2d collision test
            diag = np.arange(total_bv.shape[0])
            coll_mat[diag, diag] = False
            valid_samples = []
            for i in range(num_gt, num_gt + num_sampled):
                # if any collision in that row (one object collides with any other object) than set all the other vars to false in that row
                if coll_mat[i].any():
                    coll_mat[i] = False
                    coll_mat[:, i] = False# set the complete column to false since that sample will not gonna be in the final sample list
                # if no collision was detected 
                else:
                    valid_samples.append(sampled_sampler[i - num_gt])
            
            # ------------------------------------------------------------------------------------------------------
            # get non-colliding boxes
            # ------------------------------------------------------------------------------------------------------

            sampled_cls = valid_samples
            sampled += sampled_cls
            if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0)
                    sampled_gt_boxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)
    
    
   
    # ------------------------------------------------------------------------------------------------------
    # if we have successfully sampled some bounding boxes laod their points from file 
    # ------------------------------------------------------------------------------------------------------

    if len(sampled) > 0:
        sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)
        s_points_list = []
        sampled_no_collison = []
        sampled_gt_boxes_no_collison = []
        for i, info in enumerate(sampled):
            if custom_dataset:
                with open(str(pathlib.Path(root_path) / info["path"])[:-3]+"pkl", 'rb') as file:
                    s_points = pickle.load(file, encoding='latin1')

                # ------------------------------------------------------------------------------------------------------
                # Check if the sampled gt boxes have too much overlay with other points in the pc
                # This is kind of a collsion test for other objects than the annotated ones
                # ------------------------------------------------------------------------------------------------------

                gt_coords = np.array(info["box3d_lidar"])[np.newaxis,:]
                indices = points_in_rbbox(points, gt_coords)
                num_points_in_gt = indices.sum(0)

                # ------------------------------------------------------------------------------------------------------
                # Add sampled gt boxes only if they have not much overlay with points in the pc
                # ------------------------------------------------------------------------------------------------------
                sample_distance = math.sqrt(abs(info["box3d_lidar"][0])**2 + abs(info["box3d_lidar"][1])**2) # distance to camera
                low_likelyhood =  bool(random.getrandbits(1)) and bool(random.getrandbits(1)) and bool(random.getrandbits(1))
                if num_points_in_gt < sampler_max_point_collision and (num_points_in_gt >= sampler_min_point_collision or (sample_distance < 2.5 and low_likelyhood)) and len(s_points)>0:
                    sampled_no_collison.append(info)
                    sampled_gt_boxes_no_collison.append(sampled_gt_boxes[i])
                    s_points[:, :3] += info["box3d_lidar"][:3] # move the bbox in the correct position (it was centered in the file)
                    
                    # ------------------------------------------------------------------------------------------------------
                    # Apply random trunctation in height to simulate when the object is near to camera and therefore outside the image boundaries
                    # ------------------------------------------------------------------------------------------------------
                    
                    sample_height_half = info["box3d_lidar"][5]/2.0 # get half size of sample
                    truncation_weight = (2.5-sample_distance)/2.5 # calc the intensity of truncation (truncation TRUE, if sample closer than X)
                    if(truncation_weight>0):
                        points_z_max = np.max(s_points[:,2])
                        points_z_min = np.min(s_points[:,2])
                        points_z_max_new = points_z_max - sample_height_half*truncation_weight
                        points_z_min_new = points_z_min + sample_height_half*truncation_weight
                        max_filter = s_points[:,2] < points_z_max_new 
                        min_filter = s_points[:,2] > points_z_min_new 
                        s_points = s_points[np.logical_and(max_filter,min_filter)]
                    
                    # points_z_mean = np.mean(s_points[:,2])
                    # points_z_max = np.max(s_points[:,2])
                    # points_z_min = np.min(s_points[:,2])
                    # points_z_max_new = points_z_max - (np.random.uniform()*points_z_mean/3)
                    # points_z_min_new = points_z_min + (np.random.uniform()*points_z_mean/3)
                    # max_filter = s_points[:,2] < points_z_max_new 
                    # min_filter = s_points[:,2] > points_z_min_new 
                    # s_points = s_points[np.logical_and(max_filter,min_filter)]

                    



                    s_points_list.append(s_points)
                #debug
                #debug_save_points_func(points,gt_coords)
                
            else:
                s_points = np.fromfile(
                    str(pathlib.Path(root_path) / info["path"]), dtype=np.float32, count=-1).reshape([-1, num_point_features])
                s_points[:, :3] += info["box3d_lidar"][:3] # move the bbox in the correct position (it was centered in the file)
                s_points_list.append(s_points)

        # ------------------------------------------------------------------------------------------------------
        # create the object which contains the sampled boxes for return
        # ------------------------------------------------------------------------------------------------------
        if len(sampled_no_collison) > 0:
            ret = {
                "gt_names": np.array([s["name"] for s in sampled_no_collison]), # in sampled are all the infos
                "difficulty": np.array([s["difficulty"] for s in sampled_no_collison]),
                "gt_boxes": np.array(sampled_gt_boxes_no_collison), #  in sampled_gt_boxes are just the gt_boxes
                "points": np.concatenate(s_points_list, axis=0), # in s_points_list are the gt points
                "gt_masks": np.ones((len(sampled_no_collison), ), dtype=np.bool_)
            }

            ret["group_ids"] = np.arange(gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled))
        else:
            ret = None
    else:
        ret = None
    return ret


# debug purpose
# ==================================================
# def debug_save_points_func(points, gt_boxes, message):
    
#     self.pub_points.publish(point_cloud2.create_cloud(self.header, self.fields, points))
    
    # ------------------------------------------------------------------------------------------------------ 
    #  old functionality of this function
    # ------------------------------------------------------------------------------------------------------
    
    # with open(os.getcwd() + "/scripts/debug_rviz/points/points.pkl", 'wb') as file:
    #     pickle.dump(np.array(points), file, 2)
    # origin = [0.5, 0.5, 0]
    # gt_boxes_corners = center_to_corner_box3d(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=origin, axis=2)
    # with open(os.getcwd() + "/scripts/debug_rviz/points/bbox.pkl", 'wb') as file:
    #     pickle.dump(np.array(gt_boxes_corners), file, 2)


# This is the class which holds the functions to get the dataloader
# ==================================================
class dataLoader():

    def __init__(self, training, sampler_class, config, sample_val_dataset_mode=False, no_annos_mode=False, iterate_samples_in_debug_mode=False):
     
        
        


        self.config = config
        self.config_input_reader = config["train_input_reader"] if training == True else config["eval_input_reader"]
        self.config_voxel_generator = config["model"]["second"]["voxel_generator"]
        self.config_rpn = config["model"]["second"]["rpn"]
        self.config_anchor_generator = config["model"]["second"]["target_assigner"]["anchor_generators"]["anchor_generator_stride"]
        self.config_target_assigner = config["model"]["second"]["target_assigner"]


        # ------------------------------------------------------------------------------------------------------ 
        #  fetch params from the config file
        # ------------------------------------------------------------------------------------------------------


        # regular params
        
        self.dataset_root_path = self.config_input_reader["dataset_root_path"]
        self.num_point_features = self.config_input_reader["num_point_features"]
        self.feature_map_size =  self.config_input_reader["feature_map_size"]
        self.desired_objects = self.config_input_reader["desired_objects"] 
        self.cache = config["cache"] 
        self.shuffle_buffer_size = config["shuffle_buffer_size"] 
        self.batch_size = self.config_input_reader["batch_size"]
        self.buffer_size = config["buffer_size"] 
        self.custom_dataset = config["custom_dataset"]
        self.training = training
        self.sample_classes = self.config_input_reader["sample_classes"]
        self.sample_max_nums = self.config_input_reader["sample_max_nums"]
        self.sampler_max_point_collision = self.config_input_reader["sampler_max_point_collision"]
        self.sampler_min_point_collision = self.config_input_reader["sampler_min_point_collision"]
        self.sampler_class = sampler_class
        self.no_annos_mode = self.config_input_reader["no_annos_mode"]
        self.img_list_and_infos_path = self.config_input_reader["img_list_and_infos_path_no_annos"] if self.no_annos_mode else self.config_input_reader["img_list_and_infos_path"]
        
        # debug params
        self.iterate_samples_in_debug_mode = config["iterate_samples_in_debug_mode"] 
        self.debug_save_points = config["debug_save_points"] 
        self.sample_val_dataset_mode = sample_val_dataset_mode
        
        # just important for sample_val_dataset_mode
        if self.sample_val_dataset_mode:
            self.alpha = 0.0
            self.bbox= [[700., 100., 800., 300.]]
            self.difficulty = 0
            self.group_ids = 0
            self.index = 0
            self.name = "Pedestrian"
            self.num_points_in_gt = 5000
            self.occluded = 0
            self.score = 0
            self.truncated = 0.0

        # production mode params
        self.production_mode = config["production_mode"] 



        # ------------------------------------------------------------------------------------------------------ 
        #  load the file which hold all the meta data and labels 
        # ------------------------------------------------------------------------------------------------------


        if self.no_annos_mode == False:
            with open(self.img_list_and_infos_path, 'rb') as f:
                self.img_list_and_infos_ori = pickle.load(f)
                self.img_list_and_infos = self.img_list_and_infos_ori
                if self.training:
                    self.img_list_and_infos = self.img_list_and_infos_ori#[750:] #LIMIT
        else:
            velodyne_path = self.dataset_root_path + "/testing_live" + "/velodyne" 
            velodyne_number_images = len([name for name in os.listdir(velodyne_path)])
            self.img_list_and_infos = list(range(velodyne_number_images))
            with open(self.img_list_and_infos_path, 'rb') as f:
                self.img_infos_dummy = pickle.load(f) # we just load a dummy in case we do not have annos
                
        # ------------------------------------------------------------------------------------------------------ 
        #  setup ros publisher (for rviz visualization) and listener (to receiver point clouds live from sensor)
        # ------------------------------------------------------------------------------------------------------   
        
        if self.production_mode or self.debug_save_points:
            
            self.production_pc = "Placeholder"
            self.production_pc_new = False

            rospy.init_node('listener', anonymous=True)
            from sensor_msgs.msg import PointCloud2, PointField
            import std_msgs.msg
            self.pub_points = rospy.Publisher('debug_points', PointCloud2)
            self.debug_load_data_bb = rospy.Publisher("debug_load_data_bb", BoundingBoxArray)
            self.debug_load_data_pillars = rospy.Publisher("debug_load_data_pillars", BoundingBoxArray)
            self.calib = {"R0_rect" : np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3,3),
                        "Tr_velo_to_cam" : np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0]).reshape(3,4)}

            self.subscriber = rospy.Subscriber("/camera/depth/color/points", msg_PointCloud2, self.production_pc_update) # source: to capture from depth_image_proc/point_cloud_xyz package
            
            self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    ]
            self.header = std_msgs.msg.Header()
            self.header.stamp = rospy.Time.now()
            self.header.frame_id = 'camera_color_frame'

    # debug purpose
    # ==================================================
    def debug_save_points_func(self, points, message, gt_boxes=None, pillars=None):
        
        # print("DEBUG RVIZ: "+message)
        
        self.pub_points.publish(point_cloud2.create_cloud(self.header, self.fields, points))
        
        if gt_boxes is not None:
            centers,dims,angles = gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6] # [a,b,c] -> [c,a,b] (camera to lidar coords)

            centers = centers + [0.0,0.0,0.9]
            send_3d_bbox(centers, dims, angles, self.debug_load_data_bb, self.header) 
        
        if pillars is not None:
            centers,dims,angles = pillars[:, :3], pillars[:, 3:6], pillars[:, 6] # [a,b,c] -> [c,a,b] (camera to lidar coords)

            centers = centers + [0.0,0.0,0.9]
            send_3d_bbox(centers, dims, angles, self.debug_load_data_pillars, self.header) 
            
        
        if False: print("")
        
        # ------------------------------------------------------------------------------------------------------ 
        #  old functionality of this function
        # ------------------------------------------------------------------------------------------------------
        
        # with open(os.getcwd() + "/scripts/debug_rviz/points/points.pkl", 'wb') as file:
        #     pickle.dump(np.array(points), file, 2)
        # origin = [0.5, 0.5, 0]
        # gt_boxes_corners = center_to_corner_box3d(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=origin, axis=2)
        # with open(os.getcwd() + "/scripts/debug_rviz/points/bbox.pkl", 'wb') as file:
    #     pickle.dump(np.array(gt_boxes_corners), file, 2)


    # just return the number of datapoints in the dataset
    # ==================================================
    def production_pc_update(self,pc):
        self.production_pc_new = True
        self.production_pc = pc

    # just return the number of datapoints in the dataset
    # ==================================================
    @property
    def ndata(self):
        return len(self.img_list_and_infos)


    # creates a random distribution of all samples for the iterator
    # ==================================================
    @property
    def random_list_of_data(self):
        return random.sample(range(len(self.img_list_and_infos)),len(self.img_list_and_infos))#[:1000] #LIMIT


    # saves a the newly created validation dataset 
    # ==================================================
    def save_kitti_infos_val_sampled(self):
        velodyne_reduced_path = self.dataset_root_path + "/kitti_infos_val_sampled.pkl"
        with open(velodyne_reduced_path, 'wb') as file:
                pickle.dump(self.img_list_and_infos, file, 2)



    # Prepare dataloader
    # ==================================================
    def getIterator(self):

        cache = self.config["cache"]
        shuffle_buffer_size = self.config["shuffle_buffer_size"]
        buffer_size = self.config["buffer_size"]
        batch_size = self.config_input_reader["batch_size"]
        iterate_samples_in_debug_mode = self.config["iterate_samples_in_debug_mode"]

        # merge two or more datapoints to one batch
        # =============================================
        def merge_second_batch(batch_list, _unused=False):

            # ------------------------------------------------------------------------------------------------------
            # merge the keys of the dictionary
            # ------------------------------------------------------------------------------------------------------

            example_merged = defaultdict(list) # put all key values together  
            for example in batch_list:
                for k, v in example.items():
                    example_merged[k].append(v)
            ret = {}


            # delete "num_voxels"
            example_merged.pop("num_voxels") 


            # ------------------------------------------------------------------------------------------------------ 
            # now also merge the values of dictionary
            # therefore create the correct input format for the network
            # TODO: QUESTION: How the network differentiates the concatinated lists than belonging to different point clouds?
            # ------------------------------------------------------------------------------------------------------

            for key, elems in example_merged.items():

                if key in ['voxels', 'num_points', 'num_gt', 'gt_boxes', 'voxel_labels','match_indices']:


                    # concat and add
                    ret[key] = np.concatenate(elems, axis=0)
                elif key == 'match_indices_num': # never happens?


                    # concat and add
                    ret[key] = np.concatenate(elems, axis=0)


                # add additional batch dim (0 is 0's batch, 1 is 1's batch)
                # each voxel has now 4 dims instead of 3: [voxels,4]
                elif key == 'coordinates':
                    coors = []
                    for i, coor in enumerate(elems):
                        coor_pad = np.pad(
                            coor, ((0, 0), (1, 0)),
                            mode='constant',
                            constant_values=i)
                        coors.append(coor_pad)


                    # concat and add
                    ret[key] = np.concatenate(coors, axis=0)
                
                # for every other key like rect, Trv2c, P2, anchors, anchors_mask, labels reg_targets, reg_weights, image_idx, image_shape
                # keep the samples apart by adding an addition dimention with np.stack
                else:
                    # stack and add
                    ret[key] = np.stack(elems, axis=0)


            # return merged batch, ready for the network
            return ret


        # this is the inner part of the generator which deliver all the datapoints of our dataset
        # also the main call for the prepocess (__getitem__ function) is located here
        # =============================================
        def gen():
            

            # here we fetch the list of samples we want to iterate over for training, testing etc.
            datalist = []

            if self.production_mode:

                tf.print("", output_stream=sys.stdout)
                tf.print("# ------------------------------------------------------------------------------------------------------", output_stream=sys.stdout)
                tf.print("# Waiting for ROS Input", output_stream=sys.stdout)
                tf.print("# ------------------------------------------------------------------------------------------------------", output_stream=sys.stdout)
                tf.print("", output_stream=sys.stdout)

                while True:
                    if self.production_pc_new:
                        yield [self.__getitem__(0)]
            else:

                # if training mode we want to randomize the datapoints (also in sample_val_dataset_mode)
                if self.training and not self.sample_val_dataset_mode:
                    datalist = self.random_list_of_data
                else:
                    datalist = range(0, self.ndata) # dont randomize the data in testing mode

                # iterate over the list of samples
                for i in datalist:
                    yield [self.__getitem__(i)] # call for preprocessing

            # ------------------------------------------------------------------------------------------------------
            # hier könnte ich eine art subscriber für das ros topic einbauen, welches dann elemente an __getitem__ schickt
            # ------------------------------------------------------------------------------------------------------


        # this is the outer part of the generator which collects datapoints from the inner generator 
        # and creates batches out of it if it has enough elements
        # =============================================
        def get_batch_gen(gen, batch_size):
            def batch_gen():

                #  this elements holds new samples until the batch size is reached and than merges them
                buff = []


                # we get new samples from our sample generator
                for i, x in enumerate(gen()):
                    
                    
                    # if we have enough datapoints together for the batch 
                    if i % batch_size == 0 and buff: 
                        

                        # merge the datapoints together to a batch
                        buff = merge_second_batch(buff) 


                        # take values of dictionary and convert to tuple (tensorflow datapipes cannot deal with dictionaries)
                        buff = tuple(buff.values()) 


                        # return the merged batch
                        yield buff 


                        # reset batch # TODO wrong place?
                        buff = []


                    # fill up the datapoints buffer 
                    buff += x 


                # after the generator is empty return the last elements in the buffer if there are any 
                if buff:
                    pass # TODO: skip the last elements since the follwing code raises an error
                    #yield tuple(buff[0].values()) 

            return batch_gen # this gives back the outer iterator which is including also the inner generator 

        # ------------------------------------------------------------------------------------------------------ 
        # This blog exists because of debug reasons, here we can iterator over the datatset without tensorflow datasets whoch doesnt allow breakpoints
        # ------------------------------------------------------------------------------------------------------
        
        if (iterate_samples_in_debug_mode):
            gen1 = get_batch_gen(gen,batch_size)()
            i = 0
            while True:
                try:
                    # get the next item
                    element = next(gen1)
                    i +=1
                    print(i)


                # if end of the dataset is reached, break from loop
                except StopIteration:


                    # save the newly created dataset if in sample_val_dataset_mode
                    if self.sample_val_dataset_mode:
                        self.save_kitti_infos_val_sampled()
                    sys.exit()
           
        # ------------------------------------------------------------------------------------------------------ 
        # create tensorflow dataset from our generator 
        # ------------------------------------------------------------------------------------------------------

        # depending on training or testing we have a different amount of return values from the generator which must be specified for the tensorflow dataset
        if self.training == False:
            ds = tf.data.Dataset.from_generator(get_batch_gen(gen, batch_size),
                (tf.float32,
                tf.int32,
                tf.int32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.uint8,
                tf.int32,
                tf.int32))
        else:
            ds = tf.data.Dataset.from_generator(get_batch_gen(gen, batch_size),
                (tf.float32,
                tf.int32,
                tf.int32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.uint8,
                tf.int32,
                tf.float32,
                tf.float32,
                tf.int64,
                tf.int32))


        # ------------------------------------------------------------------------------------------------------ 
        # Do we want to cache the dataset (keep in memory) for the epoch? 
        # Does not make sense for us since we have randomization in the datapipe
        # ------------------------------------------------------------------------------------------------------
    
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        # ------------------------------------------------------------------------------------------------------ 
        # set how many (shuffle_buffer_size) datapoints are in the buffer where elements are randomly selected from
        # reshuffle_each_iteration is probably without function here since I dont use "repeat" to iterate epochs
        # ------------------------------------------------------------------------------------------------------

        #ds = ds.shuffle(buffer_size=shuffle_buffer_size) 

        # ------------------------------------------------------------------------------------------------------ 
        # `prefetch` lets the dataset fetch batches in the background while the model is training.
        # ------------------------------------------------------------------------------------------------------

        if (buffer_size=="AUTOTUNE"):
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        else:
            ds = ds.prefetch(buffer_size=buffer_size)


        # ------------------------------------------------------------------------------------------------------ 
        # return the tensorflow dataset object which is iterable
        # ------------------------------------------------------------------------------------------------------
        return ds 



    # this function gets called every time a new datapoint is requested by the dataloader
    # ==================================================
    def __getitem__(self, img_id):


        # ------------------------------------------------------------------------------------------------------ 
        #  load annotations for the current example
        # ------------------------------------------------------------------------------------------------------
        
        if not self.no_annos_mode:
            img_infos = self.img_list_and_infos[int(img_id)] 
        else:
            img_infos = self.img_infos_dummy # we just load a dummy in case we do not have annos


        # we create leading zeros of the index to match the filenames of the dataset
        img_id = "%06d" % (int(img_id),) 


        # ------------------------------------------------------------------------------------------------------ 
        #  load the pointcloud
        # ------------------------------------------------------------------------------------------------------
        

        # get the folder name 
        train_or_test = "/training" if self.training == True else "/testing" # are we work on train or test data?
        train_or_test = "/testing" if self.sample_val_dataset_mode else train_or_test # take always test data if we are in sample_val_dataset_mode mode
        train_or_test = "/testing_live" if self.no_annos_mode else train_or_test # take always testing_live data if we are in no_annos_mode mode 


        # load the preprocessed reduced pointcloud 
        if self.production_mode:
            points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.production_pc)[1::4] # transform pc to numpy and take every 4 points
            # TRANSFORM to lidar coords (x,y,z)
            # - the output of the realsense are image coords
            # - rviz already shows the direct output of the realsense in lidar coords because of the topic "CameraInfo"
            r = R.from_euler('y', -90, degrees=True).as_dcm()
            r2 = R.from_euler('x', 90, degrees=True).as_dcm()
            # rotate the pc so that z is height, x is depth and y is width (TODO, is this right?)
            points = np.dot(points,r) # rotate the pc 
            points = np.dot(points,r2)
            points = points + [0.0,0.0,1.0] # elevate z coords of pointcloud to hight of the pipeline
            if self.debug_save_points: self.pub_points.publish(point_cloud2.create_cloud(self.header, self.fields, points)) # show pc in rviz 
        else:
            if self.custom_dataset:
                velodyne_reduced_path = self.dataset_root_path + train_or_test + "/velodyne" + "/" + img_id + ".pkl" 
                with open(str(velodyne_reduced_path), 'rb') as file:
                    points = pickle.load(file, encoding='latin1')
            else:
                velodyne_reduced_path = self.dataset_root_path + train_or_test + "/velodyne_reduced" + "/" + img_id + ".bin" 
                points = np.fromfile(str(velodyne_reduced_path), dtype=np.float32, count=-1).reshape([-1, self.num_point_features])


        # ------------------------------------------------------------------------------------------------------ 
        #  create a dictionary "input_dict" which contains the annotations and pointcloud both
        #  "input_dict" is the input for the "prep_pointcloud" function
        # ------------------------------------------------------------------------------------------------------


        input_dict = {
            'points': points,
            'rect': img_infos['calib/R0_rect'].astype(np.float32), # is just the unity matrix for d435i data
            'Trv2c': img_infos['calib/Tr_velo_to_cam'].astype(np.float32), # needed for the rotation of GT annos (stored in cam coord) to lidar coords
            'P2': img_infos['calib/P2'].astype(np.float32), # not used for d435i data
            'image_shape': np.array(img_infos["img_shape"], dtype=np.int32) if 'img_shape' in img_infos else None,
            'image_idx': img_infos['image_idx'], 
            'image_path': img_infos['img_path'] if 'img_path' in img_infos else None
        }


        # filter undesired_objects 
        annos = img_infos['annos']
        annos = remove_undesired_objects(annos, self.desired_objects) # filter undesired_objects 


        # continue to fill up the "input_dict"
        loc = annos["location"]
        dims = annos["dimensions"]
        rots = annos["rotation_y"]
        gt_names = annos["name"]
        gt_boxes = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        difficulty = annos["difficulty"]
        input_dict.update({
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
            'difficulty': difficulty,
        })
        if 'group_ids' in annos:    
            input_dict['group_ids'] = annos["group_ids"]


        # ------------------------------------------------------------------------------------------------------ 
        # start the main preprocessing for this sample
        # ------------------------------------------------------------------------------------------------------


        example = self.prep_pointcloud(  input_dict=input_dict,    
                                    img_id=img_id,
                                    sample_val_dataset_mode=self.sample_val_dataset_mode,
                                    img_infos=img_infos
                                    )


        # ------------------------------------------------------------------------------------------------------ 
        # add meta data to the sample 
        # ------------------------------------------------------------------------------------------------------


        example["image_idx"] = img_infos['image_idx']
        example["image_shape"] = input_dict["image_shape"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        

        # ------------------------------------------------------------------------------------------------------ 
        # return example to the data generator
        # example has this content: [0:voxels, 1:num_points, 2:coordinates, 3:rect, 4:Trv2c, 5:P2, 6:anchors, 7:anchors_mask, 8:labels, 9:reg_targets, 10:reg_weights, 11:image_idx, 12:image_shape]

        # Example:
        # "image_idx" # corresponding to the names from the pointcloud files 
        # "image_shape" # in pixel
        # 'voxels': # [M, max_points, ndim] [voxels, points in that voxel [here: 100], coords in lidar coords?] / thus must be [P(pillars),N(points),D(dimensions) from the PointPillars Paper]
        # 'num_points': # [M] [number of point per voxel] 
        # 'coordinates': # coords: [M, 3] [voxels, coords of voxel centers in feature_map_size corrdinates. feature_map_size here: [1, 248, 296]]
        # "num_voxels": # just the number of voxels
        # 'rect': rect,
        # 'Trv2c': Trv2c,
        # 'P2': P2,
        # "anchors" # [x, y, z, xdim, ydim, zdim, rad] # list of anchors wuth the given features
        # 'anchors_mask' # [boolean], for each anchor if it is valid or not based on its size and distance to gt_boxes
        # 'labels': # [int] # one int for each anchors (the labels are mapped from the gt_boxes to the anchors)
        # 'reg_targets': # [x, y, z, xdim, ydim, zdim, rad] for each anchors (most zero)
        # 'reg_weights': # [int] # 0 or 1 # one int for each anchors # however, this does not seem to be very important since each assigned anchor has a 1.0
        # ------------------------------------------------------------------------------------------------------

        return example 
        
    # Main preprocessing for the data samples
    #============================================================
    def prep_pointcloud(self,
            input_dict, 
            img_id,
            sample_val_dataset_mode,
            img_infos            
            ):
        """
        Parameters
        ----------
        input_dict: pointcloud and annotation 
        img_id: id of the pointcloud
        sample_val_dataset_mode: bool, if true a new test dataset is created
        img_infos: just used if sample_val_dataset_mode is true
        Returns
        -------
        example: pointcloud and annos prepared for either traning or testing
        """
        
        # ------------------------------------------------------------------------------------------------------ 
        # get different configs and params
        # ------------------------------------------------------------------------------------------------------

        # transformation params
        points = input_dict["points"]
        rect = input_dict["rect"]
        Trv2c = input_dict["Trv2c"]
        P2 = input_dict["P2"]


        # voxel params
        point_cloud_range = np.array(self.config_voxel_generator["point_cloud_range"])
        voxel_size = np.array(self.config_voxel_generator["voxel_size"]).astype(float)
        
        
        
        
        # point_cloud_range : [0, -2.56, -3.0, 6.08, 2.56, 3.0]
        # points[...,0]>point_cloud_range[0]
        # points[...,0]<point_cloud_range[0]
        
        
        
        
        
        

        # ------------------------------------------------------------------------------------------------------
        # calculate the grid size (here: [296, 248,   1]) based on the point cloud range (here [0.0, -19.84, -2.5, 47.36, 19.84, 0.5]) divided by the voxel_size (here [0.16, 0.16, 5.  ]) 
        # ------------------------------------------------------------------------------------------------------

        max_number_of_points_per_voxel = self.config_voxel_generator["max_number_of_points_per_voxel"]
        grid_size = (
                point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        max_voxels=self.config_voxel_generator["max_number_of_voxels"]


        # rpn params
        layer_strides = np.array(self.config_rpn["layer_strides"])
        upsample_strides = np.array(self.config_rpn["upsample_strides"])
        anchor_area_threshold = self.config_input_reader["anchor_area_threshold"]


        # ------------------------------------------------------------------------------------------------------ 
        # apply augmentation if training
        # ------------------------------------------------------------------------------------------------------
        
        # debug
        # gt_boxes = input_dict["gt_boxes"]
        # gt_boxes = box_camera_to_lidar(gt_boxes, rect, Trv2c)
        # self.debug_save_points_func(points, "input",gt_boxes)


        if self.training:


            # ------------------------------------------------------------------------------------------------------ 
            # get different configs
            # ------------------------------------------------------------------------------------------------------


            # noise_per_object_v3_ params
            groundtruth_rotation_uniform_noise = self.config_input_reader["groundtruth_rotation_uniform_noise"]
            groundtruth_localization_noise_std= self.config_input_reader["groundtruth_localization_noise_std"] #0,25 lidar(x,y,z)
            global_random_rotation_range_per_object= self.config_input_reader["global_random_rotation_range_per_object"]


            # global noice params
            global_rotation_uniform_noise= self.config_input_reader["global_rotation_uniform_noise"]
            global_scaling_uniform_noise= self.config_input_reader["global_scaling_uniform_noise"]
            global_loc_noise_std= self.config_input_reader["global_loc_noise_std"]


            # ------------------------------------------------------------------------------------------------------ 
            # get annos
            # ------------------------------------------------------------------------------------------------------


            gt_boxes = input_dict["gt_boxes"]
            gt_names = input_dict["gt_names"]
            difficulty = input_dict["difficulty"]
            group_ids = None


            # ------------------------------------------------------------------------------------------------------ 
            # project gt_boxes from camera space to lidar space
            # note: gt_boxes are stored in camera space coords 
            # note: but lidar point cloud must not be transformed to lidar coords since they are already obviously stored in lidar space
            # RECALL ANNOTATIONS: yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
            # ------------------------------------------------------------------------------------------------------

            gt_boxes = box_camera_to_lidar(gt_boxes, rect, Trv2c)

            # debug
            
            # if self.debug_save_points: self.debug_save_points_func(points, "input",gt_boxes)

            # ------------------------------------------------------------------------------------------------------ 
            # filter undesired objects
            # ------------------------------------------------------------------------------------------------------


            gt_boxes_mask = np.array(
                [n in self.desired_objects for n in gt_names], dtype=np.bool_)
            
            
            # ------------------------------------------------------------------------------------------------------ 
            # RANDOM / Randomly put objects sampler_class into the pc 
            # ------------------------------------------------------------------------------------------------------


            sampled_dict = sample_all(
                    self.sampler_class,
                    self.dataset_root_path,
                    gt_boxes, #shape(1,7)
                    gt_names, #shape(1,0)
                    self.num_point_features,
                    self.custom_dataset,
                    self.sample_classes,
                    self.sample_max_nums,
                    self.sampler_max_point_collision,
                    self.sampler_min_point_collision,
                    points=points,
                    random_crop=False)
            
            # ------------------------------------------------------------------------------------------------------
            # delete the main gt box (unsampeld one) with a likelyhood
            # ------------------------------------------------------------------------------------------------------

            # check if a main gt box exists, first
            if len(gt_boxes)>0:
                
                high_likelyhood =  bool(random.getrandbits(1)) or bool(random.getrandbits(1)) or bool(random.getrandbits(1))
                if high_likelyhood:
                    indices = points_in_rbbox(points, gt_boxes)
                    indices_invert = ~np.array(indices).flatten()
                    points = points[indices_invert]
             
                    gt_names = np.delete(gt_names,0,0)
                    gt_boxes = np.delete(gt_boxes,0,0)
                    gt_boxes_mask = np.delete(gt_boxes_mask,0,0)
                    
                    # dimensions = {"dimensions":dimensions[np.newaxis,:]}
                    # img_infos["annos"].update(dimensions)
                    # location = {"location":location[np.newaxis,:]}
                    # img_infos["annos"].update(location)
                    # rotation_y = {"rotation_y":rotation_y}
                    # img_infos["annos"].update(rotation_y)
                    
                    img_infos["annos"]["dimensions"] = np.empty(shape = [0] + [3])
                    img_infos["annos"]["location"] = np.empty(shape = [0] + [3])
                    img_infos["annos"]["rotation_y"] = np.empty(shape = [0])
                    
                    img_infos["annos"]["alpha"] = np.empty(shape = [0])
                    img_infos["annos"]["bbox"] = np.empty(shape = [0] + [4])
                    img_infos["annos"]["difficulty"] = np.empty(shape = [0])
                    img_infos["annos"]["group_ids"] = np.empty(shape = [0])
                    img_infos["annos"]["index"] = np.empty(shape = [0])
                    img_infos["annos"]["name"] = np.empty(shape = [0])
                    img_infos["annos"]["num_points_in_gt"] = np.empty(shape = [0])
                    img_infos["annos"]["occluded"] = np.empty(shape = [0])
                    img_infos["annos"]["score"] = np.empty(shape = [0])
                    img_infos["annos"]["truncated"] = np.empty(shape = [0])

            # ------------------------------------------------------------------------------------------------------ 
            # Concatentate annos with sampled objects 
            # ------------------------------------------------------------------------------------------------------

            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                # gt_names = gt_names[gt_boxes_mask].tolist()
                gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
                # gt_names += [s["name"] for s in sampled]
                gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes])
                gt_boxes_mask = np.concatenate([gt_boxes_mask, sampled_gt_masks], axis=0)
                
                if group_ids is not None:
                    sampled_group_ids = sampled_dict["group_ids"]
                    group_ids = np.concatenate([group_ids, sampled_group_ids])

                points = np.concatenate([sampled_points, points], axis=0)

            # debug
            
            # if self.debug_save_points: self.debug_save_points_func(points, "samples added",gt_boxes)
            # ------------------------------------------------------------------------------------------------------ 
            # RANDOM / Augmentation on boxes themselves
            # ------------------------------------------------------------------------------------------------------
            
            noise_per_object_v3_(
                gt_boxes,
                points,
                gt_boxes_mask,
                rotation_perturb=groundtruth_rotation_uniform_noise,
                center_noise_std=groundtruth_localization_noise_std,
                global_random_rot_range=global_random_rotation_range_per_object,
                group_ids=group_ids,
                num_try=100)


            # should remove unrelated objects after noise per object
            gt_boxes = gt_boxes[gt_boxes_mask]
            gt_names = gt_names[gt_boxes_mask]
            if group_ids is not None:
                group_ids = group_ids[gt_boxes_mask]
            gt_classes = np.array(
                [self.desired_objects.index(n) + 1 for n in gt_names], dtype=np.int32)

            # debug
            
            # if self.debug_save_points: self.debug_save_points_func(points, "Augmentation on boxes themselves",gt_boxes)
            
            # ------------------------------------------------------------------------------------------------------ 
            # RANDOM / Randomly flip boxes
            # ------------------------------------------------------------------------------------------------------
            
            gt_boxes, points = random_flip(gt_boxes, points)

            # debug
            
            # if self.debug_save_points: self.debug_save_points_func(points, "Randomly flip boxess",gt_boxes)

            # ------------------------------------------------------------------------------------------------------ 
            # RANDOM / Randomly rotates the whole pointcloud including gt boxes
            # ------------------------------------------------------------------------------------------------------
        

            gt_boxes, points = global_rotation(gt_boxes, points, rotation=global_rotation_uniform_noise)

            # debug
            
            # if self.debug_save_points: self.debug_save_points_func(points, "Randomly rotates the whole pointcloud including gt boxes",gt_boxes)

            # ------------------------------------------------------------------------------------------------------ 
            # RANDOM / Randomly scales the whole pointcloud including gt boxes
            # ------------------------------------------------------------------------------------------------------


            gt_boxes, points = global_scaling_v2(gt_boxes, points,*global_scaling_uniform_noise)

            # debug
            # if self.debug_save_points: self.debug_save_points_func(points, "Randomly scales the whole pointcloud including gt boxes",gt_boxes)

            # ------------------------------------------------------------------------------------------------------ 
            # RANDOM / Randomly translates the whole pointcloud including gt boxes
            # ------------------------------------------------------------------------------------------------------


            gt_boxes, points = global_translate(gt_boxes, points, global_loc_noise_std)

            # debug
            
            # if self.debug_save_points: self.debug_save_points_func(points, "Randomly translates the whole pointcloud including gt boxes",gt_boxes)

            # ------------------------------------------------------------------------------------------------------ 
            # Filter bboxes that are ouside the range
            # Not necessary here since the realsense does not Output a huge range (and we can save some computation)
            # ------------------------------------------------------------------------------------------------------

            # bv_range = np.array(point_cloud_range)[[0, 1, 3, 4]]
            # mask = filter_gt_box_outside_range(gt_boxes, bv_range)
            # gt_boxes = gt_boxes[mask]
            # gt_classes = gt_classes[mask]
            # if group_ids is not None:
            #     group_ids = group_ids[mask]

            
            # ------------------------------------------------------------------------------------------------------ 
            # limit rad to [-pi, pi]
            # ------------------------------------------------------------------------------------------------------

            gt_boxes[:, 6] = limit_period(
                gt_boxes[:, 6], offset=0.5, period=2 * np.pi)


            # ------------------------------------------------------------------------------------------------------ 
            # RANDOM / Randomly shuffles the list of points
            # ------------------------------------------------------------------------------------------------------

            np.random.shuffle(points)

            # debug
            # if self.debug_save_points: self.debug_save_points_func(points, "Randomly shuffles the list of points",gt_boxes)

        
            # ------------------------------------------------------------------------------------------------------ 
            # Super important: Filter all gt boxes out outside the point_cloud_range
            # ------------------------------------------------------------------------------------------------------


            mask = filter_gt_box_outside_range_by_center(gt_boxes, point_cloud_range[[0, 1, 3, 4]])
            # if all(mask) == False:
            #     print("got one")
            gt_boxes = gt_boxes[mask]


            # ------------------------------------------------------------------------------------------------------ 
            # If we are in sample_val_dataset_mode we want to save the augmented pointclouds to "/testing/velodyne_sampled"
            # as pkl file but also create a new info file. Therefore we update the "/kitti_infos_val_sampled.pkl" dictionary in the follwing loop
            # ------------------------------------------------------------------------------------------------------
            
            
    
            if sample_val_dataset_mode:


                # save the augmented pointcloud
                velodyne_reduced_path = self.dataset_root_path + "/testing" + "/velodyne_sampled" + "/" + img_id + ".pkl" 
                with open(velodyne_reduced_path, 'wb') as file:
                    pickle.dump(np.array(points), file, 2)

                
                # convert to camera coords since annotation are in camera coords in kitti style
                gt_boxes_camera = box_lidar_to_camera(gt_boxes, rect, Trv2c)


                # iterate over the gt_boxes 
                for i in range(len(gt_boxes)):
                    

                    dimensions = gt_boxes_camera[i][3:6]
                    location = gt_boxes_camera[i][0:3]
                    rotation_y = gt_boxes_camera[i][[6]]
                    
                    


                    # in the first iteration replace the original gtboxes since they have been augmented now
                    if (i == 0):
                        
                      
                        
                        dimensions = {"dimensions":dimensions[np.newaxis,:]}
                        img_infos["annos"].update(dimensions)
                        location = {"location":location[np.newaxis,:]}
                        img_infos["annos"].update(location)
                        rotation_y = {"rotation_y":rotation_y}
                        img_infos["annos"].update(rotation_y)
                        
                        img_infos["annos"]["alpha"] = np.array([self.alpha])
                        img_infos["annos"]["bbox"] = np.array(self.bbox)
                        img_infos["annos"]["difficulty"] = np.array([self.difficulty])
                        img_infos["annos"]["group_ids"] = np.array([self.group_ids])
                        img_infos["annos"]["index"] = np.array([self.index])
                        img_infos["annos"]["name"] = np.array([self.name])
                        img_infos["annos"]["num_points_in_gt"] = np.array([self.num_points_in_gt])
                        img_infos["annos"]["occluded"] = np.array([self.occluded])
                        img_infos["annos"]["score"] = np.array([self.score])
                        img_infos["annos"]["truncated"] = np.array([self.truncated])
                            
                    # after first iteration fill up with the rest of the bboxes
                    
                    else:
       
                        
                        dimensions = {"dimensions":np.append(img_infos["annos"]["dimensions"],dimensions[np.newaxis,:],axis=0)}
                        img_infos["annos"].update(dimensions)
                        location = {"location":np.append(img_infos["annos"]["location"],location[np.newaxis,:],axis=0)}
                        img_infos["annos"].update(location)
                        rotation_y = {"rotation_y":np.append(img_infos["annos"]["rotation_y"],rotation_y)}
                        img_infos["annos"].update(rotation_y)
                        
                        # if len(img_infos["annos"]["name"]) > 0 
                        alpha = {"alpha":np.append(img_infos["annos"]["alpha"],self.alpha)}
                        img_infos["annos"].update(alpha)
                        bbox = {"bbox":np.append(img_infos["annos"]["bbox"],self.bbox,axis=0)}
                        img_infos["annos"].update(bbox)
                        difficulty = {"difficulty":np.append(img_infos["annos"]["difficulty"],self.difficulty)}
                        img_infos["annos"].update(difficulty)
                        group_ids = {"group_ids":np.append(img_infos["annos"]["group_ids"],self.group_ids)}
                        img_infos["annos"].update(group_ids)
                        index = {"index":np.append(img_infos["annos"]["index"],self.index)}
                        img_infos["annos"].update(index)
                        name = {"name":np.append(img_infos["annos"]["name"],self.name)}
                        img_infos["annos"].update(name)
                        num_points_in_gt = {"num_points_in_gt":np.append(img_infos["annos"]["num_points_in_gt"],self.num_points_in_gt)}
                        img_infos["annos"].update(num_points_in_gt)
                        occluded = {"occluded":np.append(img_infos["annos"]["occluded"],self.occluded)}
                        img_infos["annos"].update(occluded)
                        score = {"score":np.append(img_infos["annos"]["score"],self.score)}
                        img_infos["annos"].update(score)
                        truncated = {"truncated":np.append(img_infos["annos"]["truncated"],self.truncated)}
                        img_infos["annos"].update(truncated)
                    
                    # print(len(img_infos["annos"]["dimensions"]))
                    # print(len(img_infos["annos"]["location"]))
                    # print(len(img_infos["annos"]["rotation_y"]))
                    # print(len(img_infos["annos"]["alpha"]))
                    # print(len(img_infos["annos"]["bbox"]))
                    # print(len(img_infos["annos"]["difficulty"]))
                    # print(len(img_infos["annos"]["group_ids"]))
                    # print(len(img_infos["annos"]["index"]))
                    # print(len(img_infos["annos"]["name"]))
                    # print(len(img_infos["annos"]["num_points_in_gt"]))
                    # print(len(img_infos["annos"]["occluded"]))
                    # print(len(img_infos["annos"]["score"]))
                    # print(len(img_infos["annos"]["truncated"]))
                    # print("DONE")
            
    
        # ------------------------------------------------------------------------------------------------------ 
        # Convert pc to voxels
        # note 1: for return look at the function
        # note 2: the points which will be voxelized also contain the sampled bounding boxes. so this can not be the reason why far off objects are not detected
        # ------------------------------------------------------------------------------------------------------

        voxels, coordinates, num_points = points_to_voxel(points, voxel_size,point_cloud_range,max_number_of_points_per_voxel,True,max_voxels)
        
        
        # debug
        # show pillars
        if self.debug_save_points:
            
            grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
            grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
            pillar_coords = coordinates[:,1:]*voxel_size[:2] 
            pillar_coords = pillar_coords[:,[1,0]] # swap x and y axis
            pillar_coords = np.concatenate((pillar_coords,np.zeros(shape=(len(coordinates),1))), axis=1) # add z axis
            pillar_coords =  pillar_coords+(point_cloud_range[0],point_cloud_range[1],0)
        
            pillar_boxes = np.concatenate((pillar_coords,np.zeros(shape=(len(coordinates),1))+voxel_size[0]), axis=1) # add width axis
            pillar_boxes = np.concatenate((pillar_boxes,np.zeros(shape=(len(coordinates),1))+voxel_size[1]), axis=1) # add length axis
            pillar_boxes = np.concatenate((pillar_boxes,np.zeros(shape=(len(coordinates),1))+voxel_size[2]), axis=1) # add height axis
            pillar_boxes = np.concatenate((pillar_boxes,np.zeros(shape=(len(coordinates),1))), axis=1) # add rotation axis
            if self.debug_save_points: self.debug_save_points_func(points, "print pillars",pillars=pillar_boxes,gt_boxes=gt_boxes)
            

        # debug stuff
        # question: the voxel shape seems to be corresponing to the input point cloud but why are the coordininates are scaled to much?
        # answer:
        # The voxel coords are not given now in real coords but in feature_map_size coords instead.
        # Also it must be noted that although feature_map_size gets calculated later on in code, a similar calculation is used before in   
        # points_to_voxel; that why its possible to get feature_map_size coords before the actual creation of this variable
        # if self.debug_save_points:
        #     with open(os.getcwd() + "/scripts/debug_rviz/points/coordinates.pkl", 'wb') as file:
        #         pickle.dump(np.array(coordinates), file, 2)
        #     with open(os.getcwd() + "/scripts/debug_rviz/points/voxels.pkl", 'wb') as file:
        #         pickle.dump(np.array(voxels), file, 2)
        
        # ------------------------------------------------------------------------------------------------------ 
        # Create and fill-up Variable which will be returned 
        # ------------------------------------------------------------------------------------------------------

        example = {
            'voxels': voxels, # [M, max_points, ndim] [voxels, points in that voxel [here: 100], coords of those points relative to voxel center] / thus must be [P(pillars),N(points),D(dimensions) from the PointPillars Paper]
            'num_points': num_points, # [M] [number of point per voxel] 
            'coordinates': coordinates, # coords: [M, 3] [voxels, coords of voxel centers in feature_map_size corrdinates [1, 248, 296]]
            "num_voxels": np.array([voxels.shape[0]], dtype=np.int64), # 
            'rect': rect,
            'Trv2c': Trv2c,
            'P2': P2,
        }


        # ------------------------------------------------------------------------------------------------------ 
        # create feature_map_size: the result is probably very similar to grid_size if out_size_factor will be 1
        # ------------------------------------------------------------------------------------------------------
        
        
        # get feature_map_size
        out_size_factor = layer_strides[0] // upsample_strides[0]
        # if not lidar_input:
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1] # feature_map_size here: [1, 248, 296]

        # ------------------------------------------------------------------------------------------------------
        # generate_anchors
        # ------------------------------------------------------------------------------------------------------
        
        ret = generate_anchors(feature_map_size, self.config_anchor_generator) # return: [anchors,matched_thresholds,unmatched_thresholds]
        anchors = ret["anchors"] # anchors: [x, y, z, xdim, ydim, zdim, rad]
        
        # ------------------------------------------------------------------------------------------------------
        # just flatten the whole anchor tile situtation into an array of anchors with its 7 features [x, y, z, xdim, ydim, zdim, rad]
        # ------------------------------------------------------------------------------------------------------
        
        anchors = anchors.reshape([-1, 7])
        example["anchors"] = anchors # [x, y, z, xdim, ydim, zdim, rad] # list of anchors with the given features
        
        # ------------------------------------------------------------------------------------------------------
        # get the corners of all anchors in 2d (bev)
        # ------------------------------------------------------------------------------------------------------

        anchors_bv = rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]]) # return bboxes: [N, 4(xmin, ymin, xmax, ymax)] 

        # ------------------------------------------------------------------------------------------------------ 
        # filter anchors which are too small
        # ------------------------------------------------------------------------------------------------------

        if anchor_area_threshold >= 0:
            coors = coordinates
            dense_voxel_map = sparse_sum_for_anchors_mask(
                coors, tuple(grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)

            # ------------------------------------------------------------------------------------------------------
            # I think this gets the 2d size of the anchor 
            # However,  since voxel coordinates (through dense_voxel_map) are also considered, I guess that this mask 
            # also considers that only the anchors are interesting that lie in the neighborhood of voxels
            # ------------------------------------------------------------------------------------------------------
            
            anchors_area = fused_get_anchors_area(
                dense_voxel_map, anchors_bv, voxel_size, point_cloud_range, grid_size)

            # ------------------------------------------------------------------------------------------------------
            # Create a mask where there is a true for a big enough anchor and false for a too small
            # ------------------------------------------------------------------------------------------------------
            
            anchors_mask = anchors_area > anchor_area_threshold
            # example['anchors_mask'] = anchors_mask.astype(np.uint8)
            example['anchors_mask'] = anchors_mask # [boolean], for each anchor if it is valid or not based on its size and distance to gt_boxes

        # ------------------------------------------------------------------------------------------------------ 
        # return sample if in eval mode
        # ------------------------------------------------------------------------------------------------------

        if not self.training:
            return example

        # ------------------------------------------------------------------------------------------------------ 
        # assign gt_boxes to anchors if in self.training mode
        # ------------------------------------------------------------------------------------------------------


        if self.training:
            targets_dict = assign(
                anchors, # [x, y, z, xdim, ydim, zdim, rad]
                gt_boxes, # [x, y, z, xdim, ydim, zdim, rad]
                anchors_mask, # [boolean] # one boolean for each anchor
                gt_classes=gt_classes, # [int] # one int for each gt_boxes
                matched_thresholds=ret["matched_thresholds"], # [float] # one float for each anchors
                unmatched_thresholds=ret["unmatched_thresholds"], # [float] # one float for each anchors
                config_target_assigner=self.config_target_assigner)

            # update example
            example.update({
                'labels': targets_dict['labels'], # [int] # one int for each anchors (the labels are mapped from the gt_boxes to the anchors)
                'reg_targets': targets_dict['bbox_targets'], # [x, y, z, xdim, ydim, zdim, rad] for each anchors (most zero)
                'reg_weights': targets_dict['bbox_outside_weights'], # [int] # 0 or 1 # one int for each anchors # however, this does not seem to be very important since each assigned anchor has a 1.0
            })

            # 'voxels': # [M, max_points, ndim] [voxels, points in that voxel [here: 100], coords of those points relative to voxel center] / thus must be [P(pillars),N(points),D(dimensions) from the PointPillars Paper]
            # 'num_points': # [M] [number of point per voxel] 
            # 'coordinates': # coords: [M, 3] [voxels, coords of voxel centers in feature_map_size corrdinates [1, 248, 296]]
            # "num_voxels": # just the number of voxels
            # 'rect': rect,
            # 'Trv2c': Trv2c,
            # 'P2': P2,
            # "anchors" # [x, y, z, xdim, ydim, zdim, rad] # list of anchors wuth the given features
            # 'anchors_mask' # [boolean], for each anchor if it is valid or not based on its size and distance to gt_boxes
            # 'labels': # [int] # one int for each anchors (the labels are mapped from the gt_boxes to the anchors)
            # 'reg_targets': # [x, y, z, xdim, ydim, zdim, rad] for each anchors (most zero) # maps to the ground truth boxes
            # 'reg_weights': # [int] # 0 or 1 # one int for each anchors # however, this does not seem to be very important since each assigned anchor has a 1.0

        return example
        
# DEBUG 
# The main function has just the purpose to create a validation set with data augmentation
# However, I should put this in an extra file called "create_augmented_validation_dataset.py" # TODO
# =======================================================
if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------ 
    #  load the config from file and set Variables
    # ------------------------------------------------------------------------------------------------------ 


    config_path = "configs/train.yaml"
    # load the config from file
    with open(config_path) as f1:    # load the config file 
        config = yaml.load(f1, Loader=yaml.FullLoader)


    # change specific variabeles in the config to put the pipeline into a custom mode

    kitti_infos_val_path = sys.argv[1]
    kitti_dbinfos_val_path = sys.argv[2]
    config["train_input_reader"]["img_list_and_infos_path"] = kitti_infos_val_path
    config["train_input_reader"]["sampler_info_path"] = kitti_dbinfos_val_path
    config["train_input_reader"]["batch_size"] = 1  
    config["iterate_samples_in_debug_mode"] = True
    config["eval_input_reader"]["no_annos_mode"] = False
    config["custom_dataset"] = True
    config["production_mode"] = False
    config["debug_save_points"] = True # can also set to be true here
    

    # ------------------------------------------------------------------------------------------------------
    #  Load dataloader parameter and create dataloader
    # ------------------------------------------------------------------------------------------------------ 


    training = True # training mode is needed here 
    sample_val_dataset_mode = True # if True, sampled pointclouds are stored in "/testing/velodyne_sampled" and "/kitti_infos_val_sampled.pkl". Also loads the test pc folder is laoded
    sampler_info_path = config["train_input_reader"]["sampler_info_path"] # we changed the sampler_info_path before (so val data ist used and not training data)


    # Create a class which which creates new augmented examples of objects which are placed in the dataset
    sampler_class = DataBaseSamplerV2(sampler_info_path, config=config["train_input_reader"]) # the sampler 


    # create the dataLoader object which is reponsible for loading datapoints
    # contains not much logic and basically just holds some variables
    dataset_ori = dataLoader(training, sampler_class, config, sample_val_dataset_mode=sample_val_dataset_mode)


    # initializes the dataset object (batch creating etc.)
    dataset = dataset_ori.getIterator()

    # the program will exit earlier with "sys.exit()"
    # created files are "./kitti_infos_val_sampled.pkl" and files in "./testing/velodyne_sampled" folder


