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


# These import are needed RTX GPU'S ?:(
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import tensorflow_addons as tfa

# imports from tensorflow 
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate, Dropout, Input, MaxPooling2D, Softmax, Conv2D, Flatten, Dense, BatchNormalization
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50


class Scalar(tf.keras.Model):
    def __init__(self):
        super(Scalar, self).__init__()
        self.total = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32, name="total")
        self.count = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32, name="count")

    def call(self, scalar):
        if not tf.equal(scalar,0.0):
            self.count.assign_add(1)
            self.total.assign_add(tf.cast(scalar, tf.float32))
        return self.total / self.count


class Accuracy(tf.keras.Model):
    def __init__(self,
                config):

        super(Accuracy, self).__init__()

        self._encode_background_as_zeros = config["model"]["second"]["encode_background_as_zeros"]
        self._dim = -1
        self.ignore_idx=-1
        self.threshold=0.5
        self.total = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32, name="total")
        self.count = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32, name="count")


    def call(self, labels, preds, weights=None):

        if self._encode_background_as_zeros:
            scores = tf.math.sigmoid(preds)
            labels_pred = tf.argmax(preds, axis=self._dim) + 1
            pred_labels = tf.where(
                condition = tf.keras.backend.any((scores > self.threshold), axis=self._dim), 
                x=labels_pred, 
                y=tf.constant(0, labels_pred.dtype)) 
        else:   
            labels_pred = tf.argmax(preds, axis=self._dim)[1]
        N, *Ds = labels.shape
        labels = tf.reshape(labels,(N, int(np.prod(Ds))))
        pred_labels = tf.reshape(pred_labels,(N, int(np.prod(Ds))))
        if weights is None:
            weights = tf.cast((labels != self.ignore_idx),tf.float32)
        else:
            weights =  tf.cast(weights,tf.float32)
        num_examples = tf.reduce_sum(weights)
        num_examples = tf.cast((tf.clip_by_value(num_examples, clip_value_min=1.0, clip_value_max=1000000)),tf.float32)
        total = tf.reduce_sum(tf.cast((pred_labels == tf.cast(labels, tf.int64)),tf.float32))
        self.count.assign_add(num_examples)
        self.total.assign_add(total)
        return self.total / self.count


class PrecisionRecall(tf.keras.Model):
    def __init__(self,
                config):

        super(PrecisionRecall, self).__init__()

        self._ignore_idx = -1
        self._dim = -1
        self._thresholds = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        self._use_sigmoid_score = config["model"]["second"]["use_sigmoid_score"]
        self._encode_background_as_zeros = config["model"]["second"]["encode_background_as_zeros"]

        self.prec_total = tf.Variable(initial_value=tf.zeros(len(self._thresholds)), trainable=False, dtype=tf.float32, name="prec_total")
        self.prec_count = tf.Variable(initial_value=tf.zeros(len(self._thresholds)), trainable=False, dtype=tf.float32, name="prec_count")
        self.rec_total = tf.Variable(initial_value=tf.zeros(len(self._thresholds)), trainable=False, dtype=tf.float32, name="rec_total")
        self.rec_count = tf.Variable(initial_value=tf.zeros(len(self._thresholds)), trainable=False, dtype=tf.float32, name="rec_count")

    def call(self, labels, preds, weights=None):
        # labels: [N, ...]
        # preds: [N, ..., C]
        if self._encode_background_as_zeros:
            # this don't support softmax
            assert self._use_sigmoid_score is True
            total_scores = tf.math.sigmoid(preds)
            # scores, label_preds = torch.max(total_scores, dim=1)
        else:
            # Matthes: still pytorch
            if self._use_sigmoid_score:
                total_scores = torch.sigmoid(preds)[..., 1:]
            else:
                total_scores = F.softmax(preds, dim=-1)[..., 1:]
        
        scores = tf.math.reduce_max(total_scores, axis=-1)
        if weights is None:
            weights = tf.cast((labels != self._ignore_idx),tf.float32)
        else:
            weights =  tf.cast(weights,tf.float32)
        for i, thresh in enumerate(self._thresholds):
            tp, tn, fp, fn = _calc_binary_metrics(labels, scores, weights,
                                                  self._ignore_idx, thresh)
            rec_count = tp + fn
            prec_count = tp + fp
            if rec_count > 0:
                self.rec_count.scatter_nd_add([[i]],[rec_count])
                self.rec_total.scatter_nd_add([[i]],[tp])
            if prec_count > 0:
                self.prec_count.scatter_nd_add([[i]],[prec_count])
                self.prec_total.scatter_nd_add([[i]],[tp])


        
        prec_count = tf.clip_by_value(self.prec_count, 1.0,100000)
        rec_count = tf.clip_by_value(self.rec_count, 1.0,100000)
        return ((self.prec_total / prec_count),
                (self.rec_total / rec_count))


def _calc_binary_metrics(labels,
                         scores,
                         weights,
                         ignore_idx,
                         threshold):

    pred_labels = tf.cast(scores > threshold, tf.int64)
    N, *Ds = labels.shape
    labels = tf.reshape(labels, (N,int(np.prod(Ds))))
    pred_labels = tf.reshape(pred_labels, (N,int(np.prod(Ds))))
    pred_trues = pred_labels > 0
    pred_falses = pred_labels == 0
    trues = labels > 0
    falses = labels == 0
    true_positives = tf.reduce_sum(weights * tf.cast((trues & pred_trues),tf.float32))
    true_negatives = tf.reduce_sum(weights * tf.cast((falses & pred_falses),tf.float32))
    false_positives = tf.reduce_sum(weights * tf.cast((falses & pred_trues),tf.float32))
    false_negatives = tf.reduce_sum(weights * tf.cast((trues & pred_falses),tf.float32))
    return true_positives, true_negatives, false_positives, false_negatives


def update_metrics(config,
                cls_loss,
                loc_loss,
                cls_preds,
                labels,
                sampled,
                rpn_acc,
                rpn_metrics,
                rpn_cls_loss,
                rpn_loc_loss):

    num_class = config["model"]["second"]["num_class"]
    _encode_background_as_zeros = config["model"]["second"]["encode_background_as_zeros"]
    batch_size = cls_preds.shape[0]
    if not _encode_background_as_zeros:
        num_class += 1
    cls_preds = tf.reshape(cls_preds, (batch_size, -1, num_class))
    rpn_acc = rpn_acc(labels, cls_preds, sampled).numpy()
    prec, recall = rpn_metrics(labels, cls_preds, sampled)
    prec = prec.numpy()
    recall = recall.numpy()
    rpn_cls_loss = rpn_cls_loss(cls_loss).numpy()
    rpn_loc_loss = rpn_loc_loss(loc_loss).numpy()
    ret = {
        "cls_loss": float(rpn_cls_loss),
        "cls_loss_rt": float(cls_loss.numpy()),
        'loc_loss': float(rpn_loc_loss),
        "loc_loss_rt": float(loc_loss.numpy()),
        "rpn_acc": float(rpn_acc),
    }
    for i, thresh in enumerate(rpn_metrics._thresholds):
        ret[f"prec@{int(thresh*100)}"] = float(prec[i])
        ret[f"rec@{int(thresh*100)}"] = float(recall[i])

    return ret