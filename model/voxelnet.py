import numpy as np
import time


# These import are needed RTX GPU'S ?:(
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# imports from tensorflow 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate, Dropout, Input, MaxPooling2D, Softmax, Conv2D, Flatten, Dense, BatchNormalization
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50

# imports from own codebase
from model.pointpillars import PillarFeatureNet, PointPillarsScatter
from libraries.eval_helper_functions import second_box_decode, nms, tf_to_np_dtype, box_lidar_to_camera, project_to_image
from load_data import center_to_corner_box2d, corner_to_standup_nd_jit

import sys


# this function returns the direction targets
# which is a one hot vector
# its mainly done by checking if the tagets have a postitve rotation, after
# the baseline rotation of the anchor was added
# the direction target basically says that the rotation prediction should be positive, if
# the direcion target is 1 (1 = 0° to 180° & 0 = -180° to 0°).
# otherwise the rotation prediction must be rotated by +180°
# =========================================
def get_direction_target(anchors, reg_targets, one_hot_bool=True):
    batch_size = tf.shape(reg_targets)[0]
    anchors = tf.reshape(anchors,(batch_size, -1, 7))
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = tf.cast((rot_gt > 0),tf.int64)
    if one_hot_bool:
        dir_cls_targets = one_hot(
            dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets

def _get_pos_neg_loss(cls_loss, labels): 
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = tf.cast(tf.shape(cls_loss)[0],tf.float32)
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2: # here True
        
        cls_pos_loss = tf.cast((labels > 0), cls_loss.dtype) * tf.reshape(cls_loss,(batch_size, -1))
        cls_neg_loss = tf.cast((labels == 0), cls_loss.dtype) * tf.reshape(cls_loss,(batch_size, -1))
        cls_pos_loss = tf.reduce_sum(cls_pos_loss) / batch_size
        cls_neg_loss = tf.reduce_sum(cls_neg_loss) / batch_size
    else:
        cls_pos_loss = tf.math.reduce_sum(cls_loss[..., 1:]) / batch_size
        cls_neg_loss = tf.math.reduce_sum(cls_loss[..., 0]) / batch_size 
    return cls_pos_loss, cls_neg_loss

def add_sin_difference(boxes1, boxes2):
    # sin(a - b) = sina*cosb-cosa*sinb
    rad_pred_encoding = tf.sin(boxes1[..., -1:]) * tf.cos(boxes2[..., -1:])
    rad_tg_encoding = tf.cos(boxes1[..., -1:]) * tf.sin(boxes2[..., -1:])
    boxes1 = tf.concat([boxes1[..., :-1], rad_pred_encoding], axis=-1)
    boxes2 = tf.concat([boxes2[..., :-1], rad_tg_encoding], axis=-1)
    return boxes1, boxes2

def one_hot(tensor, depth, dim=-1, on_value=1.0, dtype=tf.float32):    
    return tf.one_hot(tensor,depth)

def create_loss(config,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                weighted_smooth_l1_localization_loss,
                batch_size):

    # ------------------------------------------------------------------------------------------------------
    # Get Variables
    # ------------------------------------------------------------------------------------------------------

    config = config
    num_class = config["num_class"]     
    encode_rad_error_by_sin = config["encode_rad_error_by_sin"] 
    encode_background_as_zeros = config["encode_background_as_zeros"] 
    box_code_size = 7 # [y,x,z,h,l,w,r]

    # ------------------------------------------------------------------------------------------------------
    # Reshape box_preds to have them in one list [batch_size, width, height, box_code_size*2] -> [batch_size, anchors, box_code_size]
    # ------------------------------------------------------------------------------------------------------

    box_preds = tf.reshape(box_preds,(batch_size, -1, box_code_size))

    # ------------------------------------------------------------------------------------------------------
    # Reshape cls_preds to have them in one list [batch_size, width, height, number of classes*2] -> [batch_size, anchors, number of classes]
    # ------------------------------------------------------------------------------------------------------

    if encode_background_as_zeros: # (here true)
        cls_preds = tf.reshape(cls_preds,(batch_size, -1, num_class))
    else:
        cls_preds = tf.reshape(cls_preds,(batch_size, -1, num_class + 1))
        
    cls_targets = tf.squeeze(cls_targets, axis=-1)

    # ------------------------------------------------------------------------------------------------------
    # reshape cls_targets that [0 OR 1] becomes to [0,1] OR [1,0]
    # ------------------------------------------------------------------------------------------------------

    one_hot_targets = one_hot( 
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    
    # ------------------------------------------------------------------------------------------------------
    # just take the object classes 
    # NOTE: if number of classes = 1 and encode_background_as_zeros = True we are getting left with cls_targets before one_hot presentation
    # ------------------------------------------------------------------------------------------------------
    
    if encode_background_as_zeros: # here true
        one_hot_targets = one_hot_targets[..., 1:]

    # ------------------------------------------------------------------------------------------------------
    # Here, the loss for rotation (sin(a - b)) is prepared so that it can be forwarded to the same L1 loss function 
    # as size and posititon (next in code)
    # Therefore, [preds(a)] and [targets(b)] are converted to [sina*cosb] and [cosa*sinb] because:
    # sin(a - b) = sina*cosb - cosa*sinb
    # ------------------------------------------------------------------------------------------------------

    if encode_rad_error_by_sin: # here true
        # sin(a - b) = sina*cosb-cosa*sinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)

    # ------------------------------------------------------------------------------------------------------
    # Get weighted_smooth_l1_localization_loss for all box_preds 
    # Note: With the use of reg_weights this only focusses on objects (loss for bg will be muliplied with 0)
    # Note: This loss does not differentiate between pred: 0.5, pred: 1.0, pred: 0.0 if label is positive
    # ------------------------------------------------------------------------------------------------------

    loc_losses = weighted_smooth_l1_localization_loss(
        box_preds, reg_targets, weights=reg_weights)  # [batch_size, n_anchors, n_locations]

    # ------------------------------------------------------------------------------------------------------
    # Get sigmoid_focal_classification_loss for all cls_preds
    # Note: With the use of cls_weights this only focusses on objects (not background)
    # ------------------------------------------------------------------------------------------------------

    cls_losses = sigmoid_focal_classification_loss(
        config["loss"]["classification_loss"]["weighted_sigmoid_focal"], cls_preds, one_hot_targets, weights=cls_weights) # [batch_size, n_anchors, n_classes]
        
    return loc_losses, cls_losses 

#@tf.function
def _softmax_cross_entropy_with_logits(logits, labels):

    # ------------------------------------------------------------------------------------------------------
    # Transpose the logits to the second postion:
    # [N, ..., C] -> [N, C, ...]
    # Does not really has an effect in this pipeline
    # ------------------------------------------------------------------------------------------------------

    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = tf.transpose(logits,perm=transpose_param) # [N, ..., C] -> [N, C, ...]

    # ------------------------------------------------------------------------------------------------------
    # Get loss per anchor
    # ------------------------------------------------------------------------------------------------------

    #logits and labels must be broadcastable
    loss = tf.nn.softmax_cross_entropy_with_logits(labels,logits) # [n_anchor]

    #loss = tf.nn.softmax_cross_entropy_with_logits(tf.reduce_max(labels,axis=-1),logits)[1]
    return loss

#@tf.function
def weighted_softmax_classification_loss(prediction_tensor, target_tensor, weights):
    """Softmax loss function."""

    """Constructor.

    Args:
      logit_scale: When this value is high, the prediction is "diffused" and
                   when this value is low, the prediction is made peakier.
                   (default 1.0)

    """

    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors]
        representing the value of the loss function.
    """

    logit_scale=1.0

    # ------------------------------------------------------------------------------------------------------
    # Get Num_classes: Here the 2 classes stand for the 2 directions to classify
    # ------------------------------------------------------------------------------------------------------

    num_classes = prediction_tensor.shape[-1]

    # ------------------------------------------------------------------------------------------------------
    # apply some scaling
    # ------------------------------------------------------------------------------------------------------

    prediction_tensor = tf.math.divide(
        prediction_tensor, logit_scale)

    # ------------------------------------------------------------------------------------------------------
    # Get Loss per anchor 
    # ------------------------------------------------------------------------------------------------------
    
    per_row_cross_ent = (_softmax_cross_entropy_with_logits( 
        labels=tf.reshape(target_tensor,(-1, num_classes)),
        logits=tf.reshape(prediction_tensor,(-1, num_classes)))) # [n_anchor]

    # ------------------------------------------------------------------------------------------------------
    # return reshaped loss [batch_size, n_anchor] and apply weights
    # ------------------------------------------------------------------------------------------------------
    
    return tf.reshape(per_row_cross_ent, tf.shape(weights)) * weights


#@tf.function
def _sigmoid_cross_entropy_with_logits(logits, labels):
    # if NANs occur here this bugfix may help:
    # https://github.com/traveller59/second.pytorch/issues/144
    # logits_max = tf.reduce_max(logits, 1, keepdims=True)
    # logits = logits - logits_max

    # ------------------------------------------------------------------------------------------------------
    # Create initial Loss:
    # abs(class predictions) - class predictions[ground truth]
    # This calc has the effect that negative predictions for gt>0 get punished but positive does not...
    # ... So how do we encourage positive labels to be even get even higher (appoaching 1)?
    # ------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------
    # Apply sigmoid_cross_entropy
    # Formuala here: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    # ------------------------------------------------------------------------------------------------------

    loss = tf.clip_by_value(logits, clip_value_min=0, clip_value_max=10000) - logits * tf.cast(labels,logits.dtype)

    loss += tf.math.log1p(tf.exp(-tf.math.abs(logits)))
 
    return loss
    
#@tf.function
def sigmoid_focal_classification_loss(config,
                                        prediction_tensor,
                                        target_tensor,
                                        weights,
                                        class_indices=None):
    
    """Sigmoid focal cross entropy loss.

    Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """
    """ Args:
      gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
      alpha: optional alpha weighting factor to balance positives vs negatives.
      all_zero_negative: bool. if True, will treat all zero as background.
        else, will treat first label as background. only affect alpha.
    """
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """

    # ------------------------------------------------------------------------------------------------------
    # get Params
    # ------------------------------------------------------------------------------------------------------

    config = config
    alpha = config["alpha"] # here 0,25
    gamma = config["gamma"] # here 2.0

    weights = tf.expand_dims(weights, axis=2) 

    if class_indices is not None: # is None here
        # not used
        weights *= indices_to_dense_vector(class_indices,
                prediction_tensor.shape[2]).view(1, 1, -1).type_as(prediction_tensor)

    # ------------------------------------------------------------------------------------------------------
    # Get the sigmoid_cross_entropy loss per anchor (-log(pt))
    # is the cross entropy (CE) from the paper "https://arxiv.org/pdf/1708.02002.pdf" or second part in 
    # pointpillars formlula Lcls
    # ------------------------------------------------------------------------------------------------------
    
    per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))

    # ------------------------------------------------------------------------------------------------------
    # sigmoid for predictions
    # ------------------------------------------------------------------------------------------------------

    prediction_probabilities = tf.math.sigmoid(prediction_tensor) 

    # ------------------------------------------------------------------------------------------------------
    # Calc modulating_factor
    # 1. Calc p_t from the paper "https://arxiv.org/pdf/1708.02002.pdf"
    # ------------------------------------------------------------------------------------------------------

    p_t = ((target_tensor * prediction_probabilities) +
           ((1 - target_tensor) * (1 - prediction_probabilities))) 

    # ------------------------------------------------------------------------------------------------------
    # Calc modulating_factor (1-p_t)^gamma
    # 2. Calc pt from the paper "https://arxiv.org/pdf/1708.02002.pdf"
    # ------------------------------------------------------------------------------------------------------
    modulating_factor = 1.0
    if gamma:
      modulating_factor = tf.math.pow(1.0 - p_t, gamma)

    # ------------------------------------------------------------------------------------------------------
    # Calc alpha_weight_factor
    # I dont know why they do this since it seems to give the background class an extra loss (*0.75) compared to 
    # the main class (*0.25)
    # ------------------------------------------------------------------------------------------------------

    alpha_weight_factor = 1.0
    if alpha is not None:
      alpha_weight_factor = (target_tensor * alpha +
                              (1 - target_tensor) * (1 - alpha))

    # ------------------------------------------------------------------------------------------------------
    # Multiply modulating_factor, alpha_weight_factor and per_entry_cross_ent
    # ------------------------------------------------------------------------------------------------------

    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)

    # ------------------------------------------------------------------------------------------------------
    # return classification loss 
    # ------------------------------------------------------------------------------------------------------

    return focal_cross_entropy_loss * weights

#class weighted_smooth_l1_localization_loss(config, prediction_tensor, target_tensor, weights=None):
class WeightedSmoothL1LocalizationLoss(tf.keras.Model):
    """Smooth L1 localization loss function.

    The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    """Compute loss function.

    Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the (encoded) predicted locations of objects.
        target_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the regression targets
        weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
        loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """

    def __init__(self, config):
        super(WeightedSmoothL1LocalizationLoss, self).__init__()

        # ------------------------------------------------------------------------------------------------------
        # get Params
        # ------------------------------------------------------------------------------------------------------

        config = config
        self.sigma = config["model"]["second"]["loss"]["localization_loss"]["weighted_smooth_l1"]["sigma"]
        code_weights = config["model"]["second"]["loss"]["localization_loss"]["weighted_smooth_l1"]["code_weight"]

        # ------------------------------------------------------------------------------------------------------
        # init code_weights # trainable set to false since currently not good for accuracy
        # ------------------------------------------------------------------------------------------------------

        self.code_weights = tf.Variable(initial_value=code_weights,trainable=False, name="code_weights")
        
        
    def call(self, prediction_tensor, target_tensor, weights=None):

        # ------------------------------------------------------------------------------------------------------
        # simple difference between prediction and annotation
        # ------------------------------------------------------------------------------------------------------

        diff = prediction_tensor - target_tensor

        # ------------------------------------------------------------------------------------------------------
        # Multiply the code_weights (one per filter, for each anchor same) with the difference
        # TODO do this in the initialization
        # ------------------------------------------------------------------------------------------------------

        if self.code_weights is not None:
            code_weights = tf.cast(self.code_weights,prediction_tensor.dtype)
            diff = tf.reshape(code_weights,(1, 1, -1)) * diff

        # ------------------------------------------------------------------------------------------------------
        # Make diff only positive
        # ------------------------------------------------------------------------------------------------------

        abs_diff = tf.math.abs(diff)

        # ------------------------------------------------------------------------------------------------------
        # Apply Equation (3) (smooth L1)
        # ------------------------------------------------------------------------------------------------------

        abs_diff_lt_1 = tf.math.less_equal(abs_diff, 1 / (self.sigma**2))
        abs_diff_lt_1 = tf.cast(abs_diff_lt_1, abs_diff.dtype)
        loss = abs_diff_lt_1 * 0.5 * tf.math.pow(abs_diff * self.sigma, 2) \
        + (abs_diff - 0.5 / (self.sigma**2)) * (1. - abs_diff_lt_1)


        anchorwise_smooth_l1norm = loss

        # ------------------------------------------------------------------------------------------------------
        # Apply additional weights (per anchor) 
        # - loss for other classes than objects (eg. backround) are multiplied with 0
        # ------------------------------------------------------------------------------------------------------
        
        anchorwise_smooth_l1norm *= tf.expand_dims(weights, axis=-1)

        # ------------------------------------------------------------------------------------------------------
        # return loss
        # ------------------------------------------------------------------------------------------------------

        return anchorwise_smooth_l1norm

# Output:
# cls_weights: The classification weights for each anchor (0 for dont cares) [batch_size, n_anchors]
# reg_weights: The registration weights (0 for dont cares) [batch_size, n_anchors]
# cared: True if Object or Background, False if dont care [batch_size, n_anchors]
# Note: cls_weights and reg_weights are very similar if pos_cls_weight is 1.0: Than, reg_weights = cls_weights - negative_cls_weights
# =========================================
def prepare_loss_weights(config,
                         labels,
                         dtype):
    """
    This class creates the weights for classification and regression targets. 
    The classification targets (cls_weights) will have only weights for classes that are not dont_cares (inclusive background)
    The regression targets (reg_weights) will have only weights for classes without background and not dont_cares
    Also a boolean list will returned which indicates only is true for positions that are not dont_cares (inclusive background)
    """

    config = config
    pos_cls_weight = config["pos_class_weight"] 
    neg_cls_weight = config["neg_class_weight"] 
    loss_norm_type = config["loss_norm_type"] 

    # ------------------------------------------------------------------------------------------------------
    # Filter out the dont cares in labels to get "cared"
    # ------------------------------------------------------------------------------------------------------

    cared = labels >= 0 # [batch_size, n_anchors]

    # ------------------------------------------------------------------------------------------------------
    # Get cls_weights
    # We are just multiplying pos_class_weight and neg_class_weight with its corresponding entries in labels and sum that up again
    # ------------------------------------------------------------------------------------------------------

    # label = 1 is positive, 0 is negative, -1 is don't care (ignore)
    positives = labels > 0 # [batch_size, num_anchors] # shape=(2, 146816)
    negatives = labels == 0 # [batch_size, num_anchors] # shape=(2, 146816)
    positive_cls_weights = tf.cast(positives, tf.float32) * pos_cls_weight # shape=(2, 146816)
    negative_cls_weights = tf.cast(negatives, tf.float32) * neg_cls_weight # shape=(2, 146816)
    cls_weights = negative_cls_weights + positive_cls_weights # cls weights for all classes but dont_care # shape=(2, 146816)

    # ------------------------------------------------------------------------------------------------------
    # convert the weights of all classes but background into floats
    # ------------------------------------------------------------------------------------------------------

    reg_weights = tf.cast(positives, tf.float32) # shape=(2, 146816)

    # ------------------------------------------------------------------------------------------------------
    # We are normalizing the reg_weights & cls_weights with the number of postive items in labels (pos_normalizer) per batch
    # (Remember that number of positive labels is at least the number of ground truth boxes, see create_target_np.
    # Thats, why pos_normalizer has higher numbers than the real number of classes per sample) 
    # ------------------------------------------------------------------------------------------------------

    if loss_norm_type == "NormByNumPositives":  # is true here
        positives = tf.cast(positives, tf.float32)
        pos_normalizer = tf.math.reduce_sum(positives, axis=1, keepdims=True)
        reg_weights /= tf.clip_by_value(pos_normalizer, clip_value_min=1.0, clip_value_max=100000.0)
        cls_weights /= tf.clip_by_value(pos_normalizer, clip_value_min=1.0, clip_value_max=100000.0)

    return cls_weights, reg_weights, cared

# This model contains the params and layers for the RPN -> it is actually an SSD
# The RPN contains downsampling, upsampling and detection heads
# =========================================
class RPN(tf.keras.Model):
    def __init__(self, 
                config,
                num_input_filters,
                training):
        
        super(RPN,self).__init__()

        self.config = config

        # ------------------------------------------------------------------------------------------------------
        # params
        # ------------------------------------------------------------------------------------------------------

        # set the layers to training or eval mode
        self.training = training
        anchor_generator_stride = self.config["model"]["second"]["target_assigner"]["anchor_generators"]["anchor_generator_stride"]
        rotations = anchor_generator_stride["rotations"]
        sizes = anchor_generator_stride["sizes"]
        num_rot = len(rotations)
        num_size = tf.convert_to_tensor(np.array(sizes).reshape([-1, 3]).shape[0])
        self._num_anchor_per_loc = (num_rot * num_size) # rotations (2) * sizes (1) = 2 in this case
        num_class = self.config["model"]["second"]["num_class"] # num_class = model_cfg.num_class
        layer_nums = self.config["model"]["second"]["rpn"]["layer_nums"] #list(model_cfg.rpn.layer_nums) 
        layer_strides = self.config["model"]["second"]["rpn"]["layer_strides"]#list(model_cfg.rpn.layer_strides)
        num_filters = self.config["model"]["second"]["rpn"]["num_filters"]# list(model_cfg.rpn.num_filters)
        upsample_strides = self.config["model"]["second"]["rpn"]["upsample_strides"]#list(model_cfg.rpn.upsample_strides)
        num_upsample_filters = self.config["model"]["second"]["rpn"]["num_upsample_filters"]#list(model_cfg.rpn.num_upsample_filters)
        num_input_filters=num_input_filters
        encode_background_as_zeros = self.config["model"]["second"]["encode_background_as_zeros"]#model_cfg.encode_background_as_zeros
        self._use_direction_classifier = self.config["model"]["second"]["use_direction_classifier"]#model_cfg.use_direction_classifier
        use_groupnorm = self.config["model"]["second"]["rpn"]["use_groupnorm"]#model_cfg.rpn.use_groupnorm
        num_groups = self.config["model"]["second"]["rpn"]["num_groups"]#model_cfg.rpn.num_groups
        self.box_code_size = 7 # [y,x,z,h,l,w,r]
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])

        # ------------------------------------------------------------------------------------------------------
        # extra params
        # ------------------------------------------------------------------------------------------------------

        use_bias_Conv2d = False
        use_bias_ConvTranspose2d = False

        # ------------------------------------------------------------------------------------------------------
        # Block 1
        # ------------------------------------------------------------------------------------------------------

        self.block1 = tf.keras.Sequential(name="block1")
        self.block1.add(tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1))))
        self.block1.add(tf.keras.layers.Conv2D(num_filters[0],3,strides=layer_strides[0], use_bias=use_bias_Conv2d, kernel_initializer=keras.initializers.he_uniform(seed=None)))
        self.block1.add(tf.keras.layers.BatchNormalization(axis=-1, trainable=True))
        self.block1.add(tf.keras.layers.ReLU())

        for i in range(layer_nums[0]):
            #self.block1.add(tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1))))
            self.block1.add(tf.keras.layers.Conv2D(num_filters[0],3,use_bias=use_bias_Conv2d, kernel_initializer=keras.initializers.he_uniform(seed=None),padding="same"))
            self.block1.add(tf.keras.layers.BatchNormalization(axis=-1, trainable=True))
            self.block1.add(tf.keras.layers.ReLU())

        self.deconv1 = tf.keras.Sequential(name="deconv1")
        self.deconv1.add(tf.keras.layers.Conv2DTranspose(
                filters=num_upsample_filters[0],
                kernel_size=upsample_strides[0],
                strides=upsample_strides[0],
                kernel_initializer=keras.initializers.he_uniform(seed=None),
                use_bias=use_bias_ConvTranspose2d))
        self.deconv1.add(tf.keras.layers.BatchNormalization(axis=-1, trainable=True))
        self.deconv1.add(tf.keras.layers.ReLU())

        # ------------------------------------------------------------------------------------------------------
        # Block 2
        # ------------------------------------------------------------------------------------------------------

        self.block2 = tf.keras.Sequential(name="block2")
        self.block2.add(tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1))))
        self.block2.add(tf.keras.layers.Conv2D(num_filters[1],3,strides=layer_strides[1], use_bias=use_bias_Conv2d, kernel_initializer=keras.initializers.he_uniform(seed=None)))
        self.block2.add(tf.keras.layers.BatchNormalization(axis=-1, trainable=True))
        self.block2.add(tf.keras.layers.ReLU())

        for i in range(layer_nums[1]):
            #self.block2.add(tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1))))
            self.block2.add(tf.keras.layers.Conv2D(num_filters[1],3,use_bias=use_bias_Conv2d, kernel_initializer=keras.initializers.he_uniform(seed=None),padding="same"))
            self.block2.add(tf.keras.layers.BatchNormalization(axis=-1, trainable=True))
            self.block2.add(tf.keras.layers.ReLU())

        self.deconv2 = tf.keras.Sequential(name="deconv2")
        self.deconv2.add(tf.keras.layers.Conv2DTranspose(
                filters=num_upsample_filters[1],
                kernel_size=upsample_strides[1],
                strides=upsample_strides[1],
                kernel_initializer=keras.initializers.he_uniform(seed=None),
                use_bias=use_bias_ConvTranspose2d))
        self.deconv2.add(tf.keras.layers.BatchNormalization(axis=-1, trainable=True))
        self.deconv2.add(tf.keras.layers.ReLU())

        # ------------------------------------------------------------------------------------------------------
        # Block 3
        # ------------------------------------------------------------------------------------------------------

        self.block3 = tf.keras.Sequential(name="block3")
        self.block3.add(tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1))))
        self.block3.add(tf.keras.layers.Conv2D(num_filters[2],3,strides=layer_strides[2], use_bias=use_bias_Conv2d, kernel_initializer=keras.initializers.he_uniform(seed=None)))
        self.block3.add(tf.keras.layers.BatchNormalization(axis=-1, trainable=True))
        self.block3.add(tf.keras.layers.ReLU())

        for i in range(layer_nums[2]):
            #self.block3.add(tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1))))
            self.block3.add(tf.keras.layers.Conv2D(num_filters[2],3,use_bias=use_bias_Conv2d, kernel_initializer=keras.initializers.he_uniform(seed=None),padding="same"))
            self.block3.add(tf.keras.layers.BatchNormalization(axis=-1, trainable=True))
            self.block3.add(tf.keras.layers.ReLU())

        self.deconv3 = tf.keras.Sequential(name="deconv3")
        self.deconv3.add(tf.keras.layers.Conv2DTranspose(
                filters=num_upsample_filters[2],
                kernel_size=upsample_strides[2],
                strides=upsample_strides[2],
                kernel_initializer=keras.initializers.he_uniform(seed=None),
                use_bias=use_bias_ConvTranspose2d))
        self.deconv3.add(tf.keras.layers.BatchNormalization(axis=-1, trainable=True))
        self.deconv3.add(tf.keras.layers.ReLU())

        # ------------------------------------------------------------------------------------------------------
        # set the number of predicted classes depending on if we want an extra class for the background
        # ------------------------------------------------------------------------------------------------------

        if encode_background_as_zeros: # here true
            num_cls = self._num_anchor_per_loc * num_class # num_cls is here 2 (2*1) # one class for each anchor and there are two anchors per pillar
        else:   
            num_cls = self._num_anchor_per_loc * (num_class + 1)

        # ------------------------------------------------------------------------------------------------------
        # Set detection head for box 
        # filters: _num_anchor_per_loc (here 2) * box_code_size (here 7) = 14
        # ------------------------------------------------------------------------------------------------------

        self.conv_box = tf.keras.layers.Conv2D(filters=self._num_anchor_per_loc * self.box_code_size,kernel_size=1,kernel_initializer=keras.initializers.he_uniform(seed=None),strides=(1,1), name="conv_box",use_bias=True) # TODO use_bias should be true I think
        
        # ------------------------------------------------------------------------------------------------------
        # Set detection head for classification
        # filters: num_cls (here 2) 
        # ------------------------------------------------------------------------------------------------------

        self.conv_cls = tf.keras.layers.Conv2D(filters=num_cls,kernel_size=1,kernel_initializer=keras.initializers.he_uniform(seed=None),strides=(1,1), name="conv_cls",use_bias=True) # TODO use_bias should be true I think

        # ------------------------------------------------------------------------------------------------------
        # Set direction detection head 
        # filters: _num_anchor_per_loc (here 2) * 2 = 4 # accoring to second paper we need to detect the direction
        # ------------------------------------------------------------------------------------------------------

        if self._use_direction_classifier: # (here true)
            self.conv_dir_cls = tf.keras.layers.Conv2D(self._num_anchor_per_loc * 2,kernel_size=1,kernel_initializer=keras.initializers.he_uniform(seed=None),strides=(1,1), name="conv_dir_cls",use_bias=True)# TODO use_bias should be true I think

    # This calls the RPN 
    # input: [batch_size, filter/channel, width, height] 
    def call(self, inputs, bev=None):# TensorShape([1, 64, 248, 296]) 

        inputs = tf.transpose(inputs,(0,2,3,1)) 
        b1 = self.block1(inputs, training=self.training) # TensorShape([1, 248, 296, 64])


        b1_up = self.deconv1(b1, training=self.training) # TensorShape([1, 248, 296, 128])
        b2 = self.block2(b1, training=self.training) # shape=(1, 124, 148, 128) 
        b2_up = self.deconv2(b2, training=self.training) #shape=(1, 248, 296, 128) 
        b3 = self.block3(b2, training=self.training) # shape=(1, 62, 74, 256) 
        b3_up = self.deconv3(b3, training=self.training) # shape=(1, 248, 296, 128)
        b_conc = tf.concat([b1_up, b2_up, b3_up], axis=3) # shape=(1, 248, 296, 384) 

        box_preds = self.conv_box(b_conc, training=self.training) # shape=(1, 248, 296, 14) 
        cls_preds = self.conv_cls(b_conc, training=self.training) # shape=(1, 248, 296, 2) 

        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(b_conc) 
            ret_dict["dir_cls_preds"] = dir_cls_preds # TensorShape([1, 248, 296, 8])
        return ret_dict 

 # debug
current_milli_time = lambda: int(round(time.time() * 1000))

def sigmoid_array(x):                                        
    return 1 / (1 + np.exp(-x))

class VoxelNet(tf.keras.Model):

    def __init__(self, config, writer, training=True):
        super(VoxelNet, self).__init__()

        # ------------------------------------------------------------------------------------------------------
        # Set params
        # ------------------------------------------------------------------------------------------------------

        self.config = config
        self.training = training
        if training:
            self.batch_size = self.config["train_input_reader"]["batch_size"]
        else:
            self.batch_size = self.config["eval_input_reader"]["batch_size"]

        # predict params

        self.encode_background_as_zeros = self.config["model"]["second"]["encode_background_as_zeros"]
        self.use_direction_classifier = config["model"]["second"]["use_direction_classifier"]
        self.use_multi_class_nms = self.config["model"]["second"]["use_multi_class_nms"]
        self.nms_score_threshold = self.config["model"]["second"]["nms_score_threshold"]
        self.nms_pre_max_size = self.config["model"]["second"]["nms_pre_max_size"]
        self.nms_post_max_size = self.config["model"]["second"]["nms_post_max_size"]
        self.nms_iou_threshold = self.config["model"]["second"]["nms_iou_threshold"]
        self.num_class = self.config["model"]["second"]["num_class"]

        # eval params
        self.measure_time_extended = self.config["measure_time_extended"]
        
        #debug
        if self.measure_time_extended:
            self.t_nms_func_list = []
            self.t_voxel_features_list = []
            self.t_spatial_features_list = []
            self.t_rpn_list = []

        # ------------------------------------------------------------------------------------------------------
        # Create Loss Object
        # ------------------------------------------------------------------------------------------------------

        self.weighted_smooth_l1_localization_loss = WeightedSmoothL1LocalizationLoss(self.config)

        # ------------------------------------------------------------------------------------------------------
        # Create voxel_feature_extractor, middle_feature_extractor and rpn
        # ------------------------------------------------------------------------------------------------------

        self.voxel_feature_extractor = PillarFeatureNet(config, training)
        self.middle_feature_extractor = PointPillarsScatter(config, training)
        num_rpn_input_filters = self.middle_feature_extractor.nchannels


        self.rpn = RPN(
                config=config,
                num_input_filters=num_rpn_input_filters,
                training=training)
        
        # ------------------------------------------------------------------------------------------------------
        # Set NMS function
        # ------------------------------------------------------------------------------------------------------
        
        self.nms_func = nms
        
        # ------------------------------------------------------------------------------------------------------
        # Experiments with tflite
        # ------------------------------------------------------------------------------------------------------

        #tf.keras.models.save_model(model, filepath)

        # ------------------------------------------------------------------------------------------------------
        # laod the model
        # ------------------------------------------------------------------------------------------------------

        #self.rpn = keras.models.load_model("/home/makr/Desktop/tmp")
        
        # ------------------------------------------------------------------------------------------------------
        # convert model to tflite
        # ------------------------------------------------------------------------------------------------------

        # Convert the model
        # converter = tf.lite.TFLiteConverter.from_saved_model("/home/makr/Desktop/tmp") # path to the SavedModel directory
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # tflite_model = converter.convert()
        # # Save the model.
        # with open('/home/makr/Desktop/tmp/model_quantized.tflite', 'wb') as f:
        #     f.write(tflite_model)

        # ------------------------------------------------------------------------------------------------------
        # Load the TFLite model
        # ------------------------------------------------------------------------------------------------------

        # Load the TFLite model and allocate tensors.
        # self.rpn = tf.lite.Interpreter(model_path="/home/makr/Desktop/tmp/model_quantized.tflite")
        # self.rpn.allocate_tensors()
        # # Get input and output tensors.
        # self.rpn_output_details = self.rpn.get_output_details()

    # =========================================
    def print_profile_time(self,function_name, time):
        if function_name == "t_voxel_features":
            t_voxel_features = current_milli_time() - time
            self.t_voxel_features_list.append(t_voxel_features)
            if len(self.t_voxel_features_list) > 1:
                t_voxel_features_avg = round(sum(self.t_voxel_features_list[1:])/len(self.t_voxel_features_list[1:]),2)
                tf.print(f't_voxel_features: {t_voxel_features_avg}')
        if function_name == "t_spatial_features":
            t_spatial_features = current_milli_time() - time
            self.t_spatial_features_list.append(t_spatial_features)
            if len(self.t_spatial_features_list) > 1:
                t_spatial_features_avg = round(sum(self.t_spatial_features_list[1:])/len(self.t_spatial_features_list[1:]),2)
                tf.print(f't_spatial_features: {t_spatial_features_avg}')
        if function_name == "t_rpn":
            t_rpn = current_milli_time() - time
            self.t_rpn_list.append(t_rpn)
            if len(self.t_rpn_list) > 1:
                t_rpn_avg = round(sum(self.t_rpn_list[1:])/len(self.t_rpn_list[1:]),2)
                tf.print(f't_rpn: {t_rpn_avg}')
        if function_name == "t_nms_func":
            t_nms_func = current_milli_time() - t_nms_func
            self.t_nms_func_list.append(t_nms_func)
            if len(self.t_nms_func_list) > 1:
                t_nms_func_avg = round(sum(self.t_nms_func_list[1:])/len(self.t_nms_func_list[1:]),2)
                print(f't_nms_func: {t_nms_func_avg}')
    
    # =========================================
    def call(self, voxels,num_points,coors,batch_anchors,labels= None,reg_targets=None):

        # ------------------------------------------------------------------------------------------------------
        # set params
        # ------------------------------------------------------------------------------------------------------

        batch_size_dev = tf.shape(batch_anchors)[0] # = 2
        self.box_code_size = 7

        # ------------------------------------------------------------------------------------------------------
        # Apply PillarFeatureNet
        # Output: [voxels, filters]
        # ------------------------------------------------------------------------------------------------------

        # debug
        if self.measure_time_extended: t_voxel_features = current_milli_time()

        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)

        # debug
        if self.measure_time_extended: 
            self.print_profile_time("t_voxel_features",t_voxel_features)
            
        # ------------------------------------------------------------------------------------------------------
        # Apply Backbone / Middle feature Extractor
        # Output: [batch_size, filter/channel, width, height] 
        # ------------------------------------------------------------------------------------------------------

        # debug
        if self.measure_time_extended: t_spatial_features = current_milli_time()

        spatial_features = self.middle_feature_extractor(
                voxel_features, coors)

        # debug
        if self.measure_time_extended:
            self.print_profile_time("t_spatial_features",t_spatial_features) 
      

        # ------------------------------------------------------------------------------------------------------
        # Apply RPN
        # Outputs: 
        # preds_dict: [batch_size, width, height, filter(14)] 
        # cls_preds: [batch_size, width, height, filter(2)] 
        # dir_cls_preds: [batch_size, width, height, filter(4)] 
        # preds_dict: [batch_size, width, height, filter] 
        # ------------------------------------------------------------------------------------------------------

        if self.measure_time_extended: t_rpn = current_milli_time()
       
        preds_dict = self.rpn(spatial_features)

        if self.measure_time_extended: 
            self.print_profile_time("t_rpn",t_rpn) 
         
        # ------------------------------------------------------------------------------------------------------
        # use tflite rpn model instead
        # ------------------------------------------------------------------------------------------------------

        # self.rpn.set_tensor(0, spatial_features)
        # self.rpn.invoke()
        # preds_dict = {
        #     "box_preds" : self.rpn.get_tensor(self.rpn_output_details[2]['index']),
        #     "cls_preds" : self.rpn.get_tensor(self.rpn_output_details[0]['index']),
        #     "dir_cls_preds" : self.rpn.get_tensor(self.rpn_output_details[1]['index'])
        # }

        
        # ------------------------------------------------------------------------------------------------------
        # if we are in training mode -> calc loss
        # ------------------------------------------------------------------------------------------------------

        if self.training:

            box_preds = preds_dict["box_preds"] # shape=(2, 248, 296, 14)
            cls_preds = preds_dict["cls_preds"] # shape=(2, 248, 296, 2)

            # ------------------------------------------------------------------------------------------------------
            # Get cls_weights, reg_weights, cared
            # ------------------------------------------------------------------------------------------------------

            cls_weights, reg_weights, cared = prepare_loss_weights(
                config=self.config["model"]["second"], labels=labels, dtype=voxels.dtype)

            # ------------------------------------------------------------------------------------------------------
            # Get cls_targets by filter the labels with cared (basically filter out non_cares(is -1 in the gt annotations))
            # ------------------------------------------------------------------------------------------------------

            cls_targets = labels * tf.cast(cared, labels.dtype)

            # ------------------------------------------------------------------------------------------------------
            # Add additional dim to cls_targets WHY?
            # ------------------------------------------------------------------------------------------------------

            cls_targets = tf.expand_dims(cls_targets, axis=-1)
            
            # ------------------------------------------------------------------------------------------------------
            # Get losses for Location and Classification
            # loc_loss: [batch_size, anchors, n_filter]
            # cls_loss: [batch_size, anchors, n_classes]
            # ------------------------------------------------------------------------------------------------------

            loc_loss, cls_loss = create_loss(
                self.config["model"]["second"],
                box_preds=box_preds,
                cls_preds=cls_preds,
                cls_targets=cls_targets,
                cls_weights=cls_weights,
                reg_targets=reg_targets,
                reg_weights=reg_weights,
                weighted_smooth_l1_localization_loss = self.weighted_smooth_l1_localization_loss,
                batch_size = self.batch_size
            )

            # ------------------------------------------------------------------------------------------------------
            # LOCATION LOSS # Sum Up all location losses and normalize
            # ------------------------------------------------------------------------------------------------------

            loc_loss_reduced = tf.math.reduce_sum(loc_loss) / self.batch_size 
            loc_loss_reduced *= self.config["model"]["second"]["loss"]["localization_weight"] 

            # ------------------------------------------------------------------------------------------------------
            # Split negative and positive anchor classifications and sum them up
            # ------------------------------------------------------------------------------------------------------

            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)  # Not used, just for debugging

            # ------------------------------------------------------------------------------------------------------
            # CLASSIFICATION LOSS # Sum Up all classification losses, normalize and apply weight
            # ------------------------------------------------------------------------------------------------------
            
            cls_loss_reduced = tf.math.reduce_sum(cls_loss) / self.batch_size
            cls_loss_reduced *= self.config["model"]["second"]["loss"]["classification_weight"]

            # ------------------------------------------------------------------------------------------------------
            # Sum up both losses
            # ------------------------------------------------------------------------------------------------------

            loss = loc_loss_reduced + cls_loss_reduced

            # ------------------------------------------------------------------------------------------------------
            # calculate direction classification loss
            # ------------------------------------------------------------------------------------------------------

            if self.config["model"]["second"]["use_direction_classifier"]: # here true
                
                # ------------------------------------------------------------------------------------------------------
                # Get direction targets as one hot
                # ------------------------------------------------------------------------------------------------------

                dir_targets = get_direction_target(batch_anchors, reg_targets)
                
                # ------------------------------------------------------------------------------------------------------
                # reshape direction predictions
                # ------------------------------------------------------------------------------------------------------

                dir_logits = tf.reshape(preds_dict["dir_cls_preds"],(batch_size_dev, -1, 2))

                # ------------------------------------------------------------------------------------------------------
                # Create Weights for direction Classifcation by:
                # 1. initialize weights with 1 where anchor have object labels
                # ------------------------------------------------------------------------------------------------------

                weights = tf.cast((labels > 0),dir_logits.dtype)

                # ------------------------------------------------------------------------------------------------------
                # 2. Normalize by the amount ob objects per batch
                # ------------------------------------------------------------------------------------------------------

                weights /= tf.clip_by_value(tf.math.reduce_sum(weights, -1, keepdims=True), clip_value_min=1.0, clip_value_max=9999999.0)

                # ------------------------------------------------------------------------------------------------------
                # Get direction classificaion loss
                # ------------------------------------------------------------------------------------------------------

                dir_loss = weighted_softmax_classification_loss(
                    dir_logits, dir_targets, weights=weights)

                # ------------------------------------------------------------------------------------------------------
                # DIRECTION LOSS # Sum Up all dir losses, normalize and apply weight
                # ------------------------------------------------------------------------------------------------------

                dir_loss = tf.math.reduce_sum(dir_loss) / tf.cast(batch_size_dev,tf.float32)
                dir_loss *= self.config["model"]["second"]["direction_loss_weight"]
                loss = loss + dir_loss

            # retrun loss for optimizer. Note, that just the "loss" is used for optimization
            
            return {
                "loss": loss, # Sum of Location, Classification and direction classificaion loss [float]
                "cls_loss": cls_loss, # classification losses [batch_size, n_anchor, n_classes(here 1)]
                "loc_loss": loc_loss, # classification losses [batch_size, n_anchor, n_locations(here 7)]
                "cls_pos_loss": cls_pos_loss, # Sum of positive classification losses [float]
                "cls_neg_loss": cls_neg_loss, # Sum of negative classification losses [float]
                "cls_preds": cls_preds, # classification predictions [batch_size, width, height, n_classes(here 1)]
                "dir_loss_reduced": dir_loss, # Sum of direction classification losses [float]
                "cls_loss_reduced": cls_loss_reduced, # Sum of classification losses [float]
                "loc_loss_reduced": loc_loss_reduced, # Sum of location losses [float]
                "cared": cared, # Filter for non_cares(-1) [batch_size, n_anchor]
            }
            
        # ------------------------------------------------------------------------------------------------------
        # if we are in eval mode -> do not calc loss
        # ------------------------------------------------------------------------------------------------------
        
        else:
            return preds_dict 

   
    # =========================================
    def predict(self, example, preds_dict):
        #[1:num_points, 2:coordinates, 3:rect, 4:Trv2c, 5:P2, 6:anchors, 7:anchors_mask, 8:image_idx, 9:image_shape]

        # ------------------------------------------------------------------------------------------------------
        # convert params outputs to numpy
        # ------------------------------------------------------------------------------------------------------
        
        batch_anchors = example[6].numpy()
        batch_size = batch_anchors.shape[0]
        batch_rect = example[3].numpy()
        batch_Trv2c = example[4].numpy()
        batch_P2 = example[5].numpy()
        batch_anchors_mask = example[7].numpy()
        batch_imgidx = example[8].numpy()
        num_class_with_bg = self.num_class
        if not self.encode_background_as_zeros: # (here false)
            num_class_with_bg = self.config["num_class"] + 1

        # ------------------------------------------------------------------------------------------------------
        # convert network outputs to numpy
        # ------------------------------------------------------------------------------------------------------

        batch_box_preds = preds_dict["box_preds"].numpy()
        batch_cls_preds = preds_dict["cls_preds"].numpy()
        batch_dir_preds = preds_dict["dir_cls_preds"].numpy()
        
        # ------------------------------------------------------------------------------------------------------
        # reshape network outputs that everything so that the anchor dim is flattened 
        # ------------------------------------------------------------------------------------------------------
        
        batch_box_preds = np.reshape(batch_box_preds,(batch_size, -1,self.box_code_size))
        batch_cls_preds = np.reshape(batch_cls_preds,(batch_size, -1,num_class_with_bg))

        if self.use_direction_classifier: # (here true)
            batch_dir_preds = np.reshape(batch_dir_preds, (batch_size, -1, 2))
        else:
            batch_dir_preds = [None] * batch_size
        
        # emty holder for final predictions
        predictions_dicts = []

        # ------------------------------------------------------------------------------------------------------
        # iterate over each pointcloud in batch
        # ------------------------------------------------------------------------------------------------------
        
        for batch_idx in range(batch_size): 

            # ------------------------------------------------------------------------------------------------------
            # get prediction of this single pointcloud
            # ------------------------------------------------------------------------------------------------------
            
            box_preds=batch_box_preds[batch_idx,...]
            anchors=batch_anchors[batch_idx,...]
            cls_preds=batch_cls_preds[batch_idx,...]
            dir_preds=batch_dir_preds[batch_idx,...]
            rect=batch_rect[batch_idx,...]
            Trv2c=batch_Trv2c[batch_idx,...]
            P2=batch_P2[batch_idx,...]
            img_idx=batch_imgidx[batch_idx,...]
            a_mask=batch_anchors_mask[batch_idx,...]
            a_mask_as_indices = np.where(a_mask == 1)[0]

            # ------------------------------------------------------------------------------------------------------
            # Mask box_preds and cls_preds with feature map mask from dataloader
            # ------------------------------------------------------------------------------------------------------

            if a_mask is not None:
                box_preds = box_preds[[a_mask_as_indices]]
                cls_preds = cls_preds[[a_mask_as_indices]]
                anchors = anchors[[a_mask_as_indices]]

            # ------------------------------------------------------------------------------------------------------
            # Mask dir_preds with feature map mask from dataloader
            # ------------------------------------------------------------------------------------------------------

            if self.use_direction_classifier: # (here true)
                if a_mask is not None:
                    dir_preds = dir_preds[[a_mask_as_indices]]
                    
                # ------------------------------------------------------------------------------------------------------
                # Use max to get direction predictions
                # ------------------------------------------------------------------------------------------------------
            
                dir_labels = np.argmax(dir_preds, axis=-1)

            # ------------------------------------------------------------------------------------------------------
            # Use Sigmoid to get probabilities for classes
            # ------------------------------------------------------------------------------------------------------

            if self.encode_background_as_zeros: # (here true)
                total_scores = np.apply_along_axis(sigmoid_array, 0, cls_preds)

            else: # TODO convert to numpy 
                # encode background as first element in one-hot vector
                if self.config["model"]["second"]["use_sigmoid_score"]:
                    total_scores = tf.math.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = tf.nn.softmax(cls_preds, axis=-1)[..., 1:]
            
            

            # ------------------------------------------------------------------------------------------------------
            # Set Placeholder for boxes
            # ------------------------------------------------------------------------------------------------------

            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            if self.use_multi_class_nms: # (here false)
                pass
            else:

                # ------------------------------------------------------------------------------------------------------
                # bring top_scores to correct shape and create top_labels placeholder
                # ------------------------------------------------------------------------------------------------------
                
                if num_class_with_bg == 1: # (here true)
                    top_scores = np.squeeze(total_scores,axis=-1)
                    top_labels = np.zeros(
                        total_scores.shape[0],
                        dtype=int)
                else: # TODO convert to numpy 
                    top_scores = tf.math.reduce_max(total_scores, axis=-1)
                    top_labels = tf.argmax(total_scores, axis=-1)


                # ------------------------------------------------------------------------------------------------------
                # Filter out all top_scores that are below nms_score_threshold
                # I replaced it with the succeeding paragraph 
                # ------------------------------------------------------------------------------------------------------

                # if self.nms_score_threshold > 0.0: # (here false) # TODO convert to numpy 
                #     thresh = tf.constant(
                #         [self.config["model"]["second"]["nms_score_threshold"]],
                #         dtype=total_scores.dtype)
                #     top_scores_keep = (top_scores >= thresh)
                #     top_scores = tf.boolean_mask(top_scores,top_scores_keep)
                #     box_preds = box_preds[top_scores_keep]
                #     if self.config["model"]["second"]["use_direction_classifier"]:
                #         dir_labels = dir_labels[top_scores_keep]
                #     top_labels = top_labels[top_scores_keep]

                # ------------------------------------------------------------------------------------------------------
                # only keep the top n score predictions
                # TODO why Im doin this again? wouldnt it be better to filter as much as possible?
                # ------------------------------------------------------------------------------------------------------

                top_n_scores = np.argpartition(top_scores, -100)[-100:]
                top_scores = top_scores[[top_n_scores]]
                box_preds = box_preds[[top_n_scores]]
                anchors = anchors[[top_n_scores]]
                if self.use_direction_classifier:
                    dir_labels = dir_labels[[top_n_scores]]
                top_labels = top_labels[[top_n_scores]]

                # ------------------------------------------------------------------------------------------------------
                # If there are top_scores left
                # ------------------------------------------------------------------------------------------------------

                if top_scores.shape[0] != 0:

                    # ------------------------------------------------------------------------------------------------------
                    # convers network outputs for location to lidar coords
                    # ------------------------------------------------------------------------------------------------------

                    
                    box_preds = second_box_decode(box_preds,anchors)

                    # ------------------------------------------------------------------------------------------------------
                    # We take everything from the location predictions but z and height to calc nms
                    # ------------------------------------------------------------------------------------------------------

                    boxes_for_nms = box_preds[...,[0, 1, 3, 4, 6]]
                    
                    # ------------------------------------------------------------------------------------------------------
                    # calculate x,y corners (4) of predictions based on x,y,w,l,r predictions
                    # ------------------------------------------------------------------------------------------------------
                    
                    box_preds_corners = center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4]) # [predictions, 4, 2]

                    # ------------------------------------------------------------------------------------------------------
                    # Calculate maximum x and y and minimum x and y for every predicted bb (in other words left upper and
                    # lower right corner)
                    # ------------------------------------------------------------------------------------------------------

                    boxes_for_nms = corner_to_standup_nd_jit(
                        box_preds_corners) # [predictions,4]

                    # ------------------------------------------------------------------------------------------------------
                    # Apply nms (removes overlapping boxes with "nms_iou_threshold")
                    # Shouldnt I filter out all 0 labels (background) for speed improvent?
                    # Answer: No, since 0 is not background but the label of class 1 instead
                    # ------------------------------------------------------------------------------------------------------

                    if self.measure_time_extended: t_nms_func = current_milli_time()

                    selected = self.nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self.nms_pre_max_size,
                        post_max_size=self.nms_post_max_size,
                        iou_threshold=self.nms_iou_threshold
                    )
                    
                    if self.measure_time_extended: 
                        self.print_profile_time("t_nms_func",t_nms_func) 
                     
                # ------------------------------------------------------------------------------------------------------
                # If there are no top_scores left return None
                # ------------------------------------------------------------------------------------------------------

                else:
                    selected = None

                # ------------------------------------------------------------------------------------------------------
                # If nms returned predicted bb get the acording box_preds, dir_labels, top_labels, top_scores
                # ------------------------------------------------------------------------------------------------------

                if selected is not None:
                    selected_boxes = box_preds[selected]
                    if self.use_direction_classifier:
                        selected_dir_labels = dir_labels[selected]
                    selected_labels = top_labels[selected]
                    selected_scores = top_scores[selected]
            
            # ------------------------------------------------------------------------------------------------------
            # finally generate predictions
            # ------------------------------------------------------------------------------------------------------

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                # print(scores)
                if self.use_direction_classifier:
                    dir_labels = selected_dir_labels

                    # ------------------------------------------------------------------------------------------------------
                    # When the prediction says rotation is > 0 XOR direction classification is 1 we add pi to the rotation
                    # This is according to SECOND paper
                    # ------------------------------------------------------------------------------------------------------

                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels>0
                    box_preds = box_preds
                    box_preds[..., -1] += np.where(
                        opp_labels,
                        np.pi,
                        0.0)

                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds

                # ------------------------------------------------------------------------------------------------------
                # convert predictions to camera coordinates
                # ------------------------------------------------------------------------------------------------------
                
                final_box_preds = final_box_preds
                rect = rect
                Trv2c = Trv2c


                final_box_preds_camera = box_lidar_to_camera(
                        final_box_preds, rect, Trv2c)
                
                # ------------------------------------------------------------------------------------------------------
                # extract single predictions
                # ------------------------------------------------------------------------------------------------------

                locs = final_box_preds_camera[:, :3]
                dims = final_box_preds_camera[:, 3:6]
                angles = final_box_preds_camera[:, 6]

                # # ------------------------------------------------------------------------------------------------------
                # # get the 8 3D corners (x,y,z) of the bboxes in relation to the camera
                # # TODO: This is not important 
                # # ------------------------------------------------------------------------------------------------------

                # camera_box_origin = [0.5, 1.0, 0.5]
                # box_corners = center_to_corner_box3d(
                #     locs, dims, angles, camera_box_origin, axis=1)

                # # ------------------------------------------------------------------------------------------------------
                # # get the 8 2D corners (x,y) of the bboxes projected to the image plane
                # # TODO: This is not important 
                # # ------------------------------------------------------------------------------------------------------

                # box_corners_in_image = project_to_image(
                #     box_corners, P2) # [N, 8, 2]

                # # ------------------------------------------------------------------------------------------------------
                # # get the 2 2D corners (x,y) of the bboxes projected to the image plane
                # # ------------------------------------------------------------------------------------------------------

                box_2d_preds_fake = []

                for i in angles:
                    box_2d_preds_fake.append([400., 200., 500., 400.])

                predictions_dict = {
                    "bbox": np.array(box_2d_preds_fake),
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "batch_idx": img_idx,
                }

            else:
                predictions_dict = {
                    "bbox": None,
                    "box3d_camera": None,
                    "box3d_lidar": None,
                    "scores": None,
                    "label_preds": None,
                    "batch_idx": img_idx,
                }

            # ------------------------------------------------------------------------------------------------------
            # Append predictions for this pointcloud to the predictions_dicts 
            # ------------------------------------------------------------------------------------------------------
            predictions_dicts.append(predictions_dict)

        # ------------------------------------------------------------------------------------------------------
        # return predictions 
        # ------------------------------------------------------------------------------------------------------

        return predictions_dicts