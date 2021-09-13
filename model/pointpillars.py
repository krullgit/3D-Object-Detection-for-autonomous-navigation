import numpy as np

# These import are needed RTX GPU'S ?:(
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# imports from tensorflow 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate, Dropout, Input, MaxPooling2D, Softmax, Conv2D, Flatten, Dense, BatchNormalization, ReLU
from tensorflow import keras

#  ███████╗    ██╗   ██╗    ███╗   ██╗     ██████╗    ████████╗    ██╗     ██████╗     ███╗   ██╗    ███████╗
#  ██╔════╝    ██║   ██║    ████╗  ██║    ██╔════╝    ╚══██╔══╝    ██║    ██╔═══██╗    ████╗  ██║    ██╔════╝
#  █████╗      ██║   ██║    ██╔██╗ ██║    ██║            ██║       ██║    ██║   ██║    ██╔██╗ ██║    ███████╗
#  ██╔══╝      ██║   ██║    ██║╚██╗██║    ██║            ██║       ██║    ██║   ██║    ██║╚██╗██║    ╚════██║
#  ██║         ╚██████╔╝    ██║ ╚████║    ╚██████╗       ██║       ██║    ╚██████╔╝    ██║ ╚████║    ███████║
#  ╚═╝          ╚═════╝     ╚═╝  ╚═══╝     ╚═════╝       ╚═╝       ╚═╝     ╚═════╝     ╚═╝  ╚═══╝    ╚══════╝

def get_paddings_indicator(actual_num, max_num):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    axis=0

    actual_num = tf.expand_dims(actual_num, axis = axis + 1)

    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(tf.shape(actual_num))
    max_num_shape[axis + 1] = -1
    max_num = tf.range(
        max_num, dtype=tf.int32)

    max_num = tf.reshape(max_num, max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = tf.cast(actual_num, tf.int32) > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


#  ██████╗      ██████╗     ██╗    ███╗   ██╗    ████████╗              ██████╗     ██╗    ██╗         ██╗          █████╗     ██████╗     ███████╗
#  ██╔══██╗    ██╔═══██╗    ██║    ████╗  ██║    ╚══██╔══╝              ██╔══██╗    ██║    ██║         ██║         ██╔══██╗    ██╔══██╗    ██╔════╝
#  ██████╔╝    ██║   ██║    ██║    ██╔██╗ ██║       ██║       █████╗    ██████╔╝    ██║    ██║         ██║         ███████║    ██████╔╝    ███████╗
#  ██╔═══╝     ██║   ██║    ██║    ██║╚██╗██║       ██║       ╚════╝    ██╔═══╝     ██║    ██║         ██║         ██╔══██║    ██╔══██╗    ╚════██║
#  ██║         ╚██████╔╝    ██║    ██║ ╚████║       ██║                 ██║         ██║    ███████╗    ███████╗    ██║  ██║    ██║  ██║    ███████║
#  ╚═╝          ╚═════╝     ╚═╝    ╚═╝  ╚═══╝       ╚═╝                 ╚═╝         ╚═╝    ╚══════╝    ╚══════╝    ╚═╝  ╚═╝    ╚═╝  ╚═╝    ╚══════╝
#                                                                                                                                                  

# voxel_feature_extractor according to voxelnet
# The PillarFeatureNet takes the augmented voxelized pointcloud [voxels, points, point features] and 
# encodes the point features to num_filters with one linear layer per point
# Output: [voxels, filters]
# =========================================
class PillarFeatureNet(tf.keras.Model):
    def __init__(self, config, training):

        super(PillarFeatureNet,self).__init__()

        self.config = config
        self.training = training
        num_point_features = self.config["model"]["second"]["num_point_features"]
        num_filters = self.config["model"]["second"]["voxel_feature_extractor"]["num_filters"] 
        self.with_distance = self.config["model"]["second"]["voxel_feature_extractor"]["with_distance"]
        voxel_size = self.config["model"]["second"]["voxel_generator"]["voxel_size"]
        point_cloud_range = self.config["model"]["second"]["voxel_generator"]["point_cloud_range"] 
        #self.training = True

        # ------------------------------------------------------------------------------------------------------
        # increase num_point_features since we apply feature augmentation according to the paper (3(mean points) + 2 (mean voxels) new features)
        # ------------------------------------------------------------------------------------------------------

        num_point_features += 5

        # not used
        if self.with_distance: # is false here
            num_point_features += 1

        # not used
        # num_filters = [num_point_features] + list(num_filters) # num_point_features:9 num_filters:64 // =[9,64]

        # ------------------------------------------------------------------------------------------------------
        # Create PillarFeatureNet
        # Input tensor: [voxels, points, point features]
        # ------------------------------------------------------------------------------------------------------

        self.pfn_layer = tf.keras.Sequential()

        # ------------------------------------------------------------------------------------------------------
        # A linear layer is automatically applied to the second last dim (points).
        # That means for every point there is applied a linear layer [features -> filters]
        # ------------------------------------------------------------------------------------------------------
        self.pfn_layer.add(tf.keras.layers.Dense(num_filters, use_bias=False, kernel_initializer=keras.initializers.he_uniform(seed=None)))
        
        # ------------------------------------------------------------------------------------------------------
        # BN is applied on the last layer (filters)
        # ------------------------------------------------------------------------------------------------------

        self.pfn_layer.add(tf.keras.layers.BatchNormalization(trainable=True, axis=-1, name="batch",epsilon=1e-3, momentum=0.01))
        
        # ------------------------------------------------------------------------------------------------------
        # Relu according to paper
        # ------------------------------------------------------------------------------------------------------

        self.pfn_layer.add(tf.keras.layers.ReLU())

        # ------------------------------------------------------------------------------------------------------
        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        # ------------------------------------------------------------------------------------------------------
        
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]


    # @tf.function(input_signature = [tf.TensorSpec(shape=[None,100,4], dtype=tf.float32),tf.TensorSpec(shape=[None,], dtype=tf.int32),tf.TensorSpec(shape=[None,4], dtype=tf.int32)])
    def call(self, voxels, num_points, coors):
        '''
        Args:
            voxels: [n_voxels,n_points_in_voxel,[x/y/z]] # x/y/z are in lidar space
            coors: [n_voxel, b1, b2, w, h] coords in feature map space # b* are booleans for the belonging batch # # w/h are in lidar space
        '''

        # ------------------------------------------------------------------------------------------------------
        # Augments the 4 input voxels (x,y,z) to "arithmetic mean of all points in the pillar" and "offset from the pillar x, y center"
        # ------------------------------------------------------------------------------------------------------ 

        # ------------------------------------------------------------------------------------------------------
        # Find mean (x/y/z) of each voxel (depending on its points)
        # ------------------------------------------------------------------------------------------------------
    
        points_mean = tf.reduce_sum(voxels[:, :, :3], axis=1, keepdims=True) / tf.reshape(tf.cast(num_points, voxels.dtype), (-1, 1, 1))

        # ------------------------------------------------------------------------------------------------------
        # augments voxels to the distance to the point means
        # ------------------------------------------------------------------------------------------------------
   
        f_cluster = voxels[:, :, :3] - points_mean

        # ------------------------------------------------------------------------------------------------------
        # Find mean of each voxel in 2D (x/y)) (depending on its points)
        # In tf super hard to do, thats why it looks a little messy, should be fast anyways though
        # ------------------------------------------------------------------------------------------------------

        f_center = tf.zeros(tf.shape(voxels[:, :, :2]))
        a = voxels[:, :, 0]
        b = tf.cast(coors[:, 3], tf.float32) # take the x coords
        c = tf.expand_dims(b, axis=1)
        d = c * self.vx # from position in voxel count to position in actual distance
        e = d + self.x_offset 
        # e: now we have the centers of voxels globally in normal metrics 
        f_center = f_center+(tf.expand_dims((a-e),-1)@[[1,0]]) # we calculate the difference between the global coords and all voxel centers to get coords associated to local voxel mean
        #f_center[:, :, 0].assign(voxels[:, :, 0] - (tf.expand_dims((tf.cast(coors[:, 3], tf.float32)), axis=1) * self.vx + self.x_offset))
        a,b,c,d,e = None,None,None,None,None
        a = voxels[:, :, 1]
        b = tf.cast(coors[:, 2], tf.float32)
        c = tf.expand_dims(b, axis=1)
        d = c * self.vy
        e = d + self.y_offset

        # ------------------------------------------------------------------------------------------------------
        # augments voxels to the distance to the voxel means
        # ------------------------------------------------------------------------------------------------------

        f_center = f_center+(tf.expand_dims((a-e),-1)@[[0,1]])
        
        # ------------------------------------------------------------------------------------------------------
        # Combine voxels
        # ------------------------------------------------------------------------------------------------------
        
        voxels_ls = [voxels, f_cluster, f_center]

        # not used:
        if self.with_distance: # is false here
            points_dist = tf.norm(voxels[:, :, :3], ord=2, axis=2, keepdims=True)
            #points_dist = torch.norm(voxels[:, :, :3], 2, 2, keepdim=True)
            voxels_ls.append(points_dist)
                
        # ------------------------------------------------------------------------------------------------------
        # Concatenate voxels
        # Difference to paper is that reflectance is missing, hence only 8 voxels
        # ------------------------------------------------------------------------------------------------------

        voxels = tf.concat(voxels_ls, axis=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = tf.shape(voxels)[1]
        mask = get_paddings_indicator(num_points, voxel_count)
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.cast(mask, dtype=voxels.dtype)
        voxels *= mask

        # ------------------------------------------------------------------------------------------------------
        # Apply PillarFeatureNet
        # voxels: [voxels, point, point features]
        # return: [voxels, point, filters]
        # ------------------------------------------------------------------------------------------------------

        voxels = self.pfn_layer(voxels, training=self.training)

        # ------------------------------------------------------------------------------------------------------
        # Get max filter accross all points
        # This deletes the points dimension
        # Output: [voxels, points(len=1), filters]
        # ------------------------------------------------------------------------------------------------------

        voxels = tf.math.reduce_max(voxels, axis=1, keepdims=True) # shape [1,64]
        
        # ------------------------------------------------------------------------------------------------------
        # delete the points dimension since len is only 1
        # ------------------------------------------------------------------------------------------------------

        return tf.squeeze(voxels)

# middle_feature_extractor according to voxelnet
# Takes Input and scatters it to the intended feature_map_size.
# This works since for every voxel in the input we also have coords which include
# the position in the original feature_map and its inner_batch number
# Input: [voxels, filters]
# Output: [batch_size, filter/channel, width, height] 
# =========================================


 # debug
import time
current_milli_time = lambda: int(round(time.time() * 1000))

class PointPillarsScatter(tf.keras.Model):
    def __init__(self, config, training):

        super(PointPillarsScatter,self).__init__()

        # ------------------------------------------------------------------------------------------------------
        # params
        # ------------------------------------------------------------------------------------------------------

        self.config = config
        vfe_num_filters = self.config["model"]["second"]["voxel_feature_extractor"]["num_filters"] 
        voxel_size = np.array(self.config["model"]["second"]["voxel_generator"]["voxel_size"])
        point_cloud_range = np.array(self.config["model"]["second"]["voxel_generator"]["point_cloud_range"])
        
        self.nchannels = vfe_num_filters
        if training:
            self.batch_size = self.config["train_input_reader"]["batch_size"]
        else:
            self.batch_size = self.config["eval_input_reader"]["batch_size"]

        # ------------------------------------------------------------------------------------------------------
        # calculate the grid size, same as in load_data.py
        # ------------------------------------------------------------------------------------------------------
        
        grid_size = (
                point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        # ------------------------------------------------------------------------------------------------------
        # create output shape [?,?, width, height, filter]
        # ------------------------------------------------------------------------------------------------------
        dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters]
        output_shape = dense_shape

        self.ny = output_shape[2]
        self.nx = output_shape[3]


    # Input:
    # voxel_features: [voxels,filters]
    # coors: [voxels, element_in_batch_idx + coords in feature map space] 
    # Output: 
    # [batch_size, filter/channel, width, height] 
    # =========================================
    #@tf.function(input_signature = [tf.TensorSpec(shape=[None,64], dtype=tf.float32),tf.TensorSpec(shape=[None,4], dtype=tf.int32)])
    def call(self, voxel_features, coords):


        # batch_canvas will be the final output.
        batch_canvas = []

        # iterate over the batches
        for batch_itt in range(self.batch_size):
            
            # ------------------------------------------------------------------------------------------------------
            # Create the canvas for this sample
            # ------------------------------------------------------------------------------------------------------

            canvas = tf.zeros((self.nchannels, self.nx * self.ny), dtype=voxel_features.dtype)
            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = tf.boolean_mask(coords,batch_mask)
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = tf.cast(indices, tf.int64)
            #Shapes (128,) and (1,) are incompatible

            voxels = tf.boolean_mask(voxel_features,batch_mask)
            voxels = tf.transpose(voxels, perm=[1, 0])  

            # ------------------------------------------------------------------------------------------------------
            # Now scatter the blob back to the canvas.
            # the next lines are doing this: canvas[:, indices] = voxels # here are more lines of code since tf has no build in function for this
            # ------------------------------------------------------------------------------------------------------

            indices3 = tf.cast(tf.expand_dims(indices,axis=-1), dtype=tf.int64); 
            updates3 = tf.transpose(voxels)
            shape3 = tf.constant([self.nx * self.ny, self.nchannels], dtype=tf.int64); 
            scatter3 = tf.scatter_nd(indices3, updates3, shape3)
            canvas = tf.transpose(scatter3)

            # indices = tf.expand_dims(indices,axis=-1)
            # indices = tf.linalg.matmul(indices, tf.constant([[0,1]],dtype=tf.int64))
            # indices_extended = [self.nchannels,tf.shape(indices)[0],2] # self.nchannels, self.nx * self.ny, 2
            # indices_extended = tf.zeros(indices_extended, dtype=tf.int32) # self.nchannels, self.nx * self.ny, 2
            # for i in range(indices_extended.shape[0]): indices_extended = tf.concat([indices_extended,[indices +[i,0]]],axis=-3)
            # canvas = tf.scatter_nd(indices_extended[self.nchannels:], voxels, [self.nchannels, self.nx * self.ny])

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = tf.stack(batch_canvas, axis=0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = tf.reshape(batch_canvas, (self.batch_size, self.nchannels, self.ny, self.nx))

        # ------------------------------------------------------------------------------------------------------
        # return feature map
        # [batch_size, filter/channel, width, height] 
        # ------------------------------------------------------------------------------------------------------

        return batch_canvas