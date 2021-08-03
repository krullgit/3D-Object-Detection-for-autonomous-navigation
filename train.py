#import from general python libraries
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import math
import pickle
import os
import datetime
import fire
import shutil
import yaml
import json
import time
import tensorboard
import wandb
# from wandb.keras import WandbCallback


import gc

# disable GPU
# also search for "# Matthes GPU"
# import os
# import tensorflow as tf
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

# These import are needed RTX GPU'S?
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

import tensorflow_addons as tfa

# imports from tensorflow 
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate, Dropout, Input, MaxPooling2D, Softmax, Conv2D, Flatten, Dense, BatchNormalization
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50

# imports from own codebase
from load_data import dataLoader, DataBaseSamplerV2
# metrics is currently not used

from libraries.eval_helper_functions import predict_kitti_to_anno, progressBar, send_3d_bbox, box_camera_to_lidar, remove_low_score
from libraries.train_helper_functions import create_out_dir_base, create_model_dirs_training, create_model_dirs_eval, create_wandb_config, log_wandb_loss, log_wandb_eval
from model.voxelnet import VoxelNet
from second.utils.eval import get_official_eval_result, get_coco_eval_result

# ros
import rospy
import std_msgs
from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_recognition_msgs.msg import BoundingBox
from scipy.spatial.transform import Rotation as R

def set_trainable(net, trainable):
    
    # PillarFeatureNet
    net.layers[1].layers[0].layers[0].trainable = trainable
    net.layers[1].layers[0].layers[1].trainable = trainable
    net.layers[1].layers[0].layers[2].trainable = trainable
    # PointPillarsScatter
    
    # RPN
    net.layers[3].layers[0].layers[0].trainable = trainable
    net.layers[3].layers[0].layers[1].trainable = trainable
    net.layers[3].layers[0].layers[2].trainable = trainable
    net.layers[3].layers[0].layers[3].trainable = trainable
    net.layers[3].layers[0].layers[4].trainable = trainable
    net.layers[3].layers[0].layers[5].trainable = trainable
    net.layers[3].layers[0].layers[6].trainable = trainable
    net.layers[3].layers[0].layers[7].trainable = trainable
    net.layers[3].layers[0].layers[8].trainable = trainable
    net.layers[3].layers[0].layers[9].trainable = trainable
    
    net.layers[3].layers[2].layers[0].trainable = trainable
    net.layers[3].layers[2].layers[1].trainable = trainable
    net.layers[3].layers[2].layers[2].trainable = trainable
    net.layers[3].layers[2].layers[3].trainable = trainable
    net.layers[3].layers[2].layers[4].trainable = trainable
    net.layers[3].layers[2].layers[5].trainable = trainable
    net.layers[3].layers[2].layers[6].trainable = trainable
    net.layers[3].layers[2].layers[7].trainable = trainable
    net.layers[3].layers[2].layers[8].trainable = trainable
    net.layers[3].layers[2].layers[9].trainable = trainable
    
    net.layers[3].layers[4].layers[0].trainable = trainable
    net.layers[3].layers[4].layers[1].trainable = trainable
    net.layers[3].layers[4].layers[2].trainable = trainable
    net.layers[3].layers[4].layers[3].trainable = trainable
    net.layers[3].layers[4].layers[4].trainable = trainable
    net.layers[3].layers[4].layers[5].trainable = trainable
    net.layers[3].layers[4].layers[6].trainable = trainable
    net.layers[3].layers[4].layers[7].trainable = trainable
    net.layers[3].layers[4].layers[8].trainable = trainable
    net.layers[3].layers[4].layers[9].trainable = trainable
    
    return net


#  ████████╗    ██████╗      █████╗     ██╗    ███╗   ██╗    
#  ╚══██╔══╝    ██╔══██╗    ██╔══██╗    ██║    ████╗  ██║    
#     ██║       ██████╔╝    ███████║    ██║    ██╔██╗ ██║    
#     ██║       ██╔══██╗    ██╔══██║    ██║    ██║╚██╗██║    
#     ██║       ██║  ██║    ██║  ██║    ██║    ██║ ╚████║    
#     ╚═╝       ╚═╝  ╚═╝    ╚═╝  ╚═╝    ╚═╝    ╚═╝  ╚═══╝        


# Training Function
# =========================================
def train(config_path):

    # ------------------------------------------------------------------------------------------------------ 
    #  load the config from file and set Variables
    # ------------------------------------------------------------------------------------------------------ 

    # load the config from file
    with open(config_path) as f1:    # load the config file 
        config = yaml.load(f1, Loader=yaml.FullLoader)

    # ------------------------------------------------------------------------------------------------------ 
    #  Load directory parameter and create directories
    # ------------------------------------------------------------------------------------------------------  

    project_dir_base = config["project_dir_base"] # path to base project dir (where the whole code is stored)                                                                                
    training = True # we are in training mode?
    #load_checkpoints = config["#"] # we continue old learning process?
    # This is the ID of the current training. So it must match a previous 
    # training to "load_checkpoints" (if True) or if "load_checkpoints" is False the model_id 
    # is automatically increased in "create_out_dir_base" until a not existing directory is found.
    model_id = config["model_id"]

    
    # path to out base dir (where the training logs are stored)
    out_dir_base, model_id = create_out_dir_base(project_dir_base, training, model_id)
    

    # create the subdir where the training is stored
    out_dir_logs, out_dir_images, out_dir_train_images, out_dir_checkpoints = create_model_dirs_training(out_dir_base)


    # make a copy (for logging) of the config file  
    shutil.copyfile(config_path, out_dir_base + "/train.yaml")
   

    # ------------------------------------------------------------------------------------------------------
    #  Load dataloader parameter and create dataloader
    # ------------------------------------------------------------------------------------------------------ 

    batch_size = config["train_input_reader"]["batch_size"]
    sampler_info_path = config["train_input_reader"]["sampler_info_path"]
    num_point_features = config["train_input_reader"]["num_point_features"]
    do_evaluate = config["do_evaluate"]


    # Create a class which which creates new augmented examples of objects which are placed in the dataset
    sampler_class = DataBaseSamplerV2(sampler_info_path, config["train_input_reader"])


    # create the dataLoader object which is reponsible for loading datapoints
    # contains not much logic and basically just holds some variables
    dataset_ori = dataLoader(training, sampler_class, config)


    # initializes the dataset object (batch creating etc.)
    dataset = dataset_ori.getIterator()


    # makes the dataset object iterable
    data_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)


    print("**********************************************")
    print("* (1) Dataset Size: {}".format(str(dataset_ori.ndata)))
    print("* (2) Batch Size: {}".format(str(batch_size)))
    print("* Batches per Epoch ( 1 / 2 ): {}".format(str(int(dataset_ori.ndata/batch_size))))
    print("**********************************************")


    # ------------------------------------------------------------------------------------------------------
    #  Setup Logging with wandb and tensorboard
    # ------------------------------------------------------------------------------------------------------ 

    # setup wandb
    wandb_config=create_wandb_config(config)
    wandb.init(project='thesis', entity='makr06', config=wandb_config, name=model_id+"")

    # setup tensorbaord
    # writer = tf.summary.create_file_writer(out_dir_logs)
    writer = None # I use wandb instead of tensorboard now
    
    # ------------------------------------------------------------------------------------------------------
    #  Create network
    # ------------------------------------------------------------------------------------------------------ 
    
    net = VoxelNet(config, writer)



    
    
    
        
    # ------------------------------------------------------------------------------------------------------
    #  Create Optimizer
    # ------------------------------------------------------------------------------------------------------ 

    if list(config["train_config"]["optimizer"].keys())[0] == 'adam_optimizer':

        if list(config["train_config"]["optimizer"]["adam_optimizer"]["learning_rate"].keys())[0] == "exponential_decay_learning_rate":

            # create learning rate decay scheduler
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = config["train_config"]["optimizer"]["adam_optimizer"]["learning_rate"]["exponential_decay_learning_rate"]["initial_learning_rate"],
                decay_steps =           config["train_config"]["optimizer"]["adam_optimizer"]["learning_rate"]["exponential_decay_learning_rate"]["decay_steps"]/batch_size,
                decay_rate =            config["train_config"]["optimizer"]["adam_optimizer"]["learning_rate"]["exponential_decay_learning_rate"]["decay_factor"],
                staircase =             config["train_config"]["optimizer"]["adam_optimizer"]["learning_rate"]["exponential_decay_learning_rate"]["staircase"])
            
            # create the optimizer with weight decay and learning rate decay
            optimizer = tfa.optimizers.AdamW(
                learning_rate =         lr_schedule, 
                weight_decay =          config["train_config"]["optimizer"]["adam_optimizer"]["weight_decay"],
                epsilon=                1e-08
                )
    
    # DEBUG PARAMS
    log_weights_and_bias = True
    from_file_mode = False # True will repeat the element form test_batch_in_file
    take_first = False # True will repeat the first element form data_iterator
    example_first = None # do not change
    
    load_weights = config["load_weights"]
    load_weights_finished = False # This variable is part of a tensorflow workaround caused by subclassing keras.model
    if from_file_mode:
        with open("test_batch_in_file", "rb") as file:
            data_iterator = pickle.load(file)
            
    
    # ------------------------------------------------------------------------------------------------------
    #  We wrap a training step in a tf.function to to gain speedups (5x)
    # ------------------------------------------------------------------------------------------------------ 
    
    max_number_of_points_per_voxel = config["model"]["second"]["voxel_generator"]["max_number_of_points_per_voxel"]
    
    @tf.function(input_signature = [tf.TensorSpec(shape=[None,max_number_of_points_per_voxel,num_point_features], dtype=tf.float32),tf.TensorSpec(shape=[None,], dtype=tf.int32),tf.TensorSpec(shape=[None,4], dtype=tf.int32),tf.TensorSpec(shape=[None,None,7], dtype=tf.float32),tf.TensorSpec(shape=[None,None], dtype=tf.int32),tf.TensorSpec(shape=[None,None,7], dtype=tf.float32)])
    def trainStep(voxels,num_points,coors,batch_anchors,labels,reg_targets):
        with tf.GradientTape() as g:
            
            # Pass to Network
            ret_dict = net(voxels,num_points,coors,batch_anchors,labels,reg_targets)

            # Get loss for optimization
            loss = ret_dict["loss"]
            
            # Get other returns from network that might be interesting for debugging
            cls_loss = ret_dict["cls_loss"]
            loc_loss = ret_dict["loc_loss"]
            cls_pos_loss = ret_dict["cls_pos_loss"]
            cls_neg_loss = ret_dict["cls_neg_loss"]
            cls_preds = ret_dict["cls_preds"]
            dir_loss_reduced = ret_dict["dir_loss_reduced"]
            cls_loss_reduced = ret_dict["cls_loss_reduced"]
            loc_loss_reduced = ret_dict["loc_loss_reduced"]
            cared = ret_dict["cared"] 
 
        #Compute gradients
        gradients = g.gradient(loss, net.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.NONE) 
        
        # ------------------------------------------------------------------------------------------------------
        #  Apply Gradient Clipping to avoid exploding Gradients
        # ------------------------------------------------------------------------------------------------------

        #gradients = [tf.clip_by_value(grad, -0.1, +0.1) for grad in gradients]
        # gradients = [tf.clip_by_norm(grad, 1) for grad in gradients]
        
        # ------------------------------------------------------------------------------------------------------
        #  Backpropagation
        # ------------------------------------------------------------------------------------------------------

        #Update the weights of the model.
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))

        # return losses etc.
        return (cls_preds, loss, cls_loss_reduced, loc_loss_reduced, cared, loc_loss, dir_loss_reduced)

    # ------------------------------------------------------------------------------------------------------
    #  EPOCH ITERATOR Settings
    # ------------------------------------------------------------------------------------------------------   

    step_current = 0 # defines how many batches have been processed

    epochs_total = config["epochs_total"] # defines how many epochs are processed in total

    _total_step_time = 0.0

    best_eval_score = 0.0

    # ------------------------------------------------------------------------------------------------------
    #  EPOCH ITERATOR
    # ------------------------------------------------------------------------------------------------------ 
    
    for epoch_idx in range(epochs_total):

        # ------------------------------------------------------------------------------------------------------
        #  DATASET ITERATOR
        # ------------------------------------------------------------------------------------------------------    

        # Example: [0:voxels, 1:num_points, 2:coordinates, 3:rect, 4:Trv2c, 5:P2, 6:anchors, 7:anchors_mask, 8:labels, 9:reg_targets, 10:reg_weights, 11:image_idx, 12:image_shape]
        
        print("**********************************************")
        print("* New Epoch: {} , LR: {}".format(str(epoch_idx),str(lr_schedule(step_current).numpy())))
        print("**********************************************")

        for example in iter(dataset):
            
            # we set timestamp 1 to measure train time per batch
            t = time.time()
            
            # DEBUG
            if take_first:
                example_first = example
                take_first = False
                data_iterator = range(100)
            elif example_first != None:
                example = example_first
            if from_file_mode:
                example = list(example.values())
                example = [tf.convert_to_tensor(x) for x in example]
            
            # ------------------------------------------------------------------------------------------------------
            #  Load Model Weights and Optimizer Weights (Checkpoint)
            # ------------------------------------------------------------------------------------------------------

            if load_weights == True and load_weights_finished == False and step_current > 0: # step_current > 0  since otherwise optimizer weights will not load
                    
                # Run the Model 1 time to initiate everything
                net(example[0],example[1],example[2],example[6],example[8],example[9])
                load_weights_finished = True
                
                # load weights
                net.load_weights("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_102/out_dir_checkpoints/model_weights_21.h5")

                # load optimizer
                with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/out/model_102/out_dir_checkpoints/optimizer_weights_21.h5", "rb") as file: optimizer.set_weights(pickle.load(file))

                # freeze layer for transfer learning
                set_trainable(net, False)
            # ------------------------------------------------------------------------------------------------------
            #  Run Model and create Loss
            # ------------------------------------------------------------------------------------------------------
            
            cls_preds, loss, cls_loss_reduced, loc_loss_reduced, cared, loc_loss, dir_loss_reduced = trainStep(example[0],example[1],example[2],example[6],example[8],example[9])

            # ------------------------------------------------------------------------------------------------------
            # log the progress of the training
            # ------------------------------------------------------------------------------------------------------

            print_learning_rate = str(lr_schedule(step_current).numpy())
            if step_current % 200 == 0:
                print_steps_left = str(-1*step_current+int((dataset_ori.ndata/batch_size)*(epoch_idx+1)))
                print("**********************************************")
                print("* LOSS: {} Step/left: {}/{} Epoch/total: {}/{} LR: {}".format(str(loss.numpy()),str(step_current),print_steps_left,str(epoch_idx),str(epochs_total),print_learning_rate))            
                print("**********************************************")
                
            if log_weights_and_bias:
                log_wandb_loss(step_current, wandb,epoch_idx,loc_loss_reduced,cls_loss_reduced,dir_loss_reduced,loss,print_learning_rate)  

            if step_current is not 0: # skip step 1 since it takes always longer
                _total_step_time += time.time() - t 

            # Update step
            step_current += 1


        # ------------------------------------------------------------------------------------------------------
        # END EPOCH
        # ------------------------------------------------------------------------------------------------------
        # Save Model Weights and Optimizer Weights (Checkpoint)
        # ------------------------------------------------------------------------------------------------------ 

        if do_evaluate:

            # save weights temporarily to check first the eval
            if load_weights: set_trainable(net, True)
            net.save_weights(out_dir_checkpoints+"/model_weights_temp.h5")
            if load_weights: set_trainable(net, False)

            # eval weights
            mAP3d, result, mAPbev, mAPaos = evaluate(config_path, model_id, epoch_idx=str(epoch_idx))
            
            # ------------------------------------------------------------------------------------------------------
            # log eval result
            # ------------------------------------------------------------------------------------------------------
            
            # get avg score
            eval_score = (mAP3d[0][0].sum()+mAPaos[0][0].sum()+mAPbev[0][0].sum())/18
            
            if log_weights_and_bias:
                log_wandb_eval(wandb,mAP3d,mAPaos,mAPbev,eval_score)


            print("**********************************************")
            print("* Old Score: {} New Score: {}".format(str(best_eval_score),str(eval_score)))
            print("**********************************************")

            # save Model weights if eval_score is better than last best score
            if eval_score > best_eval_score:

                print("**********************************************")
                print("* Save Model Weights and Optimizer weights")
                print("**********************************************")

                if load_weights: set_trainable(net, True)
                net.save_weights(out_dir_checkpoints+"/model_weights_{}.h5".format(str(epoch_idx)))
                if load_weights: set_trainable(net, False)

                # update best eval score
                best_eval_score = eval_score

            # save eval results
            with open(out_dir_checkpoints+"/model_result_{}.txt".format(str(epoch_idx)), 'w') as file:
                file.write(result)
        
        # ------------------------------------------------------------------------------------------------------
        # DEBUG: always save weights, no matter if good or not
        # ------------------------------------------------------------------------------------------------------
        
        if load_weights: set_trainable(net, True)
        # save Optimizer weights TODO
        with open(out_dir_checkpoints+"/optimizer_weights_{}.h5".format(str(epoch_idx)), "wb") as file: 
            pickle.dump(optimizer.get_weights(), file)
            
        # just save the last model indepeneding of its performance out of curiosity
        net.save_weights(out_dir_checkpoints+"/model_weights_{}.h5".format(str(epoch_idx)))
        if load_weights: set_trainable(net, False)



# avg forward time per example: 0.160
# avg postprocess time per example: 0.011


# old data
# Pedestrian AP@0.50, 0.50, 0.50:
# bbox AP:0.01, 0.01, 0.01
# bev  AP:52.74, 52.74, 52.74
# 3d   AP:29.79, 29.79, 29.79
# aos  AP:0.01, 0.01, 0.01
# Pedestrian AP@0.50, 0.25, 0.25:
# bbox AP:0.01, 0.01, 0.01
# bev  AP:84.13, 84.13, 84.13
# 3d   AP:83.41, 83.41, 83.41
# aos  AP:0.01, 0.01, 0.01

# Pedestrian coco AP@0.25:0.05:0.70:
# bbox AP:0.35, 0.35, 0.35
# bev  AP:52.32, 52.32, 52.32
# 3d   AP:40.27, 40.27, 40.27
# aos  AP:0.20, 0.20, 0.20


# new data batcg_s 4
# Pedestrian AP@0.50, 0.50, 0.50:
# bbox AP:0.00, 0.00, 0.00
# bev  AP:77.83, 77.83, 77.83
# 3d   AP:59.23, 59.23, 59.23
# aos  AP:0.00, 0.00, 0.00
# Pedestrian AP@0.50, 0.25, 0.25:
# bbox AP:0.00, 0.00, 0.00
# bev  AP:89.45, 89.45, 89.45
# 3d   AP:89.43, 89.43, 89.43
# aos  AP:0.00, 0.00, 0.00

# Pedestrian coco AP@0.25:0.05:0.70:
# bbox AP:0.04, 0.04, 0.04
# bev  AP:66.65, 66.65, 66.65
# 3d   AP:57.23, 57.23, 57.23
# aos  AP:0.02, 0.02, 0.02

# new data batch_s 2
# Pedestrian AP@0.50, 0.50, 0.50:
# bbox AP:0.00, 0.00, 0.00
# bev  AP:77.04, 77.04, 77.04
# 3d   AP:38.31, 38.31, 38.31
# aos  AP:0.00, 0.00, 0.00
# Pedestrian AP@0.50, 0.25, 0.25:
# bbox AP:0.00, 0.00, 0.00
# bev  AP:88.86, 88.86, 88.86
# 3d   AP:88.83, 88.83, 88.83
# aos  AP:0.00, 0.00, 0.00

# Pedestrian coco AP@0.25:0.05:0.70:
# bbox AP:0.01, 0.01, 0.01
# bev  AP:62.16, 62.16, 62.16
# 3d   AP:47.56, 47.56, 47.56
# aos  AP:0.01, 0.01, 0.01


# Pedestrian AP@0.50, 0.50, 0.50:
# bbox AP:0.00, 0.00, 0.00
# bev  AP:77.08, 77.08, 77.08
# 3d   AP:10.89, 10.89, 10.89
# aos  AP:0.00, 0.00, 0.00
# Pedestrian AP@0.50, 0.25, 0.25:
# bbox AP:0.00, 0.00, 0.00
# bev  AP:90.05, 90.05, 90.05
# 3d   AP:82.20, 82.20, 82.20
# aos  AP:0.00, 0.00, 0.00

# Pedestrian coco AP@0.25:0.05:0.70:
# bbox AP:1.09, 1.09, 1.09
# bev  AP:62.58, 62.58, 62.58
# 3d   AP:26.49, 26.49, 26.49
# aos  AP:0.66, 0.66, 0.66


# Pedestrian AP@0.50, 0.50, 0.50:
# bbox AP:0.00, 0.00, 0.00
# bev  AP:79.00, 79.00, 79.00
# 3d   AP:63.81, 63.81, 63.81
# aos  AP:0.00, 0.00, 0.00
# Pedestrian AP@0.50, 0.25, 0.25:
# bbox AP:0.00, 0.00, 0.00
# bev  AP:89.90, 89.90, 89.90
# 3d   AP:89.90, 89.90, 89.90
# aos  AP:0.00, 0.00, 0.00

# Pedestrian coco AP@0.25:0.05:0.70:
# bbox AP:0.01, 0.01, 0.01
# bev  AP:69.27, 69.27, 69.27
# 3d   AP:59.02, 59.02, 59.02
# aos  AP:0.01, 0.01, 0.01


# Pedestrian AP@0.50, 0.50, 0.50:
# bev  AP:63.70, 63.70, 63.70
# 3d   AP:37.41, 37.41, 37.41
# aos  AP:38.73, 38.73, 38.73
# Pedestrian AP@0.50, 0.25, 0.25:
# bev  AP:95.29, 95.29, 95.29
# 3d   AP:89.91, 89.91, 89.91
# aos  AP:54.50, 54.50, 54.50


# bev  AP:79.17, 79.17, 79.17
# 3d   AP:41.01, 41.01, 41.01
# aos  AP:53.68, 53.68, 53.68
# Pedestrian AP@0.50, 0.25, 0.25:
# bev  AP:93.63, 93.63, 93.63
# 3d   AP:93.63, 93.63, 93.63
# aos  AP:63.87, 63.87, 63.87

# Pedestrian AP@0.50, 0.50, 0.50:
# bev  AP:82.75, 82.75, 82.75
# 3d   AP:48.25, 48.25, 48.25
# aos  AP:40.66, 40.66, 40.66
# Pedestrian AP@0.50, 0.25, 0.25:
# bev  AP:95.54, 95.54, 95.54
# 3d   AP:95.51, 95.51, 95.51
# aos  AP:46.76, 46.76, 46.76

# bev  AP:85.91, 85.91, 85.91
# 3d   AP:63.61, 63.61, 63.61
# aos  AP:53.01, 53.01, 53.01
# Pedestrian AP@0.50, 0.25, 0.25:
# bev  AP:93.62, 93.62, 93.62
# 3d   AP:93.58, 93.58, 93.58
# aos  AP:57.33, 57.33, 57.33

# bev  AP:88.29, 88.29, 88.29
# 3d   AP:67.61, 67.61, 67.61
# aos  AP:68.34, 68.34, 68.34
# Pedestrian AP@0.50, 0.25, 0.25:
# bev  AP:94.25, 94.25, 94.25
# 3d   AP:94.25, 94.25, 94.25
# aos  AP:72.95, 72.95, 72.95


#  ███████╗    ██╗   ██╗     █████╗     ██╗     
#  ██╔════╝    ██║   ██║    ██╔══██╗    ██║     
#  █████╗      ██║   ██║    ███████║    ██║     
#  ██╔══╝      ╚██╗ ██╔╝    ██╔══██║    ██║     
#  ███████╗     ╚████╔╝     ██║  ██║    ███████╗
#  ╚══════╝      ╚═══╝      ╚═╝  ╚═╝    ╚══════╝


            

# Evaluation Function
# =========================================
def evaluate(config_path, model_id=None, from_file_mode = False, epoch_idx=None):
    """
    Args:
        config_path     (str)   : Path to the config yaml
        model_id        (int)   : Which model id should be evaluated? "None" means the one in the config is used
        from_file_mode  (bool)  : "True" means that eval data from a file is used
        limit           (int)   : OPTIONS: None, int // after how many datapoints we want to exit 
    """

    print("**********************************************")
    print("* Start Evaluation")
    print("**********************************************")

    
    # This variable "model_id_memory" just helps to remember the original state of "model_id" 
    model_id_memory = 0
    if model_id == None: model_id_memory = None

    # ------------------------------------------------------------------------------------------------------ 
    #  load the config from file and set Variables
    # ------------------------------------------------------------------------------------------------------ 

    with open(config_path) as f1:   
        config = yaml.load(f1, Loader=yaml.FullLoader)


    # If no model_id is given we take the one from the config
    if model_id == None: model_id = config["eval_model_id"]
    eval_checkpoint = config["eval_checkpoint"]
    print("**********************************************")
    print("* Load Model ID {}".format(str(model_id)))
    print("**********************************************")

    # set training to false -> eval mode
    training = False

    # ------------------------------------------------------------------------------------------------------ 
    #  Load directory parameter and create directories
    # ------------------------------------------------------------------------------------------------------  
    
    project_dir_base = config["project_dir_base"] # path to base project dir (where the whole code is stored)

    
    # path to out base dir (where the training logs are stored)
    out_dir_base = create_out_dir_base(project_dir_base, training, model_id)
    

    # create the subdir where the training is stored
    out_dir_eval_results, out_dir_checkpoints = create_model_dirs_eval(out_dir_base)
   

    # ------------------------------------------------------------------------------------------------------
    #  Load dataloader parameter and create dataloader
    # ------------------------------------------------------------------------------------------------------ 

    production_mode = bool(config["production_mode"])
    limit_begin = 0 # Options: {Int} # begin idx, to limit the amount of test data to be evaluated 
    limit =None # Options: {None,Int} # end idx
    if production_mode:
        limit_begin = 0 # Options: {Int} # begin idx, to limit the amount of test data to be evaluated 
        limit =None # Options: {None,Int} # end idx
    batch_size = config["eval_input_reader"]["batch_size"] 
    num_point_features = config["train_input_reader"]["num_point_features"]
    center_limit_range = config["model"]["second"]["post_center_limit_range"]
    desired_objects = config["eval_input_reader"]["desired_objects"]
    no_annos_mode = config["eval_input_reader"]["no_annos_mode"]
    prediction_min_score = config["prediction_min_score"]
    load_weights = config["load_weights"]


    # create the dataLoader object which is reponsible for loading datapoints
    # contains not much logic and basically just holds some variables
    dataset_ori = dataLoader(training, None, config)    


    # initializes the dataset object (batch creating etc.)
    dataset = dataset_ori.getIterator()


    # makes the dataset object iterable
    data_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)


    # ------------------------------------------------------------------------------------------------------
    #  Create network
    # ------------------------------------------------------------------------------------------------------ 

    # create tensorboard writer for logging
    writer = tf.summary.create_file_writer(out_dir_base)
    
    # create network
    net = VoxelNet(config, writer, training=training)


    # DEBUG
    if from_file_mode:
        with open("eval_dataloader_limit200", "rb") as file:
            data_iterator = pickle.load(file)

    # ------------------------------------------------------------------------------------------------------
    #  We wrap a training step in a tf.function to to gain speedups (5x)
    # ------------------------------------------------------------------------------------------------------
    
    max_number_of_points_per_voxel = config["model"]["second"]["voxel_generator"]["max_number_of_points_per_voxel"] 

    @tf.function(input_signature = [tf.TensorSpec(shape=[None,max_number_of_points_per_voxel,num_point_features], dtype=tf.float32),tf.TensorSpec(shape=[None,], dtype=tf.int32),tf.TensorSpec(shape=[None,4], dtype=tf.int32),tf.TensorSpec(shape=[None,None,7], dtype=tf.float32)])
    def trainStep(voxels,num_points,coors,batch_anchors):
        preds_dict = net(voxels,num_points,coors,batch_anchors)
        return preds_dict

    # ------------------------------------------------------------------------------------------------------
    #  EPOCH ITERATOR Settings
    # ------------------------------------------------------------------------------------------------------

    # Helper Variable to load weights                                      
    load_weights_finished = False # This variable is part of a tensorflow workaround caused by subclassing keras.model




    # # Create a model using low-level tf.* APIs
    # class Squared(tf.Module):
    #     @tf.function
    #     def __call__(self, x):
    #         return tf.square(x)
    # model = Squared()
    # # (ro run your model) result = Squared(5.0) # This prints "25.0"
    # # (to generate a SavedModel) tf.saved_model.save(model, "saved_model_tf_dir")
    # concrete_func = model.__call__.get_concrete_function()

    # # Convert the model
    # converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # tflite_model = converter.convert()




    # Helper Variable to save the output of the network
    dt_annos = []
    
    # ------------------------------------------------------------------------------------------------------
    # Profile Network TIme
    # ------------------------------------------------------------------------------------------------------

    # eval params
    measure_time = config["measure_time"]

    # Helper Variables for Time Bechmarks
    current_milli_time = lambda: int(round(time.time() * 1000))
    t_full_sample_list = []
    t_preprocess_list = []
    t_network_list = []
    t_predict_list = []
    t_anno_list = []
    t_anno_list = []
    t_rviz_list = []


    # ------------------------------------------------------------------------------------------------------
    # ROS setup, in case the prediction should be sent to rviz
    # ------------------------------------------------------------------------------------------------------
    
    if production_mode:
        bb_pred_guess_1_pub = rospy.Publisher("bb_pred_guess_1", BoundingBoxArray)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_color_frame'
        calib = {"R0_rect" : np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3,3),
                    "Tr_velo_to_cam" : np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0]).reshape(3,4)}

    # ------------------------------------------------------------------------------------------------------
    #  EPOCH ITERATOR
    # ------------------------------------------------------------------------------------------------------  

    #example: [0:voxels, 1:num_points, 2:coordinates, 3:rect, 4:Trv2c, 5:P2, 6:anchors, 7:anchors_mask, 8:image_idx, 9:image_shape]
    for i, example in enumerate(data_iterator): 
        
        # debug limit eval data
        if i >= limit_begin and example[0].shape[0]>1:
            
            # debug measure time
            if measure_time: 
                if i > 0:
                    t_preprocess = current_milli_time() - t_preprocess
                    t_preprocess_list.append(t_preprocess)
                else:
                    t_preprocess_list.append(0.0)

            # save starting time to measure network speed
            if measure_time: t_full_sample = current_milli_time()

            # Progess Bar
            if not production_mode:
            
                if limit is None: 
                    progressBar(i-limit_begin,(dataset_ori.ndata-limit_begin)//batch_size)
                else:
                    progressBar(i-limit_begin,limit-limit_begin)
                    if i == limit: break # consider the case for early exit
        
            # ------------------------------------------------------------------------------------------------------
            #  Load Model Weights and Optimizer Weights (Checkpoint)
            # ------------------------------------------------------------------------------------------------------

            if load_weights_finished == False: # support variable, to check if the weights are loaded

                # run model one time to initialze it
                net(example[0],example[1],example[2],example[6])
                load_weights_finished = True

                # load the weights depending on if we are in training mode since
                # if yes the "model_weights_temp" file needs to be evaluated 
                model_dir = ""
                if load_weights: set_trainable(net, True)
                if model_id_memory == None: # for explaination, see variable "model_id_memory"
                    model_dir = out_dir_checkpoints + eval_checkpoint  # for eval mode
                    net.load_weights(model_dir)
                else:
                    model_dir = out_dir_checkpoints + "/model_weights_temp.h5" # for training mode
                    net.load_weights(model_dir)
                if load_weights: set_trainable(net, False)
                print("**********************************************")
                print("* Model Loaded from Path: {}".format(model_dir))
                print("**********************************************")

            # ------------------------------------------------------------------------------------------------------
            #  Run Network 
            # ------------------------------------------------------------------------------------------------------

            # debug
            if measure_time: t_network = current_milli_time()

            preds_dict = trainStep(example[0],example[1],example[2],example[6])

            # debug
            if measure_time:
                t_network = current_milli_time() - t_network
                t_network_list.append(t_network)

            # ------------------------------------------------------------------------------------------------------
            # Convert Network Output to predictions by applying the direction classifier to predictions of rotation 
            # use nms to get the final bboxes
            # ------------------------------------------------------------------------------------------------------

            # debug
            if measure_time: t_predict = current_milli_time()

            # t_full_sample: 23.91, t_preprocess: 0.85, t_network: 12.54, t_predict: 10.46, t_anno: 0.84, t_rviz: 0.0
            # t_full_sample: 23.54, t_preprocess: 1.13, t_network: 11.08, t_predict: 11.45, t_anno: 0.82, t_rviz: 0.01
            predictions_dicts = net.predict(example,preds_dict) # list(predictions) of ['name', 'truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score', 'image_idx'])

            # debug
            if measure_time:
                t_predict = current_milli_time() -t_predict
                t_predict_list.append(t_predict)
            
            # ------------------------------------------------------------------------------------------------------
            # Convert Predictions to Kitti Annotation style 
            # ------------------------------------------------------------------------------------------------------
            
            # debug
            if measure_time: t_anno = current_milli_time()

            dt_anno = predict_kitti_to_anno(example, desired_objects, predictions_dicts, None, False)

            # debug
            if measure_time:
                t_anno = current_milli_time() -t_anno
                t_anno_list.append(t_anno)
            
            # ------------------------------------------------------------------------------------------------------
            # OPTIONAL: Send annotation to RVIZ
            # ------------------------------------------------------------------------------------------------------

            # debug
            if measure_time: t_rviz = current_milli_time()

            if production_mode:
        
                dt_anno = remove_low_score(dt_anno[0], float(prediction_min_score))
                # if len(dt_anno["score"]) > 0:
                #    print(dt_anno["score"])
                dims = dt_anno['dimensions']
                loc = dt_anno['location']
                rots = dt_anno['rotation_y']
                boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                boxes_lidar = box_camera_to_lidar(boxes_camera, calib['R0_rect'],calib['Tr_velo_to_cam'])
                centers,dims,angles = boxes_lidar[:, :3], boxes_lidar[:, 3:6], boxes_lidar[:, 6] # [a,b,c] -> [c,a,b] (camera to lidar coords)
                if len(dt_anno["score"])>0:
                    print(dt_anno["score"])
                # LIFT bounding boxes 
                # - the postition of bboxes in the pipeline is at the z buttom of the bb and ros needs it at the z center
                # - TODO: lift by height/2 and not just 1.0

                # centers = centers + [0.0,0.0,1.0]
                send_3d_bbox(centers, dims, angles, bb_pred_guess_1_pub, header) 
            
            # ------------------------------------------------------------------------------------------------------
            # OPTIONAL: save annotations for evaluation
            # ------------------------------------------------------------------------------------------------------
            else:
                dt_annos += dt_anno
            
            # debug
            if measure_time:
                t_rviz = current_milli_time() -t_rviz
                t_rviz_list.append(t_rviz)
                t_full_sample = current_milli_time() -t_full_sample
                t_full_sample_list.append(t_full_sample)
                t_preprocess = current_milli_time()

            # ------------------------------------------------------------------------------------------------------
            # DEBUG: Print times
            # ------------------------------------------------------------------------------------------------------
        
            if i > 0 and measure_time: # we scipt the first network iteration (initialization)
                t_full_sample_avg = round(sum(t_full_sample_list[1:])/len(t_full_sample_list[1:]),2)
                t_preprocess_avg = round(sum(t_preprocess_list[1:])/len(t_preprocess_list[1:]),2)
                t_network_avg = round(sum(t_network_list[1:])/len(t_network_list[1:]),2)
                t_predict_avg = round(sum(t_predict_list[1:])/len(t_predict_list[1:]),2)
                t_anno_avg = round(sum(t_anno_list[1:])/len(t_anno_list[1:]),2)
                t_rviz_avg = round(sum(t_rviz_list[1:])/len(t_rviz_list[1:]),2)
            
                print(f't_full_sample: {t_full_sample_avg}, t_preprocess: {t_preprocess_avg}, t_network: {t_network_avg}, t_predict: {t_predict_avg}, t_anno: {t_anno_avg}, t_rviz: {t_rviz_avg}')

    # ------------------------------------------------------------------------------------------------------
    # save the results in a file
    # ------------------------------------------------------------------------------------------------------

    if epoch_idx is not None: # if epoch index is given include in the name (typically evaluation during training)
        with open(out_dir_eval_results + "/result_epoch_{}.pkl".format(str(epoch_idx)), 'wb') as f:
            pickle.dump(dt_annos, f, 2)
            
    else: # if epoch index is not given use generic name (typically while evaluation of certain models, epochs, and testing sets)
        with open(out_dir_eval_results + "/result.pkl", 'wb') as f:
            pickle.dump(dt_annos, f, 2)

    # ------------------------------------------------------------------------------------------------------
    # Exit the program if we run in no_annos_mode (since we dont have annotations we cannot do the following evaluation.
    # ------------------------------------------------------------------------------------------------------

    if no_annos_mode:
        return (np.array([0]),"no evaluation") 

    # get all gt in dataset
    gt_annos = [info["annos"] for info in dataset_ori.img_list_and_infos]
    
    # filter gt annos if we set a limit
    if limit is not None:
        gt_annos = gt_annos[limit_begin:limit]
    else:
        gt_annos = gt_annos[limit_begin:limit_begin+len(dt_annos)]

    # ------------------------------------------------------------------------------------------------------
    # evaluate the predictions in KITTI style
    # - AOS is average orientation similarity, bbox is 2D
    # - Results for the columns (difficulties) will be equal since we do not have OCCLUSION and TRUNCATION 
    #   annos in out ground truth which have influence on the difficulties
    # - Input: dt_annos, gt_annos are in camera coors (in Lidar coords)
    # ------------------------------------------------------------------------------------------------------

    compute_bbox = False # Since we dont have 2D box ground truth we don want to compute 2D bboxes results

    result1, mAPbbox, mAPbev, mAP3d, mAPaos = get_official_eval_result(gt_annos, dt_annos, desired_objects, compute_bbox = compute_bbox) 

    # ------------------------------------------------------------------------------------------------------
    # Print the evaluation result which depicts the overlaps between the gt and predictions 
    # - if multiple evals of the same class (e.g. Pedestrian,Pedestrian) occur, these depict separate evaluations 
    #   with different >overlap settings<
    # - in each of those evaluations: columns are difficulties (dependent on OCCLUSION and TRUNCATION) and 
    #   rows are [bbox,bev,3D]. The >overlap settings< are specified right to the class name e.g.:
    #   Pedestrian AP@0.50, 0.25, 0.25 -> [bbox, bev, 3D] and refer to the rows (NOT colums!). In this example
    #   bbox is 0.50, bev is 0.25 and 3D is 0.25 overlap. Also note, that bbox is missing in our evaluation.
    # ------------------------------------------------------------------------------------------------------
    print(result1)

    # ------------------------------------------------------------------------------------------------------
    # evaluate the predictions in COCO style
    # ------------------------------------------------------------------------------------------------------

    # result2 = get_coco_eval_result(gt_annos, dt_annos, desired_objects)
    # print(result2)

    # ------------------------------------------------------------------------------------------------------
    # return results (only used during training)
    # ------------------------------------------------------------------------------------------------------
    
    # get avg score
    eval_score = (mAP3d[0][0].sum()+mAPaos[0][0].sum()+mAPbev[0][0].sum())/18

    print("**********************************************")
    print("* score {}".format(str(eval_score)))
    print("**********************************************")
    
    return (mAP3d,result1, mAPbev, mAPaos) 



if __name__ == '__main__':
    fire.Fire()
    

# Pedestrian AP@0.50, 0.50, 0.50:
# bbox AP:0.00, 0.00, 0.00
# bev  AP:99.10, 99.10, 99.10
# 3d   AP:98.66, 98.66, 98.66
# aos  AP:57.04, 57.04, 57.04
# Pedestrian AP@0.50, 0.25, 0.25:
# bbox AP:0.00, 0.00, 0.00
# bev  AP:99.33, 99.33, 99.33
# 3d   AP:99.33, 99.33, 99.33
# aos  AP:57.28, 57.28, 57.28