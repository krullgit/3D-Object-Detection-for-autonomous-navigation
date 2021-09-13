# **3d** Object Detection for Pedestrian with Pointpillars, Tensorflow, Intel Realsense d435i at 120 HZ

:rotating_light:	The documentation given here **is not complete**, due to time constraints. Since the model works very well however, it would be nice if people use it and therefore I'm happy to improve the documentation and code, if there is demand. Since the code is part of my master thesis, I'm thinking about releasing that as well (on request).


<p float="center">
  <img src="doc/assets/gif1.gif" width="50%" />
</p>

Video Links (Youtube):
- [Final Evaluation](https://youtu.be/C4lSZe1UTkw) (same as GIF above)
- [Train Set 1](https://youtu.be/HQPMiLqrlv4)
- [Train Set 2](https://youtu.be/0yEBxNRi1fc)
- [Test Set](https://youtu.be/EqNhPrcwK7o)
- [Augmentation Visualization](https://youtu.be/dEYM5AUP3Jo)

Master Thesis Link:
- On Request

# Features 
This is an implementation of the Pointpillars algorithm. It can predict **3d** bounding boxes on pointclouds produced by the [Intel Realsense d435i](https://www.intelrealsense.com/depth-camera-d435i/) sensor. For network details, please refer to the original paper of the authors [arXiv](https://arxiv.org/abs/1812.05784). The many customizations made are explained in my master thesis, which I release eventually (on request probably). Everything was trained and evaluated on custom data, since no public data is available for the sensor. However, with my method of a two stage training, only a couple of hundred GT annotations are needed for very good results. 

This implementation:

- has the code to:
    - create and annotate your own dataset with the d435i,
    - train and eval,
    - put the model into production where it fetches ROS messages from the d435i and publishes the detections to another ROS topic.
- uses Tensorflow 2.0,
- can be converted to a tflite model to run faster on edge devices,
- is currentely unique in its ability to detect pedestians on the d435i sensor, 
- works on arbitrary videos with multiple people,
- supports both CPU and GPU inference,
- runs at 120 FPS on a 3090 RTX
- incorporates and expands many functions for pre- and post processing from the official [POINTPILLARS](https://github.com/traveller59/second.pytorch) implementation especially for data-augmentation,
- applies changes to the originally proposed architecture to better utilize the greater resultion of the d435i in comparison to the lidar scanner used in [KITTI](http://www.cvlibs.net/datasets/kitti/)
- is able to cope with the massive noise produced by the sensor
- code documentation

Provided Data:
- Model for inference
- Test Data (On Request)
- Train Data (On Request)

Provided Research:
- Master Thesis (On Request)

# Overview
1. Installation
1. Quickstart 
1. Create Dataset
    1. Capture Data for Training 
1. Prepare Data for Training 
    1. Create Info Files
    1. Create Ground Truth database
1. Train + Evaluate Model
    1. Evaluate Model on Test Dataset (with Annotations)
    1. Evaluate Model on Live Dataset (without Annotations)
1. Visualize Results in RVIZ

# Installation

Currently, the installation of ROS Melodic, Tensorflow 2 and NVIDIA GPU drivers is obligatory:
1. Install Python 3.6.9. (Other versions of 3 may work as well)
1. Install [ROS Melodic for Python 3](https://dhanoopbhaskar.com/blog/2020-05-07-working-with-python-3-in-ros-kinetic-or-melodic/).
(Python 3 is needed for Tensoflow 2.0.)
1. Install CUDA and CUDNN drivers, if not already present.
1. Install the rest of the needed packages like tensorflow:
    - ```pip install -r configs/pip/requirements_short.txt```
    - if something is missing, see configs/pip/requirements.txt for an extensive list of my environment
4. Install additional ROS packages for RVIZ visualization:
    - ```sudo apt-get install -y ros-melodic-ros-numpy```
    - ```sudo apt-get install -y ros-melodic-jsk-visualization```
    - ```sudo apt-get install -y ros-melodic-depth-image-proc```
5. Install the d435i: [Installation here](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md).
Or just use these commands:
    - ```sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE```
    - ```sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u```
    - ```sudo apt-get install librealsense2-dkms```
    - ```sudo apt-get install librealsense2-utils```
    - ```sudo apt-get install ros-melodic-realsense2-camera```
    - Test installation: ```realsense-viewer```
# Quickstart

**PRODUCTION MODE** If you have a Realsense d435i sensor you can simply run inference on its ROS output stream. First you need to follow the steps under [Installation](#Installation) and than run the following commands in separate terminal windows:
1. !!! If any problem accours, it might be due to wrong paths set in the train.yaml (configuration file)
1. Start ROS:
    - ```roscore```
1. Start RVIZ for visualization:
    - ```rviz```
    - In RVIZ, load the config: configs/rviz/production_mode.rviz for proper visualization.
1. Start the camera and ROS stream:
    - ```roslaunch realsense2_camera rs_camera.launch filters:=pointcloud```
    - this will publish ROS messages of type "msg_PointCloud2" to the topic "/camera/depth/color/points" 
1. Start pipeline:
    - ```python train.py evaluate configs/train.yaml```
1. Now you should see the pointcloud of the sensor and **3d** detections in RVIZ 
    - predictions of type "BoundingBoxArray" are published to the topic "bb_pred_guess_1"

# Create Datasets

It's possible to create our your dataset, if you want to train for a different type of objects. Therefore, we simply record pointclouds from the realsense sensor and save annotation files along with the pointclouds. However, to skip all the time consuming annotation work, we use a simple approach by placing the object in predefined locations with different (also predefined) rotations (in total 8 rotations, rotated by 45 degree each time and covering 360 degrees in total). This procedure can be repeated for multiple scenes to increase the variety in the training data. The following section examines this concept. However, any train data is suitable and this approach does not have to be used.


## 1. Capture Data with Annotations (for training and testing)

<p float="center">
  <img src="doc/assets/Dataset_Illustration.png" width="70%" />
</p>

```
python scripts/realsense_make_dataset.py live_mode_off DATASETPATH ROTATION START_IDX END_IDX train

For example: 
python scripts/realsense_make_dataset.py live_mode_off data/ -3.14 0 150 train
```

Adjust the following params:
* For DATASETPATH, you can just use your absolute path to the "dataset" folder within this repo or wherever you want to store the dataset.

* For ROTATION you need to run this command 8 times to get data for 8 rotations. The ROTATIONS are: -3.14, -2.35, -1.57, -0.78, 2.35, 1.57, 0.78, 0.00 (also you need to change START_IDX and END_IDX for every run) 

* START_IDX and END_IDX: defines the naming of the annotations but also how many pointcloud you capture. For example: 0, 150 for rotation -3.14 and next time 150, 300 for rotation -2.35.

In total you have to run the command 8 times (8 rotations) to get training data for every rotation. 
You can repeat this procedure several times for different scenes, e.g. 2 times as shown in the illustration above. You have to place the object you want to record inside the bounding box shown in rviz during the execution of this file. Therefore you have to load the config: configs/rviz/production_mode.rviz in RVIZ.

This procedure will store data under:
* DATASETPATH/training/calib
* DATASETPATH/training/label_2
* DATASETPATH/training/velodyne


# 2. Prepare Data for Training
For training, your custom data needs to be prepared. The first step collects the data from all locations and accumulates them into one big file to save time while training. The second step extracts all ground truth annotations from our training data, to be used while training for data augmentation.

## 1. Create Info Files

The training needs ONE FILE which accumulates all information (like camera calibration and bounding box coordinates). 
We create it with:
```
python create_data.py create_kitti_info_file DATASETPATH
```
* change DATASETPATH accordingly
* in create_data.py, you probably need to change "train_img_ids" according to the image ids you want to include for training

This will store the file here:
* DATASETPATH/kitti_infos_train.pkl


## 2. Create Ground Truth database

For the data augementation in training we need to extract annotations and their points. 
```
python create_data.py create_groundtruth_database DATASETPATH train
```
This will store data under:
* DATASETPATH/gt_database (points are stored here)
* DATASETPATH/kitti_dbinfos_train.pkl (annotations are stored here)


# Train Model

After you created your dataset or downloaded mine, you can train the model.

First, open /configs/train.yaml and change params:
- production_mode: False
- train_input_reader.img_list_and_infos_path: absolute path to kitti_infos_train.pkl
- train_input_reader.dataset_root_path: absolute DATASETPATH
- train_input_reader.sampler_info_path: absolute path to kitti_dbinfos_train.pkl
- eval_input_reader.img_list_and_infos_path: absolute path to kitti_infos_val_sampled.pkl
- eval_input_reader.dataset_root_path: absolute DATASETPATH

Than:

- ```python train.py train configs/train.yaml```

- This will train the model for 160 epochs and evaluates after each one. 
- eval results are published to wandb, so check there the eval results if you want to use early stopping
- Also, the model is saved after each epoch in case you find certain epochs working better for you than others. 
- The weights and evaluation results are stored under the folder "/out"

# Evaluate Model 


## Evaluate Model on Test Dataset (with Annotations)

To evaluate the test dataset:

Open /configs/train.yaml and change the params:
- eval_model_id: is the model number you want evaluate (found in "/out"), the provided model is 345 
- eval_checkpoint: is the epoch you want to evaluate (found in "/out/model_x/out_dir_checkpoints/"), , the provided epoch is 48
- production_mode: False
- eval_input_reader.no_annos_mode: False
- eval_input_reader.img_list_and_infos_path: absolute path to kitti_infos_val_sampled.pkl
- eval_input_reader.dataset_root_path: DATASETPATH
- measure_time: True

Than: 

- ```python train.py evaluate configs/train.yaml```

- The evaluation output in KITTI style will look like this:

t_full_sample: 8.67, t_preprocess: 0.33, t_network: 4.67, t_predict: 3.33, t_anno: 0.56, t_rviz: 0.0

Pedestrian AP@0.70, 0.50, 0.50:\
**bev**&nbsp;&nbsp;  AP: 89.15\
**3d**&nbsp;&nbsp;&nbsp;&nbsp;   AP: 88.40\
**aos**&nbsp;&nbsp;  AP: 65.62\
Pedestrian AP@0.75, 0.55, 0.55:\
**bev**&nbsp;&nbsp;  AP: 88.34\
**3d**&nbsp;&nbsp;&nbsp;&nbsp;   AP: 86.94\
**aos**&nbsp;&nbsp;  AP: 65.07\
Pedestrian AP@0.80, 0.60, 0.60:\
**bev**&nbsp;&nbsp;  AP: 86.58\
**3d**&nbsp;&nbsp;&nbsp;&nbsp;   AP: 71.66\
**aos**&nbsp;&nbsp;  AP: 63.90\
Pedestrian AP@0.85, 0.65, 0.65:\
**bev**&nbsp;&nbsp;  AP: 74.34\
**3d**&nbsp;&nbsp;&nbsp;&nbsp;   AP: 44.22\
**aos**&nbsp;&nbsp;  AP: 55.49\
Pedestrian AP@0.90, 0.70, 0.70:\
**bev**&nbsp;&nbsp;  AP: 54.31\
**3d**&nbsp;&nbsp;&nbsp;&nbsp;   AP: 22.44\
**aos**&nbsp;&nbsp;  AP: 41.28\
Pedestrian AP@0.95, 0.75, 0.75:\
**bev**&nbsp;&nbsp;  AP: 27.94\
**3d**&nbsp;&nbsp;&nbsp;&nbsp;   AP: 6.48\
**aos**&nbsp;&nbsp;  AP: 22.58\



<br/>

- The first lines are refereeing to different difficulties, e.g. "Pedestrian AP@0.70, 0.50, 0.50:\" sets the difficulties for bev, 3d and aos to 0.70, 0.50, 0.50, respectively (numbers refer to IoU). Also, I have set the whole evaluation harder than KITTI (higher IoU), due to the great performance.
- As seen, also the time is measured. "t_full_sample" means the milliseconds for one sample. 
- **AP**: Mean average precision (mAP). Average of the maximum precision at different recall values.
- **bev**: Bird's eye view overlap (2D) of prediction and ground truth.
- **3d**: 3d overlap of prediction and ground truth.
- **aos**: Average orientation similarity of prediction and ground truth jointly with object detection.


- Predictions are stored in the "out/model_\<modelNumber>" folder 
- For visualization see [Visualize Results in RVIZ](#Visualize-Results-in-RVIZ)

## Evaluate Model on Test Dataset (without Annotations)

There is the possibility to eval a dataset that has no annotations (the setting needed are not completely given here, will be done later): 

Open /configs/train.yaml and change the params:
- eval_model_id: is the model number you want evaluate (found in "/out")
- eval_checkpoint: is the epoch you want to evaluate (found in "/out/model_x/out_dir_checkpoints/")
- production_mode: False
- eval_input_reader.no_annos_mode: True
- eval_input_reader.dataset_root_path: DATASETPATH

Than: 

- ```python train.py evaluate configs/train.yaml```

- There is no output here since we don't have annotation.
- Predictions are stored in the "out/model_\<modelNumber>" folder 
    - For visualization see [Visualize Results in RVIZ](#Visualize-Results-in-RVIZ)

# Visualize Results in RVIZ

If you evaluated either with or without annotations, you can show the results in RVIZ.

- To switch between the dataset with and without annotations, open "rviz_show_predictions.py" and change the "mode" variable to "live" (without annotations) or "testing_sampled" (with annotation).

- Go to the /scripts folder and run following in terminal:

    ```python rviz_show_predictions.py```

- Load the configs/rviz/rviz_show_predictions.rviz in RVIZ. (visualization starts now)



