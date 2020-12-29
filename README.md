# 3D Object Detection for Pedestrian with Pointpillars, Tensorflow, Intel Realsense d435i

<p float="center">
  <img src="doc/assets/gif1.gif" width="70%" />
</p>

# Features 
This is an implementation of the Pointpillars algorithm. It can predict a 3D bounding box on pointclouds produced by the Intel Realsense d435i sensor. Pleaser refer to [arXiv report](https://arxiv.org/abs/1812.05784) for further details.

This implementation:

- has the code to:
    - create and annotate your own dataset (train + eval) with the d435i,
    - train and eval on this datatset,
    - put the model into production where it fetches ROS messages from the d435i and publishes the detections to another ROS topic.
- is fully implemented in Tensorflow,
- can be converted to a tflite model to run faster on edge devices,
- is currentely unique in its ability to detect pedestians on the d435i sensor, 
- works on arbitrary videos with multiple people,
- supports both CPU and GPU inference,
- runs at 50 FPS on a 3090 RTX (so its 2x as fast as the official implementation without sacrificing precision),
- incorporates and expands many functions for pre- and post processing from the official [POINTPILLARS](https://github.com/traveller59/second.pytorch) implementation especially for data-augmentation,
- applies changes to the network to better utilize great resultion of the sensor
- is greatly documented and commented 

# Overview
1. Summary
1. Environment 
1. Create Dataset
    1. Capture Data with Annotations
    1. Capture Data without Annotations
1. Prepare Data for Training 
    1. Create Info Files
    1. Create Ground Truth database
    1. Create Sampled Dataset for Testing 

# Installation

Currently, the installation of ROS Melodic, Tensorflow 2 and NVIDIA GPU drivers is obligatory:
1. Install [ROS Melodic for Python 3](https://dhanoopbhaskar.com/blog/2020-05-07-working-with-python-3-in-ros-kinetic-or-melodic/).
(Python 3 is needed to work with Tensoflow 2.0.)
2. Install CUDA and CUDNN drivers if not already present.
3. Install the rest of the needed packages like tensorflow:
    - ```pip install -r configs/pip/requirements.txt```
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
    - test installation: ```realsense-viewer```
# Quickstart production mode

If you have a Realsense d435i sensor you can simply run inference on its ROS output stream. First you need to follow the steps under >Installation< and after run following terminal commands in separate windows:
1. Start ROS:
    - ```roscore```
1. Start RVIZ for visualization:
    - ```rviz```
    - in RVIZ load the config: configs/rviz/production_mode.rviz
1. Start the camera and ROS stream:
    - ```roslaunch realsense2_camera rs_camera.launch filters:=pointcloud```
1. Start pipeline:
    - ```python train.py evaluate configs/train.yaml```
1. Now you should see the pointcloud of the sensor and 3D detections in RVIZ

# Create Datasets

It's possible to create our own dataset, if we want to train for different objects than pedestrians. Therefore we simply record pointclouds from the realsense sensor and save annotation files along with the pointclouds. To skip all the time consuming annotation work, we use a simple approach by placing the object in predefined locations with different (also predefined) rotations (in total 8 rotations, rotated by 45 degree each time and covering 360 degrees in total). The follwing illustrates the concept for two different predefined locations.



## 1. Capture Data with Annotations (for training and testing)

<p float="center">
  <img src="doc/assets/Dataset_Illustration.png" width="70%" />
</p>

```
python scripts/realsense_make_dataset.py live_mode_on DATASETPATH ROTATION START_IDX END_IDX TRAIN_OR_TEST

For example: 
python scripts/realsense_make_dataset.py live_mode_on data/ -3.14 0 150 train
```

Adjust the following params:
* For DATASETPATH you can just use your absolute path to the "dataset" folder within this repo or wherever you want to store the dataset.

* For ROTATION you need to run this command 8 times to get data for 8 rotations. The ROTATIONS are: -3.14, -2.35, -1.57, -0.78, 2.35, 1.57, 0.78, 0.00 (also you need to change START_IDX and END_IDX for every run) 

* START_IDX and END_IDX: defines the naming of the annotations but also how many pointcloud you capture. For example: 0, 150 for rotation -3.14 150 and next time 150, 300 for rotation -2.35.

* TRAIN_OR_TEST is set by you to "train" or "test" depending on if you need train or test data.

In total you have to run the command 16 times (8 rotations * 2 (train + test)) to get training and test data for every rotation. 
You can repeat this procedure several times for different environment, e.g. 2 times as shown in the illustration above. You have to place the object you want to record inside the bounding box shown in rviz during the execution of this file. Therefore you have to load the config: configs/rviz/production_mode.rviz in RVIZ.

This procedure will store data under:
* DATASETPATH/training/calib
* DATASETPATH/training/label_2
* DATASETPATH/training/velodyne
* DATASETPATH/testing/calib
* DATASETPATH/testing/label_2
* DATASETPATH/testing/velodyne

## 2. Capture Data without Annotations (purely for testing, NOT training)

<p float="center">
  <img src="doc/assets/gif2.gif" width="350px" />
</p>

In case we want to capture data from the realsense sensor without annotations use this:
```
python scripts/realsense_make_dataset.py live_mode_off DATASETPATH  
```
* change DATASETPATH accordingly 

This procedure will store data under:
* DATASETPATH/testing_live/velodyne

# 2. Prepare Data for Training
For training, your custom data needs to be prepared. The first step collects the data from all locations and accumulates them into one big file that this has not to be done while training and save time. The second step creates a new dataset with all ground truth annotations we have, to be used while training for data augmentation. In the third step, we combine test pointclouds and augmented test ground truths (from step two), which simulates a test dataset.

## 1. Create Info Files

The training needs TWO FILES (one for training and one for testing) which accumulates all information (like camera calibration and bounding box coordinates). 
Therefore we create these files with:
```
python create_data.py create_kitti_info_file DATASETPATH
```
* change DATASETPATH accordingly

This will store data under:
* DATASETPATH/kitti_infos_train.pkl
* DATASETPATH/kitti_infos_val.pkl

## 2. Create Ground Truth database

For the data augementation during the training we need to extract annoations and point clouds from our data. 
```
python create_data.py create_groundtruth_database DATASETPATH train
python create_data.py create_groundtruth_database DATASETPATH test
```
This will store data under:
* DATASETPATH/gt_database
* DATASETPATH/gt_database_val
* DATASETPATH/kitti_dbinfos_train.pkl
* DATASETPATH/kitti_dbinfos_val.pkl

## 3. Create Sampled Dataset for Testing

<p float="center">
  <img src="doc/assets/gif4.gif" width="350px" />
</p>

1. We create a test dataset to be able to apply a quantitative evaluation later on.

    ```
    python load_data.py KITTI_INFOS_VAL_PATH KITTI_DBINFOS_VAL_PATH
    ```

    * KITTI_INFOS_VAL_PATH is the absolute path to your kitti_infos_val.pkl
    * KITTI_DBINFOS_VAL_PATH is the absolute path to your kitti_dbinfos_val.pkl 
    * Also change train_input_reader.dataset_root_path in configs/train.yaml to your DATASETPATH

    This will store data under:
    * DATASETPATH/gt_database
    * DATASETPATH/kitti_infos_val_sampled.pkl
    * DATASETPATH/testing/velodyne_sampled

1. Rename the "velodyne" folder to "velodyne_unsampeld"
1. Make a copy of the "velodyne_sampled" folder and rename the copied folder to "velodyne".
    * Now you have the folders "velodyne", "velodyne_sampled" and "velodyne_unsampled" under the folder "testing"



# Train + Evaluate Model

After you created your dataset or downloaded mine, you can train the model.

First, ppen /configs/train.yaml and change params:
- production_mode: False
- train_input_reader -> img_list_and_infos_path: absolute path to kitti_infos_train.pkl
- train_input_reader -> dataset_root_path: DATASETPATH
- train_input_reader -> sampler_info_path: absolute path to kitti_dbinfos_train.pkl
- eval_input_reader -> no_annos_mode: False
- eval_input_reader -> img_list_and_infos_path: absolute path to kitti_infos_val_sampled.pkl
- eval_input_reader -> dataset_root_path: DATASETPATH

Than:

```python train.py train configs/train.yaml```

- This will train the model for 160 epochs and evaluates after each one. 
- Also, the model is saved after each epoch in case you find certain epochs working better for you than others. 
- The weights and evaluation results are stored under the folder "/out"

# Evaluate Model 

In case you just want to evaluate your model I show you how to do that here.

## Evaluate Model on Sampled Test Dataset (with Annotations)

To evaluate the test dataset created in "Create Sampled Dataset for Testing":

Open /configs/train.yaml and change the params:
- eval_model_id: is the model number you want evaluate (found in "/out")
- eval_checkpoint: is the epoch you want to evaluate (found in "/out/model_x/out_dir_checkpoints/")
- production_mode: False
- eval_input_reader -> no_annos_mode: False
- eval_input_reader -> img_list_and_infos_path: absolute path to kitti_infos_val_sampled.pkl
- eval_input_reader -> dataset_root_path: DATASETPATH

Than: 

```python train.py evaluate configs/train.yaml```

## Evaluate Model on Test Dataset (without Annotations)

To evaluate the test dataset created in "Capture Data with Annotations (for training and testing)":

Open /configs/train.yaml and change the params:
- eval_model_id: is the model number you want evaluate (found in "/out")
- eval_checkpoint: is the epoch you want to evaluate (found in "/out/model_x/out_dir_checkpoints/")
- production_mode: False
- eval_input_reader -> no_annos_mode: True
- eval_input_reader -> dataset_root_path: DATASETPATH

Than: 

```python train.py evaluate configs/train.yaml```

# Visualize Results in RVIZ

<p float="center">
  <img src="doc/assets/gif5.gif" width="350px" />
  <img src="doc/assets/gif6.gif" width="350px" />
</p>

Red are ground-truth annotations and green the predictions. Left with annotations and right without. 

If you evaluated either with or without annotations, you can show the results in RVIZ.
- Therefore, go to the /scripts folder and run following in terminal:

```python rviz_show_predictions.py```

- and load the configs/rviz/rviz_show_predictions.rviz in RVIZ.
- If you want to switch between the dataset with and without annotations, open rviz_show_predictions.py and change the "mode" parameter to "live" (without annotations) or "testing_sampled" (with annotation).
    - consequently you also need to change the corresponding data paths in the according "if" statements following the "mode" parameter (so "if mode == "live":" OR "if mode == "testing_sampled":")



# Install 
- python 3.6.9
- pip3 install fire
- pip3 install numba
- pip3 install tensorflow-addons
- pip install pybind11
- python -m pip install -U scikit-image
- pip install -U PyYAML
- pip install defusedxml
- maybe https://dhanoopbhaskar.com/blog/2020-05-07-working-with-python-3-in-ros-kinetic-or-melodic/
- sudo apt-get install -y ros-melodic-ros-numpy
- sudo apt-get install -y ros-melodic-jsk-visualization
- sudo apt-get install -y ros-melodic-depth-image-proc

for 3090
install tensorflow 2.5
install cuda 11.1 (or 11.0)
and corresponding cudnn version 

install realsense
- https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
- sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
- sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u
- sudo apt-get install librealsense2-dkms
- sudo apt-get install librealsense2-utils
- sudo apt-get install ros-melodic-realsense2-camera
- realsense-viewer


ffmpeg
ffmpeg -i realsense_pointpillars_tensorflow_3d_detection.mkv -c:a copy -c:v copy -ss 00:03:40 -t 00:00:12 test.mkv
ffmpeg -i in.mp4 -filter:v "crop=640:480:100:100" out.mp4
fmpeg -i test.mkv -s 800x420 test1_1.mkv

ffmpeg -i test1_2.mkv  -r 25 'frames/frame-%03d.jpg'

cd frames
convert -delay 4 -loop 0 *.jpg myimage.gif
