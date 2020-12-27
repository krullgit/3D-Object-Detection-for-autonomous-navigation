# Master Thesis - Pointpillars, Tensorflow, Intel Realsense Implementation 

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


# Summary
This repository contains the code for recording a dataset, train [Pointpillars](https://arxiv.org/pdf/1812.05784.pdf) on this dataset and visualize the results in ROS rviz. There is also a clear *code structure* and *extensive comments*. 

# Environment

Setup 2 Conda environments for (1) tensorflow training and (2) ROS visualisation.

> conda env create -f configs/conda/tf37.yml

> conda env create -f configs/conda/ros27.yml

# Create Datasets

We create our own dataset for pedestrians. Therefore we simply record pointclouds from the realsense sensor and save annotation files along with the pointclouds. This only works because because we know beforehand where the person will stand and just change the rotations (in total 8 rotations, rotated by 45 degree each time and covring 360 degrees in total). 

For the secon command you have to change DATASETPATH, ROTATION, START_IDX, END_IDX, TRAIN_OR_TEST. 
* For DATASETPATH you can just use your absolute path to the dataset folder within this repo or wherever you want to store the dataset. For ROTATION you need to run this command 8 times to get data for 8 rotations. The ROTATIONS are: -3.14, -2.35, -1.57, -0.78, 2.35, 1.57, 0.78, 0.00

* Also you need to change START_IDX and END_IDX according to how many pointclouds you want to capture and (END_IDX, START_IDX) and under which name they are stored. For example: 0, 150 for rotation -3.14 and after 150, 300 for rotation -2.35.

* TRAIN_OR_TEST is set by you to "train" or "test" depeneding on if you need train or test data.

You can repeat this procedure for several times for different environment. 

*
## 1. Capture Data with Annotations
```
conda activate ros27
python scripts/realsense_make_dataset.py live_mode_on DATASETPATH ROTATION START_IDX, END_IDX TRAIN_OR_TEST
```

This will store data under:
* DATASETPATH/training/calib
* DATASETPATH/training/label_2
* DATASETPATH/training/velodyne
* DATASETPATH/testing/calib
* DATASETPATH/testing/label_2
* DATASETPATH/testing/velodyne


## 2. Capture Data without Annotations
In case we want to capture data from the realsense sensor without annotations (so the person can move freely) use this:
```
python scripts/realsense_make_dataset.py live_mode_on DATASETPATH  
```
This will store data under:
* DATASETPATH/testing_live/velodyne

# Prepare Data for Training 
## 1. Create Info Files

We need to create two info files for training and testing which accumulate all information like calibration, bounding box coordinates. 
```
conda activate tf37
python create_data.py create_kitti_info_file DATASETPATH
```
This will store data under:
* DATASETPATH/kitti_infos_train.pkl
* DATASETPATH/kitti_infos_val.pkl

## 2. Create Ground Truth database

We need to create a dataset with ground truth points and annoatations (according to the paper). This will be used for sampling and data augmentation during the training. 
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

1. We create sampled annotations for the test dataset to be able to apply a quantitative evaluation later on.

    * KITTI_INFOS_VAL_PATH is the absolute path to your kitti_infos_val.pkl
    * KITTI_DBINFOS_VAL_PATH is the absolute path to your kitti_dbinfos_val.pkl 
    * Also change train_input_reader.dataset_root_path in configs/train.yaml to your DATASETPATH

    ```
    python load_data.py KITTI_INFOS_VAL_PATH KITTI_DBINFOS_VAL_PATH
    ```

    This will store data under:
    * DATASETPATH/gt_database
    * DATASETPATH/kitti_infos_val_sampled.pkl
    * DATASETPATH/testing/velodyne_sampled

1. Rename the "velodyne" folder to "velodyne_unsampeld"
1. Make a copy of the "velodyne_sampled" folder and rename the copied folder to "velodyne".
    * Now you have the folders "velodyne", "velodyne_sampled" and "velodyne_unsampled" under the folder "testing"



# Train + Evaluate Model

# Evaluate Model 

## Evaluate Model on Sampled Test Dataset (with Annotations)

## Evaluate Model on Test Dataset (without Annotations)

# Visualize Results in rviz

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

install tensorflow 2.5

install cuda 11.1 (or 11.0)
and corresponding cudnn version 

install realsense
- https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
- sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u
- sudo apt-get install librealsense2-dkms
- sudo apt-get install librealsense2-utils
- realsense-viewer
- sudo apt-get install ros-melodic-realsense2-camera


