import copy
import pathlib
import pickle

import fire
import numpy as np
#from skimage import io as imgio

# import libraries.box_np_ops as box_np_ops
# import libraries.kitti_common as kitti
# from libraries.progress_bar import list_bar as prog_bar

from second.core import box_np_ops
from second.data import kitti_common as kitti
from second.utils.progress_bar import list_bar as prog_bar
"""
Note: tqdm has problem in my system(win10), so use my progress bar
try:
    from tqdm import tqdm as prog_bar
except ImportError:
    from second.utils.progress_bar import progress_bar_iter as prog_bar
"""



# Here we calculate the number of points in the pointcloud which fall into each gt box
# =========================================
def _calculate_num_points_in_gt(data_path, infos, relative_path, remove_outside=True, num_features=4):

    # custom dataset parameter
    custom_dataset = True
    # comment strating form here
    import pickle
    num_features=3
   
    # iterate over the datapoints
    for info in infos:
        if relative_path:
            v_path = str(pathlib.Path(data_path) / info["velodyne_path"])
        else:
            v_path = info["velodyne_path"]

        if custom_dataset:
            with open(v_path[:-3]+"pkl", 'rb') as file:
                points_v = pickle.load(file, encoding='latin1')
        else:
            points_v = np.fromfile(
                v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = info['calib/R0_rect']
        Trv2c = info['calib/Tr_velo_to_cam']
        P2 = info['calib/P2']

        # debug
        # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/debug_rviz/points/points.pkl", 'wb') as file:
        #     pickle.dump(np.array(points_v), file, 2)


        # remove points outside a frustum defined by the shape of the image 
        if remove_outside and not custom_dataset:
            points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
                                                        info["img_shape"])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']


        # filter DontCare
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]


        # get gt boxes in cameras and lidar coords
        gt_boxes_camera = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)


        # check which points are in the gt boxes
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)


        # fill the list up with -1s for the DontCares        
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])


        annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)
        
# ------------------------------------------------------------------------------------------------------ 
# creates the files "kitti_infos_train.pkl" and "kitti_infos_val.pkl" which contains the gt annotations and calib info for every pointcloud in training- and test-dataset
# here, the annos are transformed and saved in the camera coord system, see next line:
# RECALL ANNOTATIONS: yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
# ------------------------------------------------------------------------------------------------------
# example output:
# ------------------------------------------------------------------------------------------------------
# 'annos':{'name': array(['Pedestrian']...pe='<U10'), 'truncated': array([0.]), 'occluded': array([0]), 'alpha': array([0.]), 'bbox': array([[700., 100., ...., 300.]]), 'dimensions': array([[0.6, 1.8, 0.4]]), 'location': array([[-0.1,  0. ,  2. ]]), 'rotation_y': array([1.57079633]), 'score': array([0.]), 'index': array([0], dtype=int32), 'group_ids': array([0], dtype=int32), 'difficulty': array([0], dtype=int32), 'num_points_in_gt': array([3768], dtype=int32)}
# 'calib/Tr_imu_to_velo':array([[ 9.999976e-01,  7.553071e-04, -2.035826e-03, -8.086759e-01],
#        [-7.854027e-04,  9.998898e-01, -1.482298e-02,  3.195559e-01],
#        [ 2.024406e-03,  1.482454e-02,  9.998881e-01, -7.997231e-01],
#        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
# 'calib/Tr_velo_to_cam':array([[ 0., -1.,  0.,  0.],
#        [ 0.,  0., -1.,  0.],
#        [ 1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.]])
# 'calib/R0_rect':array([[1., 0., 0., 0.],
#        [0., 1., 0., 0.],
#        [0., 0., 1., 0.],
#        [0., 0., 0., 1.]])
# 'calib/P3':array([[ 7.070493e+02,  0.000000e+00,  6.040814e+02, -3.341081e+02],
#        [ 0.000000e+00,  7.070493e+02,  1.805066e+02,  2.330660e+00],
#        [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  3.201153e-03],
#        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
# 'calib/P2':array([[ 7.070493e+02,  0.000000e+00,  6.040814e+02,  4.575831e+01],
#        [ 0.000000e+00,  7.070493e+02,  1.805066e+02, -3.454157e-01],
#        [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  4.981016e-03],
#        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
# 'calib/P1':array([[ 707.0493,    0.    ,  604.0814, -379.7842],
#        [   0.    ,  707.0493,  180.5066,    0.    ],
#        [   0.    ,    0.    ,    1.    ,    0.    ],
#        [   0.    ,    0.    ,    0.    ,    1.    ]])
# 'calib/P0':array([[707.0493,   0.    , 604.0814,   0.    ],
#        [  0.    , 707.0493, 180.5066,   0.    ],
#        [  0.    ,   0.    ,   1.    ,   0.    ],
#        [  0.    ,   0.    ,   0.    ,   1.    ]])
# 'img_shape':array([ 800, 1280], dtype=int32)
# 'img_path':'training/image_2/000810.png'
# 'velodyne_path':'training/velodyne/000810.pkl'
# 'image_idx':810
# 'pointcloud_num_features':4
# special variables
# function variables
# 'name':array(['Pedestrian'], dtype='<U10')
# 'truncated':array([0.])
# 'occluded':array([0])
# 'alpha':array([0.])
# 'bbox':array([[700., 100., 800., 300.]])
# 'dimensions':array([[0.6, 1.8, 0.4]])
# 'location':array([[-0.1,  0. ,  2. ]])
# 'rotation_y':array([1.57079633])
# 'score':array([0.])
# 'index':array([0], dtype=int32)
# 'group_ids':array([0], dtype=int32)
# 'difficulty':array([0], dtype=int32)
# 'num_points_in_gt':array([3768], dtype=int32)
# ------------------------------------------------------------------------------------------------------
# =========================================
def create_kitti_info_file(data_path,
                           save_path=None,
                           create_trainval=False,
                           relative_path=True):

    
    train_img_ids = list(range(0,4440)) # indices of training images that will appear in the info file
    val_img_ids = list(range(0,1200)) # indices of test images that will appear in the info file
    # trainval_img_ids = list(range(0,7480)) # only kitti
    # test_img_ids = list(range(0,7517)) # only kitti

    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = pathlib.Path(data_path)
    else:
        save_path = pathlib.Path(save_path)
        
    mode = "train" # option: train, test, test_real

    # ------------------------------------------------------------------------------------------------------ 
    #  Train
    # ------------------------------------------------------------------------------------------------------
    
    if mode == "train":
        # get all the annos from the file
        kitti_infos_train = kitti.get_kitti_image_info(
            data_path,
            training=True,
            velodyne=True,
            calib=True,
            image_ids=train_img_ids,
            relative_path=relative_path)


        # calculate the number of points in the pointcloud which fall into each gt box
        _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    

        filename = save_path / 'kitti_infos_train.pkl'
        print(f"Kitti info train file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_train, f,2)
        
    # ------------------------------------------------------------------------------------------------------ 
    #  test fake anno
    # ------------------------------------------------------------------------------------------------------

    if mode == "test":
        val_img_ids = list(range(0,360)) # indices of test images that will appear in the info file
        kitti_infos_val = kitti.get_kitti_image_info(
            data_path,
            training=False,
            velodyne=True,
            calib=True,
            image_ids=val_img_ids,
            relative_path=relative_path)
        _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
        filename = save_path / 'kitti_infos_val.pkl'
        print(f"Kitti info val file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_val, f,2)
        
        
    # ------------------------------------------------------------------------------------------------------ 
    #  test real anno
    # ------------------------------------------------------------------------------------------------------
    
    if mode == "test_real":
        val_img_ids = list(range(0,120)) # indices of test images that will appear in the info file
        
        kitti_infos_val = kitti.get_kitti_image_info(
            data_path,
            training=False,
            velodyne=True,
            calib=True,
            image_ids=val_img_ids,
            relative_path=relative_path)
        _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
        filename = save_path / 'kitti_infos_val.pkl'
        print(f"Kitti info val file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_val, f,2)
    
    
    
    # ------------------------------------------------------------------------------------------------------
    # old KITTI stuff
    # ------------------------------------------------------------------------------------------------------
    """
    if create_trainval:
        kitti_infos_trainval = kitti.get_kitti_image_info(
            data_path,
            training=True,
            velodyne=True,
            calib=True,
            image_ids=trainval_img_ids,
            relative_path=relative_path)
        filename = save_path / 'kitti_infos_trainval.pkl'
        print(f"Kitti info trainval file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_trainval, f)
    """
    # filename = save_path / 'kitti_infos_trainval.pkl'
    # print(f"Kitti info trainval file is saved to {filename}")
    # with open(filename, 'wb') as f:
    #     pickle.dump(kitti_infos_train + kitti_infos_val, f)

    # kitti_infos_test = kitti.get_kitti_image_info(
    #     data_path,
    #     training=False,
    #     label_info=False,
    #     velodyne=True,F
    #     calib=True,
    #     image_ids=test_img_ids,
    #     relative_path=relative_path)
    # filename = save_path / 'kitti_infos_test.pkl'
    # print(f"Kitti info test file is saved to {filename}")
    # with open(filename, 'wb') as f:
    #     pickle.dump(kitti_infos_test, f)


# helper function for create_reduced_point_cloud
# =========================================
def _create_reduced_point_cloud(data_path,
                                info_path,
                                save_path=None,
                                back=False):
    
    # custom dataset parameter
    custom_dataset = True

    
    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    for info in prog_bar(kitti_infos):
        v_path = info['velodyne_path']
        v_path = pathlib.Path(data_path) / v_path


        if custom_dataset:
            with open(str(v_path)[:-3]+"pkl", 'rb') as file:
                points_v = pickle.load(file, encoding='latin1')
        else:
            points_v = np.fromfile(
                str(v_path), dtype=np.float32, count=-1).reshape([-1, 4])


        rect = info['calib/R0_rect']
        P2 = info['calib/P2']
        Trv2c = info['calib/Tr_velo_to_cam']
        # first remove z < 0 points
        # keep = points_v[:, -1] > 0
        # points_v = points_v[keep]
        # then remove outside.
        if back:
            points_v[:, 0] = -points_v[:, 0]

        if not custom_dataset:
            points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
                                                        info["img_shape"])

        if save_path is None:
            save_filename = v_path.parent.parent / (v_path.parent.stem + "_reduced") / v_path.name
            # save_filename = str(v_path) + '_reduced'
            if back:
                save_filename += "_back"
        else:
            save_filename = str(pathlib.Path(save_path) / v_path.name)
            if back:
                save_filename += "_back"
        with open(save_filename, 'w') as f:
            points_v.tofile(f)

# deletes and saves pc with all points deleteted outside the image frustum
# for our custom realsense dataset we do not use this since the range of this sensor is not so high
#======================================================================
def create_reduced_point_cloud(data_path,
                               train_info_path=None,
                               val_info_path=None,
                               test_info_path=None,
                               save_path=None,
                               with_back=False):
    if train_info_path is None:
        train_info_path = pathlib.Path(data_path) / 'kitti_infos_train.pkl'
    if val_info_path is None:
        val_info_path = pathlib.Path(data_path) / 'kitti_infos_val.pkl'
    if test_info_path is None:
        test_info_path = pathlib.Path(data_path) / 'kitti_infos_test.pkl'

    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(
            data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, test_info_path, save_path, back=True)


# ------------------------------------------------------------------------------------------------------
# create the file "kitti_dbinfos_val.pkl" or  "kitti_dbinfos_train.pkl" which contains gt annotations for the groundtruth_database
# saves the point belonging to that groundtruth database annotations to the path database_save_path
# here, the annos are saved lidar coord system, see next line
# RECALL ANNOTATIONS: yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
# ------------------------------------------------------------------------------------------------------
# example output:
# ------------------------------------------------------------------------------------------------------
# 0000:{'name': 'Pedestrian', 'path': 'gt_database/000000_P...rian_0.bin', 'image_idx': '000000', 'gt_idx': 0, 'box3d_lidar': array([ 2.        , ...14159265]), 'num_points_in_gt': 5248, 'difficulty': 0, 'group_id': 0, 'score': 0.0}
# 0001:{'name': 'Pedestrian', 'path': 'gt_database/000001_P...rian_0.bin', 'image_idx': '000001', 'gt_idx': 0, 'box3d_lidar': array([ 2.        , ...14159265]), 'num_points_in_gt': 5261, 'difficulty': 0, 'group_id': 1, 'score': 0.0}
# 0002:{'name': 'Pedestrian', 'path': 'gt_database/000002_P...rian_0.bin', 'image_idx': '000002', 'gt_idx': 0, 'box3d_lidar': array([ 2.        , ...14159265]), 'num_points_in_gt': 5210, 'difficulty': 0, 'group_id': 2, 'score': 0.0}
# ------------------------------------------------------------------------------------------------------
def create_groundtruth_database(data_path,
                                train_or_test,
                                info_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None):

    # custom dataset parameter
    custom_dataset = True  # if true, it indicates that we are not operating on the kitti data
    sample_val_dataset_mode = True if train_or_test == "test" else False # if true, we are creating a gt database for the test set (instead of the train set)

    # ------------------------------------------------------------------------------------------------------ 
    #  create path where the gt boxes are stored
    # -----------------------------------------------------------------------------------------------------

    root_path = pathlib.Path(data_path)
    if info_path is None:
        if sample_val_dataset_mode:
            info_path = root_path / 'kitti_infos_val.pkl'
        else:
            info_path = root_path / 'kitti_infos_train.pkl'
    if database_save_path is None:
        if sample_val_dataset_mode:
            database_save_path = root_path / 'gt_database_val'
        else:
            database_save_path = root_path / 'gt_database'
    else:
        database_save_path = pathlib.Path(database_save_path)
    if db_info_save_path is None:
        if sample_val_dataset_mode:
            db_info_save_path = root_path / "kitti_dbinfos_val.pkl"
        else:
            db_info_save_path = root_path / "kitti_dbinfos_train.pkl"
    database_save_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------ 
    #  load kitti infos
    # -----------------------------------------------------------------------------------------------------

    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    all_db_infos = {}


    # get the classnames we are intered in
    if used_classes is None:
        used_classes = list(kitti.get_classes())
        used_classes.pop(used_classes.index('DontCare'))
    for name in used_classes:
        all_db_infos[name] = []
    group_counter = 0

    # ------------------------------------------------------------------------------------------------------    
    #  iterate over kitti_infos
    # -----------------------------------------------------------------------------------------------------

    for info in prog_bar(kitti_infos):

        # ------------------------------------------------------------------------------------------------------ 
        #  load pc
        # -----------------------------------------------------------------------------------------------------

        velodyne_path = info['velodyne_path']
        if relative_path:
            # velodyne_path = str(root_path / velodyne_path) + "_reduced"
            velodyne_path = str(root_path / velodyne_path)
        num_features = 4
        if 'pointcloud_num_features' in info:
            num_features = info['pointcloud_num_features']


        if custom_dataset:
            with open(str(velodyne_path)[:-3]+"pkl", 'rb') as file:
                points = pickle.load(file, encoding='latin1')
        else:
            points = np.fromfile(
                str(velodyne_path), dtype=np.float32, count=-1).reshape([-1, 4])


        image_idx = info["image_idx"]
        rect = info['calib/R0_rect']
        P2 = info['calib/P2']
        Trv2c = info['calib/Tr_velo_to_cam']

        # ------------------------------------------------------------------------------------------------------ 
        #  remove boxes outside the frustum of the image
        # -----------------------------------------------------------------------------------------------------

        if not lidar_only and not custom_dataset:
            points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2,
                                                        info["img_shape"])
        
        # ------------------------------------------------------------------------------------------------------ 
        #  get the bboxes and transform (annos not points)
        # -----------------------------------------------------------------------------------------------------

        annos = info["annos"]
        names = annos["name"]
        bboxes = annos["bbox"]
        difficulty = annos["difficulty"]
        gt_idxes = annos["index"]
        num_obj = np.sum(annos["index"] >= 0)
        rbbox_cam = kitti.anno_to_rbboxes(annos)[:num_obj]
        rbbox_lidar = box_np_ops.box_camera_to_lidar(rbbox_cam, rect, Trv2c)
        if bev_only: # set z and h to limits
            assert coors_range is not None
            rbbox_lidar[:, 2] = coors_range[2]
            rbbox_lidar[:, 5] = coors_range[5] - coors_range[2]
        
        # ------------------------------------------------------------------------------------------------------ 
        #  other stuff
        # -----------------------------------------------------------------------------------------------------

        group_dict = {}
        group_ids = np.full([bboxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(bboxes.shape[0], dtype=np.int64)

        # ------------------------------------------------------------------------------------------------------ 
        #  test which points are in bboxes
        # -----------------------------------------------------------------------------------------------------

        point_indices = box_np_ops.points_in_rbbox(points, rbbox_lidar)

        # ------------------------------------------------------------------------------------------------------ 
        #  iterate over all objects in the annos
        # -----------------------------------------------------------------------------------------------------

        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{gt_idxes[i]}.bin"
            filepath = database_save_path / filename
            gt_points = points[point_indices[:, i]]

            gt_points[:, :3] -= rbbox_lidar[i, :3]

            # ------------------------------------------------------------------------------------------------------ 
            #  save points of gt boxes to files
            # -----------------------------------------------------------------------------------------------------

            with open(str(filepath)[:-3]+"pkl", 'wb') as file:
                pickle.dump(np.array(gt_points), file, 2)

            # debug
            # with open("/home/makr/Documents/uni/TU/3.Master/experiments/own/tf_3dRGB_pc/debug_rviz/points/bbox_pixels.pkl", 'wb') as file:
            #     pickle.dump(np.array(gt_points), file, 2)

            # ------------------------------------------------------------------------------------------------------ 
            #  save infos of gt boxes to single file
            # -----------------------------------------------------------------------------------------------------

            if names[i] in used_classes:
                if relative_path:
                    db_path = str(database_save_path.stem + "/" + filename)
                else:
                    db_path = str(filepath)
                db_info = {
                    "name": names[i],
                    "path": db_path,
                    "image_idx": image_idx,
                    "gt_idx": gt_idxes[i],
                    "box3d_lidar": rbbox_lidar[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }

                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                all_db_infos[names[i]].append(db_info)
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

if __name__ == '__main__':
    fire.Fire()
