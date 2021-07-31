import os
import numpy as np

# logs wandb loss
# =========================================
def log_wandb_loss(step_current, wandb,epoch_idx,loc_loss_reduced,cls_loss_reduced,dir_loss_reduced,loss,print_learning_rate):
    if step_current % 10 == 0:
        # log metrics using wandb.log
        wandb.log({'epochs': epoch_idx,
        'loc_loss_reduced': np.mean(loc_loss_reduced),
        'cls_loss_reduced': float(cls_loss_reduced), 
        'dir_loss_reduced': np.mean(dir_loss_reduced),
        'loss':float(loss),
        'learning_rate':float(print_learning_rate)})
        
# logs wandb eval
# =========================================   
def log_wandb_eval(wandb,mAP3d,mAPaos,mAPbev,log_wandb_eval):
    wandb.log({
        'ev_3d_50': mAP3d[0][0][0],
        'ev_3d_55': mAP3d[0][0][1],
        'ev_3d_60': mAP3d[0][0][2],
        'ev_3d_65': mAP3d[0][0][3],
        'ev_3d_70': mAP3d[0][0][4],
        'ev_3d_75': mAP3d[0][0][5],
        'ev_aos_50': mAPaos[0][0][0],
        'ev_aos_55': mAPaos[0][0][1],
        'ev_aos_60': mAPaos[0][0][2],
        'ev_aos_65': mAPaos[0][0][3],
        'ev_aos_70': mAPaos[0][0][4],
        'ev_aos_75': mAPaos[0][0][5],
        'ev_bev_70': mAPbev[0][0][0],
        'ev_bev_75': mAPbev[0][0][1],
        'ev_bev_80': mAPbev[0][0][2],
        'ev_bev_85': mAPbev[0][0][3],
        'ev_bev_90': mAPbev[0][0][4],
        'ev_bev_95': mAPbev[0][0][5],
        'avg':log_wandb_eval
    })

# creates config for wandb
# =========================================
def create_wandb_config(config):

    # 1. Start a W&B run
    wandb_config={"epochs": config["epochs_total"], 
                       "load_weights": config["load_weights"],
                       "sample_max_nums": config["train_input_reader"]["sample_max_nums"],
                       "sampler_max_point_collision": config["train_input_reader"]["sampler_max_point_collision"],
                       "sampler_min_point_collision": config["train_input_reader"]["sampler_min_point_collision"],
                       "sampler_noise_x_closer": config["train_input_reader"]["sampler_noise_x_closer"],
                       "sampler_noise_x_farther": config["train_input_reader"]["sampler_noise_x_farther"],
                       "sampler_noise_x_point": config["train_input_reader"]["sampler_noise_x_point"],
                       "sampler_noise_y": config["train_input_reader"]["sampler_noise_y"],
                       "sampler_noise_y": config["train_input_reader"]["sampler_noise_y"],
                       "batch_size": config["train_input_reader"]["batch_size"],
                       "groundtruth_rotation_uniform_noise": config["train_input_reader"]["groundtruth_rotation_uniform_noise"],
                       "groundtruth_localization_noise_std": config["train_input_reader"]["groundtruth_localization_noise_std"],
                       "global_random_rotation_range_per_object": config["train_input_reader"]["global_random_rotation_range_per_object"],
                       "global_rotation_uniform_noise": config["train_input_reader"]["global_rotation_uniform_noise"],
                       "global_scaling_uniform_noise": config["train_input_reader"]["global_scaling_uniform_noise"],
                       "global_loc_noise_std": config["train_input_reader"]["global_loc_noise_std"],
                       "anchor_area_threshold": config["train_input_reader"]["anchor_area_threshold"],
                       "voxel_size": config["model"]["second"]["voxel_generator"]["voxel_size"],
                       "max_number_of_points_per_voxel": config["model"]["second"]["voxel_generator"]["max_number_of_points_per_voxel"],
                       "max_number_of_voxels": config["model"]["second"]["voxel_generator"]["max_number_of_voxels"],
                       "num_class": config["model"]["second"]["num_class"],
                       "vfe_num_filters": config["model"]["second"]["voxel_feature_extractor"]["num_filters"],
                       "rpn_num_filters": config["model"]["second"]["rpn"]["num_filters"],
                       "rpn_num_upsample_filters": config["model"]["second"]["rpn"]["num_upsample_filters"],
                       "loss_alpha": config["model"]["second"]["loss"]["classification_loss"]["weighted_sigmoid_focal"]["alpha"],
                       "loss_gamma": config["model"]["second"]["loss"]["classification_loss"]["weighted_sigmoid_focal"]["gamma"],
                       "loss_sigma": config["model"]["second"]["loss"]["localization_loss"]["weighted_smooth_l1"]["sigma"],
                       "loss_code_weight": config["model"]["second"]["loss"]["localization_loss"]["weighted_smooth_l1"]["code_weight"],
                       "loss_classification_weight": config["model"]["second"]["loss"]["classification_weight"],
                       "loss_localization_weight": config["model"]["second"]["loss"]["localization_weight"],
                       "loss_direction_loss_weight": config["model"]["second"]["direction_loss_weight"],
                       "loss_pos_class_weight": config["model"]["second"]["pos_class_weight"],
                       "loss_neg_class_weight": config["model"]["second"]["neg_class_weight"],
                       "nms_pre_max_size": config["model"]["second"]["nms_pre_max_size"],
                       "nms_post_max_size": config["model"]["second"]["nms_post_max_size"],
                       "nms_post_max_size": config["model"]["second"]["nms_post_max_size"],
                       "nms_score_threshold": config["model"]["second"]["nms_score_threshold"],
                       "nms_iou_threshold": config["model"]["second"]["nms_iou_threshold"],
                       "opt_initial_learning_rate": config["train_config"]["optimizer"]["adam_optimizer"]["learning_rate"]["exponential_decay_learning_rate"]["initial_learning_rate"],
                       "opt_decay_steps": config["train_config"]["optimizer"]["adam_optimizer"]["learning_rate"]["exponential_decay_learning_rate"]["decay_steps"],
                       "opt_decay_factor": config["train_config"]["optimizer"]["adam_optimizer"]["learning_rate"]["exponential_decay_learning_rate"]["decay_factor"],
                       "opt_staircase": config["train_config"]["optimizer"]["adam_optimizer"]["learning_rate"]["exponential_decay_learning_rate"]["staircase"],
                       "opt_weight_decay": config["train_config"]["optimizer"]["adam_optimizer"]["weight_decay"]
                       }
    return wandb_config

# Creates the dirs for the training
# =========================================
def create_out_dir_base(project_dir_base, training, model_id):
    
    if not training:
        return project_dir_base + "/out" + "/model_" + model_id

    if training:
        out_dir_base = project_dir_base + "/out" + "/model_" + model_id
        if not os.path.exists(out_dir_base):
            os.makedirs(out_dir_base)
            print("SUCCESSFULLY CREATED DIR: {}".format(out_dir_base))
            print("***************************")
            return out_dir_base, model_id
        else:
            model_id = str(int(model_id) + 1)
            return create_out_dir_base(project_dir_base, training, model_id)


# Creates the subdirs for the training
# =========================================
def create_model_dirs_training(out_dir_base):
    out_dir_logs = out_dir_base + "/out_dir_logs"
    out_dir_images = out_dir_base + "/out_dir_images"
    out_dir_train_images = out_dir_base + "/out_dir_train_images"
    out_dir_checkpoints = out_dir_base + "/out_dir_checkpoints"

    if not os.path.exists(out_dir_logs):
        os.makedirs(out_dir_logs)
    if not os.path.exists(out_dir_images):
        os.makedirs(out_dir_images)
    if not os.path.exists(out_dir_train_images):
        os.makedirs(out_dir_train_images)
    if not os.path.exists(out_dir_checkpoints):
        os.makedirs(out_dir_checkpoints)

    return (out_dir_logs, out_dir_images, out_dir_train_images, out_dir_checkpoints)


# Creates the subdirs for the evaluation
# =========================================
def create_model_dirs_eval(out_dir_base):
    out_dir_eval_results = out_dir_base + "/out_dir_eval_results"
    out_dir_checkpoints = out_dir_base + "/out_dir_checkpoints"

    if not os.path.exists(out_dir_eval_results):
        os.makedirs(out_dir_eval_results)
    if not os.path.exists(out_dir_checkpoints):
        os.makedirs(out_dir_checkpoints)

    return (out_dir_eval_results, out_dir_checkpoints)
