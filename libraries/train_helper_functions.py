import os

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
