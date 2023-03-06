import os
import shutil
from detectron2.config import CfgNode as CN

def setup_paths(cfg, args):
    folder_name = args.folder_name
    #dataset_name = cfg.EXP.DATASET.lower()
    experiment_folder = f"{args.exp_id}"
    base_path = os.path.join(f"./output/{folder_name}", experiment_folder)
    cfg.OUTPUT_DIR = base_path


def create_experiment_directory(base_path, eval_only, resume=False):
    if eval_only or resume:
        if os.path.exists(base_path):
            pass  # If we do evaluation we dont want to destroy our saved models
        else:  # in case we loaded a pretrained model
            os.makedirs(base_path)
            os.makedirs(os.path.join(base_path, "inference"))
    else:
        # Zip old experiment to not destroy it right away
        if os.path.exists(base_path):
            shutil.make_archive(base_path, 'zip', base_path)
            shutil.rmtree(base_path)
        os.makedirs(base_path)
        os.makedirs(os.path.join(base_path, "tensorboard"))
        os.makedirs(os.path.join(base_path, "inference"))
        os.makedirs(os.path.join(base_path, "models"))


def save_exp_setup(args, cfg):
    """
    Detectron2 overwrites saved configs when evaluating the same model.
    This saves the original configs for sanity checks later.
    """
    base_path = cfg.OUTPUT_DIR
    with open(os.path.join(base_path, 'experiment_configs.txt'), 'w') as f:
        print("Command line arguments: \n", file=f)
        print(args, "\n", file=f)
        print("Detectron arguments arguments: \n", file=f)
        print(cfg, file=f)
