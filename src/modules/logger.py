import os
import shutil
from abc import ABC, abstractmethod
import wandb
import yaml
import numpy as np

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def get_run_name(log_path: str, run_name: str = None, cleanup: bool = False) -> str:
    if run_name is None:
        prefix = "run_"
        for i in range(1, 100):
            path_cand = os.path.join(log_path, f"{prefix}{i:02d}")
            if not os.path.exists(path_cand):
                os.mkdir(path_cand)
                break
        return f"{prefix}{i:02d}"
    else:
        path = os.path.join(log_path, run_name)
        if not os.path.exists(path):
            os.mkdir(path)
        elif cleanup:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            os.mkdir(path)
        return run_name


def save_config(log_path: str, run_name: str, config: dict):
    yaml.Dumper.ignore_aliases = lambda *args: True
    config_path = os.path.join(log_path, run_name, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(file_path: str):
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def prepare_folder(log_dir, run_name):
    ckpt_save_path = os.path.join(log_dir, run_name, "checkpoints")
    misc_save_path = os.path.join(log_dir, run_name, "misc")
    if not (os.path.isdir(ckpt_save_path)):
        os.makedirs(ckpt_save_path, exist_ok=True)

    if not (os.path.isdir(misc_save_path)):
        os.makedirs(misc_save_path, exist_ok=True)

    return ckpt_save_path, misc_save_path


class AbstractLogger(ABC):
    @abstractmethod
    def log_scalar(self, tag, scalar_value, global_step):
        raise NotImplementedError

    @abstractmethod
    def log_volume(self, tag, obj3d_file_path, global_step):
        raise NotImplementedError

    @abstractmethod
    def finish(self):
        pass


class LoggerCollection(AbstractLogger):
    def __init__(self, loggers: list):
        self.loggers = loggers

    def log_scalar(self, tag, scalar_value, global_step):
        for logger in self.loggers:
            logger.log_scalar(tag, scalar_value, global_step)
            
    def log_dict(self, dict_of_values, global_step):
        for logger in self.loggers:
            logger.log_dict(dict_of_values, global_step)

    def log_volume(self, tag, obj3d_file_path, global_step):
        for logger in self.loggers:
            logger.log_volume(tag, obj3d_file_path, global_step)
            
    def log_list(self, list_of_values, epoch):
        for logger in self.loggers:
            logger.log_list(list_of_values, epoch)
    
    def log_segmentations(self, bg_imgs, pred_masks, true_masks):
        for logger in self.loggers:
            logger.log_segmentations(bg_imgs, pred_masks, true_masks)
    
    def log_images(self, images, epoch):
        for logger in self.loggers:
            logger.log_images(images, epoch)
    
    def log_histogram(self, tag, values, global_step):
        for logger in self.loggers:
            logger.log_histogram(tag, values)

    def finish(self):
        for logger in self.loggers:
            logger.finish()

class WandbLogger(AbstractLogger):
    def __init__(self, *, run_name=None, log_dir=None):
        self.wandb = wandb
        self.dir = os.path.join(log_dir, run_name)

    def log_scalar(self, tag, scalar_value, global_step):
        self.wandb.log({tag: scalar_value}, step=global_step)
        
    def log_dict(self, dict_of_values, global_step):
        self.wandb.log(dict_of_values, step=global_step)

    def log_volume(self, tag, obj3d_file_path, global_step):
        try:
            with open(obj3d_file_path) as obj_file:
                obj3d = wandb.Object3D(obj_file)
                wandb.log({tag: obj3d}, step=global_step)
        except Exception as _:
            pass
    
    def log_AUC(self, misc_save_path, value):
        file_path = os.path.join(misc_save_path, 'metric_5runs.txt')

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                values = [float(line.strip()) for line in lines if line.strip().replace('.', '', 1).isdigit()]
        else:
            values = []

        values.append(value)

        # Check if there are 5 values
        if len(values) == 5:
            # Calculate mean and standard deviation
            mean_value = np.mean(values)
            std_value = np.std(values)

            # Write the values, mean, and std to the file
            with open(file_path, 'a') as file:
                file.write(f'{value}\n')  # Write the 5th value
                file.write(f'Mean: {mean_value:.3f}\n')
                file.write(f'Std: {std_value:.3f}\n\n')  # Separate entries for readability

            # Reset values list to start a new set of 5
            values.clear()

        else:
            # If fewer than 5 values, just append the current value to the file
            with open(file_path, 'a') as file:
                file.write(f'{value}\n')
    
    def log_histogram(self, tag, values, global_step):
        pass
    
    def log_segmentations(self, bg_imgs, pred_masks, true_masks):
        """
        Args:
            _type_: _description_

        Returns:
            _type_: _description_
        """
        table = wandb.Table(columns=["ID", "Image"])
        
        ids = np.array(range(bg_imgs.shape[0]))
        
        segmentation_classes = [
            'background', 'bone' 
        ]
        
        def labels():
            l = {}
            for i, label in enumerate(segmentation_classes):
              l[i] = label
            return l
        
        def wb_mask(bg_img, pred_mask, true_mask):
            return wandb.Image(bg_img, masks={
              "prediction" : {"mask_data" : pred_mask, "class_labels" : labels()},
              "ground truth" : {"mask_data" : true_mask, "class_labels" : labels()}})
        
        for id, img, pred_mask, true_mask in zip(ids, bg_imgs, pred_masks, true_masks):
            table.add_data(id, wb_mask(img[0], pred_mask[0], true_mask[0]))
        
        self.wandb.log({"Table" : table})
    
    def log_images(self, images, epoch):
        images = np.array(images)
        #images = images / 2 + 0.5
        #images = images.clip(0, 1)
        self.wandb.log({"images": [wandb.Image(img, caption="Augmented input data") for img in images]})
        

    def log_list(self, list_of_values, epoch):
        with open(os.path.join(self.dir, "important_codebookentires.txt"), "w") as file:
            for number in list_of_values:
                file.write(f"{number}\n")

    def finish(self):
        self.wandb.finish()