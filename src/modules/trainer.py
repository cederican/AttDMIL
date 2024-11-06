import os
import torch as th
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model.mil_wrapper import AttDMILWrapper
from src.modules.logger import WandbLogger
from src.modules.utils import move_to_device


class Trainer:
    def __init__(
        self,
        *,
        device: str,
        rank: int,
        world_size: int,
        wrapper: AttDMILWrapper,
    ):
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.wrapper = wrapper
        self.model = wrapper.model

    def _configure_optimizers(
        self,
        opt_state_dict_lst: list | None = None,
        lr_scheduler_state_dict_lst: list | None = None,
    ):
        self.optimizer, self.lr_scheduler = self.wrapper.configure_optimizers()
        # if opt_state_dict_lst is not None:
        #     for i, opt in enumerate(self.optimizers):
        #         opt.load_state_dict(opt_state_dict_lst[i])
        # if lr_scheduler_state_dict_lst is not None:
        #     for i, lr in enumerate(self.lr_schedulers):
        #         lr.load_state_dict(lr_scheduler_state_dict_lst[i])
    
    def _init_val_metrics(
        self,
    ):
        self.wrapper.init_val_metrics()
    
    def train(
        self,
        *,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: WandbLogger,
        ckpt_save_path: str,
        ckpt_save_max: int,
        misc_save_path: str,
        val_every: int,
        patience: int,
    ):  
        global_step = 0
        self._configure_optimizers()
        self._init_val_metrics()
        self.logger = logger
        self.ckpt_save_path = ckpt_save_path
        self.ckpt_save_max = ckpt_save_max
        self.misc_save_path = misc_save_path
        self.val_every = val_every

        progress_bar = tqdm(
            range(1, epochs + 1),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            colour="blue",
        )
        progress_bar.set_description("Training")

        best_value = float("inf")
        no_improvement_count = 0
        for epoch in progress_bar:
            
            # Training
            global_step = self._train(train_loader, epoch, global_step)
            self.lr_scheduler.step()
    
            # Validation
            if self.val_every is not None and epoch % self.val_every == 0:
                global_step = self._validate(epoch, val_loader, global_step)
                current_val_loss, current_val_error, current_val_auc  = self.wrapper.val_metrics.compute()["val/loss"], self.wrapper.val_metrics.compute()["val/error"], self.wrapper.val_metrics.compute()["val/auc"]
                # stopping criteria based on proposed approach in the paper
                combined_stop_metric = current_val_loss + current_val_error

                if combined_stop_metric < best_value:
                    best_auc = current_val_auc
                    best_value = combined_stop_metric
                    no_improvement_count = 0

                    # Visualize
                    self.visualize(val_loader, global_step)
                    global_step += 1
                    
                    if self.ckpt_save_path is not None:
                        self._save_model(f"best_ep={epoch}_val_loss={best_value:.4f}")
                else:
                    no_improvement_count += 1

                # Early stopping condition
                if no_improvement_count >= patience:
                    self.logger.log_AUC(self.misc_save_path, best_auc)
                    print(f"Early stopping at epoch {epoch} due to no improvement in validation loss for {patience} epochs.")
                    break
                self.wrapper.val_metrics.reset()

        if self.ckpt_save_path is not None:
            self._save_model("last")

        self.logger.finish()

    def _train(
            self,
            train_loader: DataLoader,
            epoch: int, 
            global_step: int,
    ):
        self.model.train()
        loader = tqdm(
            train_loader,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            colour="blue",
            leave=False,
        )
        loader.set_description(f"Epoch {epoch}")

        for batch_idx, batch in enumerate(loader):
            
            self.optimizer.zero_grad()
            batch = move_to_device(batch, self.device)
            
            loss_dict = self.wrapper.training_step(self.model, batch)

            self.optimizer.step()
            self.lr_scheduler.step()
            global_step += 1

            for key, value in loss_dict.items():
                self.logger.log_scalar(f"train/{key}", value.item(), global_step)
            self.logger.log_scalar("lr", self.lr_scheduler.get_last_lr()[0], global_step)
                    
        return global_step

    def _validate(
            self, 
            epoch: int, 
            val_loader: DataLoader, 
            global_step: int
    ):
        self.model.eval()
        with th.no_grad():
            loader = tqdm(
                val_loader,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                colour="blue",
                leave=False,
            )
            loader.set_description(f"Validation on Epoch {epoch}")
            
            for batch_idx, batch in enumerate(loader):
                batch = move_to_device(batch, self.device)

                self.wrapper.validation_step(batch)
            global_step += 1
            computed_metrics = self.wrapper.val_metrics.compute()
            for name, value in computed_metrics.items():
                self.logger.log_scalar(f"{name}", value, global_step)
        
        return global_step

    def _save_model(self, ckpt_name: str):
        if not os.path.exists(self.ckpt_save_path):
            os.makedirs(self.ckpt_save_path)

        ckpt_files = [os.path.join(self.ckpt_save_path, ckpt_file) for ckpt_file in os.listdir(self.ckpt_save_path) if ckpt_file.endswith('.pt')]
        # logic to save just a num of checkpoints
        if len(ckpt_files) >= self.ckpt_save_max:
            file_to_remove = sorted(ckpt_files, key=os.path.getctime)[0]
            os.remove(file_to_remove)

        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

        th.save(state_dict, os.path.join(self.ckpt_save_path, f"{ckpt_name}.pt"))

    def visualize(
        self,
        val_loader: DataLoader,
        global_step: int,
    ):
        self.model.eval()
        with th.no_grad():
            loader = tqdm(
                val_loader,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                colour="blue",
                leave=False,
            )
            loader.set_description(f"Visualization")
            random_visualize_idx = random.randint(0, 20)

            for batch_idx, batch in enumerate(loader):
                batch = move_to_device(batch, self.device)

                if batch_idx == random_visualize_idx:
                    self.wrapper.visualize_step(self.model, batch, self.misc_save_path, global_step)
                    break
            