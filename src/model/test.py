import os
import time
import torch as th
from src.modules.config import MILModelConfig, MNISTBagsConfig, MILPoolingConfig
from src.dataset.dataset import MNISTBags
from src.model.mil_wrapper import AttDMILWrapper
from src.model.mil import MILModel
from src.modules.logger import prepare_folder, get_run_name, bcolors, WandbLogger
from src.modules.trainer import Trainer
import torch.utils.data as data_utils
from torchinfo import summary

def test(config=None):
    base_log_dir = f'/home/pml06/dev/attdmil/logs'

    ckpt_save_path = "/home/pml06/dev/attdmil/logs/mu10/embedding_poolattention_mu10_var2_num200/checkpoints/best_ep=10_val_loss=0.3138.pt"
    misc_save_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_save_path)), 'misc')
    run_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_save_path)))

    print(f"{bcolors.OKBLUE}Start test with run_name: {bcolors.BOLD}{run_name}{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}Log dir: {bcolors.BOLD}{base_log_dir}{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}Checkpoint save path: {bcolors.BOLD}{ckpt_save_path}{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}Misc save path: {bcolors.BOLD}{misc_save_path}{bcolors.ENDC}")

    # To Do: Load config with yaml file, save yaml file in training

    test_config = MILModelConfig(
        device=th.device("cuda" if th.cuda.is_available() else "cpu"),
        mode='embedding',
        epochs=200,
        batch_size=1,
        img_size=(1, 28, 28),
        train_dataset_config=MNISTBagsConfig(
            seed=1,
            positive_num=9,
            mean_bag_size=10,
            var_bag_size=2,
            num_bags=2,
            train=False,
            test_attention=True
        ),
        mil_pooling_config=MILPoolingConfig(
            pooling_type='attention',
            feature_dim=500,
            attspace_dim=128,
            attbranches=1
        ),
        ckpt_path=ckpt_save_path,
        lr=0.0005,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
        step_size=1000000,
        gamma=0.1,
        ckpt_save_path=None,
        misc_save_path=misc_save_path,
        val_every=None,
        save_max=None,
        patience=None,
    )
    test_loader = data_utils.DataLoader(
        MNISTBags(**test_config.train_dataset_config.__dict__),
        batch_size=test_config.batch_size,
        shuffle=False
    )
    model = MILModel(mil_model_config=test_config).to(test_config.device)
    wrapper = AttDMILWrapper(model=model, config=test_config, epochs=test_config.epochs)
    summary(model, input_data=th.rand(test_config.batch_size, *test_config.img_size).to(test_config.device))
    trainer = Trainer(
        device=test_config.device,
        wrapper=wrapper,
        misc_save_path=test_config.misc_save_path,
    )
    trainer.test_visualize(test_loader)


# def main_sweep():
#     sweep_config = {
#         'method': 'grid',
#         'metric': {
#             'name': 'val/loss',
#             'goal': 'minimize' 
#             },
#         'parameters': {
#             'mean_bag_size': {
#                 'value': 10             # [10, 50, 100] fixed
#             },
#             'var_bag_size': {
#                 'value': 2             # [2, 10, 20] fixed   
#             },
#             'num_bags': {
#                 'values': [50, 100, 150]     # [50, 100, 150, 200, 300, 400, 500]
#             },
#             'mode': {
#                 'values': ['embedding', 'instance']     # ['embedding', 'instance']
#             },
#             'pooling_type': {
#                 'values': ['max', 'mean', 'attention', 'gated_attention']       # ['max', 'mean', 'attention', 'gated_attention']
#             },
#         }
#     }
#     return sweep_config
   
if __name__ == "__main__":

    test()

    # for i in range(5):
    #     project_name = 'AttDMIL-PML-MNIST'
    #     # Initialize a sweep
    #     sweep_config = main_sweep()
    #     sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    #     wandb.agent(sweep_id, function=train, count=56)
    #     print(f"{bcolors.OKGREEN}Sweep {i} completed!{bcolors.ENDC}")
    #     time.sleep(4)
    print("All sweeps completed successfully!")
