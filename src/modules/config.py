from types import SimpleNamespace


class MNISTBagsConfig(SimpleNamespace):
    """
   Configuration class for MNISTBags dataset.

   Attributes:
       seed (int): Seed for random number generation.
       positive_num (int): The digit that is considered the positive label in bags (e.g., 9).
       mean_bag_size (int): The average number of instances (images) per bag.
       var_bag_size (float): The variance in the number of instances per bag.
       num_bags (int): The total number of bags in the dataset.
       train (bool): Flag indicating if this dataset instance is for training or testing.
       test_attention (bool): Flag indicating if attention testing is enabled.
   """
    seed: int
    positive_num: int
    mean_bag_size: int
    var_bag_size: float
    num_bags: int
    train: bool
    test_attention: bool


class MILPoolingConfig(SimpleNamespace):
    """
    Configuration class for MILPooling.

    Attributes:
        pooling_type (str): Type of pooling ('max', 'mean', 'attention', 'gated_attention').
        feature_dim (int): Dimension of the feature space.
        attspace_dim (int): Dimension of the attention space.
        attbranches (int): Number of attention branches.
    """
    pooling_type: str
    feature_dim: int  #M
    attspace_dim: int  #L
    attbranches: int


class MILModelConfig(SimpleNamespace):
    """
        Configuration class for MILModel.

        Attributes:
            device (str): Device to run the model on ('cpu' or 'cuda').
            mode (str): Mode of the model ('instance' or 'embedding').
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            img_size (tuple[int, int, int]): Size of the input images.
            dataset_config (MNISTBagsConfig): Configuration for the dataset.
            mil_pooling_config (MILPoolingConfig): Configuration for the MIL pooling.
            ckpt_path (str): Path to the checkpoint file.
            lr (float): Learning rate.
            betas (tuple[float, float]): Betas for Adam optimizer.
            weight_decay (float): Weight decay for Adam optimizer.
            T_0 (int): T_0 for Cosine Annealing LR scheduler.
            T_mult (int): T_mult for Cosine Annealing LR scheduler.
            eta_min (float): eta_min for Cosine Annealing LR scheduler.
            step_size (int): Step size for Step LR scheduler.
            gamma (float): Gamma for Step LR scheduler.
            ckpt_save_path (str): Path to save the checkpoint file.
            misc_save_path (str): Path to save miscellaneous files.
            val_every (int): Validation frequency.
            save_max (int): Maximum number of checkpoints to save.
            patience (int): Patience for early stopping.
    """
    device: str
    mode: str
    epochs: int
    batch_size: int
    img_size: tuple[int, int, int]
    dataset_config: MNISTBagsConfig
    mil_pooling_config: MILPoolingConfig
    ckpt_path: str
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    T_0: int
    T_mult: int
    eta_min: float
    step_size: int
    gamma: float
    ckpt_save_path: str
    misc_save_path: str
    val_every: int
    save_max: int
    patience: int
