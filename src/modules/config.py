from types import SimpleNamespace

class MNISTBagsConfig(SimpleNamespace):
    seed: int
    positive_num: int
    mean_bag_size: int
    var_bag_size: int
    num_bags: int
    train: bool

class MILPoolingConfig(SimpleNamespace):
    pooling_type: str
    feature_dim: int #M
    attspace_dim: int #L
    attbranches: int 

class MILModelConfig(SimpleNamespace):
    mode: str
    batch_size: int
    img_size: tuple[int, int, int]
    dataset_config: MNISTBagsConfig
    mil_pooling_config: MILPoolingConfig