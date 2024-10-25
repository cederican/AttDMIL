from types import SimpleNamespace

class MNISTBagsConfig(SimpleNamespace):
    seed: int
    positive_num: int
    mean_bag_size: int
    var_bag_size: int
    num_bags: int
    train: bool