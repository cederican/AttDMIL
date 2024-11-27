import torch as th
import ast
import h5py
import numpy as np
import torch.utils.data as data_utils
from torchvision import datasets, transforms

from src.modules.config import HistoBagsConfig

class HistoDataset(data_utils.Dataset):
    def __init__(
        self, 
        *,
        h5_path: str,
        color_normalize: bool,
        datatype: str,
        mode: str,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.color_normalize = color_normalize
        self.mode = mode
        self.datatype = datatype

        with h5py.File(self.h5_path, "r") as h5:
            if self.mode == 'train':
                self.name_lst = list(h5["train"].keys())
            elif self.mode == 'test':
                self.name_lst = list(h5["test"].keys())
                
           
    def __len__(self):
        pass

    def __getitem__(self, idx):
        if self.datatype == "features":
            return self._get_features(idx)
        elif self.datatype == "patches":
            return self._get_patches(idx)
        elif self.datatype == "coordinates":
            return self._get_patch_coordinates(idx)
    
    def _get_patch_coordinates(self, idx):
        
        coordinates = {}
        case_name = self.name_lst[idx]
        self.h5 = h5py.File(self.h5_path, "r")
        self.patch_lst = list(self.h5[self.mode][case_name].keys())
        self.patch_lst.remove("features")
        self.patch_lst.remove("metadata")

        print(f"Case: {case_name}")
        print(f"Number of patches: {len(self.patch_lst)}")
        for patch_name in self.patch_lst:
            coordinates[patch_name] = ast.literal_eval(self.h5[self.mode][case_name][patch_name].attrs["position_abs"])

        return coordinates


    def _get_patches(self, idx):
        pass

    def _get_features(self, idx):

        case_name = self.name_lst[idx]
        self.h5 = h5py.File(self.h5_path, "r")
        features = self.h5[self.mode][case_name]["features"]['feature_matrix'][:]
        label = self.h5[self.mode][case_name].attrs["label"]
        label = self.label_to_logits(label)
        cls = self.h5[self.mode][case_name].attrs["class"]
        self.h5.close()
        
        return th.tensor(features, dtype=th.float32), th.tensor(label, dtype=th.float32), th.tensor(cls, dtype=th.float32)
    
    def label_to_logits(self, label):

        if label == "normal":
            return th.tensor(0, dtype=th.float32)
        elif label == "tumor":
            return th.tensor(1, dtype=th.float32)
    

if __name__ == "__main__":
    dataset = HistoDataset(
        h5_path="/home/pml06/dev/attdmil/HistoData/camelyon16_20x.h5",
        color_normalize=False,
        datatype="coordinates",
        mode="train",
    )
    data = dataset[0]
    print(data.shape)