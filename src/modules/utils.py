import torch as th
import os
from PIL import Image
import numpy as np

def move_to_device(nested_list, device):
    """
    Moves a nested list of tensors to the specified device.

    Args:
        nested_list (list): The nested list of tensors.
        device (torch.device): The device to which the tensors should be moved.
    """
    if isinstance(nested_list, th.Tensor):
        return nested_list.to(device)
    elif isinstance(nested_list, list):
        return [move_to_device(item, device) for item in nested_list]
    elif isinstance(nested_list, dict):
        return {key: move_to_device(value, device) for key, value in nested_list.items()}
    elif isinstance(nested_list, str):
        return nested_list
    else:
        raise TypeError("All elements must be either tensors or lists of tensors.")
    

def label_to_logits(label):

        if label == "normal":
            return th.tensor(0, dtype=th.float32)
        elif label == "tumor":
            return th.tensor(1, dtype=th.float32)

def logits_to_label(logit):
    """
    Convert a logit (0.0 or 1.0) back to a label.
    """
    if logit == 0:
        return "normal"
    elif logit == 1:
        return "tumor"
    else:
        raise ValueError(f"Invalid logit for label: {logit}")
        
def cls_to_logits(cls):
        if cls == "negative":
            return th.tensor(0, dtype=th.float32)
        elif cls == "micro":
            return th.tensor(1, dtype=th.float32)
        elif cls == "macro":
            return th.tensor(2, dtype=th.float32)
        
def logits_to_cls(logit):
    """
    Convert a logit (0.0, 1.0, or 2.0) back to a class.
    """
    if logit == 0:
        return "negative"
    elif logit == 1:
        return "micro"
    elif logit == 2:
        return "macro"
    else:
        raise ValueError(f"Invalid logit for class: {logit}")
    

def get_tumor_annotation(
        case_name: str,
):
    annotations_path = "/home/space/datasets/camelyon16/annotations"
    annotations_path = f"{annotations_path}/{case_name}.png"
    if not os.path.isfile(annotations_path):
        print(f"Image {case_name} not found in {annotations_path}")
        return None
    with Image.open(annotations_path) as img:
        rgba_array = np.array(img)
        r_channel = rgba_array[:, :, 0]
        g_channel = rgba_array[:, :, 1]
        b_channel = rgba_array[:, :, 2]

        red_mask = (r_channel == 255) & (g_channel == 0) & (b_channel == 0)
        positions = np.argwhere(red_mask)
        red_positions = {i: (x, y) for i, (y, x) in enumerate(positions)}
    return rgba_array 

def cut_off(
        y_instance_pred,
        top_k=10,
        threshold=0.9,
):
    if not isinstance(y_instance_pred, th.Tensor):
        y_instance_pred = th.tensor(y_instance_pred)
    top_values, top_indices = th.topk(y_instance_pred, top_k)
    cumulative_sum = th.sum(top_values)
    if cumulative_sum*100 > threshold:
        modified_array = y_instance_pred.clone()
        modified_array[top_indices] = 0.0
        print(f"Attention Map cut off at {top_k} positions")
    else:
        modified_array = y_instance_pred
        print(f"Attention Map not cut off at {top_k} positions")
    return modified_array
    