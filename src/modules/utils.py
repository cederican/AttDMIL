import torch as th

def move_to_device(nested_list, device):
    if isinstance(nested_list, th.Tensor):
        return nested_list.to(device)
    elif isinstance(nested_list, list):
        return [move_to_device(item, device) for item in nested_list]
    else:
        raise TypeError("All elements must be either tensors or lists of tensors.")
