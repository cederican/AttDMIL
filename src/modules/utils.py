import torch as th

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
    else:
        raise TypeError("All elements must be either tensors or lists of tensors.")
