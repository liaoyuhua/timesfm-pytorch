from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "process_group",
    "moving_average",
    "freq_map",
    "masked_mean_std",
    "shift_padded_seq",
]


def process_group(key, group, value_name, forecast_context_len):
    group = group.tail(forecast_context_len)
    return np.array(group[value_name], dtype=np.float32), key


def moving_average(arr, window_size):
    """Calculates the moving average using NumPy's convolution function."""
    # Pad with zeros to handle initial window positions
    arr_padded = np.pad(arr, (window_size - 1, 0), "constant")
    smoothed_arr = np.convolve(arr_padded, np.ones(window_size), "valid") / window_size
    return [smoothed_arr, arr - smoothed_arr]


def freq_map(freq: str):
    """Returns the frequency map for the given frequency string."""
    freq = str.upper(freq)
    if (
        freq.endswith("H")
        or freq.endswith("T")
        or freq.endswith("MIN")
        or freq.endswith("D")
        or freq.endswith("B")
        or freq.endswith("U")
    ):
        return 0
    elif freq.endswith(("W", "M", "MS")):
        return 1
    elif freq.endswith("Y") or freq.endswith("Q"):
        return 2
    else:
        raise ValueError(f"Invalid frequency: {freq}")


def masked_mean_std(
    inputs: torch.Tensor, paddings: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates mean and standard deviation of arr across axis 1.

    It should exclude values where pad is 1.

    Args:
        inputs: A PyTorch tensor of shape [b, n, p].
        paddings: A PyTorch tensor of shape [b, n, p] with values 0 or 1.

    Returns:
        A tuple containing the mean and standard deviation of arr. We return the
        statistics of the first patch with more than three non-padded values.
    """
    # Selecting the first patch with more than 3 unpadded values.
    pad_sum = torch.sum(1 - paddings, dim=2)

    def _get_patch_index(arr: torch.Tensor):
        indices = torch.argmax((arr >= 3).int(), dim=1)
        row_sum = (arr >= 3).sum(dim=1)
        return torch.where(row_sum == 0, arr.shape[1] - 1, indices)

    patch_indices = _get_patch_index(pad_sum)
    bidxs = torch.arange(inputs.shape[0])

    arr = inputs[bidxs, patch_indices, :]
    pad = paddings[bidxs, patch_indices, :]

    # Create a mask where P is 0
    mask = 1 - pad

    # Calculate the number of valid elements
    num_valid_elements = torch.sum(mask, dim=1)

    num_valid_elements = torch.where(num_valid_elements == 0, 1, num_valid_elements)

    # Calculate the masked sum and squared sum of M
    masked_sum = torch.sum(arr * mask, dim=1)
    masked_squared_sum = torch.sum((arr * mask) ** 2, dim=1)

    # Calculate the masked mean and standard deviation
    masked_mean = masked_sum / num_valid_elements
    masked_var = masked_squared_sum / num_valid_elements - masked_mean**2
    masked_var = torch.where(masked_var < 0.0, 0.0, masked_var)
    masked_std = torch.sqrt(masked_var)

    return masked_mean, masked_std


def shift_padded_seq(mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    """Shifts rows of seq based on the first 0 in each row of the mask."""
    num = seq.shape[1]

    # Find the index of the first 0 in each row of the mask
    first_zero_idx = torch.argmin(mask, dim=1)

    # Create a range array for indexing
    idx_range = torch.arange(num)

    def shift_row(seq_row, shift):
        shifted_idx = (idx_range - shift) % num
        shifted_row = seq_row[shifted_idx]
        return shifted_row

    # Apply the shift_row function to each row of seq based on the corresponding first_zero_idx
    shifted_seq = torch.stack(
        [shift_row(seq_row, shift) for seq_row, shift in zip(seq, first_zero_idx)]
    )

    return shifted_seq
