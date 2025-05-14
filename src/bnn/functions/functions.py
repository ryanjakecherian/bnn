import torch
import torch.autograd
import numpy as np

import bnn.type

__all__ = [
    'binarise',
    'ternarise',
    'int_matmul',
    'one_hot_argmax',
    'image_to_binary_vector',
]


def binarise(x: torch.Tensor, threshold: int = 0) -> torch.Tensor:
    out = torch.ones_like(x)
    out[x < threshold] = 0 #changed to -1 for new network

    return out.to(bnn.type.INTEGER)


def ternarise(
    x: torch.Tensor,
    threshold_lo: int = 0,
    threshold_hi: int = 0,
) -> torch.Tensor:
    """Ternarise Tensor, numbers on the threshold round up"""
    if threshold_hi < threshold_lo:
        raise ValueError('lo thresh cannot be larger than hi thresh!')

    out = torch.zeros_like(x)
    out[x >= threshold_hi] = 1
    out[x < threshold_lo] = -1

    return out


TORCH_FLOAT_TYPE = torch.float16


# HACK - this is technically "unsafe", but should be fine for reasonable layer sizes!
def int_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    AB_float = torch.matmul(A.to(TORCH_FLOAT_TYPE), B.to(TORCH_FLOAT_TYPE))
    AB_int = torch.round(AB_float).to(bnn.type.INTEGER)
    return AB_int


def one_hot_argmax(x: torch.Tensor) -> torch.Tensor:
    argmax = torch.argmax(x, dim=-1, keepdim=True)

    # empty array
    out = torch.full_like(x, fill_value=-1)
    # add 1s at argmax
    out.scatter_(dim=-1, index=argmax, value=1)
    return out


def one_hot_argmax0(x: torch.Tensor) -> torch.Tensor:
    argmax = torch.argmax(x, dim=-1, keepdim=True)
    # empty array
    out = torch.full_like(x, fill_value=0) #switched from -1 to 0 for new network
    # add 1s at argmax
    out.scatter_(dim=-1, index=argmax, value=1)
    return out



def image_to_binary_vector(images):
    # Ensure the images are a PyTorch tensor of type uint8
    images = images.to(torch.uint8)

    # Check if the images have the correct shape
    if images.shape[1:] != (1, 28, 28):
        raise ValueError("Input images must be of shape (batch_size, 1, 28, 28)")

    batch_size = images.shape[0]
    binary_vectors = torch.zeros((batch_size, 28 * 28 * 8), dtype=torch.uint8)

    # Convert each pixel in each image to an 8-bit binary representation
    for b in range(batch_size):
        for i in range(28):
            for j in range(28):
                pixel_value = images[b, 0, i, j]  # Get the pixel value
                binary_representation = torch.tensor([int(bit) for bit in f"{pixel_value.item():08b}"], dtype=torch.uint8)
                binary_vectors[b, (i * 28 + j) * 8:(i * 28 + j + 1) * 8] = binary_representation

    return binary_vectors
    
    # Ensure the image is a PyTorch tensor and is of the correct type
    image = (image*255).to(torch.uint8)

    # Check if the image is of the correct shape
    if image.shape != (28, 28):
        raise ValueError("Input image must be of shape (28, 28)")

    # Create an empty binary vector of length 28 * 28 * 8
    binary_vector = torch.zeros((28 * 28 * 8,), dtype=torch.uint8)

    # Convert each pixel to an 8-bit binary representation
    for i in range(28):
        for j in range(28):
            pixel_value = image[i, j]
            # Convert pixel value to binary and flatten it into the vector
            binary_representation = torch.tensor(list(f"{pixel_value.item():08b}"), dtype=torch.uint8)
            binary_vector[(i * 28 + j) * 8:(i * 28 + j + 1) * 8] = binary_representation

    return binary_vector
