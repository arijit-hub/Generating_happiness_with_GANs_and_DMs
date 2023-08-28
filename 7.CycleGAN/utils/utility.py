"""Implements various utility functions.

Code for weights initialization taken from: 
    https://github.com/HReynaud/CycleGan_PL/blob/main/utils/helpers.py
"""
import sys

sys.path.append("data")

import torch
import torch.nn as nn
import numpy as np

import os
import matplotlib.pyplot as plt
# from data import EnhanceDataModule


def init_weights(m):
    """Initializes the weights of the network.

    Parameters
    ----------
    m : torch.nn.Module
        The layer whose weights are to be initialized.
    """

    ## Setting the weights of convolutional layers ##
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

    ## Setting the weights of batch normalization layers ##
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    ## Setting the weights of transposed convolutional layers ##
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


def normalize_imgs(batch_imgs):
    """Normalizes images to be equally spaced brightness-wise.

    Parameters
    ----------
    batch_imgs : torch.tensor or np.ndarray
        A batch of images.
    """

    ## Normalizing the images ##
    if type(batch_imgs) != np.ndarray:
        batch_imgs = batch_imgs.detach()
    normalized_imgs = batch_imgs * 0.5 + 0.5
    return normalized_imgs


def make_img_grid(
    imgs: list,
    imgs_labels: list,
    normalize: bool = True,
    save: bool = True,
    output_dir: str = "generated_imgs",
    output_file_name: str = "image.png",
):
    """Makes a grid of image.

    Parameters
    ----------
    imgs : list[torch.tensor]
        List of images.
        Each element in img must be a batch of images.
        Each batch of images must be a torch.tensor.

    imgs_labels : list[str]
        List of labels for each column of output image.

    normalize : bool
        Whether to normalize the images or not.
        [Default : True]

    save : bool
        Whether to save the image or not.
        [Default : True]

    output_dir : str
        Location where to save the image.
        [Default : ""]

    output_file_name : str
        Name of the image to be saved.
        [Default : "image.png"]
    """

    assert len(imgs) == len(imgs_labels), "Number of images and labels must be same."

    ## Making the image grid ##
    num_cols = len(imgs)

    if normalize:
        imgs = [normalize_imgs(each_img) for each_img in imgs]

    for i, each_img_stack in enumerate(imgs):
        if len(each_img_stack.shape) == 3:
            img_stack = each_img_stack.cpu().detach().numpy()
            img_stack = img_stack.reshape(
                img_stack.shape[1], img_stack.shape[2] * img_stack.shape[0]
            )

        elif len(each_img_stack.shape) == 4:
            img_stack = each_img_stack.permute(0, 2, 3, 1).detach().cpu().numpy()
            if img_stack.shape[1] != 1:
                img_stack = img_stack.reshape(
                    img_stack.shape[1] * img_stack.shape[0],
                    img_stack.shape[2],
                    img_stack.shape[3],
                )

            else:
                img_stack = img_stack.reshape(
                    img_stack.shape[0], img_stack.shape[2], img_stack.shape[3]
                )

        plt.subplot(1, num_cols, i + 1)

        if len(img_stack.shape) == 3:
            plt.imshow(img_stack, cmap="gray")
        else:
            plt.imshow(img_stack)
        plt.axis("off")
        plt.title(imgs_labels[i])

    if save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, output_file_name)
        plt.savefig(filename, dpi=300)