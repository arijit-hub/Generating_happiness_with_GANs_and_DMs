"""Implementation of loss functions.
"""

import torch
import torch.nn as nn

## Implementation of the generator loss ##


def generator_loss(
    discriminator: nn.Module,
    current_num_block: int,
    alpha: float,
    predicted_images: torch.Tensor,
):
    """Implements the generator loss which is nothing but
    maximizing the discriminator score for the predicted
    images.

    Parameters
    ----------
    discriminator : nn.Module
        The discriminator module.

    current_num_block : int
        The current number of blocks in the discriminator.

    alpha : float
        The current alpha value.

    predicted_images : torch.Tensor
        The predicted images from the generator.
    """

    ## Calculating the discriminator score ##
    discriminator_score = discriminator(predicted_images, current_num_block, alpha)

    ## Calculating the loss ##
    loss = -torch.mean(discriminator_score)

    return loss


## Implementation of the discriminator loss ##


def discriminator_loss(
    discriminator: nn.Module,
    current_num_block: int,
    alpha: float,
    real_images: torch.Tensor,
    predicted_images: torch.Tensor,
    c: float = 10,
):
    """Implements the discriminator loss which is nothing but
    minimizing the discriminator score for the real images
    and maximizing the discriminator score for the predicted
    images.

    Parameters
    ----------
    discriminator : nn.Module
        The discriminator module.

    current_num_block : int
        The current number of blocks in the discriminator.

    alpha : float
        The current alpha value.

    real_images : torch.Tensor
        The real images.

    predicted_images : torch.Tensor
        The predicted images from the generator.

    c : float
        The gradient penalty coefficient.
        [Default : 10]
    """

    ## Calculating the discriminator score for real images ##
    discriminator_score_real = discriminator(real_images, current_num_block, alpha)

    ## Calculating the discriminator score for fake images ##
    discriminator_score_fake = discriminator(
        predicted_images.detach(), current_num_block, alpha
    )

    ## Calculating gradient penalty ##
    ## 1. Setting an epsilon ##
    epsilon = torch.rand(len(real_images), 1, 1, 1, device=real_images.device)

    ## 2. Mixing the images ##
    mixed_images = epsilon * real_images + (1 - epsilon) * predicted_images.detach()
    mixed_images.requires_grad_(True)

    ## 3. Calculating the discriminator score for mixed images ##
    discriminator_score_mixed = discriminator(mixed_images, current_num_block, alpha)

    ## 4. Calculating the gradient of the discriminator score for mixed images ##
    gradients = torch.autograd.grad(
        inputs=mixed_images,
        outputs=discriminator_score_mixed,
        grad_outputs=torch.ones_like(discriminator_score_mixed),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    ## 5. Calculating the gradient penalty ##
    gradients = gradients.view(len(gradients), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

    ## Calculating the loss ##
    loss = (
        -torch.mean(discriminator_score_real)
        + torch.mean(discriminator_score_fake)
        + c * gradient_penalty
    )

    return loss
