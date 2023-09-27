"""Implementation of the Progressive Growing Generator and Dsicriminator"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util_layers import (
    ProGANGeneratorBlock,
    ProGANDiscriminatorBlock,
    InitialProGANGeneratorBlock,
    FinalProGANDiscriminatorLayer,
)

## Implementation of the Progressive Growing Generator ##


class ProGANGenerator(nn.Module):
    """Implements Progressive Growing Generator."""

    def __init__(
        self,
        num_blocks: int = 8,
        latent_dim: int = 512,
        num_initial_filters: int = 16,
        out_channels: int = 3,
    ):
        """Constructor.

        Parameters
        ----------
        num_blocks : int
            The number of blocks in the generator.
            [Default : 8]

        latent_dim : int
            The latent dimension of the input noise.
            [Default : 512]

        num_initial_filters : int
            The number of initial filters in the generator.
            [Default : 16]

        out_channels : int
            The number of output channels.
            [Default : 3]

        Note
        ----
        Please check the num_initial_filters and num_blocks
        and make sure that num_initial_filters * 2**num_blocks
        is greater than or equal to latent dim.
        """

        super().__init__()

        ## Saving the num blocks ##
        self.num_blocks = num_blocks

        ## Setting the filters ##
        filters = list(
            reversed(
                [
                    min(2**i * num_initial_filters, latent_dim)
                    for i in range(num_blocks + 1)
                ]
            )
        )

        ## Setting the blocks ##
        self.blocks = nn.ModuleList(
            [
                InitialProGANGeneratorBlock(
                    in_channels=filters[i], out_channels=filters[i + 1]
                )
                if i == 0
                else ProGANGeneratorBlock(
                    in_channels=filters[i], out_channels=filters[i + 1]
                )
                for i in range(num_blocks)
            ]
        )

        ## Setting the toRGB layers ##
        self.toRGB = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=filters[i + 1],
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: torch.tensor, current_num_blocks: int, alpha: float):
        """Generic Forward Pass

        Parameters
        ----------
        x : torch.tensor
            The input noise.

        current_num_blocks : int
            The current number of blocks of the model.
            This ensures which block to use.

        alpha : float
            The value of alpha for fade-in.
        """

        assert current_num_blocks <= (
            self.num_blocks
        ), "Current Number of blocks cannot be greater than the number of blocks!"

        if current_num_blocks == 1:
            x = self.blocks[0](x)
            return torch.tanh(self.toRGB[0](x))

        else:
            for i in range(current_num_blocks):
                ## Getting the output of the previous block ##
                if i == current_num_blocks - 1:
                    previous_x = self.blocks[i].block[0](
                        x
                    )  # Doing this since to get the convtranspose output

                ## Current Output ##
                x = self.blocks[i](x)

            ## Returning the output as the weighted sum of the previous and current block output ##
            return torch.tanh(
                alpha * self.toRGB[current_num_blocks - 2](previous_x)
                + (1 - alpha) * self.toRGB[current_num_blocks - 1](x)
            )


## Implementing the Progressive Growing Discriminator ##


class ProGANDiscriminator(nn.Module):
    """Implements the Progressive Growing Discriminator."""

    def __init__(
        self,
        num_blocks: int = 8,
        latent_dim: int = 512,
        num_initial_filters: int = 16,
        in_channels: int = 3,
    ):
        """Constructor.

        Parameters
        ----------
        num_blocks : int
            The number of blocks in the discriminator.
            [Default : 8]

        latent_dim : int
            The latent dimension of the input noise.
            [Default : 512]

        num_initial_filters : int
            The number of initial filters in the discriminator.
            [Default : 16]

        in_channels : int
            The number of input channels.
            [Default : 3]
        """

        super().__init__()

        ## Saving the num blocks ##
        self.num_blocks = num_blocks

        ## Setting the filters ##
        filters = [
            min(2**i * num_initial_filters, latent_dim) for i in range(num_blocks + 1)
        ]

        ## Setting the blocks ##
        self.blocks = nn.ModuleList(
            [
                ProGANDiscriminatorBlock(
                    in_channels=filters[i], out_channels=filters[i + 1]
                )
                for i in range(num_blocks)
            ]
        )

        ## Setting the fromRGB layers ##
        self.fromRGB = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters[i],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for i in range(num_blocks + 1)
            ]
        )

        ## Setting the final layer ##
        self.final_layer = FinalProGANDiscriminatorLayer(
            in_channels=filters[-1], out_channels=1
        )

    def forward(self, x: torch.tensor, current_num_blocks: int, alpha: float):
        """Generic Forward Pass.

        Parameters
        ----------
        x : torch.tensor
            The input image.

        current_num_blocks : int
            The current number of blocks of the model.
            This ensures which block to use.

        alpha : float
            The value of alpha for fade-in.
        """

        assert current_num_blocks <= (
            self.num_blocks
        ), "Current Number of blocks cannot be greater than the number of blocks!"

        ## Getting the downsampled from RGB ##
        prev_x = self.fromRGB[-(current_num_blocks)](x)

        if current_num_blocks == 1:
            return self.final_layer(prev_x)

        else:
            ## Looping over the blocks ##
            for i in range(current_num_blocks - 1):
                ## Getting current output ##
                if i == 0:
                    x = self.fromRGB[-(current_num_blocks + 1)](
                        x
                    )  ## We must use the next fromRGB layer ##

                x = self.blocks[-(current_num_blocks - i)](x)

                ## Doing the weighted sum of the previous and current block output ##
                if i == 0:
                    x = alpha * x + (1 - alpha) * F.avg_pool2d(
                        prev_x, kernel_size=2, stride=2
                    )

            ## Returning the final output ##
            return self.final_layer(x)
