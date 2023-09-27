"""Implementations of various layers used in the model.
"""

import torch
import torch.nn as nn

import torch.nn.functional as F

## Implementing Equalized Learning Rate ##

## Making the Equalized Convolutional Layer ##


class EqualizedConv2d(nn.Conv2d):
    """Implements the Equalized Convolutional Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = True,
    ):
        """Constructor.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The kernel size.
        stride : int
            The stride.
        padding : int
            The padding.
        bias : bool
            Whether to use bias or not.
        """

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        ## Initializing the weights and bias ##
        nn.init.normal_(self.weight, mean=0.0, std=1.0)

        if bias:
            nn.init.zeros_(self.bias)

        ## Setting the fan in ##
        fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]

        ## Setting the equalized learning constant ##
        self.constant = (2.0 / fan_in) ** 0.5

    def forward(self, x: torch.Tensor):
        """Generic Forward Pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """

        return F.conv2d(
            x,
            self.weight * self.constant,
            self.bias,
            self.stride,
            self.padding,
        )


## Setting the Equalized ConvTranspose2d Layer ##


class EqualizedConvTranspose2d(nn.ConvTranspose2d):
    """Implements the Equalized ConvTranspose2d Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int = 1,
        bias: bool = True,
    ):
        """Constructor.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
            [Default: 3]

        out_channels : int
            The number of output channels.
            [Default: 3]

        kernel_size : int
            The kernel size.
            [Default: 3]

        stride : int
            The stride.
            [Default: 1]

        padding : int
            The padding.
            [Default: 1]

        output_padding : int
            The output padding.
            [Default: 1]

        bias : bool
            Whether to use bias or not.
            [Default: True]
        """

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

        ## Initializing the weights and bias ##
        nn.init.normal_(self.weight, mean=0.0, std=1.0)

        if bias:
            nn.init.zeros_(self.bias)

        ## Setting the fan in ##
        fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]

        ## Setting the equalized learning constant ##
        self.constant = (2.0 / fan_in) ** 0.5

    def forward(self, x: torch.Tensor):
        """Generic Forward Pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """

        return F.conv_transpose2d(
            x,
            self.weight * self.constant,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )

## Setting the Pixelwise Normalization Layer ##


class PixelNorm(nn.Module):
    """Implements the pixel normalization layer."""

    def __init__(self):
        """Constructor."""

        super().__init__()

    def forward(self, x: torch.Tensor):
        """Generic forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """

        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


## Setting the ConvNormAct Block ##


class ConvNormActBlock(nn.Module):
    """Implements the ConvNormAct Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int = 1,
        bias: bool = True,
        use_pixel_norm: bool = True,
        activation: bool = True,
        upsample: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
            [Default: 3]

        out_channels : int
            The number of output channels.
            [Default: 3]

        kernel_size : int
            Kernel size for convolution.
            [Default: 3]

        stride : int
            Stride for convolution.
            [Default: 1]

        padding : int
            The padding.
            [Default: 1]

        output_padding : int
            The output padding.
            [Default: 1]

        bias : bool
            Whether to use bias or not.
            [Default: True]

        use_pixel_norm : bool
            Whether to use pixel norm or not.
            [Default: True]

        activation : bool
            Whether to use activation or not.
            [Default: True]

        upsample : bool
            Whether to upsample or not.
            [Default: False]
        """

        super().__init__()

        ## Setting the layers ##
        self.layers = nn.ModuleList()

        ## Setting the convolutional layer ##
        self.layers.append(
            EqualizedConv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            )
            if not upsample
            else EqualizedConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                output_padding=output_padding,
                bias=bias,
            )
        )

        ## Setting the pixelwise normalization layer ##
        if use_pixel_norm:
            self.layers.append(PixelNorm())

        ## Setting the activation layer ##
        if activation:
            self.layers.append(nn.LeakyReLU(0.2))

    def forward(self, x: torch.Tensor):
        """Generic forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """

        for layer in self.layers:
            x = layer(x)

        return x


## Implementation of a Generator Block ##


class ProGANGeneratorBlock(nn.Module):
    """Implements the ProGAN Generator Block which is basically a
    ConvTranspose2d Block and 2 repeated convolutional blocks. All
    blocks have PixelNorm and activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Constructor.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
            [Default: 3]

        out_channels : int
            The number of output channels.
            [Default: 3]
        """

        super().__init__()

        ## Setting the layers ##

        self.block = nn.Sequential(
            ConvNormActBlock(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                use_pixel_norm=False,
                upsample=True,
            ),
            ConvNormActBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_pixel_norm=True,
                upsample=False,
            ),
            ConvNormActBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_pixel_norm=True,
                upsample=False,
            ),
        )

    def forward(self, x: torch.Tensor):
        """Generic forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """

        return self.block(x)


## Implementation of the initial ProGAN Generator Block ##


class InitialProGANGeneratorBlock(nn.Module):
    """Implements the initial ProGAN Generator block which is basically
    a 4x4 ConvTranspose2d Block followed by a convolutional block.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        """Constructor.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
            [Default: 3]

        out_channels : int
            The number of output channels.
            [Default: 3]
        """

        super().__init__()

        ## Setting the layers ##

        self.block = nn.Sequential(
            ConvNormActBlock(
                in_channels,
                in_channels,
                kernel_size=4,
                stride=1,
                padding=0,
                output_padding=0,
                use_pixel_norm=False,
                upsample=True,
            ),
            ConvNormActBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_pixel_norm=True,
                upsample=False,
            ),
        )

    def forward(self, x: torch.Tensor):
        """Generic forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """

        return self.block(x)


## Implementation of the ProGAN Discriminator Block ##


class ProGANDiscriminatorBlock(nn.Module):
    """Implementation of the ProGAN Discriminator Block which is basically
    2 Convolutional Blocks followed by an average pooling block.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        """Constructor.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
            [Default: 3]

        out_channels : int
            The number of output channels.
            [Default: 3]
        """
        super().__init__()

        ## Setting the block ##

        self.block = nn.Sequential(
            ConvNormActBlock(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_pixel_norm=False,
                upsample=False,
            ),
            ConvNormActBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_pixel_norm=False,
                upsample=False,
            ),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor):
        """Generic forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """

        return self.block(x)


## Implements the final ProGAN Discriminator Layer ##


class FinalProGANDiscriminatorLayer(nn.Module):
    """Implements the final ProGAN Discriminator Layer which is
    the mean std layer followed by a 3x3 convolutional layer
    and then a 4x4 convolutional layer.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        """Constructor.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
            [Default: 3]

        out_channels : int
            The number of output channels.
            [Default: 3]
        """

        super().__init__()

        ## Setting the layers ##

        self.layers = nn.Sequential(
            MinibatchSTDLayer(),
            ConvNormActBlock(
                in_channels + 1,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_pixel_norm=False,
                upsample=False,
            ),
            ConvNormActBlock(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=1,
                padding=0,
                use_pixel_norm=False,
                upsample=False,
            ),
        )

    def forward(self, x: torch.Tensor):
        """Generic forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """

        x = self.layers(x)

        x = x.view(x.shape[0], -1)

        return x


## Implements the Minibatch Standard Deviation Layer ##


class MinibatchSTDLayer(nn.Module):
    """Implements the minibatch std layer which calculates
    the std across the batch for each feature and spatial location
    and then calculates the mean of all of them to get a single value.
    Then this value is tiled across the spatial dimensions to
    get a tensor of same size as the input.
    """

    def __init__(self):
        """Constructor."""

        super().__init__()

    def forward(self, x: torch.Tensor):
        """Generic forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """

        ## Getting the std across the batch ##
        std = torch.std(x, dim=0)

        ## Getting the mean of the std ##
        mean_std = torch.mean(std)

        ## Tiling the mean_std across the spatial dimensions ##
        mean_std = mean_std.repeat(x.shape[0], 1, x.shape[2], x.shape[3])

        ## Concatenating the mean_std with the input ##
        x = torch.cat([x, mean_std], dim=1)

        return x
