"""Implements the Generator and the Discriminator networks."""

import torch
import torch.nn as nn
from .util_layers import ConvNormActBlock, ResnetBlock, UpsampleLayer, DownsampleLayer, VGGBlock


class Generator(nn.Module):
    """Implements the UNet type Generator module."""

    def __init__(
        self,
        depth: int,
        num_res_blocks: int,
        in_channels: int,
        out_channels: int,
        initial_num_filters: int,
        norm: str,
        act: str,
        padding_mode: str = "reflect",
        use_dropout: bool = False,
        dropout_rate: float = 0.0,
    ):
        """Constructor.

        Parameters
        ----------
        depth : int
            Depth of the encoder/decoder block of
            the unet generator. For example : If depth is 3,
            then the encoder has three layers/blocks in
            total, with 3 downsampling connections. Similarly,
            the decoder has three/blocks with 3 upsampling
            layers.

        num_res_blocks : int
            Number of residual blocks in the bridge part of the
            unet.

        in_channels : int
            The number of the input channels of unet
            generator.

        out_channels : int
            The number of the output channels of unet
            generator.

        initial_num_filters : int
            The number of filters after the output of the
            first layer of the encoder. All the other blocks/
            layers have filters that are taken as 2**i factor
            of the initial_num_filters.

        norm : str
            The desired Normalization layer that is
            applied throughtout the unet generator.
            [Options : 'instance' , 'batch']

        act : str
            The desired activation function that is
            imposed throughout the unet generator.
            [Options : "relu" , "leaky", "elu", "prelu",
            "selu", "glu"]

        padding_mode : str
            The desired padding type to internal conv
            layers.
            [Default : "reflect"]

        use_dropout : bool
            Sets the dropout layer.
            [Default : False]

        dropout_rate : float
            The amount of dropout to apply after passing
            through the residual block.
            [Default : 0.0]
        """

        super().__init__()

        ## Setting the filters/num_channels ##
        filters = [initial_num_filters * (2**i) for i in range(depth)]

        self.input_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters[0],
            kernel_size=7,
            stride=1,
            padding="same",
            padding_mode=padding_mode,
        )

        ## Setting the encoder block ##
        # The encoder layer has depth-1 downsample layers. #

        self.encoder = nn.ModuleList()

        for i in range(len(filters[:-1])):
            self.encoder.append(
                DownsampleLayer(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    norm=norm,
                    act=act,
                    padding_mode=padding_mode,
                )  # Downsample
            )

        ## Bridge block which contains the (n)-Resnet blocks ##

        self.bridge = nn.ModuleList(
            [
                ResnetBlock(
                    channels=filters[-1],
                    norm=norm,
                    act=act,
                    padding_mode=padding_mode,
                    use_dropout=use_dropout,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_res_blocks)
            ]
        )

        ## Setting the decoder block ##
        # Each decoder layer has depth-1 upsample layers #

        self.decoder = nn.ModuleList([])

        ## Reversing the filters ##
        reversed_filters = filters[::-1]
        for i in range(len(reversed_filters[:-1])):
            self.decoder.append(
                UpsampleLayer(
                    in_channels=reversed_filters[i],
                    out_channels=reversed_filters[i + 1],
                    norm=norm,
                    act=act,
                    padding_mode=padding_mode,
                )
            )  # Upsample

        ## Final output mapping layer ##
        self.output_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=reversed_filters[-1],
                out_channels=out_channels,
                kernel_size=7,
                stride=1,
                padding="same",
                padding_mode=padding_mode,
            ),
            # nn.InstanceNorm2d(out_channels),
            nn.Tanh(),
        )

    def forward(self, x):
        """Implements the forward pass through
        the unet generator.

        Parameters
        ----------
        x : torch.tensor
            The input batch of images.

        Returns
        -------
        torch.tensor
            The output map.
        """

        ## Passing through input layer ##
        x = self.input_layer(x)

        ## Passing through encoder ##
        for layer in self.encoder:
            x = layer(x)

        ## Passing through bridge ##
        for layer in self.bridge:
            x = layer(x)

        ## Passing through decoder ##
        for layer in self.decoder:
            x = layer(x)

        ## Passing through output layer ##
        x = self.output_layer(x)

        return x


## PatchGAN Discriminator architecture ##


class Discriminator(nn.Module):
    """PatchGAN based discriminator."""

    def __init__(
        self,
        in_channels: int,
        initial_num_filters: int,
        num_blocks: int,
        norm: str,
        act: str,
        kernel_size: int or tuple,
        stride: int or tuple = 2,
    ):
        """Constructor.

        Parameters
        ----------
        in_channels : int
            The number of input channels.

        initial_num_filters : int
            The number of filters for the first
            convolutional layer.

        num_blocks : int
            The number of blocks in the discriminator.

        norm : str
            The normalization layer to attach to the
            convolutional layers.

        act : str
            The activation layer to attach to the
            convolutional layers.

        kernel_size : int or tuple
            The kernel size for each of the convolution
            layers.

        stride : int
            The stride of each convolutional layers.
            [Default : 2]
        """

        super().__init__()

        ## Setting the filters/num_channels ##

        filters = [initial_num_filters * (2**i) for i in range(num_blocks)]
        filters.insert(0, in_channels)  # adding the input channel

        ## Convolutional blocks ##

        self.discriminator_net = nn.ModuleList(
            [
                ConvNormActBlock(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    norm=norm,
                    act=act,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1 if i > 0 else 0,
                    add_norm=True
                    if i > 0
                    else False,  # sets no norm for first conv layer
                )
                for i in range(len(filters[:-1]))
            ]
        )

        ## Final output mapper ##
        self.output_layer = nn.Conv2d(
            in_channels=filters[-1],
            out_channels=1,
            kernel_size=kernel_size,
            padding=1,
            stride=1,
        )

    def forward(self, x):
        """Generic Forward pass.

        Parameters
        ----------
        x : torch.tensor
            Input batch of images.

        Returns
        -------
        torch.tensor
            NxN receptive field matrix.
        """

        for layer in self.discriminator_net:
            x = layer(x)

        return self.output_layer(x)

## Unet network for generator ##

class UNet(nn.Module):
    """Implements a basic UNET model.
    Code highly taken from: 
        https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py.
    """
    def __init__(self, input_channels: int=3, output_channels: int =3,**kwargs):
        """Constructor.

        Parameters
        ----------
        input_channels : int
            Number of input channels of the block.

        output_channels : int
            Number of output channels for the block.
        """
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)


    def forward(self, x):
        """Generic Forward Pass.

        Parameters
        ----------
        x : torch.tensor
            Input tensor.
        """
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


## Unetpp network for the Generator ##

class NestedUNet(nn.Module):
    """Implements a basic UNET model.
    Code highly taken from: 
        https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py.
    """
    def __init__(self, input_channels: int=1, output_channels: int=1, **kwargs):
        """Constructor.

        Parameters
        ----------
        input_channels : int
            Number of input channels.

        output_channels : int
            Number of output channels.
        """
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        
        self.final = nn.Sequential(
            nn.Conv2d(nb_filter[0], output_channels, kernel_size=1),
            nn.Tanh()
        )


    def forward(self, x):
        """Generic Forward Pass.

        Parameters
        ----------
        x : torch.tensor    
            The input tensor.
        """
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        
        output = self.final(x0_4)
        return output
        