"""Builds the core layers to be used for building the
Generator and Discriminator model."""

import torch
import torch.nn as nn


def get_norm_layer(norm: str, num_features: int):
    """Given a normalization layer type,
    returns its nn.Module layer.

    Parameters
    ----------
    norm : str
        The normalization layer type.
        [Options : 'instance' , 'batch']

    num_features : int
        The number of channels for the norm
        layer.

    Returns
    -------
    nn.Module
        The normalization layer.
    """

    NORMALIZATION = {
        "batch": nn.BatchNorm2d(num_features),
        "instance": nn.InstanceNorm2d(num_features),
    }

    return NORMALIZATION[norm]


def get_act_layer(act: str):
    """Given an activation layer type,
    returns its nn.Module layer.

    Parameters
    ----------
    act : str
        The activation layer type.
        [Options : "relu" , "leaky", "elu", "prelu",
                   "selu", "glu"]

    Returns
    -------
    nn.Module
        The activation layer.
    """
    ACTIVATION = {
        "relu": nn.ReLU(),
        "leaky": nn.LeakyReLU(negative_slope=0.2),
        "elu": nn.ELU(),
        "prelu": nn.PReLU(),
        "selu": nn.SELU(),
        "glu": nn.GLU(),
    }

    return ACTIVATION[act]


class ConvNormActBlock(nn.Module):
    """Implements a collective block of
    Convolution, specific normalization,
    and Activation function block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int or tuple,
        stride: int,
        padding: int or str = 0,
        padding_mode: str = "reflect",
        add_act: bool = True,
        act: str = None,
        add_norm: bool = True,
        norm: str = None,
        upsampling: bool = False,
        output_padding: int = 0,
    ):
        """Constructor.

        Parameters
        ----------
        in_channels : int
            The number of input channels fed to the
            block.

        out_channels : int
            The number of output channels which
            feeds out from the block.

        kernel_size : int or tuple
            The kernel size for the conv layer.

        stride : int or str
            The stride for the conv layer.

        padding : str
            The type of padding to apply for the
            conv layer.
            [Options : "same", "valid"]

        padding_mode : str
            The desired padding type to apply to the
            conv layer.
            [Default : "reflect"]

        add_act : bool
            Flag to add the activation layer.
            [Default : True]

        act : str
            The desired activation layer that is applied
            after batch normalization layer..
            [Options : "relu" , "leaky", "elu", "prelu",
            "selu", "glu"]
            [Default : None]

        add_norm : bool
            Flag to add the normalization layer.
            [Default : True]

        norm : str
            The desired Normalization layer that is
            applied after convolutional layer.
            [Options : 'instance' , 'batch']
            [Default : None]

        upsampling : bool
            Flag to set for upsampling layer.
            [Default : False]

        output_padding : int
            Additional shape matching padding for
            the convtranspose2d layer.
        """

        super().__init__()

        ## Instantiating the layers ##
        layers = [
            ## For downsampling conv-layer ## 
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                bias=False if norm == "batchnorm" else True,
            )
            if upsampling == False
            ## For upsample convtranspose-layer ##
            else nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    padding_mode=padding_mode,
                    bias=False if norm == "batchnorm" else True,
                ),
            )
        ]

        ## Add norm layer if needed ##
        if add_norm:
            layers.append(get_norm_layer(norm, out_channels))

        ## Add activation layer if needed ##
        if add_act:
            layers.append(get_act_layer(act))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """Implements the forward pass."""

        return self.block(x)


## Implements the bridge residual block layer ##


class ResnetBlock(nn.Module):
    """Implements the resnet block with two
    3x3 convolutional layers of same filters.
    Additively, both of these blocks have an
    activation function and a normalization
    layer coupled with them.
    """

    def __init__(
        self,
        channels: int,
        norm: str,
        act: str,
        padding_mode: str = "reflect",
        dropout_rate: float = 0.1,
        use_dropout: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        channels : int
            The number of input and output channels for
            the residual block.

        norm : str
            The desired Normalization layer that is
            applied after each convolution.
            [Options : 'instance' , 'batch']

        act : str
            The desired activation layer that is applied
            after each convolution.
            [Options : "relu" , "leaky", "elu", "prelu",
            "selu", "glu"]

        padding_mode : str
            The desired padding type to apply to the
            conv layers.
            [Default : "reflect"]

        dropout_rate : float
            The amount of dropout to apply after passing
            through the residual block.
            [Default : 0.1]

        use_dropout : bool
            Flag to use dropout or not.
            [Default : False]
        """

        super().__init__()

        ## Setting the two conv layers ##
        layers = [
            ConvNormActBlock(
                in_channels=channels,
                out_channels=channels,
                norm=norm,
                act=act,
                kernel_size=3,
                stride=1,
                padding="same",
                padding_mode=padding_mode,
            ),
            ConvNormActBlock(
                in_channels=channels,
                out_channels=channels,
                norm=norm,
                add_act=False,
                act=act,
                kernel_size=3,
                stride=1,
                padding="same",
                padding_mode=padding_mode,
            ),
        ]

        if use_dropout:
            layers.append(nn.Dropout2d(p=dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """Implements the residual connection
        setting by passing the input through
        both the convolutional layers and adding
        it to the input map.

        Parameters
        ----------
        x : torch.tensor
            Input feature map.

        Returns
        -------
        torch.tensor
            Output feature map.
        """

        out = self.block(x)
        return out + x


## Downsampling Layer which downsamples the spatial
# dimension by a factor of 2. ##


class DownsampleLayer(nn.Module):
    """Implements the downsampling procedure.
    It consists of a 3x3 convolutional layer,
    coupled with an user defined normalization
    and activation layer. The downsampling is
    done via the convolutional kernel which
    has a stride of 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        act: str,
        padding_mode: str = "reflect",
    ):
        """Constructor.

        Parameters
        ----------
        in_channels : int
            The number of input channels.

        out_channels : int
            The number of output channels.

        norm : str
            The desired Normalization layer that is
            applied. This acts as the key to fetch
            the corresponding nn.Module layer from the
            NORMALIZATION dict.
            [Options : 'instance' , 'batch']

        act : str
            The desired activation layer. This acts
            as the key to fetch the corresponding
            nn.Module layer from the ACTIVATION
            dict.
            [Options : "relu" , "leaky", "elu", "prelu",
            "selu", "glu"]

        padding_mode : str
            The desired padding type to apply to the
            conv layers.
            [Default : "reflect"]
        """

        super().__init__()

        self.downsample_conv = nn.Sequential(
            ConvNormActBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                norm=norm,
                act=act,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
            ),
        )
    def forward(self, x):
        """Downsamples the input spatial dimension.

        Parameters
        ----------
        x : torch.tensor
            Input feature map.

        Returns
        -------
        torch.tensor
            Output feature map.
        """

        return self.downsample_conv(x)


## Upsampling Layer which upsamples the spatial
# dimension by a factor of 2. ##


class UpsampleLayer(nn.Module):
    """Implements the upsampling procedure.
    It consists of a 3x3 transposed convolutional
    layer, coupled with an user defined normalization
    and activation layer. The upsampling is
    done for a factor of 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        act: str,
        padding_mode: str = "reflect",
    ):
        """Constructor.

        Parameters
        ----------
        in_channels : int
            The number of input channels.

        out_channels : int
            The number of output channels.

        norm : str
            The desired Normalization layer that is
            applied. This acts as the key to fetch
            the corresponding nn.Module layer from the
            NORMALIZATION dict.
            [Options : 'instance' , 'batch']

        act : str
            The desired activation layer. This acts
            as the key to fetch the corresponding
            nn.Module layer from the ACTIVATION
            dict.
            [Options : "relu" , "leaky", "elu", "prelu",
            "selu", "glu"]

        padding_mode : str
            The desired padding type to apply to the
            conv layers.
            [Default : "reflect"]
        """

        super().__init__()

        self.upsample = nn.Sequential(
            ConvNormActBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                norm=norm,
                act=act,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
                upsampling=True,
                output_padding=1,
            ),
        )

    def forward(self, x):
        """Upsamples the input spatial dimension.

        Parameters
        ----------
        x : torch.tensor
            Input feature map.

        Returns
        -------
        torch.tensor
            Output feature map.
        """
        return self.upsample(x)

## Implements a two conv-relu-bn block ##
class VGGBlock(nn.Module):
    """Implements a 2x repeated conv-batchnorm-relu
    block.
    """
    def __init__(self, in_channels : int, middle_channels : int, out_channels : int):
        """Constructor. 

        Parameters
        ----------
        in_channels : int
            Input channels of the block.

        middle_channels : int
            Middle channels of the block, i.e., channels after first 
            convolutional layer.

        out_channels : int
            Final output channels.
        """
        super().__init__()

        self.block = nn.Sequential(
            ConvNormActBlock(
                in_channels=in_channels,
                out_channels=middle_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                add_act=True,
                act="relu",
                add_norm=True,
                norm="batch",
            ),
            ConvNormActBlock(
                in_channels=middle_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                add_act=True,
                act="relu",
                add_norm=True,
                norm="batch",
            )
        )

    def forward(self, x):
        """Generic Forward Pass.

        Parameters
        ----------
        x : torch.tensor
            Input tensor.
        """
        out = self.block(x)
        return out
