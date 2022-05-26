import torch
import torch.nn as nn
import itertools as it
from torch import Tensor, dropout
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        activation_type = ACTIVATIONS[self.activation_type](**self.activation_params)
        pooling_type = POOLINGS[self.pooling_type](**self.pooling_params)

        # conv part
        channels = [in_channels] + self.channels
        for i in range(len(channels) - 1):
            conv = nn.Conv2d(channels[i], channels[i+1], **self.conv_params)
            layers = layers + [conv, activation_type]
            if (i+1) % self.pool_every == 0:
                layers.append(pooling_type)
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            test_im = torch.ones([1, *self.in_size])
            out = self.feature_extractor(test_im)
            out_size = torch.prod(torch.tensor(out.shape)).item()
            return out_size
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        # TODO:
        #  - Create the MLP part of the model: (FC -> ACT)*M -> Linear
        #  - Use the the MLP implementation from Part 1.
        #  - The first Linear layer should have an input dim of equal to the number of
        #    convolutional features extracted by the convolutional layers.
        #  - The last Linear layer should have an output dim of out_classes.
        mlp: MLP = None
        # ====== YOUR CODE: ======
        in_dim = self._n_features()
        hidden_dims = [*self.hidden_dims, self.out_classes]
        # lesson from bug: read the entire documentation... all of the networks were different except for the 1st
        activations = [ACTIVATIONS[self.activation_type](**self.activation_params)]*(len(hidden_dims) - 1) + ['none']
        mlp = MLP(in_dim, hidden_dims, activations)
        # ========================
        return mlp

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        out: Tensor = None
        # ====== YOUR CODE: ======
        extracted_feats = self.feature_extractor(x)
        extracted_feats = extracted_feats.view(extracted_feats.size(0), -1) 
        out = self.mlp(extracted_feats)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        main_layers = []
        channels = [in_channels] + channels
        for i in range(len(channels) - 1):
            k = kernel_sizes[i]
            main_layers.append(nn.Conv2d(channels[i], channels[i+1], k, padding=k//2, bias=True))                         
            if i != len(channels) - 2:
                main_layers += [nn.Dropout2d(dropout)]
                if batchnorm:
                    main_layers += [nn.BatchNorm2d(channels[i+1])]
                main_layers += [ACTIVATIONS[activation_type](**activation_params)]
                
        
        # main_layers.append(nn.Conv2d(channels[-1], channels[0], kernel_size=1))
        self.main_path = nn.Sequential(*main_layers)
        self.shortcut_path = nn.Sequential(nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False))
        # self.shortcut_path = nn.Sequential(nn.Identity())
        

        # ========================

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass. Save the main and residual path to `out`.
        out: Tensor = None
        # ====== YOUR CODE: ======
        out = self.main_path(x) + self.shortcut_path(x)
        # ========================
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions, excluding the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->10.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        # TODO:
        #  Initialize the base class in the right way to produce the bottleneck block
        #  architecture.
        # ====== YOUR CODE: ======
        # maybe between every downsize in the bottleneck we do
        # channels = []
        # kernel_sizes = []
        # channels = [inner_channels[0]]*2 
        # kernel_sizes = [1] + [inner_kernel_sizes[0]]
        # for i in range(len(inner_channels) - 1):
        #     if inner_channels[i] != inner_channels[i + 1]:
        #         channels += [inner_channels[i + 1]]*2
        #         kernel_sizes += [1] + [inner_kernel_sizes[i + 1]]
        #     else:
        #         channels += [inner_channels][i + 1]
        #         kernel_sizes += kernel_sizes[i + 1]
        # channels += [in_out_channels]
        # kernel_sizes += [1]

        channels = [inner_channels[0]] + inner_channels + [in_out_channels]
        kernel_sizes = [1] + inner_kernel_sizes + [1]
        super().__init__(
            in_channels= in_out_channels, 
            channels = channels, 
            kernel_sizes = kernel_sizes,
            **kwargs)
        self.shortcut_path = nn.Identity()
        # ========================


class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        # ====== YOUR CODE: ======
        
         #### VVVVVVVVVV  CNN params  VVVVVVVVVVVVVV
        # self,
        # in_size,
        # out_classes: int,
        # channels: Sequence[int],
        # pool_every: int,
        # hidden_dims: Sequence[int],
        # conv_params: dict = {},
        # activation_type: str = "relu",
        # activation_params: dict = {},
        # pooling_type: str = "max",
        # pooling_params: dict = {},

        #####  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        channels = [in_channels] + self.channels
        N = len(channels)
        P = self.pool_every
        res_block = None
        
        # build the pooling blocks
        # take all of the channels N (including the input) and divide into chunks of P 
        # (TODO: test: see that there)
        for i in range(N//P):
            # take 
            block_channels = channels[(P+1)*i : (P + 1)*(i + 1)]
            if block_channels[0] == block_channels[-1] and self.bottleneck:
                inner_channels = block_channels[1:-1]  #####
                res_block = ResidualBottleneckBlock(in_out_channels=block_channels[0], 
                                                    inner_channels=inner_channels,
                                                    inner_kernel_sizes=len(inner_channels)*[3],
                                                    batchnorm = self.batchnorm,
                                                    dropout = self.dropout,
                                                    activation_type = self.activation_type,
                                                    activation_params = self.activation_params)  #########
            else:
                res_block = ResidualBlock(in_channels= block_channels[0]
                                        , channels = block_channels[1:]
                                        , kernel_sizes= [3]*len(block_channels[1:])
                                        , batchnorm = self.batchnorm
                                        , dropout = self.dropout
                                        , activation_type = self.activation_type
                                        , activation_params = self.activation_params) #########

            layers = layers + [res_block, POOLINGS[self.pooling_type](self.pooling_params['kernel_size'])]
        
        # build the rest
        if N%P != 0:
            block_channels = channels[P*(N//P):]
            if block_channels[0] == block_channels[-1] and self.bottleneck:
                inner_channels = block_channels[1:-1]  #####
                res_block = ResidualBottleneckBlock(in_out_channels=block_channels[0], 
                                                    inner_channels=inner_channels,
                                                    inner_kernel_sizes=len(inner_channels)*[3],
                                                    batchnorm = self.batchnorm,
                                                    dropout = self.dropout,
                                                    activation_type = self.activation_type,
                                                    activation_params = self.activation_params)  #########
            else:
                res_block = ResidualBlock(in_channels= block_channels[0]
                                        , channels = block_channels[1:]
                                        , kernel_sizes= [3]*len(block_channels[1:])
                                        , batchnorm = self.batchnorm
                                        , dropout = self.dropout
                                        , activation_type = self.activation_type
                                        , activation_params = self.activation_params) #########

            layers = layers + [res_block]
        
        
        # ========================
        seq = nn.Sequential(*layers)
        return seq


    # def _make_feature_extractor(self):
    #     in_channels, in_h, in_w, = tuple(self.in_size)

    #     layers = []
    #     # TODO: Create the feature extractor part of the model:
    #     #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
    #     #   \------- SKIP ------/
    #     #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
    #     #  Apply Pooling to reduce dimensions after every P convolutions.
    #     #  Notes:
    #     #  - If N is not divisible by P, then N mod P additional
    #     #    CONV->ACT (with a skip over them) should exist at the end,
    #     #    without a POOL after them.
    #     #  - Use your own ResidualBlock implementation.
    #     #  - Use bottleneck blocks if requested and if the number of input and output
    #     #    channels match for each group of P convolutions.
    #     # ====== YOUR CODE: ======
    #     pooling = POOLINGS[self.pooling_type](self.pooling_params['kernel_size'])     
    #     # concatanate dimentions
    #     all_dims = [self.in_size[0], *self.channels]        
    #     for i in range(0,len(all_dims),self.pool_every):            
    #         if i+1+self.pool_every<=len(all_dims): # full blocks
    #             if i==0: # first
    #                 block_dims = all_dims[i:i+self.pool_every+1]
    #             else:
    #                 block_dims = all_dims[i+1:i+self.pool_every+1]
    #             if self.bottleneck and block_dims[0]==block_dims[-1]: # ResidualBottleneckBlock                
    #                 res_block = ResidualBottleneckBlock(
    #                     in_out_channels=block_dims[0], inner_channels=block_dims[1:-1],
    #                     inner_kernel_sizes=[3]*len(block_dims[1:-1]),
    #                     batchnorm=self.batchnorm, dropout=self.dropout,activation_type=self.activation_type,
    #                     activation_params=self.activation_params
    #                 )
    #             else:   # ResidualBlock
    #                 block_dims = all_dims[i:i+self.pool_every+1]
    #                 res_block = ResidualBlock(
    #                     in_channels=block_dims[0], channels=block_dims[1:],
    #                     kernel_sizes=[3]*len(block_dims[1:]),
    #                     batchnorm=self.batchnorm, dropout=self.dropout,activation_type=self.activation_type,
    #                     activation_params=self.activation_params
    #                 )
    #             layers += [res_block, pooling]
    #         elif i < (len(all_dims)-1): # partial block (last block)
    #             if i==0: # first
    #                 block_dims = all_dims[i:]
    #             else:
    #                 block_dims = all_dims[i+1:]
    #             if self.bottleneck and block_dims[0]==block_dims[-1]: # ResidualBottleneckBlock
    #                 res_block = ResidualBottleneckBlock(
    #                     in_out_channels=block_dims[0], inner_channels=block_dims[i+1:],
    #                     inner_kernel_sizes=[3]*len(block_dims[i+1:]),
    #                     batchnorm=self.batchnorm, dropout=self.dropout,activation_type=self.activation_type,
    #                     activation_params=self.activation_params
    #                 )                
    #             else:   # ResidualBlock
    #                 block_dims = all_dims[i:]
    #                 res_block = ResidualBlock(
    #                     in_channels=block_dims[0], channels=block_dims[1:],
    #                     kernel_sizes=[3]*len(block_dims[1:]),
    #                     batchnorm=self.batchnorm, dropout=self.dropout,activation_type=self.activation_type,
    #                     activation_params=self.activation_params
    #                 )
    #             layers += [res_block]

    #         # ========================
    #     seq = nn.Sequential(*layers)
    #     return seq


# class YourCNN(CNN):
#     def __init__(self, *args, **kwargs):
#         """
#         See CNN.__init__
#         """
#         super().__init__(*args, **kwargs)

#         # TODO: Add any additional initialization as needed.
#         # ====== YOUR CODE: ======
#         raise NotImplementedError()
#         # ========================

#     # TODO: Change whatever you want about the CNN to try to
#     #  improve it's results on CIFAR-10.
#     #  For example, add batchnorm, dropout, skip connections, change conv
#     #  filter sizes etc.
#     # ====== YOUR CODE: ======
#     raise NotImplementedError()

#     # ========================
