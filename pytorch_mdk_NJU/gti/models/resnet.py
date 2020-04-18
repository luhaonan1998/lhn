'''
All Rights Reserved.

Copyright (c) 2017-2019, Gyrfalcon technology Inc.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import torch.nn as nn
from gti.layers import basic_conv_block, residual_block, Flatten, ConvBlock
import gti.chip.spec as spec
from gti.models.gtinet import GtiNet

#Resnet18
def make_layers(args, use_bn=True):
    mask_bits = spec.specs[args.chip]['resnet18']
    return nn.Sequential(
        basic_conv_block(3, 64, downsample_mode="MAXPOOL", block_size=2,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[0]),

        basic_conv_block(64, 64, downsample_mode="MAXPOOL", block_size=2,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[1]),

        residual_block(64, 64, downsample_mode="MAXPOOL", block_size=2,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[2]),

        basic_conv_block(64, 128, downsample_mode=None, block_size=2,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[3]),
        residual_block(128, 128, downsample_mode="MAXPOOL", block_size=1,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[3]),

        basic_conv_block(128, 256, downsample_mode=None, block_size=2,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[4]),
        residual_block(256, 256, downsample_mode="MAXPOOL", block_size=1,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[4]),

        basic_conv_block(256, 512, downsample_mode=None, block_size=2,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[5]),
        residual_block(512, 512, downsample_mode=None, block_size=1,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[5]),
    )

class resnet18(GtiNet):
    '''A variant resnet model designed to closely mimic resnet18
        - arxiv:1512.03385'''
    def __init__(self, args):
        super(resnet18, self).__init__()
        self.chip_layer0 = make_layers(args)
        self.host_layer0 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(512, args.num_classes)
        )
        self._initialize_weights()

class resnet18nobn(GtiNet):
    '''Same, but without BN'''
    def __init__(self, args):
        super(resnet18nobn, self).__init__()
        self.chip_layer0 = make_layers(args, False)
        self.host_layer0 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(512, args.num_classes)
        )
        self._initialize_weights()

#Resnet50
def make_layers_chip0(args, use_bn=True):
    mask_bits = spec.specs[args.chip]['resnet50']
    return nn.Sequential(
        basic_conv_block(3, 64, downsample_mode="MAXPOOL", block_size=2,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[0]),

        basic_conv_block(64, 64, downsample_mode="MAXPOOL", block_size=2,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[1]),

        residual_block(64, 64, downsample_mode="MAXPOOL", block_size=3,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[2]),

        basic_conv_block(64, 128, downsample_mode=None, block_size=2,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[3]),
        residual_block(128, 128, downsample_mode="MAXPOOL", block_size=3,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[3]),

        basic_conv_block(128, 256, downsample_mode=None, block_size=2,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[4]),
        residual_block(256, 256, downsample_mode=None, block_size=1,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[4]),

        residual_block(256, 256, downsample_mode=None, block_size=6,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[5])
    )

def make_layers_chip1(args, use_bn=True):
    mask_bits = spec.specs[args.chip]['resnet50']
    return nn.Sequential(
        basic_conv_block(256, 512, downsample_mode="STRIDE2", block_size=1,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[6]),

        basic_conv_block(512, 512, downsample_mode=None, block_size=1,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[7]),

        residual_block(512, 512, downsample_mode=None, block_size=2,
            use_bn=use_bn, quant_params=args, mask_bit=mask_bits[8])
    )

class resnet50(GtiNet):
    '''A variant resnet model designed to closely mimic resnet50.
        arxiv:1512.03385  Due to the size of the model, 2 chips are required.'''
    def __init__(self, args):
        super(resnet50, self).__init__()
        self.chip_layer0 = make_layers_chip0(args)
        self.chip_layer1 = make_layers_chip1(args)
        self.host_layer0 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(512, args.num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.chip_layer0(x)
        x = self.chip_layer1(x)
        return self.host_layer0(x)

class resnet50nobn(GtiNet):
    '''Same, but without BN'''
    def __init__(self, args):
        super(resnet50nobn, self).__init__()
        self.chip_layer0 = make_layers_chip0(args, False)
        self.chip_layer1 = make_layers_chip1(args, False)
        self.host_layer0 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(512, args.num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.chip_layer0(x)
        x = self.chip_layer1(x)
        return self.host_layer0(x)
