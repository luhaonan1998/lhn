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
from gti.layers import basic_conv_block, residual_block, Flatten
import gti.chip.spec as spec
from gti.models.gtinet import GtiNet

def make_layers(args):
    mask_bits = spec.specs[args.chip]['tenbit']
    return nn.Sequential(
        basic_conv_block(3, 64, downsample_mode="MAXPOOL",
            block_size=2, quant_params=args, mask_bit=mask_bits[0]),

        basic_conv_block(64, 64, downsample_mode="MAXPOOL",
            block_size=2, quant_params=args, mask_bit=mask_bits[1]),

        residual_block(64, 64, downsample_mode="MAXPOOL",
            block_size=2, quant_params=args, mask_bit=mask_bits[2]),

        basic_conv_block(64, 128, downsample_mode=None,
            block_size=2, quant_params=args, mask_bit=mask_bits[3]),
        residual_block(128, 128, downsample_mode="MAXPOOL",
            block_size=1, quant_params=args, mask_bit=mask_bits[3]),

        basic_conv_block(128, 256, downsample_mode=None,
            block_size=2, quant_params=args, mask_bit=mask_bits[4]),
        residual_block(256, 256, downsample_mode="MAXPOOL",
            block_size=1, quant_params=args, mask_bit=mask_bits[4]),

        basic_conv_block(256, 512, downsample_mode=None,
            block_size=1, quant_params=args, mask_bit=mask_bits[5]),
    )

class tenbit(GtiNet):
    '''A truncated version of GTI resnet18.  The smaller model leaves enough
        RAM for the chip to support 10 bit activations.  Make sure to include
        ten_bit_act in args.'''
    def __init__(self, args):
        super(tenbit, self).__init__()
        self.chip_layer0 = make_layers(args)
        self.host_layer0 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(512, args.num_classes)
        )
        self._initialize_weights()
