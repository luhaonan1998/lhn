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
import torch
from gti.layers import basic_conv_block, Flatten
import gti.chip.spec as spec
from gti.models.gtinet import GtiNet

def make_layers(args):
    mask_bits = spec.specs[args.chip]['vgg16']
    return nn.Sequential(
        basic_conv_block(3, 64, downsample_mode="MAXPOOL",
            block_size=2, quant_params=args, mask_bit=mask_bits[0]),

        basic_conv_block(64, 128, downsample_mode="MAXPOOL",
            block_size=2, quant_params=args, mask_bit=mask_bits[1]),

        basic_conv_block(128, 256, downsample_mode="MAXPOOL",
            block_size=3, quant_params=args, mask_bit=mask_bits[2]),

        basic_conv_block(256, 512, downsample_mode="MAXPOOL",
            block_size=3, quant_params=args, mask_bit=mask_bits[3]),

        basic_conv_block(512, 512, downsample_mode="MAXPOOL",
            block_size=3, quant_params=args, mask_bit=mask_bits[4]),
    )

class vgg16(GtiNet):
    '''Implementation of VGG16 - arxiv:1409.1556'''
    def __init__(self, args):
        super(vgg16, self).__init__()
        self.chip_layer0 = make_layers(args)
        self.host_layer0 = nn.Sequential(
            Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, args.num_classes),
        )
        self._initialize_weights()

    @staticmethod
    def get_num_classes(checkpoint):
        return GtiNet.get_num_classes(checkpoint, name_prefix="module.host_layer0.7")

    def modify_num_classes(self, checkpoint, mode=0):
        return super(vgg16, self).modify_num_classes(checkpoint, mode,
            name_prefix="module.host_layer0.7")
