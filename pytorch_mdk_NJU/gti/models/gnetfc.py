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
from gti.layers import basic_conv_block
import gti.chip.spec as spec
from gti.models.gtinet import GtiNet

def make_layers(args):
    mask_bits = spec.specs[args.chip]['gnetfc']
    return nn.Sequential(
        basic_conv_block(3, 64, downsample_mode="MAXPOOL",
            block_size=2, quant_params=args, mask_bit=mask_bits[0]),

        basic_conv_block(64, 128, downsample_mode="MAXPOOL",
            block_size=2, quant_params=args, mask_bit=mask_bits[1]),

        basic_conv_block(128, 256, downsample_mode="MAXPOOL",
            block_size=3, quant_params=args, mask_bit=mask_bits[2]),

        basic_conv_block(256, 256, downsample_mode="MAXPOOL",
            block_size=3, quant_params=args, mask_bit=mask_bits[3]),

        basic_conv_block(256, 256, downsample_mode="MAXPOOL",
            block_size=3, quant_params=args, mask_bit=mask_bits[4]),

        basic_conv_block(256, 256, padding=0, downsample_mode=None,
            block_size=2, quant_params=args, mask_bit=mask_bits[5]),
        basic_conv_block(256, args.num_classes, padding=0, downsample_mode=None,
            block_size=1, quant_params=args, mask_bit=mask_bits[5])
    )

class gnetfc(GtiNet):
    '''A custom architecture based off of VGG16.  Its main advantage is that
        it has no host layers, so the chip outputs can be directly used
        for classification.'''
    def __init__(self, args):
        super(gnetfc, self).__init__()
        self.chip_layer0 = make_layers(args)
        self._initialize_weights()

    @staticmethod
    def get_num_classes(checkpoint):
        return GtiNet.get_num_classes(checkpoint, name_prefix="module.chip_layer0.6.0.conv")

    def forward(self, x):
        return self.chip_layer0(x).view(x.size(0), -1)

    #needs to be redefined due to extra BN
    def modify_num_classes(self, checkpoint, mode=0):
        change_flag = False
        current_num_classes = self.chip_layer0[6][0].conv.weight.shape[0]
        incoming_num_classes = gnetfc.get_num_classes(checkpoint)
        if incoming_num_classes != current_num_classes:
            change_flag = True
            keys_to_be_processed = [
                'module.chip_layer0.6.0.conv.weight',
                'module.chip_layer0.6.0.conv.bias'
            ]
            for key in checkpoint.keys():
                if "6.0.bn" in key:
                    keys_to_be_processed.append(key)
            if mode!=0:
                if incoming_num_classes > current_num_classes:
                    for key in keys_to_be_processed:
                        if "track" in key:
                            continue
                        var = checkpoint[key]
                        var = var[:current_num_classes]
                        checkpoint[key] = var
                else:
                    raise NotImplementedError("Change num classes mode != 0 not yet implemented for \
                        case where checkpoint has fewer classes than currently being trained")
                    #implement 0 padding or initialization or something
            else:
                for key in keys_to_be_processed:
                    try:
                        del checkpoint[key]
                    except KeyError:
                        pass
        return checkpoint, change_flag
