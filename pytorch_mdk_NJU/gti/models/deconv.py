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
from gti.layers import basic_conv_block, deconv_block, Upsample, Flatten
import gti.chip.spec as spec
from gti.models.gtinet import GtiNet

USE_DEFAULT_UPSAMPLING_MODE = True
if USE_DEFAULT_UPSAMPLING_MODE:
    _UPSAMPLING_MODE = "REPEAT"
    _DOWNSAMPLING_MODE = "MAXPOOL"
else:
    _UPSAMPLING_MODE = "ZERO"
    _DOWNSAMPLING_MODE = "STRIDE2"
#   1. When using ZERO fill mode, current GTI devices are constrained to top-left pooling (also known
#      as sample pooling) only (in other words, max pooling is no longer possible).
#      When possible, this MDK implements it as stride 2 for speed.
#      See gti.layers.py for more information.
#   2. When using ZERO fill mode, in deconv DAT JSON, remember to set "SamplingMethod": 1 to enable this
#      feature during chip conversion process.
#   3. When using REPEAT fill mode (default), current GTI devices are constrained to max pooling only.
#      In DAT JSON, "SamplingMethod" should be set to 0 for chip conversion.

#creates major layers 0-5
#512x14x14 output for 3x56x56 input (eg resized CIFAR)
def make_deconv_layers(args):
    mask_bits = spec.specs[args.chip]['deconv']
    return nn.Sequential(
        basic_conv_block(3, 64, downsample_mode=_DOWNSAMPLING_MODE,
            block_size=2, quant_params=args, mask_bit=mask_bits[0]),
        basic_conv_block(64, 128, downsample_mode=_DOWNSAMPLING_MODE,
            block_size=2, quant_params=args, mask_bit=mask_bits[1]),
        deconv_block(128, 128, upsampling_mode=_UPSAMPLING_MODE,
            block_size=1, quant_params=args, mask_bit=mask_bits[2]),
        deconv_block(128, 128, upsampling_mode=_UPSAMPLING_MODE,
            block_size=1, quant_params=args, mask_bit=mask_bits[3]),
        basic_conv_block(128, 256, downsample_mode=_DOWNSAMPLING_MODE,
            block_size=2, quant_params=args, mask_bit=mask_bits[4]),
        basic_conv_block(256, 256, downsample_mode=_DOWNSAMPLING_MODE,
            block_size=2, quant_params=args, mask_bit=mask_bits[5]),
    )

class deconv(GtiNet):
    '''A custom architecture to demonstrate how to use upsampling.
        It also shows the coupling between the upsampling/downsampling modes.'''
    def __init__(self, args):
        super(deconv, self).__init__()
        self.chip_layer0 = make_deconv_layers(args)
        self.host_layer0 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(256, args.num_classes)
        )
        self._initialize_weights()
