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
import gti.layers

class GtiNet(nn.Module):
    '''abstract base class that includes most of the API necessary for
        training a CNN model compatible with GTI chip.  Models that
        require special handling are free to implement methods
        that override these defaults.'''
    def __init__(self):
        super(GtiNet, self).__init__()

    #returns tuple of bools
    #assume dictionary is for net wrapped in nn.DataParallel
    @staticmethod
    def get_status_checkpoint(checkpoint):
        qa = bool(checkpoint['module.chip_layer0.0.0.relu.quantize'])
        qw = bool(checkpoint['module.chip_layer0.0.0.conv.quantize'])
        fuse = bool(checkpoint['module.chip_layer0.0.0.fuse'])
        return qw, qa, fuse

    @staticmethod
    def get_num_classes(checkpoint, name_prefix="module.host_layer0.2"):
        try:
            return checkpoint[name_prefix+'.weight'].shape[0]
        except KeyError:
            return -1

    #needed because loading checkpoint overrides these vars
    def set_status(self, qw, qa, fuse, cal=None):
        for name, child in self.named_children():
            if "chip" in name:
                for major in child:
                    for minor in major:
                        if type(minor) is gti.layers.ConvBlock:
                            minor.set_status(qw, qa, fuse, cal)

    #if checkpoint has same # of classes as # being trained, this function does nothing
    #mode=0: dump the mismatched vars
    #mode!=0: modify checkpoint to have the same number of classes
    def modify_num_classes(self, checkpoint, mode=0, name_prefix="module.host_layer0.2"):
        change_flag = False
        #this is slow, but avoids special casing or requiring defining another function
        current_num_classes = self.get_num_classes(self.state_dict())
        incoming_num_classes = self.get_num_classes(checkpoint)
        if incoming_num_classes != current_num_classes:
            change_flag = True
            keys_to_be_processed = [
                name_prefix+'.weight',
                name_prefix+'.bias'
            ]
            if mode!=0:
                if incoming_num_classes > current_num_classes:
                    for key in keys_to_be_processed:
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

    def forward(self, x):
        x = self.chip_layer0(x)
        return self.host_layer0(x)

    #BN merge must be done before gain edit
    #if done together, BN is automatically done first
    def fuse(self, do_fuse=True, do_gain=True, ten_bits=False):
        if ten_bits:
            prev_cap = 255.0
            MAX_ACTIVATION = 1023.0
        else:
            prev_cap = 31.0
            MAX_ACTIVATION = 31.0
        for m in self.modules(): #recursively goes through every module and submodule
            if isinstance(m, gti.layers.ConvBlock):
                if do_fuse and m.use_bn:
                    gamma = m.bn.weight
                    beta = m.bn.bias
                    mean = m.bn.running_mean
                    var = m.bn.running_var
                    bn_epsilon = 1e-6
                    bn_stddev = torch.sqrt(var + bn_epsilon)
                    bn_factor = gamma / bn_stddev
                    for i in range(bn_factor.shape[0]):
                        m.conv.weight.data[i] *= bn_factor[i]
                    m.conv.bias = nn.Parameter(beta - bn_factor * mean)
                    print(m.conv.weight.data)
                    #nn.Param has req_grad=True by default
                    
                if do_gain:
                    this_cap = m.relu.cap.item()

                    b_gain = MAX_ACTIVATION / this_cap
                    m.conv.bias.data *= b_gain

                    w_gain = prev_cap / this_cap
                    m.conv.weight.data *= w_gain
                    print(m.conv.weight.data)
                    prev_cap = this_cap
                    m.relu.cap.data[...] = MAX_ACTIVATION
            #assuming all FC layers come after all conv layers
            #assuming 1st encountered FC layer is also 1st FC layer executed by net
            if type(m)==nn.Linear:
                if do_gain:
                    m.weight.data *= prev_cap/MAX_ACTIVATION
                break
        return prev_cap

    #should be same for all classes
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
